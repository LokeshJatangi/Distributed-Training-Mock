# Implementation notes

Build notes for anyone reading or extending `vzero/`. These are the PyTorch behaviours that
forced specific design choices, and the measurement traps that would have produced
wrong-but-plausible numbers. The [README](README.md) is the explainer; this is the workshop.

---

## 1. Counting memory correctly

**A `.view()` is not new memory.** Counting `numel × itemsize` per tensor double-counts every
view and slice, because a view shares storage with its base. The arena keys tensors on
`untyped_storage().data_ptr()` instead, so a view costs zero bytes.

This matters more than it sounds: an inflated ledger inflates *all four modes together*, so no
comparison between them would look wrong. Nothing downstream could have caught it.

**Addresses get recycled.** Keying on an address is not enough on its own. Free a storage, and
the allocator hands the same address to the next tensor — at which point `key in live` reads as
"already counted" and the ledger silently undercounts from then on.

Fix: the arena holds a **strong reference** to each storage it charges. Address reuse then
becomes impossible while counted, and a forgotten `free()` shows up as a leak rather than as a
wrong number. The test is four lines with a `gc.collect()` between two allocations
(`tests/test_arena.py::test_freed_address_cannot_alias_a_live_record`).

**Peaks don't happen at the same time.** A stacked bar built from each bucket's own peak adds up
to more memory than was ever simultaneously live — it overstated ZeRO-2 by 200,704 bytes and
ZeRO-3 by 281,860. The arena records the full breakdown at the *instant* of peak total, and a
test asserts the bars sum to the measured peak.

---

## 2. Making ZeRO-3 actually free its weights

**The problem.** ZeRO-3's premise is that a rank holds 1/N of a weight and materialises the rest
only briefly. But `F.linear` saves its weight tensor for backward, so dropping your reference to
a gathered buffer frees nothing at all.

**The fix, which is what real FSDP does.** Resize the underlying storage to **zero bytes** after
the forward pass, then resize it back and refill before backward needs it. Saved views follow
the storage.

```python
self.full_d.untyped_storage().resize_(0)          # genuinely freed
...
self.full_d.untyped_storage().resize_(self._nbytes)  # and back, before backward
```

This is why ZeRO-3 here can use ordinary `F.linear`, `F.embedding` and `F.layer_norm` rather
than hand-written backward passes — which in turn is why it comes out **bitwise** identical to
the other three modes rather than merely close to them.

**The follow-on problem.** Writing the re-gathered weights into that storage trips autograd's
version counter:

```
one of the variables needed for gradient computation has been modified
by an inplace operation
```

Autograd saved views of that storage during forward and recorded the version it saw.

**The escape hatch**, which is not something invented here — FSDP2 wraps its own all-gather
copy-out in exactly the same context manager, in
`torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py`:

```python
with torch.autograd._unsafe_preserve_version_counter(self.full_d):
    ...
```

---

## 3. Why the model is functional, not an `nn.Module`

With an `nn.Module` the parameters always exist, so ZeRO-3's claim can't be demonstrated
honestly. Two alternatives were considered and rejected:

| approach | why not |
|---|---|
| swap `param.data` | invisible to autograd's version counter, and 32 `Parameter` objects still exist per rank |
| `torch.func.functional_call` | swaps tensors into the shared module object and restores in a `finally` — a data race across 32 threads. And 32 module copies would materialise 32 × Ψ parameters, which is exactly what ZeRO-3 denies |

So: `forward(cfg, ParamAccess, ids)`, where every weight access goes through a swappable object.
One forward pass, four modes.

---

## 4. Why all four modes share one grouping

Float addition is not associative, so the **order** in which N contributions are summed is part
of the answer.

In a ring reduce-scatter that order is fixed by the algorithm: chunk `c` is first sent by rank
`c+1`, each rank it passes through adds its own contribution, and rank `c` adds last.

Which chunk an element lands in depends on the bucketing. Change the grouping between modes and
you change the summation order, and the bit-exact comparison becomes impossible. All four modes
therefore use identical per-group buckets.

The same reasoning is why `all_reduce` is **built as** reduce-scatter + all-gather rather than
as its own algorithm: it makes ZeRO-0's summed gradient the same additions in the same order as
ZeRO-1/2/3's shard, and the all-gather contributes no arithmetic.

---

## 5. Why bit-exact was worth the effort

It looked like tidiness. It turned out to buy real discriminating power.

One of the four planted bugs — reducing with `sum` instead of `mean`, i.e. forgetting to divide
by N — shifts the loss by only **5.007e-06**. Adam normalises by the gradient's own magnitude,
so scaling every gradient by 32 almost entirely cancels; the only trace is ε becoming
effectively ε/32.

Any tolerance loose enough to survive honest float drift would wave that bug straight through.
The bit-exact assertion catches it.

---

## 6. Float drift, and where it actually lives

The bit-exact comparison is against a reference that accumulates in the same **ring order**. A
second reference accumulates in plain rank order — genuinely independent — and the difference
between the two measures how much drift the reordering causes.

- loss differs by **5.96e-08**
- individual parameters by up to **2.5e-04**

That gap looked alarming until the parameters were inspected: 0.1% of elements, concentrated in
`attn.bqkv` — biases initialised to exactly zero, where Adam's second moment is near zero, so ε
dominates the denominator and tiny reordering differences get amplified.

Not a bug, and not something a single flat parameter tolerance would have described honestly.

---

## 7. Model configuration choices

| choice | reason |
|---|---|
| vocab 509, `d_model` 64 | makes Ψ = 269,821, **not** a multiple of 32, so padding is real. A test guards this — a tidier config would delete the lesson |
| untied output head | a tied head is used twice in forward, so a post-accumulate grad hook fires twice on a half-accumulated gradient. Real FSDP handles this with an explicit counter; untying sidesteps it |
| group sizes 128 → 50,176 | a 392× spread, so ZeRO-3's transient buffer is driven by the largest group rather than the average |

---

## 8. Performance of the simulator itself

Not a result about ZeRO — a note for anyone extending this.

The activation tracker fires a Python callback per saved tensor: 112 per forward, times 32
ranks. `weakref.finalize` costs roughly 0.3 ms per call, which made measurement two thirds of
the total step time.

Replacing it with `__del__` on the wrapper object, and tracking on one rank by default
(activation memory is identical across ranks — asserted, not assumed), took a step from
**1595 ms → 524 ms**.

The remaining gap between that and the 89 ms the same arithmetic takes single-threaded is GIL
contention. Which is exactly why no timing in this project is presented as a result.

---

## 9. Deadlock safety

32 threads issuing collectives can deadlock if any rank reaches a different collective than its
peers. Mitigations:

- **Rendezvous keyed by `(collective, key, step)`.** Two ranks at different collectives land in
  different slots and can never exchange wrong data. Divergence becomes a timeout, never silent
  corruption.
- **A shape/dtype signature check** at the first arriving pair, which catches the most common
  real bug (wrong slice length) immediately.
- **Abort propagation:** one rank raising fails all 32 in milliseconds rather than 32 sequential
  timeouts.
- **A diagnosis message** naming every missing rank and what it is waiting for instead.

The timeout path is itself tested — `tests/test_fabric.py` deliberately sends one rank to a
different key and asserts it raises rather than hanging.

---

## 10. Cross-check scope

`scripts/torch_crosscheck.py` validates **numerics only**, deliberately.

gloo has no native reduce-scatter. `ProcessGroupGloo` implements `reduce_scatter_single` by
cloning the input, running a full all-reduce, and copying out this rank's chunk. So a gloo
reduce-scatter moves 2(N−1)/N bytes where the algorithm wants (N−1)/N, and real FSDP on gloo
moves roughly 4(N−1)/N per step rather than 3.

Instrumenting torch's byte counters and comparing them against the 1.5× result would look like a
refutation of a derivation that is correct. Every communication number in this project comes
from `vzero/fabric.py`, which implements a real ring and is asserted byte-identical to its own
fast path.
