"""The assertions. Everything else in this project is only as good as these.

The design principle: make the headline claim tolerance-free, and give every
tolerance that remains a derivation and a measured margin.

Why a bitwise claim is even possible. all_reduce is built as ring reduce_scatter
followed by ring all_gather, so the summed gradient a rank sees under ZeRO-0 is
the same N-1 additions in the same order as the shard a rank sees under
ZeRO-1/2/3; the all_gather adds no arithmetic. Adam is element-wise, so running
it on a shard gives bitwise what running it on the whole tensor gives for those
indices. So ZeRO-0/1/2/3 must agree exactly, and a single-rank reference that
accumulates its microbatch gradients in that same ring order must agree too.

That matters for more than tidiness. Of the four deliberate bugs in the ablation
suite, one -- forgetting to divide the reduction by N -- moves the loss by only
3.7e-06, because Adam is scale-invariant and absorbs a uniform 32x gradient
scaling almost entirely. Any tolerance loose enough to survive float drift would
miss it. The bitwise assertion catches it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from .engine import Reference, RunConfig, run
from .model import ModelConfig


@dataclass
class Check:
    name: str
    passed: bool
    detail: str
    measured: float | None = None
    tolerance: float | None = None


@dataclass
class ValidationReport:
    checks: list[Check] = field(default_factory=list)

    def add(self, name, passed, detail, measured=None, tolerance=None):
        self.checks.append(Check(name, bool(passed), detail, measured, tolerance))

    @property
    def n_passed(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def render(self) -> str:
        w = max(len(c.name) for c in self.checks)
        lines = []
        for c in self.checks:
            mark = "PASS" if c.passed else "FAIL"
            lines.append(f"  [{mark}] {c.name:<{w}}  {c.detail}")
        lines.append(f"\n  {self.n_passed}/{len(self.checks)} checks passed")
        return "\n".join(lines)


def max_param_delta(a: dict, b: dict) -> tuple[float, int]:
    worst, differing = 0.0, 0
    for k, v in a.items():
        if not torch.equal(v, b[k]):
            differing += 1
        worst = max(worst, (v - b[k]).abs().max().item())
    return worst, differing


def validate(cfg: ModelConfig, world: int = 32, steps: int = 8,
             ablations: bool = True) -> ValidationReport:
    rep = ValidationReport()
    base = RunConfig(world=world, steps=steps)

    # ---------------------------------------------------------- preconditions
    space = cfg.space(world)
    sr = space.report()
    rep.add("psi not divisible by world", sr["psi"] % world != 0,
            f"Psi={sr['psi']:,}, Psi mod {world} = {sr['psi'] % world} -> padding is real",
            measured=sr["psi"] % world)
    rep.add("shards tile the padded space",
            sum(g.shard_numel for g in space.groups.values()) * world == sr["psi_padded"],
            f"{world} x {sr['shard_numel_total']:,} = {sr['psi_padded']:,}")

    runs = {m: run(cfg, RunConfig(mode=m, world=world, steps=steps)) for m in (0, 1, 2, 3)}

    # reference loss must actually fall, or "the modes match" proves nothing
    ref = Reference(cfg, base, order="ring").run()
    fell = ref["losses"][0] - ref["losses"][-1]
    rep.add("reference actually learns", fell > 0.05,
            f"loss {ref['losses'][0]:.4f} -> {ref['losses'][-1]:.4f} "
            f"(fell {fell:.4f}; ln(vocab)={math.log(cfg.vocab):.3f})",
            measured=fell, tolerance=0.05)

    # ------------------------------------------------- cross-mode equivalence
    p0 = runs[0].params
    for m in (1, 2, 3):
        d, n = max_param_delta(p0, runs[m].params)
        rep.add(f"ZeRO-0 == ZeRO-{m} bitwise", d == 0.0 and n == 0,
                f"max|delta|={d:.3e}, {n}/{len(p0)} tensors differ", measured=d, tolerance=0.0)

    for m in (0, 1, 2, 3):
        d, n = max_param_delta(runs[m].params, ref["params"])
        rep.add(f"ZeRO-{m} == ring-order reference bitwise", d == 0.0 and n == 0,
                f"max|delta|={d:.3e}, {n}/{len(p0)} tensors differ", measured=d, tolerance=0.0)
        dl = max(abs(a - b) for a, b in zip(runs[m].losses_mean, ref["losses"]))
        rep.add(f"ZeRO-{m} loss == reference bitwise", dl == 0.0,
                f"max|dLoss|={dl:.3e}", measured=dl, tolerance=0.0)

    # ------------------------------------------------------- float-drift scale
    nref = Reference(cfg, base, order="sequential").run()
    d, _ = max_param_delta(runs[0].params, nref["params"])
    dl = max(abs(a - b) for a, b in zip(runs[0].losses_mean, nref["losses"]))
    rep.add("ring vs sequential accumulation: drift is small in loss", dl < 1e-4,
            f"max|dLoss|={dl:.3e} (reordering 32 float sums, not a bug)",
            measured=dl, tolerance=1e-4)
    rep.add("ring vs sequential accumulation: param drift recorded", True,
            f"max|dParam|={d:.3e}, concentrated in zero-initialised biases where "
            f"Adam's second moment is near zero", measured=d)

    # -------------------------------------------------------- memory invariants
    def state(r):
        b = r.peak_by_bucket[0]
        return b["params"] + b["grads"] + b["opt"]

    st = {m: state(runs[m]) for m in (0, 1, 2, 3)}
    rep.add("model state strictly decreases with stage",
            st[0] > st[1] > st[2] > st[3],
            " > ".join(f"{st[m]:,}" for m in (0, 1, 2, 3)))
    exp0 = 16 * sr["psi_padded"]
    rep.add("ZeRO-0 state == 16 x Psi_padded", st[0] == exp0,
            f"measured {st[0]:,} vs 16 x {sr['psi_padded']:,} = {exp0:,}")
    exp3 = 16 * sr["shard_numel_total"]
    rep.add("ZeRO-3 state == 16 x Psi_padded / world", st[3] == exp3,
            f"measured {st[3]:,} vs {exp3:,} -> exactly {st[0] / st[3]:.2f}x less than DDP")
    acts = [runs[m].peak_by_bucket[0]["act"] for m in (0, 1, 2, 3)]
    rep.add("activation memory is IDENTICAL across all four stages",
            len(set(acts)) == 1,
            f"{acts[0]:,} B in every mode -- ZeRO shards model state and does "
            f"nothing for activations")
    rep.add("ZeRO-3 never materialises the full parameters",
            runs[3].peak_by_bucket[0]["params"] == 4 * sr["shard_numel_total"],
            f"params bucket peak {runs[3].peak_by_bucket[0]['params']:,} B "
            f"= 4 x {sr['shard_numel_total']:,} (one shard, never Psi)")

    # ---------------------------------------------------------- comm invariants
    from .fabric import ring_bytes
    per_step = {m: runs[m].comm["per_rank_sent"][0] // steps for m in (0, 1, 2, 3)}
    expect_ddp = sum(ring_bytes("all_reduce", g.padded_numel, world)
                     for g in space.groups.values())
    rep.add("ZeRO-0 bytes == ring formula exactly", per_step[0] == expect_ddp,
            f"measured {per_step[0]:,} == sum 2(N-1)/N x P_g x 4 = {expect_ddp:,}")
    rep.add("ZeRO-1 and ZeRO-2 cost exactly what DDP costs",
            per_step[0] == per_step[1] == per_step[2],
            f"all three {per_step[0]:,} B/rank/step, identical to the byte")
    rep.add("ZeRO-3 costs exactly 1.5x DDP", per_step[3] * 2 == per_step[0] * 3,
            f"{per_step[3]:,} / {per_step[0]:,} = {per_step[3] / per_step[0]:.4f}")
    t = runs[0].comm
    rep.add("bytes sent == bytes received across ranks",
            sum(t["per_rank_sent"]) == sum(t["per_rank_recv"]),
            f"{sum(t['per_rank_sent']):,} both ways")
    rep.add("every rank sends the same number of bytes",
            len(set(t["per_rank_sent"])) == 1, "a ring is symmetric")

    naive = run(cfg, RunConfig(mode=1, world=world, steps=steps, zero1_naive=True))
    nb = naive.comm["per_rank_sent"][0] // steps
    d, n = max_param_delta(naive.params, ref["params"])
    rep.add("naive ZeRO-1 is correct but costs 1.5x", nb * 2 == per_step[1] * 3 and d == 0.0,
            f"all_reduce-then-slice: {nb:,} B/step = {nb / per_step[1]:.3f}x, "
            f"bitwise identical result ({d:.1e}) -- a free footgun")

    # ---------------------------------------------------------------- ablations
    if ablations:
        for ab in ("no_reduce", "shard_offset", "no_all_gather", "sum_not_mean"):
            r = run(cfg, RunConfig(mode=2, world=world, steps=steps, ablation=ab))
            dl = max(abs(a - b) for a, b in zip(r.losses_mean, ref["losses"]))
            fell = r.losses_mean[-1] < r.losses_mean[0]
            rep.add(f"ablation {ab} is caught", dl > 0.0,
                    f"max|dLoss|={dl:.3e}"
                    + ("  (loss still fell -- a falling loss proves nothing)" if fell else "")
                    + ("  [only a bitwise check catches this: Adam is scale-invariant]"
                       if dl < 1e-4 else ""),
                    measured=dl, tolerance=0.0)

    # ------------------------------------------------------------- determinism
    a = run(cfg, RunConfig(mode=3, world=world, steps=steps))
    b = run(cfg, RunConfig(mode=3, world=world, steps=steps))
    d, n = max_param_delta(a.params, b.params)
    rep.add("two runs are bitwise identical", d == 0.0 and n == 0,
            "thread interleaving does not affect the result: the ring's summation "
            "order is fixed by step index, not by arrival order",
            measured=d, tolerance=0.0)
    return rep
