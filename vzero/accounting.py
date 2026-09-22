"""Measuring activation memory, rather than estimating it.

Activations are the segment of the memory bar that ZeRO does NOT shard, so
guessing them would hide the most important caveat in the project. These get
measured with saved_tensors_hooks, with three details that each cost a bug if
skipped:

  * pack() returns t.detach(), never t. The docs are explicit that the return
    value must not hold a reference to the input -- doing so builds a reference
    cycle and leaks.
  * tensors are deduplicated by untyped_storage().data_ptr(). F.linear saves
    several tensors that share storage with things we already own, and counting
    numel*itemsize per saved tensor inflates the total with no assertion able to
    catch it.
  * weights are classified OUT. F.linear saves both its input and its weight; the
    weight already lives in the params bucket, so counting it as an activation
    would double-count it. It is reported separately as param_ref_bytes: "bytes
    autograd pins inside the parameter buffer".

torch.utils.checkpoint is banned in this codebase. Its non-reentrant path
installs its own saved_tensors_hooks, and the docs state that only the innermost
pair applies -- so it would silently shadow this tracker and report activation
memory as zero.
"""

from __future__ import annotations

import torch

from .vgpu import MemArena


class _Packed:
    """What we hand autograd in place of a saved tensor.

    Holds a detached alias (same storage, no reference cycle). Its death is what
    tells the arena the activation is gone, so live bytes fall at graph teardown
    and the peak is a true watermark rather than a running total.

    Release is via __del__ rather than weakref.finalize. That is not a style
    choice: finalize costs roughly 0.3 ms per call here, and with ~112 saved
    tensors per forward it was two thirds of the entire step time. There is no
    reference cycle to worry about -- _Packed holds a tensor and an int, and the
    arena does not hold _Packed.
    """

    __slots__ = ("t", "key", "_free")

    def __init__(self, t: torch.Tensor, key: int, free) -> None:
        self.t = t
        self.key = key
        self._free = free

    def __del__(self) -> None:
        try:
            self._free(self.key)
        except Exception:
            pass


class ActivationTracker(torch.autograd.graph.saved_tensors_hooks):
    """Per-rank. saved_tensors_hooks scope is thread-local, so one per worker."""

    def __init__(self, arena: MemArena) -> None:
        self.arena = arena
        self.param_ref_bytes = 0
        self.unique_storages = 0
        self.duplicate_saves = 0
        self.n_saved = 0
        super().__init__(self._pack, self._unpack)

    def _pack(self, t: torch.Tensor):
        self.n_saved += 1
        d = t.detach()
        store = d.untyped_storage()
        key = store.data_ptr()
        if self.arena.owns(key):
            bucket = self.arena.bucket_of(key)
            if bucket in ("params", "grads", "opt"):
                # autograd pinning a weight, not an activation
                self.param_ref_bytes += store.nbytes()
                return d
            self.duplicate_saves += 1
            return d
        self.arena.adopt(d, bucket="act", name="saved")
        self.unique_storages += 1
        return _Packed(d, key, self.arena.free)

    @staticmethod
    def _unpack(p):
        return p.t if isinstance(p, _Packed) else p

    def report(self) -> dict:
        return {
            "n_saved": self.n_saved,
            "unique_storages": self.unique_storages,
            "duplicate_saves": self.duplicate_saves,
            "param_ref_bytes": self.param_ref_bytes,
            "peak_act_bytes": self.arena.peak_by_bucket["act"],
        }
