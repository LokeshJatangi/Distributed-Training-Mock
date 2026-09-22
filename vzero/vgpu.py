"""A virtual GPU: a byte-accurate memory arena plus an identity.

The arena is the thing every memory number in this project rests on, so it is
deliberately boring. Two rules:

  1. Nothing is counted twice. Tensors are keyed by the base address of their
     untyped storage, so a .view() or .narrow() of a counted buffer adds zero
     bytes. Counting numel*itemsize per tensor instead would silently inflate
     every figure, and no assertion downstream would notice.

  2. Every allocation is named and phase-tagged, so a peak can be attributed to
     a line of code rather than to a mode.

Allocations are 64-byte aligned. That is physically realistic, and it matters
here for a subtler reason: CPU sgemm picks different kernels depending on
operand alignment, and this project asserts that ZeRO-0 and ZeRO-3 agree
bitwise. ZeRO-3 computes on freshly gathered buffers while ZeRO-0 computes on
views into a flat buffer, so those two have to be aligned identically or the
last bits can differ for reasons that have nothing to do with ZeRO.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch

Bucket = Literal["params", "grads", "opt", "act", "transient"]
BUCKETS: tuple[Bucket, ...] = ("params", "grads", "opt", "act", "transient")

ALIGN_BYTES = 64
ALIGN_ELEMS = ALIGN_BYTES // 4  # fp32


class VGPUOutOfMemory(RuntimeError):
    """Raised when an allocation would exceed a virtual GPU's capacity.

    This is what makes "DDP cannot train this model and ZeRO-3 can" a refused
    allocation with a traceback instead of a row in a spreadsheet.
    """

    def __init__(self, rank: int, requested: int, live: int, capacity: int,
                 bucket: str, name: str, phase: str) -> None:
        super().__init__(
            f"rank {rank}: allocating {requested:,} B for {bucket}/{name!r} "
            f"during phase {phase!r} would take live memory from {live:,} B to "
            f"{live + requested:,} B, over the {capacity:,} B capacity of this "
            f"virtual GPU"
        )
        self.rank, self.requested, self.live = rank, requested, live
        self.capacity, self.bucket, self.name, self.phase = capacity, bucket, name, phase


@dataclass
class AllocRecord:
    key: int
    nbytes: int
    bucket: Bucket
    name: str
    phase: str
    storage: object  # strong ref: see _charge



@dataclass
class MemSnapshot:
    phase: str
    step: int
    live_by_bucket: dict[str, int]
    live_total: int


class MemArena:
    """Per-rank memory ledger. Counts bytes, not objects."""

    def __init__(self, rank: int, capacity_bytes: int | None = None) -> None:
        self.rank = rank
        self.capacity_bytes = capacity_bytes
        self.phase = "init"
        self.step = -1

        self._live: dict[int, AllocRecord] = {}
        self.live_by_bucket: dict[str, int] = dict.fromkeys(BUCKETS, 0)
        self.peak_by_bucket: dict[str, int] = dict.fromkeys(BUCKETS, 0)
        self.peak_total = 0
        self.peak_total_phase = "init"
        self.peak_total_step = -1
        # The bucket breakdown AT THE INSTANT of the peak. Summing each bucket's
        # own peak instead would overstate the real peak, because the buckets do
        # not all peak at the same moment -- a stacked bar built that way adds up
        # to more memory than was ever simultaneously live.
        self.peak_breakdown: dict[str, int] = dict.fromkeys(BUCKETS, 0)

        self.n_allocs = 0
        self.n_frees = 0
        self.bytes_allocated = 0
        self.bytes_freed = 0
        self.timeline: list[MemSnapshot] = []

    # ---------------------------------------------------------------- phases

    def note_phase(self, phase: str) -> None:
        self.phase = phase

    def set_step(self, step: int) -> None:
        self.step = step

    def record(self) -> None:
        self.timeline.append(
            MemSnapshot(self.phase, self.step, dict(self.live_by_bucket), self.live_total)
        )

    # ------------------------------------------------------------ accounting

    @property
    def live_total(self) -> int:
        return sum(self.live_by_bucket.values())

    @staticmethod
    def key_of(t: torch.Tensor) -> int:
        return t.untyped_storage().data_ptr()

    def _charge(self, t: torch.Tensor, bucket: Bucket, name: str) -> None:
        """Charge a tensor's storage to this arena, once.

        The record keeps a STRONG reference to the untyped storage. Without it,
        a storage that is garbage collected frees its address, the allocator
        hands the same address to the next tensor, and `key in self._live`
        wrongly reports "already counted" -- silently under-counting from then
        on. Holding the storage makes address reuse impossible while counted,
        and turns a forgotten free() into a detectable leak (live_total does not
        return to baseline) rather than a wrong number.
        """
        store = t.untyped_storage()
        key, nbytes = store.data_ptr(), store.nbytes()
        if key in self._live:
            return  # a view of something we already own
        live = self.live_total
        if self.capacity_bytes is not None and live + nbytes > self.capacity_bytes:
            raise VGPUOutOfMemory(
                self.rank, nbytes, live, self.capacity_bytes, bucket, name, self.phase
            )
        self._live[key] = AllocRecord(key, nbytes, bucket, name, self.phase, store)
        self.live_by_bucket[bucket] += nbytes
        self.n_allocs += 1
        self.bytes_allocated += nbytes
        if self.live_by_bucket[bucket] > self.peak_by_bucket[bucket]:
            self.peak_by_bucket[bucket] = self.live_by_bucket[bucket]
        total = self.live_total
        if total > self.peak_total:
            self.peak_total = total
            self.peak_total_phase = self.phase
            self.peak_total_step = self.step
            self.peak_breakdown = dict(self.live_by_bucket)

    def alloc(self, numel: int, *, bucket: Bucket, name: str,
              dtype: torch.dtype = torch.float32, zero: bool = True) -> torch.Tensor:
        t = torch.zeros(numel, dtype=dtype) if zero else torch.empty(numel, dtype=dtype)
        assert t.untyped_storage().data_ptr() % ALIGN_BYTES == 0, (
            "torch CPU allocator is expected to return 64-byte aligned storage"
        )
        self._charge(t, bucket, name)
        return t

    def adopt(self, t: torch.Tensor, *, bucket: Bucket, name: str) -> torch.Tensor:
        """Count a tensor torch allocated for us (an activation, a grad)."""
        self._charge(t, bucket, name)
        return t

    def free(self, t: torch.Tensor | int) -> None:
        key = t if isinstance(t, int) else self.key_of(t)
        rec = self._live.pop(key, None)
        if rec is None:
            return
        self.live_by_bucket[rec.bucket] -= rec.nbytes
        self.n_frees += 1
        self.bytes_freed += rec.nbytes

    def owns(self, key: int) -> bool:
        return key in self._live

    def storage_keys(self) -> set[int]:
        return set(self._live)

    def bucket_of(self, key: int) -> str | None:
        rec = self._live.get(key)
        return rec.bucket if rec else None

    def report(self) -> dict:
        return {
            "rank": self.rank,
            "peak_total": self.peak_total,
            "peak_total_phase": self.peak_total_phase,
            "peak_by_bucket": dict(self.peak_by_bucket),
            "peak_breakdown": dict(self.peak_breakdown),
            "live_total": self.live_total,
            "n_allocs": self.n_allocs,
            "n_frees": self.n_frees,
            "bytes_allocated": self.bytes_allocated,
            "bytes_freed": self.bytes_freed,
        }


@dataclass
class VirtualGPU:
    rank: int
    world: int
    arena: MemArena = field(init=False)
    capacity_bytes: int | None = None

    def __post_init__(self) -> None:
        self.arena = MemArena(self.rank, self.capacity_bytes)

    def __repr__(self) -> str:
        return f"VirtualGPU(rank={self.rank}/{self.world}, peak={self.arena.peak_total:,}B)"
