"""Hand-written collectives over 32 worker threads.

Two things in here are load-bearing for the whole project.

THE ACCUMULATION ORDER IS PART OF THE CONTRACT
----------------------------------------------
Float addition is not associative, so the order in which 32 contributions are
summed decides the last bits. In a ring reduce-scatter the order is fixed by the
algorithm: chunk c is first sent by rank c+1, each rank it passes through adds
its own contribution, and rank c adds its own last. So for chunk c the order is

    contrib[c+1] + contrib[c+2] + ... + contrib[c+N-1] + contrib[c]

Both implementations below reproduce exactly that order, so they agree bitwise
with each other -- and, more importantly, all_reduce is built AS reduce_scatter
followed by all_gather, which means the summed gradient a rank sees under ZeRO-0
is bitwise the value it sees under ZeRO-1/2/3. That is what turns the
correctness claim from "the curves look close" into an equality.

    impl="direct"  one rendezvous, each rank sums its own chunk in ring order.
                   Same arithmetic as the ring, far fewer barriers. Byte counts
                   come from the ring's exact formula.
    impl="ring"    the real N-1 step pass-around, counting every copy_ it does.
                   Slow (31 barriers per collective), used to prove that the
                   direct path's byte counts and bits are the ring's.

DIVERGENCE FAILS LOUDLY
-----------------------
A rendezvous is keyed by (collective, key, step). Two ranks that reach different
collectives land in different slots, so they can never exchange the wrong data --
the failure becomes a timeout with a diagnosis, never silent corruption. One
rank raising aborts all 32 in milliseconds instead of 32 sequential timeouts.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import torch

ITEMSIZE = 4  # fp32 throughout


class FabricError(RuntimeError):
    pass


class FabricTimeout(FabricError):
    pass


class FabricMismatch(FabricError):
    pass


class FabricAborted(FabricError):
    pass


@dataclass
class CommStats:
    """Per-rank counters. Convention: bytes_sent is BYTES LEAVING THIS RANK."""

    bytes_sent: int = 0
    bytes_recv: int = 0
    n_collectives: int = 0
    pad_bytes_sent: int = 0
    by_coll: dict[str, list[int]] = field(default_factory=dict)  # coll -> [sent, recv, n]

    def add(self, coll: str, sent: int, recv: int, pad: int = 0) -> None:
        self.bytes_sent += sent
        self.bytes_recv += recv
        self.pad_bytes_sent += pad
        self.n_collectives += 1
        e = self.by_coll.setdefault(coll, [0, 0, 0])
        e[0] += sent
        e[1] += recv
        e[2] += 1

    def snapshot(self) -> dict:
        return {
            "bytes_sent": self.bytes_sent,
            "bytes_recv": self.bytes_recv,
            "n_collectives": self.n_collectives,
            "pad_bytes_sent": self.pad_bytes_sent,
            "by_coll": {k: list(v) for k, v in self.by_coll.items()},
        }


def ring_bytes(coll: str, padded_numel: int, world: int) -> int:
    """Bytes leaving one rank for one collective. Padding-aware and exact.

    A ring splits the payload into `world` chunks and every rank sends one chunk
    per step. reduce_scatter and all_gather are N-1 steps each; all_reduce is
    both phases back to back.
    """
    chunk = padded_numel // world
    per_phase = (world - 1) * chunk * ITEMSIZE
    if coll in ("reduce_scatter", "all_gather"):
        return per_phase
    if coll == "all_reduce":
        return 2 * per_phase
    if coll == "broadcast":
        return padded_numel * ITEMSIZE
    raise KeyError(coll)


def accumulation_order(chunk: int, world: int) -> list[int]:
    """Ring order in which chunk `chunk` is summed: c+1, c+2, ..., c+N-1, c."""
    return [(chunk + 1 + i) % world for i in range(world - 1)] + [chunk]


class _Slot:
    __slots__ = ("cv", "arrived", "departed", "payload", "sig", "ready", "done")

    def __init__(self, world: int) -> None:
        self.cv = threading.Condition()
        self.arrived = 0
        self.departed = 0
        self.payload: list = [None] * world
        self.sig: tuple | None = None
        self.ready = False
        self.done = False


class Fabric:
    def __init__(self, world: int, *, timeout_s: float = 30.0,
                 impl: str = "direct") -> None:
        assert impl in ("direct", "ring")
        self.world = world
        self.timeout_s = timeout_s
        self.impl = impl
        self.stats = [CommStats() for _ in range(world)]
        # Set by the engine so a collective charges its caller for the one chunk
        # a real ring holds in flight.
        self.arena_hook = None
        self.arena_free = None

        self._lock = threading.Lock()
        self._slots: dict[tuple, _Slot] = {}
        # The step is THREAD-LOCAL. Every worker runs all of its steps on one
        # thread, so a global step counter would flap while ranks are briefly at
        # different steps, and rendezvous keys would stop matching. Thread-local
        # keeps each rank's key consistent while still agreeing across ranks that
        # are on the same step.
        self._tl = threading.local()
        self._aborted: BaseException | None = None
        self._abort_rank: int | None = None
        self._waiting: dict[int, tuple] = {}
        self._last_done: dict[int, tuple] = {}

    # --------------------------------------------------------------- control

    @property
    def _step(self) -> int:
        return getattr(self._tl, "step", 0)

    def set_step(self, step: int) -> None:
        self._tl.step = step

    def abort(self, rank: int, exc: BaseException) -> None:
        with self._lock:
            if self._aborted is None:
                self._aborted, self._abort_rank = exc, rank
            slots = list(self._slots.values())
        for s in slots:
            with s.cv:
                s.cv.notify_all()

    def _check_abort(self) -> None:
        if self._aborted is not None:
            raise FabricAborted(
                f"aborted because rank {self._abort_rank} raised "
                f"{type(self._aborted).__name__}: {self._aborted}"
            )

    def _diagnose(self, rk: tuple, slot: _Slot) -> str:
        missing = [r for r in range(self.world) if slot.payload[r] is None]
        lines = [
            f"collective rendezvous timed out after {self.timeout_s}s.",
            f"  waiting on {rk}, arrived {slot.arrived}/{self.world} ranks",
            "  missing ranks and what they are doing instead:",
        ]
        for r in missing[:8]:
            w = self._waiting.get(r)
            if w is None:
                lines.append(f"    rank {r} -> not in any collective; "
                             f"last completed {self._last_done.get(r)}")
            else:
                lines.append(f"    rank {r} -> waiting on {w}")
        lines.append(
            "  DIAGNOSIS: ranks executed different collective sequences. Every "
            "rank must issue the same (collective, key) in the same order; the "
            "bucket drain must follow the fixed schedule, not gradient arrival "
            "order."
        )
        return "\n".join(lines)

    # ------------------------------------------------------------ rendezvous

    def _rendezvous(self, rank: int, coll: str, key: str, payload, sig: tuple) -> list:
        rk = (coll, key, self._step)
        with self._lock:
            self._check_abort()
            slot = self._slots.get(rk)
            if slot is None:
                slot = self._slots[rk] = _Slot(self.world)
            if slot.sig is None:
                slot.sig = sig
            elif slot.sig != sig:
                raise FabricMismatch(
                    f"rank {rank} joined {rk} with {sig}, peers used {slot.sig}"
                )
            slot.payload[rank] = payload
            slot.arrived += 1
            self._waiting[rank] = rk
            last = slot.arrived == self.world
        with slot.cv:
            if last:
                slot.ready = True
                slot.cv.notify_all()
            elif not slot.cv.wait_for(
                lambda: slot.ready or self._aborted is not None, timeout=self.timeout_s
            ):
                raise FabricTimeout(self._diagnose(rk, slot))
        self._check_abort()
        return slot.payload

    def _release(self, rank: int, slot_key: tuple) -> None:
        """Non-blocking slot teardown.

        There is deliberately no second barrier here. Ranks publish a CLONE of
        their send buffer, so a fast rank returning and overwriting its own
        gradient buffer cannot corrupt what a slow peer is still reading -- which
        is what a departure barrier would otherwise be protecting against. One
        barrier per collective instead of two roughly halves the cost of a step,
        and with 32 threads under the GIL the barriers, not the arithmetic, are
        what a step costs.

        The clone is a simulator artifact and is NOT charged to a virtual GPU: a
        real ring holds one chunk in flight, not a whole copy of the payload, and
        that single chunk is what gets charged.
        """
        with self._lock:
            slot = self._slots.get(slot_key)
            if slot is not None:
                slot.departed += 1
                if slot.departed == self.world:
                    self._slots.pop(slot_key, None)
            self._waiting.pop(rank, None)
            self._last_done[rank] = slot_key

    # ------------------------------------------------------------ collectives

    def barrier(self, rank: int, key: str = "b") -> None:
        rk = ("barrier", key, self._step)
        self._rendezvous(rank, "barrier", key, True, ())
        self._release(rank, rk)

    def reduce_scatter(self, rank: int, send: torch.Tensor, recv: torch.Tensor,
                       key: str, op: str = "mean", pad_numel: int = 0) -> None:
        """send: padded_numel on every rank -> recv: padded_numel/world on rank r,
        holding the reduction of chunk r."""
        N = self.world
        P = send.numel()
        assert P % N == 0, f"payload {P} not divisible by world {N}"
        C = P // N
        assert recv.numel() == C, (recv.numel(), C)
        if self.impl == "ring":
            return self._reduce_scatter_ring(rank, send, recv, key, op, pad_numel)
        rk = ("reduce_scatter", key, self._step)
        parts = self._rendezvous(rank, "reduce_scatter", key, send.clone(),
                                 (P, str(send.dtype)))
        stage = self.arena_hook(rank, C) if self.arena_hook else None

        order = accumulation_order(rank, N)
        recv.copy_(parts[order[0]].narrow(0, rank * C, C))
        for src in order[1:]:
            recv.add_(parts[src].narrow(0, rank * C, C))
        if op == "mean":
            recv.div_(N)
        self._release(rank, rk)
        if stage is not None:
            self.arena_free(rank, stage)

        b = ring_bytes("reduce_scatter", P, N)
        pb = (pad_numel * ITEMSIZE * (N - 1)) // N if pad_numel else 0
        self.stats[rank].add("reduce_scatter", b, b, pb)

    def _reduce_scatter_ring(self, rank: int, send: torch.Tensor, recv: torch.Tensor,
                             key: str, op: str, pad_numel: int) -> None:
        """The real N-1 step pass-around, counting every copy it performs.

        Rank r sends chunk (r-1-k) mod N at step k, and adds what arrives from
        rank r-1 to its own copy of that chunk. After N-1 steps rank r holds the
        complete sum of chunk r. The byte count here comes from the copies
        actually made, which is what lets us prove the cheap path's counters are
        the ring's and not a formula asserting itself.
        """
        N = self.world
        C = send.numel() // N
        work = send.clone()
        st = self.stats[rank]
        sent_total = 0
        for k in range(N - 1):
            idx = (rank - 1 - k) % N
            outgoing = work.narrow(0, idx * C, C).clone()
            inbox = self._rendezvous(
                rank, "ring_rs", f"{key}/{k}", outgoing, (C, str(send.dtype))
            )
            incoming = inbox[(rank - 1) % N]          # from my left neighbour
            recv_idx = (rank - 2 - k) % N
            work.narrow(0, recv_idx * C, C).add_(incoming)
            self._release(rank, ("ring_rs", f"{key}/{k}", self._step))
            sent_total += C * ITEMSIZE
        recv.copy_(work.narrow(0, rank * C, C))
        if op == "mean":
            recv.div_(N)
        pb = (pad_numel * ITEMSIZE * (N - 1)) // N if pad_numel else 0
        st.add("reduce_scatter", sent_total, sent_total, pb)

    def _all_gather_ring(self, rank: int, send: torch.Tensor, recv: torch.Tensor,
                         key: str, pad_numel: int) -> None:
        """Real ring all-gather: N-1 hops, each rank relaying one chunk per step."""
        N = self.world
        C = send.numel()
        recv.narrow(0, rank * C, C).copy_(send)
        sent_total = 0
        for k in range(N - 1):
            idx = (rank - k) % N
            outgoing = recv.narrow(0, idx * C, C).clone()
            inbox = self._rendezvous(
                rank, "ring_ag", f"{key}/{k}", outgoing, (C, str(send.dtype))
            )
            incoming = inbox[(rank - 1) % N]
            recv.narrow(0, ((rank - k - 1) % N) * C, C).copy_(incoming)
            self._release(rank, ("ring_ag", f"{key}/{k}", self._step))
            sent_total += C * ITEMSIZE
        pb = (pad_numel * ITEMSIZE * (N - 1)) // N if pad_numel else 0
        self.stats[rank].add("all_gather", sent_total, sent_total, pb)

    def all_gather(self, rank: int, send: torch.Tensor, recv: torch.Tensor,
                   key: str, pad_numel: int = 0) -> None:
        """send: shard of rank r -> recv: the full padded buffer on every rank."""
        N = self.world
        C = send.numel()
        assert recv.numel() == C * N, (recv.numel(), C, N)
        if self.impl == "ring":
            return self._all_gather_ring(rank, send, recv, key, pad_numel)
        rk = ("all_gather", key, self._step)
        # Write our own slice BEFORE publishing. In ZeRO-1/2 the send buffer is a
        # view of recv, so writing our own slice after peers can read it would be
        # a write to memory another rank is reading. Doing it first, then copying
        # only the other ranks' chunks, means our slice is never touched while it
        # is visible.
        recv.narrow(0, rank * C, C).copy_(send)
        parts = self._rendezvous(rank, "all_gather", key, send.clone(),
                                 (C, str(send.dtype)))
        stage = self.arena_hook(rank, C) if self.arena_hook else None
        for r in range(N):
            if r != rank:
                recv.narrow(0, r * C, C).copy_(parts[r])
        self._release(rank, rk)
        if stage is not None:
            self.arena_free(rank, stage)

        b = ring_bytes("all_gather", C * N, N)
        pb = (pad_numel * ITEMSIZE * (N - 1)) // N if pad_numel else 0
        self.stats[rank].add("all_gather", b, b, pb)

    def all_reduce(self, rank: int, buf: torch.Tensor, key: str,
                   op: str = "mean", pad_numel: int = 0) -> None:
        """Built as reduce_scatter + all_gather, deliberately.

        This is the textbook bandwidth-optimal ring all_reduce, and it is also
        what makes ZeRO-0 bitwise comparable to ZeRO-1/2/3: the reduce-scatter
        phase computes exactly the per-chunk sums that ZeRO-1/2/3 compute, and
        the all-gather phase is a pure copy that adds no arithmetic.
        """
        N = self.world
        P = buf.numel()
        C = P // N
        tmp = torch.empty(C, dtype=buf.dtype)
        self.reduce_scatter(rank, buf, tmp, key=f"{key}/rs", op=op, pad_numel=pad_numel)
        self.all_gather(rank, tmp, buf, key=f"{key}/ag", pad_numel=pad_numel)

    def broadcast(self, rank: int, buf: torch.Tensor, src: int, key: str) -> None:
        rk = ("broadcast", key, self._step)
        parts = self._rendezvous(rank, "broadcast", key, buf.clone(),
                                 (buf.numel(), str(buf.dtype)))
        if rank != src:
            buf.copy_(parts[src])
        self._release(rank, rk)
        if rank == src:
            self.stats[rank].add("broadcast", buf.numel() * ITEMSIZE * (self.world - 1), 0)
        else:
            self.stats[rank].add("broadcast", 0, buf.numel() * ITEMSIZE)

    # ---------------------------------------------------------------- report

    def totals(self) -> dict:
        return {
            "per_rank_sent": [s.bytes_sent for s in self.stats],
            "per_rank_recv": [s.bytes_recv for s in self.stats],
            "per_rank_n": [s.n_collectives for s in self.stats],
            "pad_bytes_sent": [s.pad_bytes_sent for s in self.stats],
        }

    def reset_stats(self) -> None:
        self.stats = [CommStats() for _ in range(self.world)]
