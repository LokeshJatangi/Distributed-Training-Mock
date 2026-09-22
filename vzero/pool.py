"""Run one function on each of `world` worker threads.

Threads, not processes, for two reasons: 32 processes each importing torch costs
~8-11 GB of RSS on this machine, and per-rank memory would then have to be read
out of noisy RSS instead of counted exactly. Threads let the arena count bytes
precisely, which is the whole point.

What threads cost us is honesty about time: the GIL means this is concurrency,
not 32-way parallelism. Nothing in this project claims a speedup.

The thread design is safe for a specific reason worth recording: for a CPU-only
graph PyTorch's autograd engine runs backward on the thread that called
.backward(), and saved_tensors_hooks scope is thread-local. So per-rank
attribution holds in both passes, and collectives fired from backward hooks stay
on the right rank.
"""

from __future__ import annotations

import threading


class WorkerFailed(RuntimeError):
    pass


def run_workers(world: int, fn, fabric=None, timeout_s: float = 300.0) -> list:
    """fn(rank) -> result, run on `world` threads. First exception wins."""
    results: list = [None] * world
    errors: list = [None] * world

    def body(rank: int) -> None:
        try:
            results[rank] = fn(rank)
        except BaseException as exc:  # noqa: BLE001 - we re-raise on the main thread
            errors[rank] = exc
            if fabric is not None:
                fabric.abort(rank, exc)

    threads = [threading.Thread(target=body, args=(r,), name=f"vgpu{r}", daemon=True)
               for r in range(world)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout_s)

    stuck = [t.name for t in threads if t.is_alive()]
    first = next((e for e in errors if e is not None), None)
    if first is not None:
        raise WorkerFailed(
            f"rank {errors.index(first)} raised {type(first).__name__}: {first}"
        ) from first
    if stuck:
        raise WorkerFailed(f"threads still alive after {timeout_s}s: {stuck}")
    return results
