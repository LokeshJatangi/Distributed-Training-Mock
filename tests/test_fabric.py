import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from vzero.fabric import Fabric, ring_bytes, accumulation_order, FabricError
from vzero.pool import run_workers, WorkerFailed

N = 32
P = 32 * 16 * 3          # divisible by N, multiple of ALIGN_ELEMS per chunk


def _contribs(world, numel, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(numel, generator=g) for _ in range(world)]


def test_accumulation_order_is_the_ring_order():
    # chunk c is summed starting at c+1, wrapping, with c adding last
    assert accumulation_order(0, 4) == [1, 2, 3, 0]
    assert accumulation_order(3, 4) == [0, 1, 2, 3]
    for c in range(N):
        o = accumulation_order(c, N)
        assert len(o) == N and len(set(o)) == N and o[-1] == c


def test_reduce_scatter_matches_ring_order_reference_bitwise():
    fab = Fabric(N)
    src = _contribs(N, P, seed=1)
    out = [torch.empty(P // N) for _ in range(N)]

    def work(r):
        fab.reduce_scatter(r, src[r], out[r], key="k", op="sum")

    run_workers(N, work, fab)
    C = P // N
    for r in range(N):
        ref = src[accumulation_order(r, N)[0]].narrow(0, r * C, C).clone()
        for s in accumulation_order(r, N)[1:]:
            ref = ref + src[s].narrow(0, r * C, C)
        assert torch.equal(out[r], ref), f"rank {r} not bitwise equal to ring order"


def test_all_reduce_equals_reduce_scatter_then_all_gather_bitwise():
    """The property the whole correctness argument rests on."""
    src = _contribs(N, P, seed=2)

    fab1 = Fabric(N)
    ar = [src[r].clone() for r in range(N)]
    run_workers(N, lambda r: fab1.all_reduce(r, ar[r], key="a", op="mean"), fab1)

    fab2 = Fabric(N)
    sh = [torch.empty(P // N) for _ in range(N)]
    full = [torch.empty(P) for _ in range(N)]

    def work(r):
        fab2.reduce_scatter(r, src[r].clone(), sh[r], key="rs", op="mean")
        fab2.all_gather(r, sh[r], full[r], key="ag")

    run_workers(N, work, fab2)
    for r in range(N):
        assert torch.equal(ar[r], full[r]), f"rank {r}: all_reduce != RS+AG bitwise"
    # and every rank agrees with every other
    for r in range(1, N):
        assert torch.equal(ar[0], ar[r])


def test_ring_impl_and_direct_impl_agree_bitwise_and_on_bytes():
    src = _contribs(N, P, seed=3)
    res = {}
    for impl in ("direct", "ring"):
        fab = Fabric(N, impl=impl)
        sh = [torch.empty(P // N) for _ in range(N)]
        full = [torch.empty(P) for _ in range(N)]

        def work(r, fab=fab, sh=sh, full=full):
            fab.reduce_scatter(r, src[r].clone(), sh[r], key="rs", op="mean")
            fab.all_gather(r, sh[r], full[r], key="ag")

        run_workers(N, work, fab)
        res[impl] = (full, fab.totals())
    for r in range(N):
        assert torch.equal(res["direct"][0][r], res["ring"][0][r]), f"rank {r} bits differ"
    assert res["direct"][1]["per_rank_sent"] == res["ring"][1]["per_rank_sent"], (
        "the cheap path's byte counts are not the real ring's"
    )


def test_byte_counters_equal_the_padding_aware_formula_exactly():
    fab = Fabric(N)
    src = _contribs(N, P, seed=4)
    sh = [torch.empty(P // N) for _ in range(N)]
    full = [torch.empty(P) for _ in range(N)]

    def work(r):
        fab.reduce_scatter(r, src[r].clone(), sh[r], key="rs", op="mean")
        fab.all_gather(r, sh[r], full[r], key="ag")

    run_workers(N, work, fab)
    expect = ring_bytes("reduce_scatter", P, N) + ring_bytes("all_gather", P, N)
    assert expect == ring_bytes("all_reduce", P, N)          # RS+AG == all_reduce
    for r in range(N):
        assert fab.stats[r].bytes_sent == expect, (r, fab.stats[r].bytes_sent, expect)
    t = fab.totals()
    assert sum(t["per_rank_sent"]) == sum(t["per_rank_recv"])  # conservation
    assert len(set(t["per_rank_sent"])) == 1                   # ring is symmetric


def test_divergent_ranks_raise_instead_of_hanging():
    """Rank 5 issues a different key. Must fail fast with a diagnosis."""
    fab = Fabric(N, timeout_s=3.0)
    buf = [torch.ones(P) for _ in range(N)]

    def work(r):
        fab.all_reduce(r, buf[r], key="same" if r != 5 else "different")

    with pytest.raises(WorkerFailed) as ei:
        run_workers(N, work, fab, timeout_s=60)
    msg = str(ei.value)
    assert "timed out" in msg or "Aborted" in msg or "aborted" in msg, msg


def test_one_rank_raising_aborts_all_of_them():
    fab = Fabric(N, timeout_s=30.0)
    buf = [torch.ones(P) for _ in range(N)]

    def work(r):
        if r == 7:
            raise ValueError("deliberate")
        fab.all_reduce(r, buf[r], key="k")

    import time
    t0 = time.time()
    with pytest.raises(WorkerFailed):
        run_workers(N, work, fab, timeout_s=60)
    assert time.time() - t0 < 10, "abort did not propagate; ranks waited out the timeout"


def test_shape_mismatch_is_caught_not_corrupted():
    fab = Fabric(N, timeout_s=3.0)

    def work(r):
        n = P if r != 3 else P + 32
        fab.all_reduce(r, torch.ones(n), key="k")

    with pytest.raises(WorkerFailed):
        run_workers(N, work, fab, timeout_s=60)
