"""The arithmetic, separated from the measurement.

Everything here is a closed form. The point of keeping it in its own module is
that the notebook can put a measured number next to a predicted one and assert
they agree, rather than presenting a formula as if it were evidence.

Two accounting regimes, and conflating them is the most common way write-ups
about ZeRO go wrong:

  fp32 + Adam (what this simulator runs)
      params 4Psi + grads 4Psi + Adam m,v 8Psi = 16Psi
  mixed precision (what the ZeRO paper tabulates), K = 12
      fp16 params 2Psi + fp16 grads 2Psi + (fp32 master + m + v) 12Psi = 16Psi

Both total 16Psi, which makes them look interchangeable. They are not: the
SHARDABLE FRACTION at stage 1 is 8/16 in the first and 12/16 in the second. So
the same ZeRO-1 saves 1.94x in fp32 and 3.66x in mixed precision. My measured
numbers are the weaker column, and the README says so rather than quoting the
paper's.
"""

from __future__ import annotations

from dataclasses import dataclass

GB = 1e9
GiB = 1 << 30

REGIMES = {
    # name: (param bytes, grad bytes, optimizer bytes) per parameter
    "fp32_adam": (4, 4, 8),
    "mixed_paper": (2, 2, 12),
}
STAGES = (0, 1, 2, 3)
STAGE_NAMES = {0: "ZeRO-0 (DDP)", 1: "ZeRO-1", 2: "ZeRO-2", 3: "ZeRO-3"}


def state_bytes_per_gpu(stage: int, psi: float, world: int,
                        regime: str = "fp32_adam") -> float:
    """Model-state bytes one GPU holds. Activations are NOT included -- they do
    not shard under any ZeRO stage, which is the whole reason they are a separate
    term everywhere in this project."""
    p, g, o = REGIMES[regime]
    N = world
    if stage == 0:
        return (p + g + o) * psi
    if stage == 1:
        return p * psi + g * psi + o * psi / N
    if stage == 2:
        return p * psi + (g + o) * psi / N
    if stage == 3:
        return (p + g + o) * psi / N
    raise ValueError(stage)


def comm_bytes_per_gpu(stage: int, psi: float, world: int, itemsize: int = 4,
                       reshard_after_forward: bool = True) -> float:
    """Bytes leaving one GPU per step, ring algorithms.

    all_reduce  = 2(N-1)/N . Psi . b
    reduce_scatter = all_gather = (N-1)/N . Psi . b

    ZeRO-0 is one all_reduce. ZeRO-1 and ZeRO-2 are reduce_scatter + all_gather,
    which is the same two phases carrying the same payload, hence the same bytes.
    ZeRO-3 adds a second parameter all_gather because the first one was freed
    after forward -- so 3 phases, 1.5x, exactly and for every N.
    """
    r = (world - 1) / world
    phases = {0: 2, 1: 2, 2: 2, 3: 3 if reshard_after_forward else 2}[stage]
    return phases * r * psi * itemsize


def paper_table() -> list[dict]:
    """Reproduce Table 1 of arXiv:1910.02054, as a check on the formulas.

    Paper: Psi=7.5B and 128B at N_d=64 -> 120 / 31.4 / 16.6 / 1.88 GB and
    2048 / 536 / 284 / 32 GB.
    """
    out = []
    for psi, want in ((7.5e9, (120, 31.4, 16.6, 1.88)), (128e9, (2048, 536, 284, 32))):
        row = {"psi": psi, "world": 64}
        for s in STAGES:
            row[f"stage{s}"] = state_bytes_per_gpu(s, psi, 64, "mixed_paper") / GB
            row[f"paper{s}"] = want[s]
        out.append(row)
    return out


def max_psi(stage: int, budget_bytes: float, world: int, *, regime: str = "fp32_adam",
            act_reserve_frac: float = 0.25, n_layers: int = 32,
            prefetch_depth: int = 1) -> float:
    """Largest Psi that fits one GPU's budget.

    Two terms most versions of this table leave out, both of which make the
    answer smaller:

      activations. They do not shard. Reserving nothing for them gives a number
      that is a weights-only upper bound and is NOT trainable.

      ZeRO-3's transient buffer. While a layer is gathered, that layer's full
      parameters and its full gradient are resident. With prefetch depth d you
      hold d+1 layers' parameters, so the term is (2+d) . (Psi/L) . b, and it
      does NOT shrink with N. It is why the ZeRO-3 curve flattens at large N
      instead of falling forever.
    """
    p, g, o = REGIMES[regime]
    avail = budget_bytes * (1.0 - act_reserve_frac)
    N = world
    if stage == 0:
        per = p + g + o
    elif stage == 1:
        per = p + g + o / N
    elif stage == 2:
        per = p + (g + o) / N
    else:
        per = (p + g + o) / N + (2 + prefetch_depth) * p / n_layers
    return avail / per


def model_flops(psi_matmul: float, tokens: int, *, batch: int, seq: int,
                n_layers: int, n_heads: int, d_head: int) -> dict:
    """FLOPs per rank per step, derived from shapes rather than benchmarked.

    This is the quantity that does NOT change between ZeRO stages, which is the
    point: ZeRO moves memory and bytes around, it does not change the arithmetic
    each GPU performs.
    """
    fwd_mm = 2 * tokens * psi_matmul
    fwd_attn = 2 * 2 * batch * n_layers * n_heads * seq * seq * d_head
    fwd = fwd_mm + fwd_attn
    return {"forward": fwd, "backward": 2 * fwd, "total": 3 * fwd}


@dataclass(frozen=True)
class LinkModel:
    """A MODEL of interconnect time. Nothing measured on this machine.

    Reported as an interval, never a single number: with zero overlap the
    communication is fully exposed, with perfect overlap it hides entirely behind
    compute, and a real implementation sits between. Quoting one end as "the"
    time would be picking a result.
    """

    label: str = "MODEL: 25 GB/s per link, 5 us latency -- not measured here"
    bw_bytes_per_s: float = 25e9
    latency_s: float = 5e-6

    def time_for(self, stage: int, payload_bytes: float, world: int,
                 n_collectives: int) -> float:
        hops = 2 * (world - 1) * n_collectives
        return hops * self.latency_s + payload_bytes / self.bw_bytes_per_s

    def bracket(self, comm_bytes: float, compute_s: float, world: int,
                n_collectives: int, stage: int = 0) -> tuple[float, float]:
        c = self.time_for(stage, comm_bytes, world, n_collectives)
        return (max(compute_s, c), compute_s + c)


PRESET_LINKS = {
    "nvlink3": 300e9,
    "ib_hdr_200g": 25e9,
    "pcie4_x16": 25e9,
}
