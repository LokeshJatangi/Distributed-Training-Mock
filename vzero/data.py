"""A deterministic synthetic task.

Each sequence is a random motif of `period` tokens tiled to fill the context, so
next-token prediction is learnable from position alone once the model picks up
the periodicity. Loss falls from ln(vocab) to well under 1 in a few dozen steps,
which is what makes "this is really training" checkable rather than asserted.

Determinism matters more than realism here. Every mode must see bitwise the same
batches in the same order, or a loss-curve comparison measures the data pipeline
instead of ZeRO. Batches are drawn from one generator seeded by (step, rank), so
rank r's microbatch at step t is reproducible from nothing but those two numbers
-- which is also how the single-rank reference replays all 32 microbatches.
"""

from __future__ import annotations

import torch

from .model import ModelConfig


def microbatch(cfg: ModelConfig, step: int, rank: int, batch: int,
               period: int = 16, seed: int = 4242) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed + step * 100003 + rank * 7919)
    motif = torch.randint(0, cfg.vocab, (batch, period), generator=g)
    reps = (cfg.seq + 1 + period - 1) // period
    full = motif.repeat(1, reps)[:, : cfg.seq + 1]
    return full[:, :-1].contiguous(), full[:, 1:].contiguous()
