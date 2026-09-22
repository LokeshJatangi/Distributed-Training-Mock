"""Adam over a shard.

The only interesting thing here is that it is element-wise, which is why
sharding the optimizer is free in exact terms: running Adam on elements
[lo, hi) of a tensor gives bitwise the same answer as running it on the whole
tensor and then looking at [lo, hi). That is the other half of the bitwise
equality argument -- the fabric supplies identical gradients, and this supplies
identical updates.

The update follows torch.optim.Adam's exact arithmetic, including where epsilon
goes: denom = sqrt(v)/sqrt(bc2) + eps, NOT sqrt(v/bc2) + eps. Those differ, and
getting it wrong produces drift far larger than float noise, which would then be
blamed on ZeRO rather than on Adam.
"""

from __future__ import annotations

import torch

from .vgpu import MemArena


class ShardedAdam:
    def __init__(self, numel: int, arena: MemArena, *, lr: float = 3e-3,
                 betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8,
                 name: str = "adam") -> None:
        self.numel = numel
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.t = 0
        self.m = arena.alloc(numel, bucket="opt", name=f"{name}.m")
        self.v = arena.alloc(numel, bucket="opt", name=f"{name}.v")

    def step(self, p: torch.Tensor, g: torch.Tensor) -> None:
        assert p.numel() == g.numel() == self.numel, (p.numel(), g.numel(), self.numel)
        self.t += 1
        self.m.mul_(self.b1).add_(g, alpha=1 - self.b1)
        self.v.mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
        bc1 = 1 - self.b1 ** self.t
        bc2 = 1 - self.b2 ** self.t
        denom = (self.v.sqrt() / (bc2 ** 0.5)).add_(self.eps)
        p.addcdiv_(self.m, denom, value=-(self.lr / bc1))

    def elementwise_ops(self) -> int:
        """Roughly how many element-ops this rank spends per step. ZeRO-0 pays
        this 32 times over for the same answer."""
        return 10 * self.numel
