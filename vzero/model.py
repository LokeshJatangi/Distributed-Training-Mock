"""The demo model: a pre-LN decoder-only transformer, written functionally.

There is no nn.Module here, and that is the point. With an nn.Module the
parameters always exist as Parameter objects, so ZeRO-3's claim -- that a rank
holds 1/32 of the weights and materialises a layer only for as long as it takes
to use it -- cannot be demonstrated honestly. Every weight access goes through a
ParamAccess object instead, so "where does this weight come from" is a swappable
decision and the four ZeRO modes can share one forward pass.

torch.func.functional_call was the obvious alternative and it is wrong here: it
swaps tensors into the module object and restores them in a finally block, so 32
threads sharing one module is a data race, and 32 module copies would
materialise 32 x Psi parameters -- exactly what ZeRO-3 denies.

The config is chosen, not inherited:

  vocab 509, d_model 64, 4 heads, 4 layers, seq 64, untied head

  * Psi is not a multiple of 32, so the flat buffer genuinely needs padding.
    A test guards this, because a tidier config would delete the lesson.
  * group sizes span 128 to 32,576 elements, so per-layer gather cost varies by
    254x and ZeRO-3's transient buffer is driven by the largest group rather
    than the average.
  * the head is NOT tied to the embedding. A tied head is consumed twice in the
    forward pass, so a post-accumulate gradient hook fires twice and would
    reduce-scatter a half-accumulated gradient. Real FSDP handles this with an
    explicit accumulation counter; untying sidesteps it, and the README says so.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn.functional as F

from .shard import FlatSpace


@dataclass(frozen=True)
class ModelConfig:
    vocab: int = 509
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 4
    seq: int = 64
    ff_mult: int = 4
    eps: float = 1e-5

    @property
    def d_head(self) -> int:
        assert self.d_model % self.n_heads == 0
        return self.d_model // self.n_heads

    @property
    def d_ff(self) -> int:
        return self.ff_mult * self.d_model

    def declare(self) -> list[tuple[str, tuple[int, ...], str]]:
        """Every parameter, in forward-traversal order, with its gather group."""
        d, V, T, F_ = self.d_model, self.vocab, self.seq, self.d_ff
        out: list[tuple[str, tuple[int, ...], str]] = [
            ("tok_emb", (V, d), "embed"),
            ("pos_emb", (T, d), "embed"),
        ]
        for i in range(self.n_layers):
            g = f"block{i}"
            p = f"blocks.{i}"
            out += [
                (f"{p}.ln1.g", (d,), g), (f"{p}.ln1.b", (d,), g),
                (f"{p}.attn.wqkv", (3 * d, d), g), (f"{p}.attn.bqkv", (3 * d,), g),
                (f"{p}.attn.wo", (d, d), g), (f"{p}.attn.bo", (d,), g),
                (f"{p}.ln2.g", (d,), g), (f"{p}.ln2.b", (d,), g),
                (f"{p}.mlp.w1", (F_, d), g), (f"{p}.mlp.b1", (F_,), g),
                (f"{p}.mlp.w2", (d, F_), g), (f"{p}.mlp.b2", (d,), g),
            ]
        out += [
            ("ln_f.g", (d,), "ln_f"), ("ln_f.b", (d,), "ln_f"),
            ("head.w", (V, d), "head"), ("head.b", (V,), "head"),
        ]
        return out

    def space(self, world: int) -> FlatSpace:
        return FlatSpace(self.declare(), world)

    @property
    def psi(self) -> int:
        return sum(
            math.prod(shape) for _, shape, _ in self.declare()
        )

    def groups(self) -> list[str]:
        seen: list[str] = []
        for _, _, g in self.declare():
            if g not in seen:
                seen.append(g)
        return seen


PRESETS: dict[str, ModelConfig] = {
    "xs": ModelConfig(vocab=509, d_model=64, n_heads=4, n_layers=4, seq=64),
    "s": ModelConfig(vocab=1021, d_model=128, n_heads=4, n_layers=6, seq=64),
    "m": ModelConfig(vocab=2039, d_model=256, n_heads=8, n_layers=8, seq=128),
    "l": ModelConfig(vocab=4093, d_model=512, n_heads=8, n_layers=12, seq=128),
}


# --------------------------------------------------------------------- access

class ParamAccess(Protocol):
    """How the forward pass gets at a weight. The seam between the model and
    the four ZeRO modes."""

    def enter(self, gid: str, x: torch.Tensor) -> torch.Tensor: ...
    def exit(self, gid: str, x: torch.Tensor) -> torch.Tensor: ...
    def get(self, name: str) -> torch.Tensor: ...
    def linear(self, x: torch.Tensor, w: str, b: str | None) -> torch.Tensor: ...
    def embed(self, ids: torch.Tensor, w: str) -> torch.Tensor: ...
    def norm(self, x: torch.Tensor, g: str, b: str, eps: float) -> torch.Tensor: ...


# -------------------------------------------------------------------- forward

def init_param(name: str, shape: tuple[int, ...], cfg: ModelConfig,
               seed: int = 1234) -> torch.Tensor:
    """Deterministic per-parameter init.

    Seeded by parameter NAME, not by a single stream, so every rank produces
    bitwise identical values for the pieces it owns without any broadcast --
    including ZeRO-3, which never builds the full tensor at all.
    """
    name_seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "big") & 0x7FFFFFFF
    gen = torch.Generator().manual_seed(seed + name_seed)
    if name.endswith(".g"):
        return torch.ones(shape)
    if name.endswith(".b") or name.endswith(".bo") or name.endswith(".bqkv") \
       or name.endswith(".b1") or name.endswith(".b2"):
        return torch.zeros(shape)
    if "emb" in name:
        return torch.randn(shape, generator=gen) * 0.02
    fan_in = shape[-1]
    return torch.randn(shape, generator=gen) * (1.0 / math.sqrt(fan_in))


def forward(cfg: ModelConfig, P: ParamAccess, ids: torch.Tensor) -> torch.Tensor:
    """ids: (B, T) int64 -> logits: (B, T, V)"""
    B, T = ids.shape
    d, H, dh = cfg.d_model, cfg.n_heads, cfg.d_head

    # enter/exit thread the activation through on purpose: under ZeRO-3, exit is
    # where the group's parameters are freed AND where the node that re-gathers
    # them in backward gets planted. Having them take and return x keeps that
    # mechanism visible in the forward pass instead of hidden in a hook.
    P.enter("embed", ids)
    x = P.embed(ids, "tok_emb") + P.get("pos_emb")[:T].unsqueeze(0)
    x = P.exit("embed", x)

    causal = torch.tril(torch.ones(T, T, dtype=torch.bool)).view(1, 1, T, T)

    for i in range(cfg.n_layers):
        g, p = f"block{i}", f"blocks.{i}"
        x = P.enter(g, x)

        h = P.norm(x, f"{p}.ln1.g", f"{p}.ln1.b", cfg.eps)
        qkv = P.linear(h, f"{p}.attn.wqkv", f"{p}.attn.bqkv")
        q, k, v = qkv.split(d, dim=-1)
        q = q.view(B, T, H, dh).transpose(1, 2)
        k = k.view(B, T, H, dh).transpose(1, 2)
        v = v.view(B, T, H, dh).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(dh)
        att = att.masked_fill(~causal, float("-inf")).softmax(dim=-1)
        o = (att @ v).transpose(1, 2).reshape(B, T, d)
        x = x + P.linear(o, f"{p}.attn.wo", f"{p}.attn.bo")

        h = P.norm(x, f"{p}.ln2.g", f"{p}.ln2.b", cfg.eps)
        h = F.gelu(P.linear(h, f"{p}.mlp.w1", f"{p}.mlp.b1"))
        x = x + P.linear(h, f"{p}.mlp.w2", f"{p}.mlp.b2")

        x = P.exit(g, x)

    x = P.enter("ln_f", x)
    x = P.norm(x, "ln_f.g", "ln_f.b", cfg.eps)
    x = P.exit("ln_f", x)

    x = P.enter("head", x)
    logits = P.linear(x, "head.w", "head.b")
    return P.exit("head", logits)


def loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(), targets.reshape(-1)
    )
