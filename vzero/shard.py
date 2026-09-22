"""Flat parameter space: how Psi parameters become 32 shards.

Three decisions here, each with a consequence that shows up in the results.

1. ONE FLAT BUFFER PER GROUP, not one per tensor and not one for the whole
   model. A group is a unit we gather as a whole: the embeddings, each
   transformer block, the final LayerNorm, the output head. Per-tensor
   collectives would be catastrophic at N=32 -- a 64-element LayerNorm bias
   split 32 ways sends 2 elements per rank, and the ring pads it to a full
   chunk, so the overhead is larger than the payload.

2. ALL FOUR ZeRO MODES USE THE SAME GROUPING. This is not cosmetic. In a ring
   reduce-scatter, the order in which an element's 32 contributions are summed
   depends on which CHUNK the element falls in: chunk c accumulates
   contrib[c] + contrib[c+1] + ... wrapping around. Change the bucketing and you
   change the chunk an element lands in, which changes the summation order,
   which changes the last bits. Keeping the grouping identical across modes is
   what lets us assert that ZeRO-0/1/2/3 agree BITWISE rather than "within
   tolerance".

3. PADDING IS MADE VISIBLE, NOT DESIGNED AWAY. A group is padded up to a
   multiple of world * ALIGN_ELEMS, so every shard is 64-byte aligned and every
   ring chunk starts on an alignment boundary. Param offsets inside a group are
   aligned too. The cost is wasted elements, and small groups waste a large
   fraction of themselves -- which is exactly why FSDP1 flattens a wrapping unit
   into one FlatParameter and DeepSpeed keeps small tensors replicated below
   stage3_param_persistence_threshold.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from .vgpu import ALIGN_ELEMS


def _round_up(n: int, m: int) -> int:
    return ((n + m - 1) // m) * m


@dataclass(frozen=True)
class ParamSpec:
    name: str
    shape: tuple[int, ...]
    group: str
    offset: int  # element offset inside the group's flat buffer, 64-byte aligned

    @property
    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n


@dataclass
class GroupSpec:
    gid: str
    params: list[ParamSpec] = field(default_factory=list)
    raw_numel: int = 0      # sum of param numels
    packed_numel: int = 0   # after per-param alignment, before shard padding
    padded_numel: int = 0   # after padding to world * ALIGN_ELEMS

    @property
    def shard_numel(self) -> int:
        return self.padded_numel // self.world

    world: int = 1

    @property
    def pad_numel(self) -> int:
        return self.padded_numel - self.raw_numel

    @property
    def pad_frac(self) -> float:
        return self.pad_numel / self.padded_numel


class FlatSpace:
    """The layout of all parameters, for a given world size."""

    def __init__(self, declared: list[tuple[str, tuple[int, ...], str]], world: int) -> None:
        self.world = world
        self.groups: dict[str, GroupSpec] = {}
        self.order: list[str] = []
        self.index: dict[str, ParamSpec] = {}

        for name, shape, gid in declared:
            if gid not in self.groups:
                self.groups[gid] = GroupSpec(gid=gid, world=world)
                self.order.append(gid)
            g = self.groups[gid]
            off = _round_up(g.packed_numel, ALIGN_ELEMS)
            spec = ParamSpec(name=name, shape=tuple(shape), group=gid, offset=off)
            g.params.append(spec)
            g.raw_numel += spec.numel
            g.packed_numel = off + spec.numel
            self.index[name] = spec

        chunk_align = world * ALIGN_ELEMS
        for g in self.groups.values():
            g.padded_numel = _round_up(g.packed_numel, chunk_align)

    # ---------------------------------------------------------------- totals

    @property
    def psi(self) -> int:
        return sum(g.raw_numel for g in self.groups.values())

    @property
    def psi_padded(self) -> int:
        return sum(g.padded_numel for g in self.groups.values())

    @property
    def shard_numel_total(self) -> int:
        return sum(g.shard_numel for g in self.groups.values())

    @property
    def pad_numel(self) -> int:
        return self.psi_padded - self.psi

    @property
    def max_group_numel(self) -> int:
        """Drives ZeRO-3's transient gather buffer."""
        return max(g.padded_numel for g in self.groups.values())

    # ----------------------------------------------------------------- views

    def shard_slice(self, gid: str, rank: int) -> tuple[int, int]:
        s = self.groups[gid].shard_numel
        return rank * s, (rank + 1) * s

    def view(self, buf: torch.Tensor, name: str, base: int = 0) -> torch.Tensor:
        """Shaped view of one parameter inside a group buffer."""
        sp = self.index[name]
        return buf.narrow(0, sp.offset - base, sp.numel).view(sp.shape)

    def params_of(self, gid: str) -> list[ParamSpec]:
        return self.groups[gid].params

    def owner_of(self, name: str) -> list[tuple[int, int, int]]:
        """Which ranks hold pieces of this parameter.

        Returns (rank, offset_in_shard, numel) triples. A parameter is never
        moved to avoid a shard boundary, so large tensors genuinely straddle
        several ranks -- sharding here is over bytes, not over layers.
        """
        sp = self.index[name]
        s = self.groups[sp.group].shard_numel
        out: list[tuple[int, int, int]] = []
        lo, hi = sp.offset, sp.offset + sp.numel
        for r in range(self.world):
            a, b = max(lo, r * s), min(hi, (r + 1) * s)
            if a < b:
                out.append((r, a - r * s, b - a))
        return out

    # ---------------------------------------------------------------- report

    def report(self) -> dict:
        straddle = {n: len(self.owner_of(n)) for n in self.index}
        return {
            "world": self.world,
            "psi": self.psi,
            "psi_padded": self.psi_padded,
            "pad_numel": self.pad_numel,
            "pad_frac": self.pad_numel / self.psi_padded,
            "shard_numel_total": self.shard_numel_total,
            "n_groups": len(self.groups),
            "n_params": len(self.index),
            "max_group_numel": self.max_group_numel,
            "n_straddling": sum(1 for v in straddle.values() if v > 1),
            "max_pieces": max(straddle.values()),
            "groups": {
                g.gid: {
                    "raw": g.raw_numel,
                    "padded": g.padded_numel,
                    "shard": g.shard_numel,
                    "pad": g.pad_numel,
                    "pad_frac": g.pad_frac,
                }
                for g in self.groups.values()
            },
        }

    def table(self) -> str:
        r = self.report()
        lines = [f"{'group':<10}{'raw':>10}{'padded':>10}{'shard':>9}{'pad':>7}{'pad%':>8}"]
        for gid, g in r["groups"].items():
            lines.append(
                f"{gid:<10}{g['raw']:>10,}{g['padded']:>10,}{g['shard']:>9,}"
                f"{g['pad']:>7,}{100 * g['pad_frac']:>7.2f}%"
            )
        lines.append(
            f"{'TOTAL':<10}{r['psi']:>10,}{r['psi_padded']:>10,}"
            f"{r['shard_numel_total']:>9,}{r['pad_numel']:>7,}{100 * r['pad_frac']:>7.2f}%"
        )
        return "\n".join(lines)
