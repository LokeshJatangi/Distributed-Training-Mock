import gc
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from vzero.model import PRESETS
from vzero.vgpu import ALIGN_BYTES, MemArena, VGPUOutOfMemory


def test_a_view_adds_zero_bytes():
    a = MemArena(0)
    t = a.alloc(1000, bucket="params", name="flat")
    assert a.live_total == 4000
    a.adopt(t.narrow(0, 0, 500), bucket="act", name="view")
    a.adopt(t.view(-1), bucket="act", name="view2")
    assert a.live_total == 4000, "counting numel*itemsize would double-count views"


def test_freed_address_cannot_alias_a_live_record():
    """The bug this catches: keying on data_ptr without holding the storage means
    a recycled address reads as 'already counted', undercounting from then on."""
    a = MemArena(1, capacity_bytes=4096)
    a.alloc(1000, bucket="params", name="first")   # reference deliberately dropped
    gc.collect()
    with pytest.raises(VGPUOutOfMemory):
        a.alloc(1000, bucket="params", name="second")


def test_peak_is_a_watermark_and_breakdown_sums_to_it():
    a = MemArena(2)
    x = a.alloc(1000, bucket="params", name="p")
    y = a.alloc(2000, bucket="act", name="a")
    a.free(y)
    a.alloc(100, bucket="transient", name="t")
    assert a.peak_total == 12000
    assert sum(a.peak_breakdown.values()) == a.peak_total, (
        "a stacked bar built from per-bucket peaks would overstate the real peak"
    )


def test_allocations_are_64_byte_aligned():
    a = MemArena(3)
    for n in (1, 7, 33, 1001):
        t = a.alloc(n, bucket="params", name=f"p{n}")
        assert t.untyped_storage().data_ptr() % ALIGN_BYTES == 0


def test_shards_tile_the_padded_space_exactly():
    for world in (1, 2, 4, 8, 16, 32):
        sp = PRESETS["xs"].space(world)
        assert sp.shard_numel_total * world == sp.psi_padded
        for gid, g in sp.groups.items():
            assert g.padded_numel % world == 0
            assert g.padded_numel >= g.raw_numel


def test_psi_is_not_divisible_by_32():
    """The padding demonstration depends on this; a tidier config would delete it."""
    sp = PRESETS["xs"].space(32)
    assert sp.psi % 32 != 0, "model config no longer demonstrates padding"


def test_small_groups_waste_a_lot_to_padding():
    sp = PRESETS["xs"].space(32)
    assert sp.groups["ln_f"].pad_frac > 0.5, "the ln_f padding finding has changed"
