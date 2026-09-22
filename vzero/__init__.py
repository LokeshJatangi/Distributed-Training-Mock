"""32 virtual GPUs, and ZeRO stages 1-3 built on top of them from scratch.

Read in this order:
    vgpu.py       what a virtual GPU is: an arena that counts bytes
    fabric.py     the collectives, and why all_reduce is built from the other two
    shard.py      how Psi parameters become 32 shards, and what padding costs
    model.py      a functional transformer, so ZeRO-3 has something to shard
    engine.py     the four modes
    validate.py   the assertions, and the four deliberate bugs
"""

__all__ = ["accounting", "analysis", "data", "engine", "env", "fabric",
           "model", "optim", "pool", "report", "shard", "validate", "vgpu"]
