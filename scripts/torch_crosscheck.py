"""Cross-check the simulator against real PyTorch distributed.

    torchrun --nproc_per_node=4 scripts/torch_crosscheck.py

Run at world=4, not 32, on purpose: the claim under test is "does my hand-written
sharding move parameters the way PyTorch's does", and that does not need 32 ranks.
Comparing a 32-rank simulator run against a 4-process torch run would compare
different global batches and fail for reasons unrelated to correctness.

The model is not reimplemented. Both sides call vzero.model.forward, one with
parameters backed by nn.Parameter and one with parameters backed by flat shards,
so a mismatch cannot be a difference in the model.

WHAT THIS VALIDATES, AND WHAT IT DOES NOT
-----------------------------------------
Numerics only. It deliberately makes no communication-volume claim, because gloo
does not have a native reduce_scatter: ProcessGroupGloo implements
reduce_scatter_single by cloning the input, running a full allreduce, and copying
out this rank's chunk. So a gloo reduce_scatter moves 2(N-1)/N bytes where the
algorithm wants (N-1)/N, and real FSDP on gloo therefore moves about 4(N-1)/N per
step rather than 3. Instrumenting torch's counters here and comparing them to the
1.5x result would look like a refutation of a correct derivation. The
communication numbers in this project come from our own fabric, which implements
a real ring.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist
import torch.nn as nn

from vzero.data import microbatch
from vzero.model import PRESETS, ModelConfig, forward, init_param, loss_fn

WORLD = int(os.environ.get("WORLD_SIZE", "4"))
STEPS = int(os.environ.get("CROSSCHECK_STEPS", "15"))
LR = 3e-3
TOL = 1e-3


class TorchParams(nn.Module):
    """nn.Module holding the same parameters, feeding the same functional forward."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.keys = {}
        for name, shape, gid in cfg.declare():
            flat = name.replace(".", "__")
            self.keys[name] = flat
            self.register_parameter(flat, nn.Parameter(init_param(name, shape, cfg)))
        self.groups = {}
        for name, _, gid in cfg.declare():
            self.groups.setdefault(gid, []).append(name)

    def get(self, name):
        return getattr(self, self.keys[name])

    # --- ParamAccess
    def enter(self, gid, x):
        return x

    def exit(self, gid, x):
        return x

    def linear(self, x, w, b):
        return nn.functional.linear(x, self.get(w), self.get(b) if b else None)

    def embed(self, ids, w):
        return nn.functional.embedding(ids, self.get(w))

    def norm(self, x, g, b, eps):
        return nn.functional.layer_norm(x, (x.shape[-1],), self.get(g), self.get(b), eps)

    def forward(self, ids):
        # .clone() is not decoration. FSDP2 warns that a wrapped module returning
        # a VIEW tensor silently drops the pre-backward hook, so the all-gather
        # before backward never happens and the gradients are wrong. F.linear on
        # a 3-D input returns a view of the matmul result, so without this the
        # FSDP2 run trains on stale sharded parameters -- which is exactly what
        # the first version of this script measured.
        return forward(self.cfg, self, ids).clone()


def probe() -> tuple[bool, str]:
    """Do the single-tensor collectives FSDP needs actually work on gloo/CPU?"""
    try:
        t = torch.ones(4 * WORLD)
        out = torch.empty(4)
        dist.reduce_scatter_tensor(out, t)
        g = torch.empty(4 * WORLD)
        dist.all_gather_into_tensor(g, out)
        return True, "gloo has reduce_scatter_tensor and all_gather_into_tensor"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def train(model: nn.Module, cfg: ModelConfig, rank: int, tag: str) -> list[float]:
    opt = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8)
    losses = []
    for s in range(STEPS):
        x, y = microbatch(cfg, s, rank, 1)
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        v = loss.detach().clone()
        dist.all_reduce(v, op=dist.ReduceOp.SUM)
        losses.append((v / WORLD).item())
    return losses


def main() -> None:
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    torch.set_num_threads(1)
    cfg = PRESETS["xs"]
    out: dict = {"world": WORLD, "steps": STEPS}

    ok, msg = probe()
    out["probe"] = {"ok": ok, "detail": msg}
    if rank == 0:
        print(f"[probe] {msg}")

    # ------------------------------------------------------------------- DDP
    torch.manual_seed(0)
    m = TorchParams(cfg)
    ddp = nn.parallel.DistributedDataParallel(m)       # device_ids must be omitted on CPU
    out["ddp"] = train(ddp, cfg, rank, "ddp")
    if rank == 0:
        print(f"[ddp ] final loss {out['ddp'][-1]:.6f}")

    # ----------------------------------------------------------------- FSDP2
    out["fsdp"] = None
    out["fsdp_error"] = None
    try:
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import fully_shard

        mesh = init_device_mesh("cpu", (WORLD,))
        torch.manual_seed(0)
        m2 = TorchParams(cfg)
        # Root-only wrapping: every parameter is sharded across the mesh, gathered
        # for forward, resharded, and gathered again for backward. That is ZeRO-3
        # semantics with one gather unit rather than several. Per-group wrapping
        # would need the model restructured into real submodules with real
        # forwards, and then the two sides would no longer share one forward pass
        # -- reintroducing "is the mismatch the model?" as a failure mode.
        fully_shard(m2, mesh=mesh, reshard_after_forward=True)
        out["fsdp"] = train(m2, cfg, rank, "fsdp")
        if rank == 0:
            print(f"[fsdp] final loss {out['fsdp'][-1]:.6f}")
    except Exception as exc:  # noqa: BLE001
        out["fsdp_error"] = f"{type(exc).__name__}: {exc}"
        if rank == 0:
            print(f"[fsdp] SKIPPED: {out['fsdp_error']}")

    # ------------------------------------------------ compare against ourselves
    if rank == 0:
        from vzero.engine import RunConfig, run

        ours = {}
        for mode in (0, 3):
            r = run(cfg, RunConfig(mode=mode, world=WORLD, steps=STEPS))
            ours[f"zero{mode}"] = r.losses_mean
        out["ours"] = ours

        def cmp(a, b):
            return max(abs(x - y) for x, y in zip(a, b))

        out["compare"] = {
            "ours_zero0_vs_torch_ddp": cmp(ours["zero0"], out["ddp"]),
            "ours_zero3_vs_torch_fsdp": (cmp(ours["zero3"], out["fsdp"])
                                         if out["fsdp"] else None),
            "torch_ddp_vs_torch_fsdp": (cmp(out["ddp"], out["fsdp"])
                                        if out["fsdp"] else None),
            "tolerance": TOL,
        }
        measured = {k: v for k, v in out["compare"].items()
                    if k != "tolerance" and v is not None}
        out["diagnosis"] = (
            f"At WORLD={WORLD}, all available loss-curve comparisons pass the "
            f"{TOL:g} tolerance (largest max|dLoss|={max(measured.values()):.3e})."
            if all(v < TOL for v in measured.values()) else
            f"At WORLD={WORLD}, at least one loss-curve comparison exceeds "
            f"the {TOL:g} tolerance."
        )
        print("\ncross-check (numerics only; see module docstring on gloo reduce_scatter)")
        for k, v in out["compare"].items():
            if k == "tolerance" or v is None:
                continue
            print(f"  {k:32} max|dLoss| = {v:.3e}   {'PASS' if v < TOL else 'FAIL'}")
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "results", f"crosscheck-world{WORLD}.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"  wrote {path}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
