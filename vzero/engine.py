"""The four ZeRO modes, sharing one forward pass.

What differs between them is exactly three things -- where parameters live, where
gradients live, and where optimizer state lives -- and the code is organised so
that is visible:

  mode 0 (DDP)  params full   grads full    opt full    all_reduce(grads)
  mode 1        params full   grads full    opt shard   reduce_scatter + all_gather
  mode 2        params full   grads SHARD   opt shard   same collectives, grads freed per group
  mode 3        params shard  grads shard   opt shard   gather params per group, twice

ZeRO-1 and ZeRO-2 differ ONLY in gradient residency. ZeRO-1 keeps a full
gradient buffer alive because nothing forces it not to; ZeRO-2 reduce-scatters
each group's gradient the moment it is complete and frees it. Same collectives,
same bytes, different peak. That is the entire distinction and the memory figures
have to show it as a measurement, not a relabelling.

The ZeRO-3 mechanism is worth reading carefully. The problem it solves: autograd
saves the weight tensor for backward, so simply dropping our reference to a
gathered buffer does not free it. The fix is the one real FSDP uses -- resize the
underlying storage to zero bytes after forward, then resize it back and re-gather
before backward needs it. Saved views follow the storage, so standard F.linear,
F.embedding and F.layer_norm can be used throughout, which is why ZeRO-3's
arithmetic is bitwise identical to the other modes rather than merely close.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .accounting import ActivationTracker
from .data import microbatch
from .fabric import Fabric, accumulation_order
from .model import ModelConfig, forward, init_param, loss_fn
from .optim import ShardedAdam
from .shard import FlatSpace
from .vgpu import VirtualGPU

ITEMSIZE = 4


@dataclass
class RunConfig:
    mode: int = 0                      # 0=DDP 1=ZeRO-1 2=ZeRO-2 3=ZeRO-3
    world: int = 32
    steps: int = 30
    micro_batch: int = 1
    lr: float = 3e-3
    impl: str = "direct"
    reshard_after_forward: bool = True
    zero1_naive: bool = False          # all_reduce then shard: the 1.5x footgun
    capacity_bytes: int | None = None
    # Activation memory is identical on every rank -- same model, same batch
    # shape -- so by default only rank 0 pays the ~112 saved-tensor callbacks per
    # forward that measuring it costs. track_all_ranks=True is used once, in a
    # dedicated check, to verify that claim rather than assume it.
    track_all_ranks: bool = False
    ablation: str | None = None        # "no_reduce" | "shard_offset" | "sum_not_mean" | "no_all_gather"
    preset: str = "xs"


# --------------------------------------------------------------------- groups

class GroupState:
    """One gather unit: the embeddings, one block, the final norm, or the head."""

    def __init__(self, gid: str, space: FlatSpace, cfg: ModelConfig, rc: RunConfig,
                 vgpu: VirtualGPU) -> None:
        self.gid = gid
        self.space = space
        self.cfg = cfg
        self.rc = rc
        self.vgpu = vgpu
        self.arena = vgpu.arena
        self.rank = vgpu.rank
        self.world = rc.world

        g = space.groups[gid]
        self.padded = g.padded_numel
        self.S = g.shard_numel
        mode = rc.mode

        off = rc.ablation == "shard_offset"
        self.lo = self.rank * self.S + (1 if off else 0)
        self.hi = self.lo + self.S
        if self.hi > self.padded:                      # keep the ablation in bounds
            self.lo, self.hi = self.padded - self.S, self.padded

        if mode < 3:
            self.full = self.arena.alloc(self.padded, bucket="params", name=f"{gid}.full")
            self._init_full()
            self.full.requires_grad_(True)
            self.full_d = self.full.detach()
            self.shard = None
        else:
            # ZeRO-3: only the shard is persistent. `full` exists as a tensor
            # object so views and .grad have somewhere to live, but its storage
            # is zero bytes whenever the group is resharded.
            self.shard = self.arena.alloc(self.S, bucket="params", name=f"{gid}.shard")
            tmp = torch.zeros(self.padded)
            self._init_into(tmp)
            self.shard.copy_(tmp.narrow(0, self.rank * self.S, self.S))
            del tmp
            self.full = torch.zeros(self.padded, requires_grad=True)
            self.full_d = self.full.detach()
            self._nbytes = self.full_d.untyped_storage().nbytes()
            self.full_d.untyped_storage().resize_(0)
            self.gathered = False

        # gradients
        self.grad_shard = self.arena.alloc(self.S, bucket="grads", name=f"{gid}.gshard") \
            if mode >= 1 else None
        if mode in (0, 1):
            gb = self.arena.alloc(self.padded, bucket="grads", name=f"{gid}.gfull")
            self.full.grad = gb.view_as(self.full)     # pre-bound: accumulated in place
        # mode 2 and 3 let autograd create .grad lazily, so it can be freed early

        opt_numel = self.padded if mode == 0 else self.S
        self.opt = ShardedAdam(opt_numel, self.arena, lr=rc.lr, name=f"{gid}.adam")

        self.n_rs = 0

    # ------------------------------------------------------------------ init

    def _init_into(self, buf: torch.Tensor) -> None:
        for sp in self.space.params_of(self.gid):
            buf.narrow(0, sp.offset, sp.numel).copy_(
                init_param(sp.name, sp.shape, self.cfg).reshape(-1)
            )

    def _init_full(self) -> None:
        with torch.no_grad():
            self._init_into(self.full)

    # --------------------------------------------------------------- ZeRO-3

    def materialise(self, fab: Fabric, tag: str) -> None:
        """all_gather this group's parameters into a live buffer.

        The write has to happen without bumping the version counter. Autograd
        saved views of this exact storage during forward and records the version
        it saw; re-gathering into it in backward would look like "a variable
        needed for gradient computation was modified in place" and raise. The
        escape hatch is the one real FSDP2 uses at its own all-gather copy-out --
        see torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py, which wraps
        the same operation in the same context manager.
        """
        if self.gathered:
            return
        with torch.autograd._unsafe_preserve_version_counter(self.full_d):
            self.full_d.untyped_storage().resize_(self._nbytes)
            self.arena.adopt(self.full_d, bucket="transient", name=f"{self.gid}.gathered")
            fab.all_gather(self.rank, self.shard, self.full_d,
                           key=f"ag/{self.gid}/{tag}",
                           pad_numel=self.padded - self.space.groups[self.gid].raw_numel)
        self.gathered = True

    def reshard(self) -> None:
        """Release the gathered parameters.

        Dropping our reference is not enough: autograd saved views of this buffer
        for backward, so it would stay alive. Resizing the storage to zero bytes
        is what actually frees it, and saved views follow the storage when it is
        resized back and refilled before backward needs them. This is FSDP's
        mechanism, and it is why standard F.linear / F.embedding / F.layer_norm
        can be used here -- which in turn is why ZeRO-3 is bitwise identical to
        the other three modes instead of merely close to them.
        """
        if not self.gathered:
            return
        self.arena.free(self.full_d)
        with torch.autograd._unsafe_preserve_version_counter(self.full_d):
            self.full_d.untyped_storage().resize_(0)
        self.gathered = False

    # ---------------------------------------------------------------- views

    def view(self, name: str) -> torch.Tensor:
        return self.space.view(self.full, name)

    def param_shard(self) -> torch.Tensor:
        """The slice this rank's optimizer owns.

        With ablation="shard_offset" this is deliberately off by one element
        while gather_src() stays correct -- the realistic version of the bug,
        where a rank updates a slice it does not own and the all-gather collects
        the slice it does. It produces a loss curve that still falls, which is
        the point: it is the kind of mistake that survives "the loss went down".
        """
        if self.rc.mode == 3:
            return self.shard
        return self.full_d.narrow(0, self.lo, self.S)

    def gather_src(self) -> torch.Tensor:
        """What all_gather sends: always the canonical slice for this rank."""
        if self.rc.mode == 3:
            return self.shard
        return self.full_d.narrow(0, self.rank * self.S, self.S)


# ------------------------------------------------------------------- access

class Access:
    """ParamAccess for every mode.

    Modes 0-2: enter/exit are bookkeeping. Parameters are already resident.
    Mode 3:    enter gathers the group; exit plants the node that re-gathers it
               in backward, then frees it.
    """

    def __init__(self, groups: dict, space: FlatSpace, fab: Fabric, rank: int,
                 mode: int, reshard: bool, arena) -> None:
        self.g = groups
        self.space = space
        self.fab = fab
        self.rank = rank
        self.mode = mode
        self.reshard = reshard
        self.arena = arena
        self.exposed_gathers = 0

    def enter(self, gid: str, x: torch.Tensor) -> torch.Tensor:
        self.arena.note_phase(f"fwd.{gid}")
        if self.mode == 3:
            self.g[gid].materialise(self.fab, "f")
            self.exposed_gathers += 1
        return x

    def exit(self, gid: str, x: torch.Tensor) -> torch.Tensor:
        if self.mode == 3 and self.reshard:
            x = _PreBackwardGather.apply(x, self.g[gid], self.fab)
            self.g[gid].reshard()
        return x

    def _gs(self, name: str):
        return self.g[self.space.index[name].group]

    def get(self, name: str) -> torch.Tensor:
        return self._gs(name).view(name)

    def linear(self, x, w: str, b: str | None):
        gs = self._gs(w)
        return F.linear(x, gs.view(w), gs.view(b) if b else None)

    def embed(self, ids, w: str):
        return F.embedding(ids, self._gs(w).view(w))

    def norm(self, x, gname: str, bname: str, eps: float):
        gs = self._gs(gname)
        return F.layer_norm(x, (x.shape[-1],), gs.view(gname), gs.view(bname), eps)


class _PreBackwardGather(torch.autograd.Function):
    """Identity in forward. In backward it re-gathers a group's parameters.

    It sits on the group's OUTPUT, so its backward runs before any of the group's
    internal backward nodes -- exactly when the weights are needed again. This is
    FSDP's pre-backward hook, written out. The second gather it performs per step
    is where ZeRO-3's 1.5x communication comes from: the figure is a consequence
    of this line existing, not a constant typed into a table.
    """

    @staticmethod
    def forward(ctx, x, gs, fab):
        ctx.gs, ctx.fab = gs, fab
        return x

    @staticmethod
    def backward(ctx, gy):
        ctx.gs.materialise(ctx.fab, "b")
        return gy, None, None


# ----------------------------------------------------------------- the modes

@dataclass
class StepStats:
    step: int
    loss: float
    peak_total: int
    peak_by_bucket: dict = field(default_factory=dict)


class Worker:
    """One virtual GPU's view of training. Built per rank, run in its own thread."""

    def __init__(self, rank: int, cfg: ModelConfig, rc: RunConfig,
                 space: FlatSpace, fab: Fabric) -> None:
        self.rank, self.cfg, self.rc, self.space, self.fab = rank, cfg, rc, space, fab
        self.vgpu = VirtualGPU(rank=rank, world=rc.world, capacity_bytes=rc.capacity_bytes)
        self.arena = self.vgpu.arena
        self.arena.note_phase("setup")
        self.groups = {gid: GroupState(gid, space, cfg, rc, self.vgpu) for gid in cfg.groups()}
        self.access = Access(self.groups, space, fab, rank, rc.mode,
                            rc.reshard_after_forward, self.arena)
        self.losses: list[float] = []
        self.tracker_report: dict = {}
        if rc.mode >= 2:
            self._install_grad_hooks()

    # ------------------------------------------------------- ZeRO-2 / ZeRO-3

    def _install_grad_hooks(self):
        """Reduce-scatter each group's gradient the moment it is complete, then
        free it. This is the only difference between ZeRO-1 and ZeRO-2: ZeRO-1
        leaves the full gradient buffer resident because nothing forces it not
        to, ZeRO-2 gets rid of it here."""
        for gid, gs in self.groups.items():
            def hook(p, gs=gs, gid=gid):
                g = p.grad
                self.arena.adopt(g, bucket="transient", name=f"{gid}.gfull")
                self.arena.note_phase(f"bwd.rs.{gid}")
                if self.rc.ablation != "no_reduce":
                    self.fab.reduce_scatter(
                        self.rank, g, gs.grad_shard, key=f"rs/{gid}",
                        op="sum" if self.rc.ablation == "sum_not_mean" else "mean",
                        pad_numel=gs.padded - self.space.groups[gid].raw_numel,
                    )
                else:
                    gs.grad_shard.copy_(g.narrow(0, gs.lo, gs.S))
                self.arena.free(g)
                p.grad = None
                if self.rc.mode == 3:
                    gs.reshard()
            gs.full.register_post_accumulate_grad_hook(hook)

    # ------------------------------------------------------------------ step

    def step(self, step: int) -> StepStats:
        rc, fab, arena = self.rc, self.fab, self.arena
        fab.set_step(step)
        arena.set_step(step)
        x, y = microbatch(self.cfg, step, self.rank, rc.micro_batch)

        arena.note_phase("fwd")
        if self.rank == 0 or rc.track_all_ranks:
            tracker = ActivationTracker(arena)
            with tracker:
                logits = forward(self.cfg, self.access, x)
                loss = loss_fn(logits, y)
                arena.note_phase("bwd")
                loss.backward()
            self.tracker_report = tracker.report()
        else:
            logits = forward(self.cfg, self.access, x)
            loss = loss_fn(logits, y)
            arena.note_phase("bwd")
            loss.backward()

        arena.note_phase("comm")
        if rc.mode == 0:
            self._finish_ddp()
        elif rc.mode == 1:
            self._finish_zero1()
        else:
            self._finish_sharded()

        self.losses.append(loss.item())
        return StepStats(step, loss.item(), arena.peak_total, dict(arena.peak_by_bucket))

    # ------------------------------------------------------------- per mode

    def _op(self) -> str:
        return "sum" if self.rc.ablation == "sum_not_mean" else "mean"

    def _finish_ddp(self) -> None:
        """Every rank all-reduces the whole gradient and performs the identical
        Adam update. 32 ranks doing the same arithmetic 32 times."""
        for gid, gs in self.groups.items():
            pad = gs.padded - self.space.groups[gid].raw_numel
            if self.rc.ablation != "no_reduce":
                self.fab.all_reduce(self.rank, gs.full.grad, key=f"ar/{gid}",
                                    op=self._op(), pad_numel=pad)
            self.arena.note_phase("opt")
            gs.opt.step(gs.full_d, gs.full.grad)
            gs.full.grad.zero_()

    def _finish_zero1(self) -> None:
        """Gradients stay fully resident; only the optimizer state is sharded."""
        for gid, gs in self.groups.items():
            pad = gs.padded - self.space.groups[gid].raw_numel
            if self.rc.zero1_naive:
                # The footgun: all_reduce then take a slice. Correct loss, same
                # memory as the efficient version, 1.5x the communication.
                self.fab.all_reduce(self.rank, gs.full.grad, key=f"ar/{gid}",
                                    op=self._op(), pad_numel=pad)
                gs.grad_shard.copy_(gs.full.grad.narrow(0, gs.lo, gs.S))
            elif self.rc.ablation != "no_reduce":
                self.fab.reduce_scatter(self.rank, gs.full.grad, gs.grad_shard,
                                        key=f"rs/{gid}", op=self._op(), pad_numel=pad)
            else:
                gs.grad_shard.copy_(gs.full.grad.narrow(0, gs.lo, gs.S))
            self.arena.note_phase("opt")
            gs.opt.step(gs.param_shard(), gs.grad_shard)
            if self.rc.ablation != "no_all_gather":
                self.fab.all_gather(self.rank, gs.gather_src().clone(), gs.full_d,
                                    key=f"ag/{gid}", pad_numel=pad)
            gs.full.grad.zero_()
            gs.grad_shard.zero_()

    def _finish_sharded(self) -> None:
        """ZeRO-2 and ZeRO-3. The gradients were already reduce-scattered and
        freed by the post-accumulate hooks during backward, so all that is left
        is the sharded Adam step -- and, for ZeRO-2, re-assembling the full
        parameters. ZeRO-3 leaves them sharded and re-gathers next forward."""
        for gid, gs in self.groups.items():
            pad = gs.padded - self.space.groups[gid].raw_numel
            self.arena.note_phase("opt")
            gs.opt.step(gs.param_shard(), gs.grad_shard)
            if self.rc.mode == 2 and self.rc.ablation != "no_all_gather":
                self.fab.all_gather(self.rank, gs.gather_src().clone(), gs.full_d,
                                    key=f"ag/{gid}", pad_numel=pad)
            gs.grad_shard.zero_()

    # ---------------------------------------------------------------- readout

    def local_shard(self, gid: str) -> torch.Tensor:
        gs = self.groups[gid]
        if self.rc.mode == 3:
            return gs.shard.clone()
        return gs.full_d.narrow(0, self.rank * gs.S, gs.S).clone()

    def opt_state(self) -> dict[str, torch.Tensor]:
        return {gid: (gs.opt.m.clone(), gs.opt.v.clone()) for gid, gs in self.groups.items()}


# ---------------------------------------------------------------- the runner

@dataclass
class RunResult:
    rc: RunConfig
    losses_mean: list[float]
    losses_per_rank: list[list[float]]
    params: dict
    peak_total: list[int]
    peak_by_bucket: list[dict]
    peak_breakdown: list[dict]
    comm: dict
    arena_reports: list[dict]
    tracker: dict
    exposed_gathers: int
    space_report: dict
    opt_elementwise_ops: int
    wall_s: float


def assemble_params(workers, space: FlatSpace, cfg: ModelConfig,
                    rc: RunConfig) -> dict[str, torch.Tensor]:
    """Reassemble every parameter from the 32 ranks, for comparison with the
    reference.

    Deliberately NOT a collective. Under ZeRO-3 no rank holds a whole parameter,
    so reading one out means concatenating shards -- which is a real consequence
    of stage 3 worth naming: checkpointing, state_dict() and every debugger
    breakpoint become collective operations. Here we are outside the worker
    threads, so we concatenate directly instead.
    """
    out: dict[str, torch.Tensor] = {}
    for gid in cfg.groups():
        S = space.groups[gid].shard_numel
        buf = torch.empty(space.groups[gid].padded_numel)
        for r, w in enumerate(workers):
            buf.narrow(0, r * S, S).copy_(w.local_shard(gid))
        for sp in space.params_of(gid):
            out[sp.name] = space.view(buf, sp.name).clone()
    return out


def run(cfg: ModelConfig, rc: RunConfig) -> RunResult:
    import time

    from .pool import run_workers

    space = cfg.space(rc.world)
    fab = Fabric(rc.world, impl=rc.impl)
    workers = [Worker(r, cfg, rc, space, fab) for r in range(rc.world)]

    def body(r: int):
        w = workers[r]
        for s in range(rc.steps):
            w.step(s)
        return w.losses

    t0 = time.perf_counter()
    per_rank = run_workers(rc.world, body, fab)
    wall = time.perf_counter() - t0

    # mean loss, summed in rank order so every mode computes it identically
    means = []
    for s in range(rc.steps):
        acc = 0.0
        for r in range(rc.world):
            acc += per_rank[r][s]
        means.append(acc / rc.world)

    params = assemble_params(workers, space, cfg, rc)
    return RunResult(
        rc=rc,
        losses_mean=means,
        losses_per_rank=per_rank,
        params=params,
        peak_total=[w.arena.peak_total for w in workers],
        peak_by_bucket=[dict(w.arena.peak_by_bucket) for w in workers],
        peak_breakdown=[dict(w.arena.peak_breakdown) for w in workers],
        comm=fab.totals(),
        arena_reports=[w.arena.report() for w in workers],
        tracker=workers[0].tracker_report,
        exposed_gathers=workers[0].access.exposed_gathers,
        space_report=space.report(),
        opt_elementwise_ops=sum(gs.opt.elementwise_ops() for gs in workers[0].groups.values()),
        wall_s=wall,
    )


# -------------------------------------------------------------- the reference

class Reference:
    """Single-rank ground truth.

    It processes the same `world` microbatches the distributed run does and
    accumulates their gradients -- but in the SAME per-chunk order the ring uses,
    which is what lets every ZeRO mode be compared to it bitwise rather than
    within a tolerance. For chunk c the ring sums contributions starting at rank
    c+1 and wrapping, with rank c adding last, so that is what happens here.

    `order="sequential"` instead accumulates 0..N-1 in the obvious order. That is
    the genuinely independent comparison, and the difference between the two is a
    direct measurement of how much float drift the ring's reordering introduces --
    which is the only thing the tolerances in this project are for.
    """

    def __init__(self, cfg: ModelConfig, rc: RunConfig, order: str = "ring") -> None:
        from .vgpu import MemArena

        self.cfg, self.rc, self.order = cfg, rc, order
        self.world = rc.world
        self.space = cfg.space(rc.world)
        self.arena = MemArena(-1)
        self.full: dict[str, torch.Tensor] = {}
        self.opt: dict[str, ShardedAdam] = {}
        self.acc: dict[str, list[torch.Tensor]] = {}
        for gid in cfg.groups():
            g = self.space.groups[gid]
            buf = torch.zeros(g.padded_numel)
            for sp in self.space.params_of(gid):
                buf.narrow(0, sp.offset, sp.numel).copy_(
                    init_param(sp.name, sp.shape, cfg).reshape(-1)
                )
            buf.requires_grad_(True)
            self.full[gid] = buf
            self.opt[gid] = ShardedAdam(g.padded_numel, self.arena, lr=rc.lr, name=f"ref.{gid}")
            self.acc[gid] = [torch.zeros(g.padded_numel) for _ in range(rc.world)]
        self.losses: list[float] = []

    class _Acc:
        def __init__(self, outer):
            self.o = outer

        def enter(self, gid, x):
            return x

        def exit(self, gid, x):
            return x

        def _b(self, name):
            return self.o.full[self.o.space.index[name].group]

        def get(self, name):
            return self.o.space.view(self._b(name), name)

        def linear(self, x, w, b):
            return F.linear(x, self.get(w), self.get(b) if b else None)

        def embed(self, ids, w):
            return F.embedding(ids, self.get(w))

        def norm(self, x, g, b, eps):
            return F.layer_norm(x, (x.shape[-1],), self.get(g), self.get(b), eps)

    def step(self, step: int) -> float:
        acc_obj = self._Acc(self)
        losses = []
        for r in range(self.world):
            for gid in self.full:
                self.full[gid].grad = None
            x, y = microbatch(self.cfg, step, r, self.rc.micro_batch)
            loss = loss_fn(forward(self.cfg, acc_obj, x), y)
            loss.backward()
            losses.append(loss.item())
            for gid, buf in self.full.items():
                self.acc[gid][r].copy_(buf.grad)

        N = self.world
        for gid, buf in self.full.items():
            P = buf.numel()
            C = P // N
            total = torch.empty(P)
            for c in range(N):
                order = (accumulation_order(c, N) if self.order == "ring"
                         else list(range(N)))
                chunk = self.acc[gid][order[0]].narrow(0, c * C, C).clone()
                for s in order[1:]:
                    chunk.add_(self.acc[gid][s].narrow(0, c * C, C))
                total.narrow(0, c * C, C).copy_(chunk)
            total.div_(N)
            self.opt[gid].step(buf.detach(), total)

        m = 0.0
        for v in losses:
            m += v
        m /= N
        self.losses.append(m)
        return m

    def run(self) -> dict:
        for s in range(self.rc.steps):
            self.step(s)
        return {
            "losses": self.losses,
            "params": {
                sp.name: self.space.view(self.full[gid], sp.name).detach().clone()
                for gid in self.full for sp in self.space.params_of(gid)
            },
            "opt": {gid: (o.m.clone(), o.v.clone()) for gid, o in self.opt.items()},
        }
