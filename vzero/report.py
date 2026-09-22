"""Figures. Each one has a job; none of them plot wall-clock.

Simulator wall-clock measures the GIL, not ZeRO, so it is absent from every
figure here on purpose rather than shown with a caveat -- caveated numbers get
screenshotted without their caveats.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .analysis import GiB, STAGE_NAMES, STAGES, comm_bytes_per_gpu, max_psi, state_bytes_per_gpu

BUCKETS = ["params", "grads", "opt", "act", "transient"]
LABELS = {"params": "parameters", "grads": "gradients", "opt": "Adam m, v",
          "act": "activations (NOT sharded)", "transient": "transient gather buffer"}
COLORS = {"params": "#3b6ea5", "grads": "#e08a3c", "opt": "#6aa84f",
          "act": "#b03a48", "transient": "#8a6aad"}
MB = 1 << 20
plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight", "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True})


def fig_memory_breakdown(runs: dict, path: str) -> str:
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    names = [STAGE_NAMES[m] for m in STAGES]
    bottoms = [0.0] * 4
    for b in BUCKETS:
        vals = [runs[m].peak_breakdown[0][b] / MB for m in STAGES]
        ax.bar(names, vals, bottom=bottoms, label=LABELS[b], color=COLORS[b],
               edgecolor="white", linewidth=0.6)
        bottoms = [x + y for x, y in zip(bottoms, vals)]
    base = bottoms[0]
    for i, tot in enumerate(bottoms):
        ax.text(i, tot + base * 0.015, f"{tot:.2f} MB\n{base / tot:.2f}x less",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("peak memory per virtual GPU (MB)")
    ax.set_title("Measured peak memory per rank, 32 virtual GPUs\n"
                 "the red segment is identical in all four stages", fontsize=10)
    ax.set_ylim(0, base * 1.22)
    ax.legend(fontsize=8, loc="upper right")
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_measured_vs_analytical(runs: dict, psi_padded: int, shard_total: int,
                               world: int, path: str) -> str:
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    xs, ys = [], []
    for m in STAGES:
        b = runs[m].peak_by_bucket[0]
        meas = (b["params"] + b["grads"] + b["opt"]) / MB
        pred = state_bytes_per_gpu(m, psi_padded, world, "fp32_adam") / MB
        if m == 1:  # ZeRO-1 also keeps the grad shard resident alongside the full buffer
            pred += 4 * shard_total / MB
        xs.append(pred)
        ys.append(meas)
        ax.annotate(f"ZeRO-{m}", (pred, meas), textcoords="offset points",
                    xytext=(7, -3), fontsize=8)
    lim = max(max(xs), max(ys)) * 1.12
    ax.plot([0, lim], [0, lim], "--", color="#888", lw=1, label="y = x")
    ax.scatter(xs, ys, s=55, color="#3b6ea5", zorder=3)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("predicted model state (MB)")
    ax.set_ylabel("measured arena peak (MB)")
    ax.set_title("Measured memory vs the closed form\n(the numbers were counted, not assumed)",
                 fontsize=10)
    ax.legend(fontsize=8)
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_comm_volume(per_step: dict, naive_z1: int, psi_padded: int, world: int,
                    path: str) -> str:
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    names = [STAGE_NAMES[m] for m in STAGES] + ["ZeRO-1\n(naive: all_reduce\nthen slice)"]
    vals = [per_step[m] / MB for m in STAGES] + [naive_z1 / MB]
    cols = ["#3b6ea5"] * 3 + ["#b03a48", "#9a9a9a"]
    bars = ax.bar(names, vals, color=cols, edgecolor="white")
    for m, bar in zip(list(STAGES) + [1], bars):
        pred = comm_bytes_per_gpu(m, psi_padded, world) / MB
        ax.plot([bar.get_x(), bar.get_x() + bar.get_width()], [pred, pred],
                color="black", lw=1.4, zorder=4)
    base = vals[0]
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + base * 0.02,
                f"{v:.2f} MB\n{v / base:.4f}x", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("bytes leaving each rank per step (MB)")
    ax.set_title("Measured communication volume. Black ticks are the closed form.\n"
                 "ZeRO-1 and ZeRO-2 cost what DDP costs, to the byte. ZeRO-3 costs exactly 1.5x.",
                 fontsize=10)
    ax.set_ylim(0, base * 1.9)
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_memory_vs_world(sweep: dict, psi_padded_of: dict, path: str) -> str:
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    worlds = sorted(sweep)
    for m in STAGES:
        ax.plot(worlds, [sweep[w][m] / MB for w in worlds], "o-",
                label=STAGE_NAMES[m], lw=1.6, ms=4)
    psi = psi_padded_of[worlds[0]]
    ax.axhline(8 * psi / MB, ls=":", color="#888", lw=1)
    ax.axhline(4 * psi / MB, ls=":", color="#888", lw=1)
    ax.text(worlds[-1], 8 * psi / MB, " ZeRO-1 floor 8$\\Psi$", fontsize=7.5, va="bottom", ha="right")
    ax.text(worlds[-1], 4 * psi / MB, " ZeRO-2 floor 4$\\Psi$", fontsize=7.5, va="bottom", ha="right")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("number of virtual GPUs (N)")
    ax.set_ylabel("measured model state per rank (MB)")
    ax.set_title("Only ZeRO-3 keeps falling. ZeRO-1 and ZeRO-2 hit a floor\n"
                 "set by what they do not shard (fp32 + Adam: 8$\\Psi$ and 4$\\Psi$).",
                 fontsize=10)
    ax.legend(fontsize=8)
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_loss_and_ablations(runs: dict, ref_losses: list, ablations: dict, path: str) -> str:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.0))
    steps = range(len(ref_losses))
    a1.plot(steps, ref_losses, "k-", lw=3, alpha=0.3, label="single-rank reference")
    for m in STAGES:
        a1.plot(steps, runs[m].losses_mean, lw=1.2, label=f"ZeRO-{m}")
    for name, r in ablations.items():
        a1.plot(steps, r.losses_mean, "--", lw=1.1, alpha=0.85, label=f"broken: {name}")
    a1.set_xlabel("step")
    a1.set_ylabel("loss")
    a1.set_title("All four stages lie exactly on the reference.\n"
                 "Every broken variant still has a falling loss.", fontsize=10)
    a1.legend(fontsize=7.5)

    for m in STAGES:
        d = [abs(x - y) for x, y in zip(runs[m].losses_mean, ref_losses)]
        a2.plot(steps, [max(v, 1e-18) for v in d], lw=1.2, label=f"ZeRO-{m} (exactly 0)")
    for name, r in ablations.items():
        d = [max(abs(x - y), 1e-18) for x, y in zip(r.losses_mean, ref_losses)]
        a2.plot(steps, d, "--", lw=1.1, alpha=0.85, label=f"broken: {name}")
    a2.axhline(1e-4, color="#b03a48", ls=":", lw=1.2)
    a2.text(0, 1.3e-4, "a 1e-4 tolerance would miss sum_not_mean", fontsize=7.5,
            color="#b03a48")
    a2.set_yscale("log")
    a2.set_ylim(1e-18, 10)
    a2.set_xlabel("step")
    a2.set_ylabel("|loss - reference|")
    a2.set_title("Why the assertion is bitwise, not a tolerance:\n"
                 "Adam absorbs the missing 1/N almost entirely", fontsize=10)
    a2.legend(fontsize=7.5, loc="lower right")
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_padding(space_report: dict, pad_vs_n: dict, path: str) -> str:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 3.8))
    gs = space_report["groups"]
    names = list(gs)
    fracs = [100 * gs[g]["pad_frac"] for g in names]
    cols = ["#b03a48" if f > 10 else "#3b6ea5" for f in fracs]
    a1.bar(names, fracs, color=cols, edgecolor="white")
    for i, (g, f) in enumerate(zip(names, fracs)):
        a1.text(i, f + 1.2, f"{gs[g]['raw']:,}\nto {gs[g]['padded']:,}", ha="center",
                fontsize=7)
    a1.set_ylabel("wasted by padding (%)")
    a1.set_ylim(0, max(fracs) * 1.45)
    a1.set_title("Small groups shard badly. ln_f is 128 real elements padded to 512\n"
                 "-- 75% waste, for a LayerNorm split 32 ways.", fontsize=10)
    a1.tick_params(axis="x", rotation=30)

    ws = sorted(pad_vs_n)
    a2.plot(ws, [100 * pad_vs_n[w] for w in ws], "o-", color="#b03a48", lw=1.6, ms=4)
    a2.set_xscale("log", base=2)
    a2.set_xlabel("number of virtual GPUs (N)")
    a2.set_ylabel("total padding waste (%)")
    a2.set_title("Padding grows with N: every group rounds up to a multiple of N.\n"
                 "This is why FSDP flattens a unit and DeepSpeed keeps small tensors whole.",
                 fontsize=10)
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_scaling(path: str, budget_gib: float = 80, n_layers: int = 32,
                regime: str = "mixed_paper") -> str:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.0))
    worlds = [2 ** k for k in range(3, 12)]
    for m in STAGES:
        a1.plot(worlds, [max_psi(m, budget_gib * GiB, w, regime=regime,
                                 n_layers=n_layers) / 1e9 for w in worlds],
                "o-", label=STAGE_NAMES[m], lw=1.6, ms=3.5)
    for psi, lab in ((1.5, "GPT-2 XL 1.5B"), (13, "13B"), (175, "GPT-3 175B")):
        a1.axhline(psi, ls=":", color="#888", lw=0.9)
        a1.text(worlds[0], psi * 1.05, lab, fontsize=7, color="#666")
    a1.set_xscale("log", base=2)
    a1.set_yscale("log")
    a1.set_xlabel("number of GPUs (N)")
    a1.set_ylabel("largest trainable $\\Psi$ (billions)")
    a1.set_title(f"Largest model that fits, {budget_gib:.0f} GiB per GPU,\n"
                 f"25% reserved for activations (mixed precision, K=12)", fontsize=10)
    a1.legend(fontsize=8)

    with_t = [max_psi(3, budget_gib * GiB, w, regime=regime, n_layers=n_layers) / 1e9
              for w in worlds]
    p, g, o = {"mixed_paper": (2, 2, 12), "fp32_adam": (4, 4, 8)}[regime]
    without = [budget_gib * GiB * 0.75 / ((p + g + o) / w) / 1e9 for w in worlds]
    a2.plot(worlds, with_t, "o-", color="#3b6ea5", lw=1.8, ms=3.5,
            label="with the transient gather buffer")
    a2.plot(worlds, without, "s--", color="#b03a48", lw=1.4, ms=3.5,
            label="$16\\Psi/N$ alone (the usual version)")
    a2.fill_between(worlds, with_t, without, alpha=0.12, color="#b03a48")
    a2.set_xscale("log", base=2)
    a2.set_yscale("log")
    a2.set_xlabel("number of GPUs (N)")
    a2.set_ylabel("largest trainable $\\Psi$ (billions)")
    a2.set_title("ZeRO-3 flattens. While a layer is gathered its full parameters\n"
                 "and gradient are resident, and that term does not shrink with N.",
                 fontsize=10)
    a2.annotate(f"{without[-1] / with_t[-1]:.0f}x overstated at N={worlds[-1]}",
                xy=(worlds[-1], without[-1]), xytext=(-135, -8),
                textcoords="offset points", fontsize=8, color="#b03a48")
    a2.legend(fontsize=8, loc="upper left")
    fig.savefig(path)
    plt.close(fig)
    return path
