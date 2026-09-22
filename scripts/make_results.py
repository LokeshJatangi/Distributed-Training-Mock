"""Run everything and write results/ and figures/.

Usage: python scripts/make_results.py [--steps N] [--preset xs|s|m|l]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vzero import analysis, report
from vzero.analysis import GiB, STAGES, comm_bytes_per_gpu, max_psi, state_bytes_per_gpu
from vzero.engine import Reference, RunConfig, run
from vzero.env import configure
from vzero.fabric import ring_bytes
from vzero.model import PRESETS
from vzero.validate import validate

ABLATIONS = ("no_reduce", "shard_offset", "no_all_gather", "sum_not_mean")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--preset", default="xs")
    ap.add_argument("--world", type=int, default=32)
    ap.add_argument("--validate-steps", type=int, default=5)
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()

    env = configure()
    print(env.render())
    cfg = PRESETS[args.preset]
    W, S = args.world, args.steps
    space = cfg.space(W)
    sr = space.report()
    print(f"\nmodel: preset={args.preset} Psi={sr['psi']:,} (mod {W} = {sr['psi'] % W}) "
          f"padded={sr['psi_padded']:,} shard={sr['shard_numel_total']:,}")
    print(space.table())

    figdir = os.path.join(args.outdir, "figures")
    resdir = os.path.join(args.outdir, "results")
    os.makedirs(figdir, exist_ok=True)
    os.makedirs(resdir, exist_ok=True)

    t_all = time.perf_counter()

    # ------------------------------------------------------------- main runs
    runs, timings = {}, {}
    for m in STAGES:
        t = time.perf_counter()
        runs[m] = run(cfg, RunConfig(mode=m, world=W, steps=S))
        timings[m] = time.perf_counter() - t
        print(f"  ZeRO-{m}: {timings[m]:5.1f}s")

    naive = run(cfg, RunConfig(mode=1, world=W, steps=S, zero1_naive=True))
    ref = Reference(cfg, RunConfig(world=W, steps=S), order="ring").run()
    nref = Reference(cfg, RunConfig(world=W, steps=S), order="sequential").run()
    abl = {a: run(cfg, RunConfig(mode=2, world=W, steps=S, ablation=a)) for a in ABLATIONS}

    # ----------------------------------------- activations identical per rank
    chk = run(cfg, RunConfig(mode=0, world=W, steps=2, track_all_ranks=True))
    acts = [b["act"] for b in chk.peak_by_bucket]
    act_identical = len(set(acts)) == 1
    print(f"  activation bytes identical on all {W} ranks: {act_identical} ({acts[0]:,} B)")

    # ------------------------------------------------------------ world sweep
    sweep, pad_vs_n, psi_pad_of = {}, {}, {}
    for w in (1, 2, 4, 8, 16, 32):
        sp = cfg.space(w)
        psi_pad_of[w] = sp.psi_padded
        pad_vs_n[w] = sp.report()["pad_frac"]
        sweep[w] = {}
        for m in STAGES:
            r = run(cfg, RunConfig(mode=m, world=w, steps=2))
            b = r.peak_by_bucket[0]
            sweep[w][m] = b["params"] + b["grads"] + b["opt"]
    print(f"  world sweep done ({len(sweep)} sizes)")

    # -------------------------------------------------------------- figures
    per_step = {m: runs[m].comm["per_rank_sent"][0] // S for m in STAGES}
    figs = [
        report.fig_memory_breakdown(runs, f"{figdir}/01-memory-breakdown.png"),
        report.fig_measured_vs_analytical(runs, sr["psi_padded"], sr["shard_numel_total"],
                                         W, f"{figdir}/02-measured-vs-analytical.png"),
        report.fig_comm_volume(per_step, naive.comm["per_rank_sent"][0] // S,
                               sr["psi_padded"], W, f"{figdir}/03-comm-volume.png"),
        report.fig_memory_vs_world(sweep, psi_pad_of, f"{figdir}/04-memory-vs-N.png"),
        report.fig_loss_and_ablations(runs, ref["losses"], abl,
                                     f"{figdir}/05-loss-and-ablations.png"),
        report.fig_padding(sr, pad_vs_n, f"{figdir}/06-padding.png"),
        report.fig_scaling(f"{figdir}/07-scaling.png"),
    ]
    print("  figures:", ", ".join(os.path.basename(f) for f in figs))

    # ---------------------------------------------------------------- metrics
    rows = []
    for m in STAGES:
        r = runs[m]
        b = r.peak_by_bucket[0]
        state = b["params"] + b["grads"] + b["opt"]
        for s in range(S):
            rows.append({
                "mode": f"ZeRO-{m}", "world": W, "psi": sr["psi"],
                "psi_padded": sr["psi_padded"], "step": s,
                "loss": f"{r.losses_mean[s]:.10f}",
                "ref_loss": f"{ref['losses'][s]:.10f}",
                "abs_loss_delta": f"{abs(r.losses_mean[s] - ref['losses'][s]):.3e}",
                "comm_bytes_per_rank_per_step": per_step[m],
                "n_collectives_per_step": r.comm["per_rank_n"][0] // S,
                "peak_bytes_per_rank": r.peak_total[0],
                "state_bytes_per_rank": state,
                "param_bytes": b["params"], "grad_bytes": b["grads"],
                "opt_bytes": b["opt"], "act_bytes": b["act"],
                "transient_bytes": b["transient"],
                "pad_bytes_sent_per_step": r.comm["pad_bytes_sent"][0] // S,
                "opt_elementwise_ops": r.opt_elementwise_ops,
                "wall_s_total": f"{timings[m]:.3f}",
            })
    with open(f"{resdir}/metrics.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        wr.writeheader()
        wr.writerows(rows)

    # ------------------------------------------------------------- validation
    print("\nvalidation:")
    rep = validate(cfg, world=W, steps=args.validate_steps)
    print(rep.render())

    pm = cfg.d_model
    flops = analysis.model_flops(
        cfg.n_layers * 12 * pm * pm + cfg.vocab * pm,
        tokens=cfg.seq, batch=1, seq=cfg.seq,
        n_layers=cfg.n_layers, n_heads=cfg.n_heads, d_head=cfg.d_head)

    summary = {
        "env": env.__dict__,
        "config": {"preset": args.preset, "world": W, "steps": S,
                   "vocab": cfg.vocab, "d_model": cfg.d_model,
                   "n_layers": cfg.n_layers, "n_heads": cfg.n_heads, "seq": cfg.seq},
        "space": sr,
        "per_stage": {
            f"ZeRO-{m}": {
                "peak_bytes_per_rank": runs[m].peak_total[0],
                "buckets": runs[m].peak_by_bucket[0],
                "state_bytes": sum(runs[m].peak_by_bucket[0][k]
                                   for k in ("params", "grads", "opt")),
                "comm_bytes_per_step": per_step[m],
                "comm_ratio_vs_ddp": per_step[m] / per_step[0],
                "n_collectives_per_step": runs[m].comm["per_rank_n"][0] // S,
                "opt_elementwise_ops": runs[m].opt_elementwise_ops,
                "final_loss": runs[m].losses_mean[-1],
                "wall_s": timings[m],
            } for m in STAGES},
        "naive_zero1_comm_bytes_per_step": naive.comm["per_rank_sent"][0] // S,
        "activations_identical_across_ranks": act_identical,
        "activation_bytes": acts[0],
        "flops_per_rank_per_step": flops,
        "paper_table_check": analysis.paper_table(),
        "ablations": {a: {"max_loss_delta": max(abs(x - y) for x, y in
                                                zip(r.losses_mean, ref["losses"])),
                          "loss_fell": r.losses_mean[-1] < r.losses_mean[0]}
                      for a, r in abl.items()},
        "ring_vs_sequential_loss_drift": max(abs(x - y) for x, y in
                                             zip(runs[0].losses_mean, nref["losses"])),
        "validation": {"n_passed": rep.n_passed, "n_total": len(rep.checks),
                       "all_passed": rep.all_passed,
                       "checks": [c.__dict__ for c in rep.checks]},
        "max_psi_table": {
            f"{budget}GiB": {f"N={w}": {f"ZeRO-{m}":
                             max_psi(m, budget * GiB, w, regime="mixed_paper", n_layers=32)
                             for m in STAGES} for w in (8, 32, 64, 512)}
            for budget in (40, 80)},
        "total_wall_s": time.perf_counter() - t_all,
    }
    with open(f"{resdir}/summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    print(f"\ntotal: {summary['total_wall_s']:.1f}s")
    print(f"wrote {resdir}/metrics.csv ({len(rows)} rows) and {resdir}/summary.json")
    if not rep.all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
