"""Reconcile every number quoted in README.md against results/.

A README that quotes measurements is only as good as the last time someone
checked them against the files. This does that check, and it is meant to be run
before every commit that touches either side.
"""

from __future__ import annotations

import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

README = open(os.path.join(ROOT, "README.md")).read()
S = json.load(open(os.path.join(ROOT, "results", "summary.json")))

fails: list[str] = []
checked = 0


def quoted(value: str) -> bool:
    return value in README


def check(label: str, value, fmt=lambda v: f"{v:,}") -> None:
    global checked
    checked += 1
    s = fmt(value)
    if not quoted(s):
        fails.append(f"{label}: README does not contain {s!r}")


per = S["per_stage"]
space = S["space"]

# --- model and sharding
check("Psi", space["psi"])
check("Psi mod 32", space["psi"] % 32, str)
check("psi_padded", space["psi_padded"])
check("shard total", space["shard_numel_total"])
check("max group", space["max_group_numel"])
check("ln_f padded", space["groups"]["ln_f"]["padded"], str)
check("ln_f raw", space["groups"]["ln_f"]["raw"], str)

# --- per-stage model state and communication
for m in range(4):
    check(f"ZeRO-{m} state bytes", per[f"ZeRO-{m}"]["state_bytes"])
    check(f"ZeRO-{m} comm bytes", per[f"ZeRO-{m}"]["comm_bytes_per_step"])

# --- headline ratios
z0, z3 = per["ZeRO-0"], per["ZeRO-3"]
state_ratio = z0["state_bytes"] / z3["state_bytes"]
peak_ratio = z0["peak_bytes_per_rank"] / z3["peak_bytes_per_rank"]
comm_ratio = z3["comm_bytes_per_step"] / z0["comm_bytes_per_step"]
check("ZeRO-3 state ratio", state_ratio, lambda v: f"{v:.2f}")
check("ZeRO-3 peak ratio", peak_ratio, lambda v: f"{v:.2f}")
check("ZeRO-3 comm ratio", comm_ratio, lambda v: f"{v:.4f}")

# --- activations
check("activation bytes", S["activation_bytes"])
if not S["activations_identical_across_ranks"]:
    fails.append("activations are NOT identical across ranks, but the README says they are")
checked += 1

# --- naive ZeRO-1
check("naive ZeRO-1 comm", S["naive_zero1_comm_bytes_per_step"])

# --- ablations
for name, d in S["ablations"].items():
    checked += 1
    row = next((line for line in README.splitlines()
                if line.startswith("|") and f"`{name}`" in line), None)
    if row is None:
        fails.append(f"ablation {name} not mentioned in README table")
        continue
    if f"{d['max_loss_delta']:.3e}" not in row:
        fails.append(f"ablation {name} loss error differs from results")
    if f"| {'yes' if d['loss_fell'] else 'no'} |" not in row:
        fails.append(f"ablation {name} loss trend differs from results")

# --- validation
v = S["validation"]
checked += 1
if not v["all_passed"]:
    fails.append(f"validation: only {v['n_passed']}/{v['n_total']} passed")
check("validation count", v["n_total"], lambda n: f"All {n} validation checks pass")

# --- paper table
checked += 1
for row in S["paper_table_check"]:
    for s in range(4):
        mine, paper = row[f"stage{s}"], row[f"paper{s}"]
        if abs(mine - paper) > max(0.05, 0.01 * paper):
            fails.append(f"paper table mismatch at Psi={row['psi']:g} stage {s}: "
                         f"{mine} vs {paper}")

# --- cross-check
for w in (1, 4):
    p = os.path.join(ROOT, "results", f"crosscheck-world{w}.json")
    if not os.path.exists(p):
        fails.append(f"missing {p}")
        continue
    c = json.load(open(p))["compare"]
    checked += 1
    val = f"{c['ours_zero0_vs_torch_ddp']:.3e}"
    if val not in README:
        fails.append(f"cross-check world={w}: README does not contain DDP delta {val}")

# --- figures referenced by the README must exist, and vice versa
for rel in re.findall(r"\((figures/[^)]+)\)", README):
    checked += 1
    if not os.path.exists(os.path.join(ROOT, rel)):
        fails.append(f"README references missing figure {rel}")
for fn in sorted(os.listdir(os.path.join(ROOT, "figures"))):
    checked += 1
    if fn.endswith(".png") and f"figures/{fn}" not in README:
        fails.append(f"figure {fn} is committed but never referenced")

print(f"checked {checked} facts against results/")
if fails:
    print(f"\n{len(fails)} MISMATCHES:")
    for f in fails:
        print(f"  - {f}")
    sys.exit(1)
print("README and results agree.")
