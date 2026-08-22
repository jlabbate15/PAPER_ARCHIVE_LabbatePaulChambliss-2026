#!/usr/bin/env python3
"""TR objective resolution study at fixed equilibrium (scan42 job_038 final.h5).

Sweeps each TR discretisation parameter independently while holding all others
at the baseline. stab_sacrifice=True only. Baseline bumped to p_max=q_max=8,
num_eta=40.

Usage (GPU node):
    # Full sequential run:
    python -u scan_tr_resolution.py

    # Single param (writes per-param sidecar, safe to run in parallel):
    python -u scan_tr_resolution.py --param num_transit

    # Merge all per-param sidecars into v15 JSON + plot:
    python -u scan_tr_resolution.py --merge
"""

import argparse
import json
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from desc.io import load
from desc.objectives import ObjectiveFunction, TrappedResonance

EQ_PATH = ("scan_runs/iota_scan42/"
           "job_038_nfp2_QA_AR3p5_iota0p6_stab_trw1p0_nt32_100it/final.h5")
OUT_DIR = Path("scan_runs/tr_resolution_v15_20260705_4pt_stencil")

BASE = dict(
    num_transit        = 5,
    num_rho            = 80,
    num_eta            = 40,
    num_pitch          = 80,
    num_quad           = 24,
    knots_per_transit  = 50,
    p_max              = 8,
    q_max              = 8,
    weight_method      = "linear",
    delta_omega        = 0.25,
    jac_chunk_size     = 2,
    M                  = 1,
    N                  = 0,
)

VERSION = "v15"
OUT_JSON  = OUT_DIR / f"tr_resolution_results_{VERSION}.json"
OUT_PNG   = OUT_DIR / f"tr_resolution_{VERSION}.png"

# Sweeps centred around the baseline values
SWEEPS = [
    ("num_transit",       [1, 2, 3, 4, 5, 6, 8, 10, 15],               "tr_num_transit"),
    ("num_rho",           [10, 20, 30, 40, 55, 80, 110, 140],           "tr_num_rho"),
    ("num_eta",           [10, 15, 20, 25, 30, 40, 50, 60, 75],         "tr_num_eta"),
    ("num_pitch",         [10, 20, 30, 40, 60, 80, 120, 160],           "tr_num_pitch"),
    ("num_quad",          [8, 12, 16, 24, 32, 48],                      "tr_num_quad"),
    ("knots_per_transit", [10, 25, 50, 75, 100, 150, 200, 300],         "tr_knots_per_transit"),
    ("pq_max",            [3, 4, 5, 6, 7, 8, 10, 12, 15],              "tr_p_max = tr_q_max"),
]

ALL_PARAMS = [p for p, _, _ in SWEEPS]


def eval_tr(eq, stab_sacrifice, **kwargs):
    p = {**BASE, **kwargs}
    pq = p.pop("pq_max", None)
    if pq is not None:
        p["p_max"] = p["q_max"] = pq

    t0 = time.time()
    tr = TrappedResonance(
        eq,
        num_rho=p["num_rho"], num_eta=p["num_eta"], KE_frac=1,
        num_transit=p["num_transit"], num_quad=p["num_quad"],
        num_pitch=p["num_pitch"],
        knots_per_transit=p["knots_per_transit"],
        N=p["N"], M=p["M"],
        p_max=p["p_max"], q_max=p["q_max"],
        res_range_min=-3, res_range_max=3,
        weight_method=p["weight_method"],
        Delta_Omega=p["delta_omega"],
        stab_sacrifice=stab_sacrifice,
        bt_filter_flag=True,
        fill_value=11, weight=1.0,
    )
    of = ObjectiveFunction((tr,), deriv_mode="batched",
                           jac_chunk_size=p["jac_chunk_size"])
    of.build(verbose=0)
    x   = of.x(eq)
    res = np.abs(np.array(of.compute_scaled(x)))
    elapsed = time.time() - t0
    # TR_sum = sum over rho surfaces (converges with num_rho; the proper optimization metric).
    # TR_mean = TR_sum / num_rho → 0 with resolution since resonance weights sum to 1 per
    # crossing (delta-function-like), so the mean is diluted by non-resonant surfaces.
    num_rho_used = p["num_rho"]
    tr_sum = float(res.mean()) * num_rho_used   # = sum of per-surface f_res_avg
    return float(res.max()), tr_sum, elapsed


def make_plot(all_results, sweeps, out_png):
    """Generate the 2×N resolution figure from all_results dict."""
    n_sweeps = len(sweeps)
    fig, axes = plt.subplots(2, n_sweeps, figsize=(4 * n_sweeps, 7),
                             gridspec_kw={"hspace": 0.45, "wspace": 0.35},
                             squeeze=False)
    fig.suptitle(
        f"TR resolution study {VERSION} (stab_sacrifice=True only; "
        f"baseline: num_transit={BASE['num_transit']}, num_eta={BASE['num_eta']}, "
        f"p_max=q_max={BASE['p_max']}, num_rho={BASE['num_rho']}, "
        f"num_pitch={BASE['num_pitch']}, knots_per_transit={BASE['knots_per_transit']})\n"
        f"Fixed geometry: scan42 job_038 (iota≈0.60, AR≈3.3)",
        fontsize=9,
    )

    colors = {"stab": "C0", "nostab": "C1"}
    labels = {"stab": "stab=True", "nostab": "stab=False"}

    for col, (param, vals, xlabel) in enumerate(sweeps):
        base_v = BASE.get(param, BASE["p_max"] if param == "pq_max" else None)

        for row, (key, ylabel) in enumerate([("tr_max", "TR max"), ("tr_sum", "TR sum")]):
            ax = axes[row, col]
            for stab_key in ["stab", "nostab"]:
                if stab_key not in all_results or param not in all_results[stab_key]:
                    continue
                r = all_results[stab_key][param]
                c = colors[stab_key]
                ax.plot(r["vals"], r[key], "o-", color=c, lw=1.8, ms=5,
                        label=labels[stab_key])
                if base_v in r["vals"]:
                    bv = r[key][r["vals"].index(base_v)]
                    ax.plot(base_v, bv, "*", color=c, ms=10, zorder=5)

            if base_v in vals:
                ax.axvline(base_v, color="grey", lw=1, ls="--", alpha=0.6)
            ax.set_xlabel(xlabel, fontsize=7)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, lw=0.4, alpha=0.5)
            if row == 0:
                ax.set_title(xlabel.replace("tr_", "").replace(" = tr_q_max", ""),
                             fontsize=9)
            if col == 0 and row == 0:
                ax.legend(fontsize=7)

    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_png}")


# ── argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--eq", default=EQ_PATH)
parser.add_argument("--param", type=str, default=None,
                    help="Run only this sweep (e.g. num_transit). "
                         "Writes a per-param sidecar; safe to run in parallel.")
parser.add_argument("--merge", action="store_true",
                    help="Merge all per-param sidecars into the main JSON and plot.")
args = parser.parse_args()

# ── merge mode ────────────────────────────────────────────────────────────────
if args.merge:
    all_results = {"stab": {}, "nostab": {}}
    missing = []
    for param in ALL_PARAMS:
        sidecar = OUT_DIR / f"tr_resolution_{VERSION}_{param}.json"
        if not sidecar.exists():
            missing.append(param)
            continue
        with open(sidecar) as f:
            d = json.load(f)
        for sk in ["stab", "nostab"]:
            if sk in d and param in d[sk]:
                all_results[sk][param] = d[sk][param]
    if missing:
        print(f"WARNING: missing sidecars for: {missing}")
    with open(OUT_JSON, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved merged results to {OUT_JSON}")
    make_plot(all_results, SWEEPS, OUT_PNG)
    raise SystemExit(0)

# ── single-param mode ─────────────────────────────────────────────────────────
if args.param is not None:
    sweep_subset = [(p, v, x) for p, v, x in SWEEPS if p == args.param]
    if not sweep_subset:
        raise ValueError(f"Unknown param '{args.param}'. Valid: {ALL_PARAMS}")

    sidecar = OUT_DIR / f"tr_resolution_{VERSION}_{args.param}.json"
    # Load existing sidecar if present (allows resuming)
    if sidecar.exists():
        with open(sidecar) as f:
            all_results = json.load(f)
        print(f"Resuming from sidecar: {sidecar}")
    else:
        all_results = {}

    print(f"Loading equilibrium from {args.eq} ...")
    eq = load(args.eq)
    if hasattr(eq, "__getitem__"):
        eq = eq[-1]
    print(f"  NFP={eq.NFP}  M={eq.M}  N={eq.N}")
    print(f"  Baseline: kpt={BASE['knots_per_transit']}, num_transit={BASE['num_transit']}, "
          f"num_rho={BASE['num_rho']}\n")

    for stab in [True]:
        sk = "stab" if stab else "nostab"
        if sk not in all_results:
            all_results[sk] = {}

        print(f"\n{'='*60}\nstab_sacrifice = {stab}\n{'='*60}")

        for param, vals, xlabel in sweep_subset:
            if param in all_results[sk]:
                print(f"  Skipping {xlabel} (already done)")
                continue
            print(f"\n  Sweeping {xlabel} over {vals}")
            tr_maxs, tr_means, ts = [], [], []
            base_v = BASE.get(param, BASE["p_max"] if param == "pq_max" else None)
            for v in vals:
                mx, mn, elapsed = eval_tr(eq, stab_sacrifice=stab, **{param: v})
                tr_maxs.append(mx)
                tr_means.append(mn)
                ts.append(elapsed)
                marker = " <-- baseline" if v == base_v else ""
                print(f"    {xlabel}={v:5}  tr_max={mx:.4e}  tr_sum={mn:.4e}  "
                      f"({elapsed:.1f}s){marker}")
            all_results[sk][param] = {"xlabel": xlabel, "vals": vals,
                                      "tr_max": tr_maxs, "tr_sum": tr_means, "t": ts}
            with open(sidecar, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"  Sidecar saved → {sidecar}")

    raise SystemExit(0)

# ── full sequential mode ──────────────────────────────────────────────────────
if OUT_JSON.exists():
    with open(OUT_JSON) as f:
        all_results = json.load(f)
    print(f"Resuming from checkpoint: {OUT_JSON}")
else:
    all_results = {}

print(f"Loading equilibrium from {args.eq} ...")
eq = load(args.eq)
if hasattr(eq, "__getitem__"):
    eq = eq[-1]
print(f"  NFP={eq.NFP}  M={eq.M}  N={eq.N}")
print(f"  Baseline: kpt={BASE['knots_per_transit']}, num_transit={BASE['num_transit']}, "
      f"num_rho={BASE['num_rho']}\n")

for stab in [True, False]:
    sk = "stab" if stab else "nostab"
    if sk not in all_results:
        all_results[sk] = {}

    print(f"\n{'='*60}\nstab_sacrifice = {stab}\n{'='*60}")

    for param, vals, xlabel in SWEEPS:
        if param in all_results[sk]:
            print(f"  Skipping {xlabel} (already done)")
            continue
        print(f"\n  Sweeping {xlabel} over {vals}")
        tr_maxs, tr_means, ts = [], [], []
        base_v = BASE.get(param, BASE["p_max"] if param == "pq_max" else None)
        for v in vals:
            mx, mn, elapsed = eval_tr(eq, stab_sacrifice=stab, **{param: v})
            tr_maxs.append(mx)
            tr_means.append(mn)
            ts.append(elapsed)
            marker = " <-- baseline" if v == base_v else ""
            print(f"    {xlabel}={v:5}  tr_max={mx:.4e}  tr_sum={mn:.4e}  "
                  f"({elapsed:.1f}s){marker}")
        all_results[sk][param] = {"xlabel": xlabel, "vals": vals,
                                  "tr_max": tr_maxs, "tr_sum": tr_means, "t": ts}
        with open(OUT_JSON, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  Checkpoint saved.")

with open(OUT_JSON, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved results to {OUT_JSON}")
make_plot(all_results, SWEEPS, OUT_PNG)
