"""
Compare the DESC TrappedResonance (TR) prediction against the firm3d trapped
Poincare map, at one energy, for precise QA at Bcrit = 0.98.

Follows `compare_island_width_Bcrit_0.98_1_2.ipynb` (ejp branch): the objective
predicts, for each rational Omega = p/q, a resonance location `s_res` and an
island full width `Delta_s`; those are drawn over the map as s_res and
s_res +/- Delta_s/2.

Adds a direct measurement of the island width from the map itself. A trajectory
inside an island librates across it, so its radial excursion max(s) - min(s) is
of order the island width, while a trajectory on a good surface barely moves.
The measured width at each predicted resonance is the peak excursion of
trajectories starting near it.

Distinct (p, q) with the same ratio (1/1, 2/2, 3/3 ...) are the same physical
resonance at different harmonic order and land on the same s_res; they are
collapsed to the reduced fraction, keeping the largest Delta_s.

Usage:
    python compare_tr_vs_poincare.py --Efrac 1e-3
"""

import argparse
from math import gcd
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--Efrac", default="1e-3")
parser.add_argument("--tr-dir", default="../data_tr")
parser.add_argument("--poinc-dir", default="../data")
parser.add_argument("--out-dir", default="../figures")
parser.add_argument("--tag", default="precise_QA_Bcrit_0.98")
parser.add_argument(
    "--poinc-suffix",
    default="",
    help='suffix on the Poincare npz, e.g. "_ns600" for the run at higher '
    "initial-condition sampling",
)
parser.add_argument(
    "--tr-suffix",
    default="",
    help='suffix on the TR npz filenames, e.g. "_hires" to pick the run at the '
    "converged resolution rather than the notebook baseline",
)
parser.add_argument(
    "--min-width",
    type=float,
    default=5e-3,
    help="only draw resonances predicted wider than this",
)
args = parser.parse_args()

out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

efrac = float(args.Efrac)
tr_path = Path(args.tr_dir) / f"tr_{args.tag}_Efrac_{args.Efrac}{args.tr_suffix}.npz"
pc_path = Path(args.poinc_dir) / f"{args.tag}_poincare_data_Efrac_{efrac:.1e}{args.poinc_suffix}.npz"

# --- TR prediction -----------------------------------------------------
tr = np.load(tr_path, allow_pickle=True)
val = tr["val"][()]
ok = np.isfinite(val["Delta_s"][0, 0, :])
Delta_s = val["Delta_s"][0, 0, ok]
s_res = val["s_res"][0, 0, ok]
p_arr = val["p_arr"][ok].astype(int)
q_arr = val["q_arr"][ok].astype(int)
Omega = np.asarray(val["Omega"])[:, 0, 0]
num_rho = Omega.size
s_prof = np.linspace(0, 1, num_rho + 1)[1:] ** 2

keep = (Delta_s > 0) & (s_res > 0) & np.isfinite(s_res)
Delta_s, s_res, p_arr, q_arr = (
    Delta_s[keep],
    s_res[keep],
    p_arr[keep],
    q_arr[keep],
)

# collapse (p,q) with equal ratio onto the reduced fraction, keeping max width
best = {}
for D, sr, p, q in zip(Delta_s, s_res, p_arr, q_arr):
    g = gcd(int(abs(p)), int(abs(q))) or 1
    key = (int(p) // g, int(q) // g)
    if key not in best or D > best[key][0]:
        best[key] = (D, sr)
res = sorted(
    ((D, sr, p, q) for (p, q), (D, sr) in best.items()), key=lambda t: -t[0]
)

# --- Poincare map ------------------------------------------------------
pc = np.load(pc_path, allow_pickle=True)
s_all, etas_all, nret = pc["s_all"], pc["etas_all"], pc["nreturns"]
alive = np.where(nret >= 1)[0]
s0 = np.array([np.asarray(s_all[i], float)[0] for i in alive])
dsp = np.array([np.ptp(np.asarray(s_all[i], float)) for i in alive])
s_max_confined = max(
    (np.asarray(s_all[i], float).max() for i in alive), default=np.nan
)

print(f"energy         : Ekin/E_alpha = {args.Efrac}")
print(f"TR file        : {tr_path.name}")
print(f"Poincare file  : {pc_path.name}")
print(f"trajectories   : {len(alive)} with >=1 return of {len(nret)}")
print(f"outermost s    : {s_max_confined:.4f}")
print(f"Omega range    : [{Omega[Omega < 11].min():.4f}, {Omega[Omega < 11].max():.4f}]")


# Background excursion floor. Every trajectory shows an eta-modulation of s from
# the equilibrium's own geometry (nfp = 2 here, and eta = nfp*zeta), whether or
# not it is near a resonance. A null test on resonance-free bands returns that
# modulation with 100% consensus and a stronger spectral peak than any island, so
# a "measured width" at or below this floor is not an island measurement at all.
# Calibrate it from trajectories far from every predicted resonance.
_far = np.array([min(abs(x - sr_) for _D, sr_, _p, _q in res) for x in s0]) if res else None
if _far is not None and (_far > 0.012).any():
    _quiet = dsp[(_far > 0.012) & (s0 > 0.05) & (s0 < 0.80)]
    BG_FLOOR = float(np.percentile(_quiet, 95)) if _quiet.size > 50 else np.nan
    print(f"background floor : {BG_FLOOR:.5f}  "
          f"(95th pct of {_quiet.size} excursions >0.012 from any resonance)")
else:
    BG_FLOOR = np.nan


def measured_width(sr, D):
    """Peak radial excursion among trajectories starting near s_res.

    The window is tied to the predicted width rather than fixed: at this energy
    neighbouring resonances are separated by less than a fixed 0.035 window, so
    a wide window makes adjacent resonances report the same peak trajectory.
    """
    halfwin = max(0.5 * D, 0.004)
    m = np.abs(s0 - sr) < halfwin
    return (np.max(dsp[m]), int(m.sum())) if m.any() else (np.nan, 0)


sel = [(D, sr, p, q) for D, sr, p, q in res if D >= args.min_width]

# Chirikov overlap with the nearest other resonance: S = (D_i + D_j) / (2 ds).
# S >= 1 means the two island chains overlap, i.e. the region between them is
# expected to be stochastic rather than to hold clean separated islands.
print(
    "\n  p/q     Omega    s_res    Delta_s(TR)   width(map)  n_traj   "
    "d(nearest)  Chirikov   vs bg   verdict    in map?"
)
rows = []
for D, sr, p, q in sel:
    w, n = measured_width(sr, D)
    inside = sr <= s_max_confined
    others = [(abs(sr - s2), D2) for D2, s2, _, _ in sel if s2 != sr]
    if others:
        dmin, D2 = min(others, key=lambda t: t[0])
        chir = (D + D2) / (2 * dmin) if dmin > 0 else np.inf
    else:
        dmin, chir = np.nan, np.nan
    rows.append((p, q, sr, D, w, n, inside, dmin, chir))
    if np.isfinite(BG_FLOOR) and np.isfinite(w):
        ratio_bg = w / BG_FLOOR
        flag = "OK" if ratio_bg > 3 else "BACKGROUND"
    else:
        ratio_bg, flag = np.nan, "?"
    print(
        f"  {p:2d}/{q:<2d}  {p/q:7.4f}  {sr:7.4f}   {D:9.5f}   "
        f"{w:9.5f}  {n:5d}   {dmin:9.4f}  {chir:8.2f}   "
        f"{ratio_bg:6.1f}x  {flag:10s} "
        f"{'yes' if inside else 'NO (lost region)'}"
    )

inside_rows = [r for r in rows if r[6]]
if inside_rows:
    ch = np.array([r[8] for r in inside_rows])
    print(
        f"\ninside the confined region: {len(inside_rows)} resonances, "
        f"Chirikov S median {np.nanmedian(ch):.2f}, "
        f"{int(np.sum(ch >= 1))}/{len(ch)} with S >= 1 (overlapping)"
    )

widest = (rows[0][0], rows[0][1]) if rows else (None, None)

# --- figure ------------------------------------------------------------
fig, (ax0, ax1, ax2) = plt.subplots(
    1, 3, figsize=(14.5, 4.6), gridspec_kw={"width_ratios": [1.5, 1, 1]}
)

# (a) map with predicted resonances overlaid
for i in alive:
    s = np.asarray(s_all[i], float)
    e = np.asarray(etas_all[i], float)
    n = min(s.size, e.size)
    ax0.scatter(
        np.mod(e[:n], 2 * np.pi), s[:n], marker="o", s=0.5,
        edgecolors="none", rasterized=True,
    )
ax0.set_xlim([0, 2 * np.pi])
ax0.set_ylim([0, 1])
ax0.set_xlabel(r"$\eta$")
ax0.set_ylabel(r"$s$")
for p, q, sr, D, _w, _n, inside, _dm, _ch in rows:
    if not (inside or (p, q) == widest):
        continue  # keep the map readable: confined region + the widest chain
    c = "k" if inside else "0.55"
    ax0.axhline(sr, color=c, ls="--", lw=1.1)
    ax0.axhspan(sr - 0.5 * D, sr + 0.5 * D, color=c, alpha=0.16, lw=0)
    ax0.text(
        6.15, sr, f"{p}/{q}", color=c, fontsize=8, va="center", ha="right",
        bbox=dict(fc="w", ec="none", alpha=0.6, pad=0.6),
    )
ax0.set_title(
    f"firm3d map + TR prediction, $E/E_\\alpha$ = {args.Efrac}", fontsize=10
)

# (b) Omega profile with the rationals
good = Omega < 11
ax1.plot(s_prof[good], Omega[good], "o-", ms=3, label=r"$\Omega$ (DESC)")
for p, q, sr, D, _w, _n, inside, _dm, _ch in rows:
    ax1.axhline(p / q, color="k" if inside else "0.6", ls=":", lw=0.8)
    ax1.plot([sr], [p / q], "r*", ms=9 if inside else 5)
ax1.set_xlabel("$s$")
ax1.set_ylabel(r"$\Omega = 2\omega_\zeta/\omega_b$")
ax1.set_xlim(0, 1)
ax1.grid(alpha=0.3)
ax1.set_title(r"$\Omega(s)$ and rational crossings", fontsize=10)

# (c) predicted vs measured width
ax2.semilogx(np.maximum(dsp, 1e-6), s0, ".", ms=2, color="C7",
             label="per-trajectory excursion")
for p, q, sr, D, w, _n, inside, _dm, _ch in rows:
    ax2.plot([D], [sr], "k^" if inside else "^", color="k" if inside else "0.6",
             ms=7)
    if np.isfinite(w):
        ax2.plot([w], [sr], "rv", ms=7)
ax2.plot([], [], "k^", label=r"$\Delta s$ predicted (TR)")
ax2.plot([], [], "rv", label="peak excursion (map)")
ax2.set_xlabel(r"radial width")
ax2.set_ylim(0, 1)
ax2.grid(alpha=0.3, which="both")
ax2.legend(fontsize=7, loc="upper left")
ax2.set_title("island width: predicted vs. map", fontsize=10)

if np.isfinite(s_max_confined):
    for a in (ax0, ax1, ax2):
        a.axhline(s_max_confined, color="tab:red", ls="-", lw=1.2, alpha=0.8)
    ax0.text(0.1, s_max_confined + 0.015, "outermost confined orbit",
             color="tab:red", fontsize=7)

fig.suptitle(
    f"precise QA, $B_{{\\rm crit}}$ = 0.9794 T, $E/E_\\alpha$ = {args.Efrac}: "
    "TR objective vs. firm3d trapped map",
    fontsize=11,
)
fig.tight_layout(rect=[0, 0, 1, 0.93])
out = out_dir / f"{args.tag}_TR_vs_poincare_Efrac_{args.Efrac}.png"
fig.savefig(out, dpi=220)
fig.savefig(out.with_suffix(".pdf"))
print(f"\nwrote {out.name}")
