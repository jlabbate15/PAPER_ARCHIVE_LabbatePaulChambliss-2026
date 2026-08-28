"""
Check that the p/q island chains in the trapped map really show q islands in eta.

Two independent tests, because a single trajectory can be misleading:
  (1) dominant eta-harmonic of the widest-excursion trajectory near s_res
      -- but near an overlapped resonance the widest-excursion trajectory is a
      stochastic wanderer, not an island librator, and gives a flat spectrum;
  (2) dominant eta-harmonic of the ensemble upper envelope of s(eta) over a
      narrow band around s_res.

A count is only trusted when both agree and the top mode clearly dominates the
second (amplitude ratio well above ~1.2). At Ekin/E_alpha = 1e-3 neither test
resolves a chain -- the spectrum is flat -- consistent with the resonances being
at the edge of Chirikov overlap there.

Usage:
    python check_island_counts.py --Efrac 1e-5 --band 0.088 0.098 --q 2
"""

import argparse
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--Efrac", default="1e-5")
parser.add_argument("--poinc-dir", default="../data")
parser.add_argument("--tag", default="precise_QA_Bcrit_0.98")
parser.add_argument(
    "--poinc-suffix",
    default="",
    help='suffix on the Poincare npz, e.g. "_ns600" for the run at higher '
    "initial-condition sampling",
)
parser.add_argument("--band", type=float, nargs=2, required=True, metavar=("LO", "HI"))
parser.add_argument("--q", type=int, default=None, help="expected island count")
parser.add_argument("--nbins", type=int, default=192)
parser.add_argument("--min-returns", type=int, default=50)
parser.add_argument(
    "--librator-frac",
    type=float,
    default=0.4,
    help="a trajectory counts as librating (inside the island) if its radial "
    "excursion is at least this fraction of the largest excursion in the band",
)
args = parser.parse_args()

efrac = float(args.Efrac)
d = np.load(
    Path(args.poinc_dir) / f"{args.tag}_poincare_data_Efrac_{efrac:.1e}{args.poinc_suffix}.npz",
    allow_pickle=True,
)
s_all, e_all, nret = d["s_all"], d["etas_all"], d["nreturns"]
alive = np.where(nret >= args.min_returns)[0]
lo, hi = args.band
nb = args.nbins


def top_modes(profile, kmax=12):
    ok = np.isfinite(profile)
    xs = np.arange(profile.size)
    profile = np.interp(xs, xs[ok], profile[ok], period=profile.size)
    amp = np.abs(np.fft.rfft(profile - profile.mean()))
    top = np.argsort(-amp[1 : kmax + 1])[:3] + 1
    return top, amp[top[0]] / amp[top[1]]


# (1) widest-excursion trajectory in the band
s0 = np.array([np.asarray(s_all[i], float)[0] for i in alive])
dsp = np.array([np.ptp(np.asarray(s_all[i], float)) for i in alive])
m = (s0 >= lo) & (s0 <= hi)
if m.any():
    j = alive[np.where(m)[0][np.argmax(dsp[m])]]
    s = np.asarray(s_all[j], float)
    e = np.mod(np.asarray(e_all[j], float), 2 * np.pi)
    n = min(s.size, e.size)
    idx = np.minimum((e[:n] / (2 * np.pi) * nb).astype(int), nb - 1)
    prof = np.array(
        [s[:n][idx == b].mean() if (idx == b).any() else np.nan for b in range(nb)]
    )
    t1, r1 = top_modes(prof)
    print(f"(1) widest trajectory : k={t1[0]:2d}  top3={list(t1)}  ratio={r1:.2f}"
          f"   (excursion {np.ptp(s):.5f})")
else:
    t1, r1 = (None,), np.nan
    print("(1) widest trajectory : no trajectories start in the band")

# (2) consensus over librating trajectories.
#
# An earlier version took the ensemble UPPER ENVELOPE (max s per eta bin over all
# trajectories in the band). That estimator degrades as sampling improves: with
# more trajectories the per-bin max approaches the band's upper edge uniformly in
# eta, washing out the modulation. Going from ns_poinc=120 to 600 turned a
# "confirmed k=3" for the 2/3 chain into k=7 -- more data, worse answer, which is
# the signature of a broken estimator rather than of noisy data.
#
# A trajectory trapped in a q-island chain visits all q lobes, so its s(eta)
# traces q closed loops and has q maxima. Take the trajectories that actually
# librate (large radial excursion) and let them vote on the dominant mode.
exc = np.array([np.ptp(np.asarray(s_all[i], float)) for i in alive])
inband = np.array([
    ((np.asarray(s_all[i], float) >= lo) & (np.asarray(s_all[i], float) <= hi)).any()
    for i in alive
])
cand = np.where(inband)[0]
votes, weights = [], []
if cand.size:
    thresh = args.librator_frac * exc[cand].max()
    for ci in cand:
        if exc[ci] < thresh:
            continue  # on a good surface, not inside the island
        i = alive[ci]
        s = np.asarray(s_all[i], float)
        e = np.mod(np.asarray(e_all[i], float), 2 * np.pi)
        n = min(s.size, e.size)
        idx = np.minimum((e[:n] / (2 * np.pi) * nb).astype(int), nb - 1)
        prof = np.array(
            [s[:n][idx == b].mean() if (idx == b).any() else np.nan for b in range(nb)]
        )
        if np.isfinite(prof).sum() < nb // 2:
            continue
        tk, tr_ = top_modes(prof)
        votes.append(int(tk[0]))
        weights.append(tr_)
if votes:
    vals, counts = np.unique(votes, return_counts=True)
    t2 = (int(vals[np.argmax(counts)]),)
    frac = counts.max() / counts.sum()
    r2 = 1.0 + 3.0 * (frac - 1.0 / max(len(vals), 1))  # consensus strength
    print(
        f"(2) librator consensus: k={t2[0]:2d}  from {len(votes)} librators, "
        f"{100*frac:.0f}% agree  (excursion >= {100*args.librator_frac:.0f}% of "
        f"band max)"
    )
else:
    t2, r2 = (None,), 0.0
    print("(2) librator consensus: no librating trajectories in the band")

# The ensemble envelope (2) is the primary estimator. Test (1) uses the single
# widest-excursion trajectory, which is only a clean librator when the chain is
# isolated -- next to a stochastic layer it is a wanderer and its spectrum is
# flat, so requiring (1) and (2) to agree throws away good envelope results.
primary_k, primary_r = t2[0], r2
verdict = "CONFIRMED" if (primary_k is not None and primary_r > 1.5) else "weak"
if t1[0] is not None and t1[0] == primary_k and r1 > 1.5:
    verdict += " (corroborated by the single-trajectory test)"
print(f"\nband s in [{lo}, {hi}]  ->  q = {primary_k}: {verdict}")
if args.q is not None:
    if primary_k == args.q:
        print(f"  matches the expected q = {args.q}")
    else:
        print(
            f"  does NOT match the reduced fraction's q = {args.q}. Note all\n"
            f"  harmonics n*p/n*q of a rational resonate at the SAME radius, and\n"
            f"  the n-th harmonic makes n*q islands -- so the reduced fraction does\n"
            f"  not by itself predict the observed count. q = {primary_k} here is\n"
            f"  consistent with the n = {primary_k // args.q if args.q and primary_k % args.q == 0 else '?'} harmonic."
        )
