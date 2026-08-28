"""
Run the DESC TrappedResonance (TR) objective for precise QA at the same
Bcrit as the firm3d trapped-map scan, for a given kinetic energy.

Reproduces the setup of `island_benchmarks/precise_qa/
compare_island_width_Bcrit_0.98_1_2.ipynb` (on the `ejp` branch of
DESC_TrappedRes), which validated the objective's predicted island widths
against the firm3d Poincare map at Ekin/E_alpha = 1e-5. Here the same setup is
run at other energies so the prediction can be checked against the corresponding
map from the 20260729 energy scan.

The objective predicts, for each rational Omega_eta = p/q, the resonance
location `s_res` and the island full width `Delta_s`. Passing `pitch_invs`
pins Bcrit and makes `compute` return the raw per-(rho, pitch, well) resonance
dictionary instead of the phase-space-averaged scalar.

Note the API difference between branches: the notebook passes `num_well=1`,
which does not exist on `ejp-merge-master` (the branch this repo is on). The
well axis is still present in the outputs, so shapes are unchanged.

Usage:
    python run_tr_objective.py --KE-frac 1e-3
    python run_tr_objective.py --KE-frac 1e-5 --compare-ref <desc_debug.npz>
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--eq", default="precise_QA_output.h5")
parser.add_argument(
    "--KE-frac",
    type=float,
    default=1e-3,
    help="kinetic energy as a fraction of 3.5 MeV (the objective's own "
    "normalization; firm3d uses 3.52 MeV -- see README)",
)
parser.add_argument(
    "--Bcrit",
    type=float,
    default=0.9793859573581938,
    help="mirror field strength; default is the exact value of the "
    "Bcrit_0.98 baseline and of the firm3d energy scan",
)
parser.add_argument(
    "--weight-method",
    default="linear",
    choices=["linear", "bump"],
    help="how surfaces near a resonance are weighted; 'linear' is f_L and "
    "'bump' is f_b in the Delta_res figure",
)
parser.add_argument("--num-rho", type=int, default=30)
parser.add_argument("--num-eta", type=int, default=30)
parser.add_argument("--num-transit", type=int, default=5)
parser.add_argument(
    "--knots-per-transit",
    type=int,
    default=40,
    help="INERT on the default path: it only builds the field-line zeta grid "
    "that Bounce1D consumes, and use_bounce1d defaults to False. Bounce2D "
    "resolution is set by --X/--Y instead.",
)
parser.add_argument("--num-quad", type=int, default=20)
parser.add_argument("--X", type=int, default=32, help="Bounce2D resolution")
parser.add_argument("--Y", type=int, default=32, help="Bounce2D resolution")
parser.add_argument("--nufft-eps", type=float, default=1e-10)
parser.add_argument(
    "--use-bounce1d",
    action="store_true",
    help="use the Bounce1D implementation instead of Bounce2D (~35x slower; "
    "agrees to 0.07% on the widest island -- a cross-check, not a setting)",
)
parser.add_argument(
    "--converged",
    action="store_true",
    help="use the settings the 1e-3 convergence scan justified: num_rho=120, "
    "num_eta=60, num_quad=40, X=Y=64. Overrides the individual flags.",
)
parser.add_argument("--out", default=None)
parser.add_argument(
    "--compare-ref",
    default=None,
    help="reference desc_debug.npz to check this run against (validation)",
)
args = parser.parse_args()

if args.converged:
    # Justified by tr_convergence_scan.py at Ekin/E_alpha = 1e-3: num_rho and
    # num_eta were the only parameters not converged at the notebook baseline
    # (up to 5.6% and 2.1% on the narrower chains); num_quad/X/Y were already
    # converged and the extra margin is nearly free.
    args.num_rho, args.num_eta, args.num_quad, args.X, args.Y = 120, 60, 40, 64, 64

from desc.equilibrium import Equilibrium  # noqa: E402  (after arg parsing)
from desc.objectives import TrappedResonance  # noqa: E402

eq = Equilibrium.load(args.eq)[-1]

print(f"equilibrium : {args.eq}  (NFP={eq.NFP})")
print(f"Bcrit       : {args.Bcrit!r}")
print(f"KE_frac     : {args.KE_frac:.1e}")
print(f"weight      : {args.weight_method}")
print(
    f"resolution  : num_rho={args.num_rho} num_eta={args.num_eta} "
    f"num_quad={args.num_quad} num_transit={args.num_transit} "
    f"X={args.X} Y={args.Y} nufft_eps={args.nufft_eps:g} "
    f"bounce1d={args.use_bounce1d}"
)

t0 = time.time()
obj = TrappedResonance(
    eq,
    num_rho=args.num_rho,
    num_eta=args.num_eta,
    M=1,
    N=0,  # QA
    pitch_invs=np.array([args.Bcrit]),
    num_transit=args.num_transit,
    knots_per_transit=args.knots_per_transit,
    num_quad=args.num_quad,
    KE_frac=args.KE_frac,
    weight_method=args.weight_method,
    X=args.X,
    Y=args.Y,
    nufft_eps=args.nufft_eps,
    use_bounce1d=args.use_bounce1d,
)
obj.build()
print(f"build       : {time.time() - t0:.1f} s")

t1 = time.time()
val = obj.compute(eq.params_dict)
print(f"compute     : {time.time() - t1:.1f} s")

val = {k: np.asarray(v) for k, v in val.items()}

out = args.out or f"tr_objective_precise_QA_Bcrit_0.98_KE_{args.KE_frac:.1e}.npz"
np.savez(
    out,
    val=val,
    KE_frac=args.KE_frac,
    Bcrit=args.Bcrit,
    weight_method=args.weight_method,
    resolution=json.dumps(
        dict(
            num_rho=args.num_rho,
            num_eta=args.num_eta,
            num_quad=args.num_quad,
            num_transit=args.num_transit,
            knots_per_transit=args.knots_per_transit,
            X=args.X,
            Y=args.Y,
            nufft_eps=args.nufft_eps,
            use_bounce1d=args.use_bounce1d,
        )
    ),
)
print(f"wrote       : {out}")

# --- report ------------------------------------------------------------
valid = np.isfinite(val["Delta_s"][0, 0, :])
Delta_s = val["Delta_s"][0, 0, valid]
s_res = val["s_res"][0, 0, valid]
p_arr = val["p_arr"][valid]
q_arr = val["q_arr"][valid]
Omega = val["Omega"][:, 0, 0]
rhos = np.linspace(0, 1, args.num_rho + 1)[1:]

print(f"\nOmega range over s: [{np.nanmin(Omega):.4f}, {np.nanmax(Omega):.4f}]")
print(f"{int(valid.sum())} finite resonances of {valid.size}")

order = np.argsort(-Delta_s)
print("\nwidest predicted islands:")
print("   p/q        s_res     Delta_s")
for i in order[:12]:
    print(f"  {p_arr[i]:3.0f}/{q_arr[i]:<3.0f}   {s_res[i]:8.4f}   {Delta_s[i]:.5f}")

if args.compare_ref:
    ref = np.load(args.compare_ref, allow_pickle=True)["val"][()]
    rv = np.isfinite(ref["Delta_s"][0, 0, :])
    print(f"\n--- comparison against {Path(args.compare_ref).name} ---")
    print(f"finite resonances  ref {int(rv.sum())}  vs  new {int(valid.sum())}")
    o_ref = np.asarray(ref["Omega"])[:, 0, 0]
    both = np.isfinite(o_ref) & np.isfinite(Omega)
    if both.any():
        print(
            f"max |dOmega|       {np.max(np.abs(o_ref[both] - Omega[both])):.3e}"
            f"   (Omega ~ {np.nanmax(np.abs(o_ref)):.3f})"
        )
    if rv.sum() == valid.sum():
        print(
            f"max |dDelta_s|     {np.max(np.abs(ref['Delta_s'][0,0,rv] - Delta_s)):.3e}"
        )
        print(
            f"max |ds_res|       {np.max(np.abs(ref['s_res'][0,0,rv] - s_res)):.3e}"
        )
