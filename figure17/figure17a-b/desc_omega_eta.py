"""
Omega_eta profile from DESC's TrappedResonance machinery, for comparison with
firm3d's trapped-Poincare frequencies.

What is being compared. In DESC (_trapped_resonance.py:612),

    Omega = eta_drift_avg / omega_bounce_avg

i.e. the bounce-averaged eta precession frequency NORMALISED by the bounce
frequency. That is the same dimensionless quantity as firm3d's

    Omega_eta / omega_b = <delta_eta> / 2*pi

the rotation number of the trapped Poincare map. So DESC's ``Omega`` and the
firm3d ratio are directly comparable; DESC's raw ``eta_drift_avg`` is the
unnormalised analogue of firm3d's Omega_eta.

DESC exposes only the resonance penalty through the registered
"trapped EP resonance" compute function -- Omega itself is an internal of
``_resonance_physics``. This script wraps that function to capture its return
dict while the objective runs, which avoids reimplementing the bounce
integrals.

``pitch_invs`` in DESC is Bcrit directly (see the TrappedResonance docstring),
so the firm3d Bcrit values pass straight through.

Usage:
    python desc_omega_eta.py --bcrit 5.962 6.469 --num-rho 32
"""

import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import desc.compute._trapped_resonance as tr_mod
import desc.io
from desc.objectives._trapped_resonance import TrappedResonance

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--eq",
    default=str(
        Path.home()
        / "Elizabeth Paul Dropbox/Elizabeth J. Paul/Princeton Dropbox data"
        / "March_2026/20260302_highshear_squid_island_benchmark/run_desc/beta_0"
        / "wout_v20250319_v2_HighShear_000_000000_desc.h5"
    ),
)
parser.add_argument("--bcrit", type=float, nargs="+", default=[5.962, 6.469])
parser.add_argument(
    "--M", type=int, default=0,
    help="helicity M. DESC's eta_drift denominator is (N*nfp - iota*M), so "
    "firm3d's helicity_M=0, helicity_N=nfp maps to M=0, N=1. NOTE the DESC "
    "objective defaults are M=1, N=0 (a QA-like helicity), which do NOT match "
    "the firm3d runs and give a different sign and magnitude.",
)
parser.add_argument("--N", type=int, default=1, help="helicity N; see --M")
parser.add_argument("--num-rho", type=int, default=32)
parser.add_argument("--num-eta", type=int, default=10)
parser.add_argument("--num-transit", type=int, default=5)
parser.add_argument("--num-quad", type=int, default=32)
parser.add_argument("--fill-value", type=float, default=11.0)
parser.add_argument("--out-dir", default="../figures")
parser.add_argument("--out-data", default="../data/desc_omega_eta.npz")
args = parser.parse_args()

eq = desc.io.load(args.eq)
if hasattr(eq, "__len__"):
    eq = eq[-1]
print(f"equilibrium: NFP={eq.NFP} L,M,N={eq.L},{eq.M},{eq.N}")

# --- capture Omega out of _resonance_physics --------------------------
_captured = {}
_orig = tr_mod._resonance_physics


def _spy(*a, **kw):
    out = _orig(*a, **kw)
    for k in ("Omega", "eta_drift_avg", "omega_bounce_avg", "valid_prime"):
        if k in out:
            _captured[k] = np.asarray(out[k])
    return out


tr_mod._resonance_physics = _spy

results = {}
try:
    for bc in args.bcrit:
        _captured.clear()
        obj = TrappedResonance(
            eq,
            pitch_invs=np.atleast_1d(bc),
            M=args.M,
            N=args.N,
            num_rho=args.num_rho,
            num_eta=args.num_eta,
            num_transit=args.num_transit,
            num_quad=args.num_quad,
            fill_value=args.fill_value,
        )
        obj.build(verbose=0)
        obj.compute(eq.params_dict)

        if "Omega" not in _captured:
            raise RuntimeError(
                "_resonance_physics was not called -- the objective's internals "
                "changed; re-check the capture hook."
            )

        # shapes: (rho, pitch, well); one pitch, num_well = 1
        Omega = _captured["Omega"].squeeze()
        eta_d = _captured["eta_drift_avg"].squeeze()
        om_b = _captured["omega_bounce_avg"].squeeze()
        valid = _captured.get("valid_prime")
        valid = valid.squeeze() if valid is not None else np.ones_like(Omega, bool)

        rho = obj._grid_1dr.compress(obj._grid_1dr.nodes[:, 0])
        s = rho**2

        # fill_value marks "no trapped particle / undefined"
        bad = (~valid.astype(bool)) | np.isclose(Omega, args.fill_value)
        Omega = np.where(bad, np.nan, Omega)
        eta_d = np.where(bad, np.nan, eta_d)
        om_b = np.where(bad, np.nan, om_b)

        results[f"{bc:.3f}"] = (s, Omega, eta_d, om_b)
        n_ok = int(np.sum(~bad))
        print(
            f"Bcrit {bc:.3f}: {n_ok}/{len(s)} surfaces with a valid trapped particle\n"
            f"    Omega (=Omega_eta/omega_b): "
            f"min={np.nanmin(Omega):+.5f} max={np.nanmax(Omega):+.5f} "
            f"median={np.nanmedian(Omega):+.5f}\n"
            f"    eta_drift_avg  median={np.nanmedian(eta_d):+.4e} rad/s\n"
            f"    omega_bounce_avg median={np.nanmedian(om_b):.4e} rad/s"
        )
finally:
    tr_mod._resonance_physics = _orig

np.savez(
    args.out_data,
    **{f"s_{k}": v[0] for k, v in results.items()},
    **{f"Omega_{k}": v[1] for k, v in results.items()},
    **{f"eta_drift_{k}": v[2] for k, v in results.items()},
    **{f"omega_b_{k}": v[3] for k, v in results.items()},
)

out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(figsize=(7, 4.6))
for label, (s, Om, _e, _b) in results.items():
    ax.plot(s, Om, "o-", ms=3, label=rf"DESC, $B_{{\rm crit}}={label}$ T")
ax.axhline(0, color="k", lw=0.8, ls=":")
ax.set_xlabel(r"$s$")
ax.set_ylabel(r"$\Omega \equiv \Omega_\eta/\omega_b$")
ax.set_title("DESC TrappedResonance normalized precession frequency")
ax.grid(alpha=0.3)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(out_dir / "desc_omega_eta.png", dpi=180)
print(f"\nwrote {out_dir/'desc_omega_eta.png'} and {args.out_data}")
