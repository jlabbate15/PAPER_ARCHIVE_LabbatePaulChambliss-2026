"""
Trapped-particle Poincare maps for the v2 (HighShear) zero-beta equilibrium,
refining the Bcrit grid between 5.92 and 6.30.

The March_2026 benchmark (20260302_highshear_squid_island_benchmark/
trapped_map_beta_0) sampled Bcrit on 10 points evenly spaced in
lam = 1/Bcrit between 1/maxB and 1/minB, giving v2 samples at
... 5.59, 5.93, 6.32, 6.76 ... This scan fills in the 5.92 -> 6.30 gap with
10 points evenly spaced in Bcrit itself.

Poincare resolution follows March_2026/.../refined_poincare (neta_poinc=20)
rather than the coarser neta_poinc=5 of the original beta_0 sweep.
"""

import argparse
import gc
import inspect
import os
import time

import numpy as np

from firm3d.field.boozermagneticfield import (
    BoozerRadialInterpolant,
    InterpolatedBoozerField,
)
from firm3d.field.trajectory_helpers import TrappedPoincare
from firm3d.util.constants import (
    ALPHA_PARTICLE_CHARGE,
    ALPHA_PARTICLE_MASS,
    FUSION_ALPHA_PARTICLE_ENERGY,
)
from firm3d.util.functions import proc0_print, setup_logging
from firm3d.util.mpi import comm_size, comm_world, verbose

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--equil",
    default="wout_v20250319_v2_HighShear_000_000000_desc.nc",
    help="VMEC/DESC wout file (zero-beta v2 HighShear by default)",
)
parser.add_argument("--tag", default="v2_HighShear_beta0", help="output file prefix")
parser.add_argument("--Bcrit-min", type=float, default=5.92)
parser.add_argument("--Bcrit-max", type=float, default=6.30)
parser.add_argument("--nBcrit", type=int, default=10)
parser.add_argument(
    "--Bcrit",
    type=float,
    default=None,
    help="run a single Bcrit value instead of the scan (for reruns)",
)
parser.add_argument("--resolution", type=int, default=48)
parser.add_argument("--ns-poinc", type=int, default=120)
parser.add_argument("--neta-poinc", type=int, default=20)
parser.add_argument("--Nmaps", type=int, default=3000)
parser.add_argument("--tmax", type=float, default=1e-3)
parser.add_argument("--tol", type=float, default=1e-8)
parser.add_argument(
    "--plot-pdf",
    action="store_true",
    help="also write firm3d's vector PDF on rank 0. OFF by default: "
    "TrappedPoincare.plot_poincare never calls plt.close(), so each "
    "iteration's figure (2400 PathCollections, ~7e6 points) leaks in "
    "pyplot's global registry and OOM-kills rank 0 partway through a "
    "multi-Bcrit scan. Make figures from the .npz with "
    "plot_poincare_grid.py instead.",
)
parser.add_argument(
    "--skip-existing",
    action="store_true",
    help="skip Bcrit values whose .npz is already on disk (idempotent reruns)",
)
parser.add_argument(
    "--rho-grid",
    action="store_true",
    help="space the radial initial conditions uniformly in rho = sqrt(s) "
    "instead of uniformly in s. firm3d's default grid is "
    "np.linspace(0,1,ns_poinc+1,endpoint=False)[1:] in s, which clusters "
    "initial conditions toward the edge in rho; this builds the (s, eta) "
    "meshgrid explicitly and passes it via s_init/etas_init.",
)
args = parser.parse_args()

charge = ALPHA_PARTICLE_CHARGE
mass = ALPHA_PARTICLE_MASS
Ekin = FUSION_ALPHA_PARTICLE_ENERGY

resolution = args.resolution  # Resolution for field interpolation
ns_interp = resolution  # number of radial grid points for interpolation
ntheta_interp = resolution  # number of poloidal grid points for interpolation
nzeta_interp = resolution  # number of toroidal grid points for interpolation
order = 3  # order for radial interpolation of the Boozer data
degree = 3  # Degree for Lagrange interpolation
helicity_M = 0  # helicity of field strength contours (M=0 => eta = theta)

setup_logging(f"stdout_trapped_map_{args.tag}_{resolution}_{comm_size}.txt")

time1 = time.time()

bri = BoozerRadialInterpolant(args.equil, order, no_K=True, comm=comm_world)
helicity_N = bri.nfp

field = InterpolatedBoozerField(
    bri,
    degree,
    ns_interp=ns_interp,
    ntheta_interp=ntheta_interp,
    nzeta_interp=nzeta_interp,
)

if args.Bcrit is not None:
    Bcrit_range = np.array([args.Bcrit])
else:
    Bcrit_range = np.linspace(args.Bcrit_min, args.Bcrit_max, args.nBcrit)

proc0_print("equilibrium: ", args.equil)
proc0_print("Bcrit values: ", Bcrit_range)
proc0_print(
    "ns_poinc, neta_poinc, Nmaps: ", args.ns_poinc, args.neta_poinc, args.Nmaps
)

# Explicit (s, eta) initial-condition grid, uniform in rho = sqrt(s).
# TrappedPoincare takes s_init/etas_init only if BOTH are given, and expects
# them already flattened to matching 1D arrays (see trajectory_helpers.py:563
# and initialize_trapped_map).
s_init_grid = etas_init_grid = None
if args.rho_grid:
    rho = np.linspace(0, 1, args.ns_poinc + 1, endpoint=False)[1:]
    s_vals = rho**2
    etas_vals = np.linspace(0, 2 * np.pi, args.neta_poinc, endpoint=False)
    etas2d, s2d = np.meshgrid(etas_vals, s_vals)
    s_init_grid = s2d.flatten()
    etas_init_grid = etas2d.flatten()
    proc0_print(
        f"rho grid: {len(rho)} surfaces uniform in rho, "
        f"rho=[{rho.min():.4f},{rho.max():.4f}] -> "
        f"s=[{s_vals.min():.5f},{s_vals.max():.4f}], "
        f"{len(s_init_grid)} initial conditions"
    )

for Bcrit in Bcrit_range:
    npz_name = f"{args.tag}_poincare_data_Bcrit_{Bcrit:.3f}.npz"
    if args.skip_existing and os.path.exists(npz_name):
        proc0_print(f"Bcrit {Bcrit:.3f} already on disk, skipping")
        continue

    lam = 1.0 / Bcrit
    tstart = time.time()

    # Newer firm3d takes lam=...; older forks take modBin=Bcrit plus a mirror
    # point (modBcrit is then overwritten by modBin).
    tp_params = inspect.signature(TrappedPoincare.__init__).parameters
    common = dict(
        s_init=s_init_grid,
        etas_init=etas_init_grid,
        ns_poinc=args.ns_poinc,
        neta_poinc=args.neta_poinc,
        Nmaps=args.Nmaps,
        comm=comm_world,
        solver_options={"reltol": args.tol, "abstol": args.tol, "axis": 0},
        tmax=args.tmax,
    )
    if "lam" in tp_params:
        poinc = TrappedPoincare(
            field, helicity_M, helicity_N, mass, charge, Ekin, lam=lam, **common
        )
    else:
        poinc = TrappedPoincare(
            field,
            helicity_M,
            helicity_N,
            0.5,
            0.0,
            0.0,  # dummy mirror; modBin sets pitch
            mass,
            charge,
            Ekin,
            Bcrit,
            **common,
        )

    if verbose:
        np.savez(
            npz_name,
            s_all=poinc.s_all,
            chis_all=poinc.chis_all,
            etas_all=poinc.etas_all,
            t_all=poinc.t_all,
            lam=poinc.lam,
            modBcrit=poinc.modBcrit,
            Bcrit_requested=Bcrit,
        )

        if args.plot_pdf:
            import matplotlib.pyplot as plt

            poinc.plot_poincare(
                filename=f"{args.tag}_trapped_map_Bcrit_{Bcrit:.3f}.pdf"
            )
            plt.close("all")  # plot_poincare does not; figures otherwise leak

    proc0_print(f"Bcrit {Bcrit:.3f} done in {time.time() - tstart:.1f} s")

    # Release iteration N's gathered trajectory data before iteration N+1
    # allocates its own -- otherwise both are live at the gather peak.
    del poinc
    gc.collect()

time2 = time.time()

proc0_print("poincare time: ", time2 - time1)
