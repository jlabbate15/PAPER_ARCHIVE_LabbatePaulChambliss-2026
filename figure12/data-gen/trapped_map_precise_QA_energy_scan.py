"""
Trapped-particle Poincare maps for the precise QA (Landreman-Paul) equilibrium
at fixed Bcrit = 0.98 T, scanning the particle kinetic energy.

Baseline is the Bcrit = 0.98 case of
`DESC_TrappedRes/island_benchmarks/precise_qa/trapped_map.py`, which fixed
Ekin = 1e-5 * FUSION_ALPHA_PARTICLE_ENERGY and swept lam over 10 points evenly
spaced between 1/maxB and 1/minB. This driver inverts that: lam is pinned to
the exact value the baseline used for its "Bcrit = 0.98" point
(lam = 1.0210479254750708 => modBcrit = 0.979386 T, read back from
poincare_data_Bcrit_0.98.npz), and Ekin/E_alpha is scanned over decades
1e-5 -> 1e-2.

Because lam = v_perp^2/(v^2 B) is an energy-independent pitch-angle variable,
every energy in the scan mirrors at the same |B| = 1/lam surface. What changes
is the magnetic drift, which scales with energy at fixed lam -- so the scan
isolates how the trapped-particle island structure widens with drift.

helicity (M, N) = (1, 0): precise QA is quasi-axisymmetric, so |B| contours
close toroidally and the mapping angle is eta = nfp*zeta.

PARTIAL TRAJECTORIES
--------------------
firm3d's TrappedPoincare.compute_trapped_map discards a trajectory *entirely*
if it breaks at any point -- one that survives 2999 of 3000 returns and then
leaves the domain contributes nothing. That is harmless in a Bcrit scan at
fixed low energy, but it destroys an energy scan: a local survival diagnostic
at this Bcrit found the fraction of trajectories completing 200 returns falls
94.9% -> 84.6% -> 46.2% -> 0.0% over Ekin/E_alpha = 1e-5 -> 1e-4 -> 1e-3 ->
1e-2, so the stock code returns a completely EMPTY map at 1e-2.

This driver therefore re-implements that loop (`run_map` below), keeping each
trajectory up to the point where it breaks and recording the number of returns
achieved plus the reason it stopped. Nothing else changes: the per-return
integration is firm3d's own `TrappedPoincare.trapped_map`, and the break tests
are the stock ones in the stock order. Filtering the saved data to
`nreturns == Nmaps` recovers exactly what stock firm3d would have written, so
the baseline comparison is preserved while the high-energy maps still show
structure.
"""

import argparse
import gc
import os
import time

import numpy as np

from firm3d._core.util import parallel_loop_bounds
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

# Stop-reason codes stored alongside each trajectory.
STOP_COMPLETE = 0  # reached Nmaps returns
STOP_DETRAPPED = 1  # |delta chi| > 2*pi -- barely-trapped/detrapping
STOP_DOMAIN = 2  # RuntimeError from trapped_map: left [0.01, 1.0] in s


def run_map(poinc, Nmaps):
    """TrappedPoincare.compute_trapped_map, but keeping partial trajectories.

    Returns the same four lists as the stock method plus per-trajectory
    return counts and stop reasons. The break conditions and their order are
    copied verbatim from firm3d so that full-length trajectories are identical
    to what the stock code produces.
    """
    s_all, chis_all, etas_all, t_all, nret_all, stop_all = [], [], [], [], [], []
    Ntrj = len(poinc.s_init)
    first, last = parallel_loop_bounds(poinc.comm, Ntrj)

    for itrj in range(first, last):
        tr = [poinc.s_init[itrj], poinc.chis_init[itrj], poinc.etas_init[itrj]]
        s_traj, chis_traj, etas_traj, t_traj = [tr[0]], [tr[1]], [tr[2]], [0]
        stop = STOP_COMPLETE
        for _jj in range(Nmaps):
            try:
                # Apply trapped map twice to return to same vpar = 0 plane
                tr, time1 = poinc.trapped_map(tr)
                tr, time2 = poinc.trapped_map(tr)
                if np.abs(tr[1] - chis_traj[-1]) > 2 * np.pi:
                    stop = STOP_DETRAPPED
                    break
                s_traj.append(tr[0])
                chis_traj.append(tr[1])
                etas_traj.append(tr[2])
                t_traj.append(time1 + time2)
            except RuntimeError:
                stop = STOP_DOMAIN
                break
        # Store as float64 arrays, not Python lists: a list of 3001 floats costs
        # ~32 B/element (object + pointer) against 8 B for numpy, and it pickles
        # far more expensively during the gather below.
        s_all.append(np.asarray(s_traj, dtype=np.float64))
        chis_all.append(np.asarray(chis_traj, dtype=np.float64))
        etas_all.append(np.asarray(etas_traj, dtype=np.float64))
        t_all.append(np.asarray(t_traj, dtype=np.float64))
        nret_all.append(len(s_traj) - 1)  # initial point is not a return
        stop_all.append(stop)

    if poinc.comm is not None:
        # gather to root, NOT allgather. firm3d's compute_trapped_map allgathers,
        # so every rank ends up holding every trajectory -- at ns_poinc=120,
        # neta_poinc=20, Nmaps=3000 that is ~0.9 GB per rank as Python lists,
        # which OOM-killed a 128-rank node (job 20569953, NODE_FAIL after 13 min).
        # Only rank 0 writes the npz, so the other ranks never need the data.
        def gather(x):
            out = poinc.comm.gather(x, root=0)
            return [i for o in out for i in o] if poinc.comm.rank == 0 else []

        s_all, chis_all, etas_all, t_all, nret_all, stop_all = (
            gather(s_all),
            gather(chis_all),
            gather(etas_all),
            gather(t_all),
            gather(nret_all),
            gather(stop_all),
        )

    return s_all, chis_all, etas_all, t_all, np.array(nret_all), np.array(stop_all)

# lam of the baseline "Bcrit = 0.98" dataset, taken verbatim from
# precise_qa/poincare_data_Bcrit_0.98.npz so the 1e-5 point of this scan is
# directly comparable to it. 1/lam = 0.9793859573581938 T -- the "0.98" in the
# baseline filenames is that value rounded to 2 decimals, not an exact 0.98.
LAM_BASELINE = 1.0210479254750708

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--equil",
    default="wout_precise_QA_desc.nc",
    help="VMEC/DESC wout file for the precise QA equilibrium",
)
parser.add_argument("--tag", default="precise_QA_Bcrit_0.98", help="output prefix")
parser.add_argument(
    "--Ekin-frac",
    type=float,
    nargs="+",
    default=[1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
    help="kinetic energies as fractions of FUSION_ALPHA_PARTICLE_ENERGY "
    "(3.52 MeV). Default spans 1e-5 (baseline) -> 1e-2, with half-decade "
    "points at 3e-4 and 3e-3 where the local survival diagnostic showed the "
    "map breaking up.",
)
parser.add_argument(
    "--lam",
    type=float,
    default=LAM_BASELINE,
    help="pitch-angle variable lam = 1/Bcrit (default: the baseline value)",
)
parser.add_argument(
    "--Bcrit",
    type=float,
    default=None,
    help="specify the mirror field strength directly instead of --lam",
)
parser.add_argument("--resolution", type=int, default=48)
parser.add_argument("--ns-poinc", type=int, default=120)
parser.add_argument("--neta-poinc", type=int, default=20)
parser.add_argument("--Nmaps", type=int, default=3000)
parser.add_argument("--tmax", type=float, default=1e-1)
parser.add_argument("--tol", type=float, default=1e-8)
parser.add_argument(
    "--out-suffix",
    default="",
    help='appended to the output .npz names, e.g. "_ns600" for a run at higher '
    "Poincare sampling, so it sits alongside the standard-resolution map "
    "rather than overwriting it",
)
parser.add_argument(
    "--skip-existing",
    action="store_true",
    help="skip energies whose .npz is already on disk (idempotent reruns)",
)
args = parser.parse_args()

charge = ALPHA_PARTICLE_CHARGE
mass = ALPHA_PARTICLE_MASS

lam = 1.0 / args.Bcrit if args.Bcrit is not None else args.lam
modBcrit = 1.0 / lam

resolution = args.resolution  # Resolution for field interpolation
ns_interp = resolution  # number of radial grid points for interpolation
ntheta_interp = resolution  # number of poloidal grid points for interpolation
nzeta_interp = resolution  # number of toroidal grid points for interpolation
order = 3  # order for radial interpolation of the Boozer data
degree = 3  # Degree for Lagrange interpolation
helicity_M = 1  # |B| contours close toroidally (QA) => eta = nfp*zeta
helicity_N = 0

setup_logging(f"stdout_trapped_map_{args.tag}_{resolution}_{comm_size}.txt")

time1 = time.time()

bri = BoozerRadialInterpolant(args.equil, order, no_K=True, comm=comm_world)

field = InterpolatedBoozerField(
    bri,
    degree,
    ns_interp=ns_interp,
    ntheta_interp=ntheta_interp,
    nzeta_interp=nzeta_interp,
)

# Global field extrema, on the same (s, theta, zeta) grid the baseline used --
# reported so it is on record that modBcrit sits inside [minB, maxB], i.e. that
# this lam corresponds to a genuinely trapped population.
s_grid = np.linspace(0, 1, 100)
theta_grid = np.linspace(0, 2 * np.pi, 100)
zeta_grid = np.linspace(0, 2 * np.pi / field.nfp, 100)
s_grid, theta_grid, zeta_grid = np.meshgrid(
    s_grid, theta_grid, zeta_grid, indexing="ij"
)
points = np.zeros((s_grid.size, 3))
points[:, 0] = s_grid.flatten()
points[:, 1] = theta_grid.flatten()
points[:, 2] = zeta_grid.flatten()
field.set_points(points)
modB = field.modB()
minB = float(np.min(modB))
maxB = float(np.max(modB))
del points, modB, s_grid, theta_grid, zeta_grid
gc.collect()

proc0_print("equilibrium: ", args.equil)
proc0_print("nfp: ", field.nfp)
proc0_print("minB, maxB: ", minB, maxB)
proc0_print("lam, modBcrit: ", lam, modBcrit)
proc0_print(
    "normalized (modBcrit - minB)/(maxB - minB): ", (modBcrit - minB) / (maxB - minB)
)
if not (minB < modBcrit < maxB):
    raise SystemExit(
        f"modBcrit = {modBcrit} lies outside [{minB}, {maxB}]: no trapped "
        "population at this lam."
    )
proc0_print("Ekin fractions: ", args.Ekin_frac)
proc0_print(
    "ns_poinc, neta_poinc, Nmaps, tmax: ",
    args.ns_poinc,
    args.neta_poinc,
    args.Nmaps,
    args.tmax,
)

for frac in args.Ekin_frac:
    npz_name = f"{args.tag}_poincare_data_Efrac_{frac:.1e}{args.out_suffix}.npz"
    if args.skip_existing and os.path.exists(npz_name):
        proc0_print(f"Efrac {frac:.1e} already on disk, skipping")
        continue

    Ekin = frac * FUSION_ALPHA_PARTICLE_ENERGY
    tstart = time.time()

    # Nmaps=0: run the mirror-point initialization only. The initial conditions
    # solve |B| = 1/lam and so are identical at every energy; run_map below does
    # the tracing, keeping partial trajectories.
    poinc = TrappedPoincare(
        field,
        helicity_M,
        helicity_N,
        mass,
        charge,
        Ekin,
        lam=lam,
        ns_poinc=args.ns_poinc,
        neta_poinc=args.neta_poinc,
        Nmaps=0,
        comm=comm_world,
        solver_options={"reltol": args.tol, "abstol": args.tol, "axis": 0},
        tmax=args.tmax,
    )

    s_all, chis_all, etas_all, t_all, nret, stop = run_map(poinc, args.Nmaps)

    if verbose:
        ntraj = len(nret)
        ncomplete = int(np.sum(stop == STOP_COMPLETE))
        proc0_print(
            f"  Ninit = {ntraj}, completed all {args.Nmaps} returns: "
            f"{ncomplete} ({100.0 * ncomplete / max(ntraj, 1):.1f}%)  "
            f"[detrapped {int(np.sum(stop == STOP_DETRAPPED))}, "
            f"left domain {int(np.sum(stop == STOP_DOMAIN))}]"
        )
        proc0_print(
            f"  returns per traj: median = {np.median(nret):.0f}, "
            f"mean = {nret.mean():.1f}"
        )

        np.savez(
            npz_name,
            s_all=np.array(s_all, dtype=object),
            chis_all=np.array(chis_all, dtype=object),
            etas_all=np.array(etas_all, dtype=object),
            t_all=np.array(t_all, dtype=object),
            nreturns=nret,
            stop_reason=stop,
            s_init=poinc.s_init,
            lam=poinc.lam,
            modBcrit=poinc.modBcrit,
            Ekin=Ekin,
            Ekin_frac=frac,
            Nmaps=args.Nmaps,
            minB=minB,
            maxB=maxB,
            stop_codes="0=completed, 1=detrapped, 2=left domain",
        )

    proc0_print(f"Efrac {frac:.1e} done in {time.time() - tstart:.1f} s")

    # Release iteration N's gathered trajectory data before iteration N+1
    # allocates its own -- otherwise both are live at the gather peak.
    del poinc, s_all, chis_all, etas_all, t_all
    gc.collect()

time2 = time.time()

proc0_print("poincare time: ", time2 - time1)
