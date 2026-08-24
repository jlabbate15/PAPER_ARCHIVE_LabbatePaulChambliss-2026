"""
Fusion-born alpha birth-distribution tracing, adapted from firm3d's
examples/fusion_distribution/fusion_distribution.py for the scan60/scan62
QA NFP=2 equilibria (Rmajor ~7.5 m, B0 ~5.3-5.4 T).

Particles are initialized proportional to the fusion reactivity profile
(Bader et al., Nucl. Fusion 61 116060 (2021)) with a Jacobian-weighted
Boozer-coordinate position sample and uniformly distributed parallel
velocities, then traced (collisionless) until they reach s=1 or tmax.

Also computes, for post-run trapped-particle Bcrit analysis: each
particle's critical mirror field Bcrit = B_init / (1 - xi_init^2) (from
conservation of the magnetic moment and kinetic energy), classifies
trapped vs. passing via Bcrit vs. the local flux-surface Bmax(s), and
saves Bmin(s)/Bmax(s) profiles plus per-particle initial conditions so the
(Bcrit-Bmin)/(Bmax-Bmin) loss-fraction plot can be built without rerunning
the trace.

Usage:
    srun -n <ranks> python -u fusion_distribution_scan.py --boozmn boozmn.nc --tag scan60
"""

import argparse
import time

import numpy as np

# Ensure mpi4py is imported and initialized before firm3d modules
from mpi4py import MPI  # noqa: F401

from firm3d.field.boozermagneticfield import InterpolatedBoozerField
from firm3d.field.tracing import MaxToroidalFluxStoppingCriterion, trace_particles_boozer
from firm3d.field.tracing_helpers import initialize_position_profile, initialize_velocity_uniform
from firm3d.util.constants import (
    ALPHA_PARTICLE_CHARGE,
    ALPHA_PARTICLE_MASS,
    FUSION_ALPHA_PARTICLE_ENERGY,
)
from firm3d.util.functions import proc0_print, setup_logging
from firm3d.util.mpi import comm_size, comm_world, verbose

parser = argparse.ArgumentParser()
parser.add_argument("--boozmn", default="boozmn.nc")
parser.add_argument("--nparticles", type=int, default=10000)
parser.add_argument("--resolution", type=int, default=48, help="ns=ntheta=nzeta for the Boozer spline")
parser.add_argument("--tol", type=float, default=1e-8, help="abstol=reltol for the orbit integrator")
parser.add_argument("--tmax", type=float, default=1e-2, help="integration time [s]")
parser.add_argument("--tag", default="scan")
parser.add_argument("--ns-bcrit", type=int, default=200, help="s-grid points for Bmin(s)/Bmax(s)")
parser.add_argument("--ntheta-bcrit", type=int, default=100, help="theta-grid points for Bmin(s)/Bmax(s)")
parser.add_argument("--nzeta-bcrit", type=int, default=100, help="zeta-grid points for Bmin(s)/Bmax(s)")
parser.add_argument("--seed", type=int, default=1000, help="base RNG seed (offset by MPI rank for positions)")
args = parser.parse_args()

order = 3  # radial interpolation order
degree = 3  # 3d interpolation degree

setup_logging(f"stdout_{args.tag}_{args.nparticles}_{args.resolution}_{comm_size}.txt")

## Setup field interpolation
field = InterpolatedBoozerField.from_booz_xform(
    args.boozmn,
    degree=degree,
    ns=args.resolution,
    ntheta=args.resolution,
    nzeta=args.resolution,
    comm=comm_world,
)

# Fusion reactivity profile (Bader et al. 2021)
nD = lambda s: 1 - s**5
nT = nD
T = lambda s: 11.5 * (1 - s)  # keV


def sigmav(T):
    if T > 0:
        return T ** (-2 / 3) * np.exp(-19.94 * T ** (-1 / 3))
    return 0


reactivity = lambda s: nD(s) * nT(s) * sigmav(T(s))

## NOTE: initialize_position_profile applies `np.random.seed(seed)` identically
## on EVERY MPI rank, then each rank draws only its own parallel_loop_bounds
## slice from that identical stream. Passing a single global seed therefore
## makes all ranks generate the SAME positions, and the allgather returns
## nparticles/comm_size unique points each replicated comm_size times (verified:
## seed=0 at n=10000 on 128 ranks gave only 79 unique positions). Offsetting the
## seed by rank gives each rank an independent-but-reproducible stream.
## (initialize_velocity_uniform is unaffected -- it draws on rank 0 and bcasts.)
rank = comm_world.rank if comm_world is not None else 0
points_init = initialize_position_profile(
    field, args.nparticles, reactivity, comm=comm_world, seed=args.seed + rank
)

Ekin = FUSION_ALPHA_PARTICLE_ENERGY
mass = ALPHA_PARTICLE_MASS
charge = ALPHA_PARTICLE_CHARGE
vpar0 = np.sqrt(2 * Ekin / mass)
vpar_init = initialize_velocity_uniform(vpar0, args.nparticles, comm=comm_world, seed=args.seed)

proc0_print(
    f"Tracing {args.nparticles} alphas on {args.tag} "
    f"(res={args.resolution}, tol={args.tol}, tmax={args.tmax} s) ..."
)
time1 = time.time()
res_tys, res_zeta_hits = trace_particles_boozer(
    field,
    points_init,
    vpar_init,
    tmax=args.tmax,
    mass=mass,
    charge=charge,
    comm=comm_world,
    Ekin=Ekin,
    stopping_criteria=[MaxToroidalFluxStoppingCriterion(1.0)],
    forget_exact_path=True,
    abstol=args.tol,
    reltol=args.tol,
)
time2 = time.time()
proc0_print(f"Elapsed time for tracing = {time2 - time1:.1f} s")

## Post-process (rank 0)
if verbose:
    from firm3d.field.trajectory_helpers import compute_loss_fraction

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times, loss_frac = compute_loss_fraction(res_tys, tmin=1e-5, tmax=args.tmax)

    plt.figure()
    plt.loglog(times, loss_frac)
    plt.xlim([1e-5, args.tmax])
    plt.ylim([1e-3, 1])
    plt.xlabel("Time [s]")
    plt.ylabel("Fraction of lost particles")
    plt.savefig(f"loss_fraction_{args.tag}.png", dpi=150)

    t_end = np.array([ty[-1, 0] for ty in res_tys])
    lost = t_end < 0.999 * args.tmax

    ## Bcrit trapped-particle classification.
    ## Bcrit = B_init / (1 - xi_init^2): the field strength at which a
    ## particle mirrors (vpar=0), from conservation of Ekin and the
    ## magnetic moment mu = m*vperp^2/(2B) along its (unperturbed) orbit.
    ## Trapped particles have Bcrit < Bmax(s_init) (they mirror before
    ## reaching the flux surface's field maximum); passing particles have
    ## Bcrit >= Bmax(s_init) (Bcrit -> inf as xi_init -> +-1).
    nfp = field.nfp
    s_init = points_init[:, 0]
    xi_init = vpar_init / vpar0

    field.set_points(points_init)
    B_init = field.modB()[:, 0]
    Bcrit = B_init / (1 - xi_init**2)

    s_grid = np.linspace(0, 1, args.ns_bcrit)
    theta_grid = np.linspace(0, 2 * np.pi, args.ntheta_bcrit, endpoint=False)
    zeta_grid = np.linspace(0, 2 * np.pi / nfp, args.nzeta_bcrit, endpoint=False)
    Bmin_s = np.empty(args.ns_bcrit)
    Bmax_s = np.empty(args.ns_bcrit)
    grid_pts = np.zeros((args.ntheta_bcrit * args.nzeta_bcrit, 3))
    tt, zz = np.meshgrid(theta_grid, zeta_grid, indexing="ij")
    grid_pts[:, 1] = tt.flatten()
    grid_pts[:, 2] = zz.flatten()
    for i, s in enumerate(s_grid):
        grid_pts[:, 0] = s
        field.set_points(grid_pts)
        modB = field.modB()[:, 0]
        Bmin_s[i] = modB.min()
        Bmax_s[i] = modB.max()

    Bmin_at_particle = np.interp(s_init, s_grid, Bmin_s)
    Bmax_at_particle = np.interp(s_init, s_grid, Bmax_s)

    trapped = Bcrit < Bmax_at_particle
    Bcrit_norm = (Bcrit - Bmin_at_particle) / (Bmax_at_particle - Bmin_at_particle)

    ## Validation -- these must hold if the sampling and Bmin/Bmax grids are sane
    n_unique = len(np.unique(points_init, axis=0))
    proc0_print(f"[check] unique initial positions: {n_unique}/{args.nparticles}")
    if n_unique < args.nparticles:
        proc0_print("[check] *** WARNING: duplicate initial conditions (seeding bug?) ***")
    below = (B_init < Bmin_at_particle * (1 - 1e-9)).sum()
    above = (B_init > Bmax_at_particle * (1 + 1e-9)).sum()
    proc0_print(f"[check] B_init outside [Bmin(s), Bmax(s)]: {below} below, {above} above")
    proc0_print(
        f"[check] Bcrit_norm range (trapped): "
        f"[{Bcrit_norm[trapped].min():.4f}, {Bcrit_norm[trapped].max():.4f}]"
    )
    proc0_print(
        f"[check] xi_init: min={xi_init.min():.4f} max={xi_init.max():.4f} "
        f"mean={xi_init.mean():.4f} (expect ~uniform on [-1,1])"
    )
    proc0_print(
        f"[check] mirror ratio Bmax/Bmin: axis={Bmax_s[0]/Bmin_s[0]:.4f} "
        f"mid={Bmax_s[len(s_grid)//2]/Bmin_s[len(s_grid)//2]:.4f} "
        f"edge={Bmax_s[-1]/Bmin_s[-1]:.4f}"
    )

    np.savez(
        f"fusion_distribution_{args.tag}.npz",
        t_end=t_end,
        lost=lost,
        loss_times=times,
        loss_fraction=loss_frac,
        nparticles=args.nparticles,
        resolution=args.resolution,
        tol=args.tol,
        tmax=args.tmax,
        s_init=s_init,
        theta_init=points_init[:, 1],
        zeta_init=points_init[:, 2],
        xi_init=xi_init,
        B_init=B_init,
        Bmin_at_particle=Bmin_at_particle,
        Bmax_at_particle=Bmax_at_particle,
        Bcrit=Bcrit,
        Bcrit_norm=Bcrit_norm,
        trapped=trapped,
        s_grid=s_grid,
        Bmin_s=Bmin_s,
        Bmax_s=Bmax_s,
    )
    proc0_print(
        f"final loss fraction ({args.tag}): {loss_frac[-1]:.3f} "
        f"({lost.sum()}/{args.nparticles} particles); "
        f"trapped: {trapped.sum()}/{args.nparticles}"
    )
