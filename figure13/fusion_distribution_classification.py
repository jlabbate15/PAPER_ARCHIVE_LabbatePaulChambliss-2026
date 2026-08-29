import sys
import time
from pathlib import Path

import numpy as np

from firm3d._core.util import parallel_loop_bounds
from firm3d.field.boozermagneticfield import (
    BoozerRadialInterpolant,
    InterpolatedBoozerField,
)
from firm3d.field.tracing import (
    MaxToroidalFluxStoppingCriterion,
    trace_particles_boozer,
)
from firm3d.field.tracing_helpers import (
    initialize_position_profile,
)
from firm3d.plotting.orbit_classification import OrbitClassification
from firm3d.util.constants import (
    ALPHA_PARTICLE_CHARGE,
    ALPHA_PARTICLE_MASS,
    FUSION_ALPHA_PARTICLE_ENERGY,
)
from firm3d.util.functions import proc0_print

try:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    verbose = comm.rank == 0
    comm_size = comm.size
except ImportError:
    comm = None
    verbose = True
    comm_size = 1

time1 = time.time()

resolution = 48  # Resolution for field interpolation
nParticles_per_Bcrit = 1000  # Number of particles to trace
reltol = 1e-8  # Relative tolerance for the ODE solver
abstol = 1e-8  # Absolute tolerance for the ODE solver
order = 3  # Order for radial interpolation
degree = 3  # Degree for 3d interpolation


equil_filename = sys.argv[1] if len(sys.argv) > 1 else ""
if not equil_filename:
    raise ValueError("Usage: python fusion_distribution_classification.py <wout_or_boozmn_file.nc>")
# Resolve to absolute path so the file is found regardless of cwd
equil_path = Path(equil_filename)
if not equil_path.is_absolute():
    equil_path = Path.cwd() / equil_path
if not equil_path.exists():
    raise FileNotFoundError(f"Equilibrium file not found: {equil_path}")
equil_filename = str(equil_path.resolve())
output_prefix = equil_path.stem

tmax = 1e-1  # Time for integration
ns_interp = resolution
ntheta_interp = resolution
nzeta_interp = resolution
# Helicity of the field strength for classifying ripple and barely-trapped
helicity_M = 0
dt_save = 1e-7  # Time interval for saving trajectory points

# Redirect stdout to file for entire script duration
stdout_file = open(  # noqa: SIM115
    f"stdout_{output_prefix}_{nParticles_per_Bcrit}_{resolution}_{comm_size}.txt", "a", buffering=1
)
sys.stdout = stdout_file

## Setup radial interpolation
bri = BoozerRadialInterpolant(equil_filename, order, no_K=True, comm=comm)

helicity_N = bri.nfp

## Setup 3d interpolation
field = InterpolatedBoozerField(
    bri,
    degree,
    ns_interp=ns_interp,
    ntheta_interp=ntheta_interp,
    nzeta_interp=nzeta_interp,
)

# Define fusion birth distribution
# Bader, A., et al. "Modeling of energetic particle transport in optimized
# stellarators." Nuclear Fusion 61.11 (2021): 116060.
nD = lambda s: 1 - s**5  # Normalized density
nT = nD
T = lambda s: 11.5 * (1 - s)  # Temperature in keV


# D-T cross-section
def sigmav(T):
    if T > 0:
        return T ** (-2 / 3) * np.exp(-19.94 * T ** (-1 / 3))
    else:
        return 0


# Reactivity profile
reactivity = lambda s: nD(s) * nT(s) * sigmav(T(s))

Ekin = FUSION_ALPHA_PARTICLE_ENERGY
mass = ALPHA_PARTICLE_MASS
charge = ALPHA_PARTICLE_CHARGE
# Initialize uniformly distributed parallel velocities
vpar0 = np.sqrt(2 * Ekin / mass)

Bcrit = np.array([6.24])
loss_count_per_Bcrit = np.zeros(len(Bcrit), dtype=float)
loss_fraction_per_Bcrit = np.zeros(len(Bcrit), dtype=float)
rng = np.random.default_rng(seed=comm.rank if comm is not None else 0)

max_init_tries_per_particle = 100
for ilam in range(len(Bcrit)):
    this_Bcrit = Bcrit[ilam]
    bcrit_tag = f"{this_Bcrit:.4f}".replace(".", "p")
    bcrit_initialization_failed_local = False
    loss_count_local = 0
    traced_count_local = 0
    first, last = parallel_loop_bounds(comm, nParticles_per_Bcrit)
    for iParticle in range(first, last):
        proc0_print('Initializing particle', iParticle, 'of', nParticles_per_Bcrit)
        particle_initialized = False
        for _ in range(max_init_tries_per_particle):
            point_init = initialize_position_profile(
                field, 1, reactivity, seed=int(rng.integers(0, 2**31 - 1))
            )
            field.set_points(point_init)
            modB = field.modB()[0,0]
            if modB < this_Bcrit:
                vpar_particle = rng.choice([-1, 1]) * vpar0 * np.sqrt(1 - modB / this_Bcrit)
                particle_initialized = True
                break
        if not particle_initialized:
            proc0_print(
                f"Failed to initialize particle {iParticle} for Bcrit={this_Bcrit:.4f} "
                f"after {max_init_tries_per_particle} tries. Skipping this Bcrit."
            )
            bcrit_initialization_failed_local = True
            break

        point = np.zeros((1, 3))
        point[0, :] = point_init[0, :]
        traced_count_local += 1
        ## Trace alpha particles in Boozer coordinates until they hit the s = 1 surface
        res_tys, res_hits = trace_particles_boozer(
            field,
            point,
            [vpar_particle],
            tmax=tmax,
            mass=mass,
            charge=charge,
            Ekin=Ekin,
            vpars=[0],
            vpars_stop=False,
            stopping_criteria=[MaxToroidalFluxStoppingCriterion(1.0)],
            forget_exact_path=False,
            abstol=abstol,
            reltol=reltol,
            dt_save=dt_save,
        )
    
        res_hit = res_hits[0]
        res_ty = res_tys[0]
        bounce_times = []
        if len(res_hit) > 0:
            if np.any(res_hit[:, 1] == -1):  # Particle was lost to the wall
                loss_count_local += 1
                np.savetxt(f"{output_prefix}_Bcrit_{bcrit_tag}_particle_{iParticle}_traj.txt", res_ty)
                np.savetxt(f"{output_prefix}_Bcrit_{bcrit_tag}_particle_{iParticle}_hits.txt", res_hit)
            else:
                continue  # Particle was not lost to the wall, skip
    
            oc = OrbitClassification(field, Ekin, mass, charge, helicity_M, helicity_N)
            particle_dict = oc.classify_orbit(res_ty, res_hit)
            np.savez(f"{output_prefix}_Bcrit_{bcrit_tag}_particle_{iParticle}.npz", **particle_dict)

    if comm is not None:
        bcrit_initialization_failed_global = bool(
            comm.allreduce(int(bcrit_initialization_failed_local), op=MPI.MAX)
        )
    else:
        bcrit_initialization_failed_global = bcrit_initialization_failed_local

    if bcrit_initialization_failed_global:
        loss_count_per_Bcrit[ilam] = np.nan
        loss_fraction_per_Bcrit[ilam] = np.nan
        proc0_print(
            f"Bcrit={this_Bcrit:.4f} skipped because one or more ranks failed "
            f"particle initialization within {max_init_tries_per_particle} tries."
        )
        continue

    if comm is not None:
        loss_count_global = comm.allreduce(loss_count_local, op=MPI.SUM)
        traced_count_global = comm.allreduce(traced_count_local, op=MPI.SUM)
    else:
        loss_count_global = loss_count_local
        traced_count_global = traced_count_local

    loss_fraction = (
        loss_count_global / traced_count_global if traced_count_global > 0 else np.nan
    )
    loss_count_per_Bcrit[ilam] = loss_count_global
    loss_fraction_per_Bcrit[ilam] = loss_fraction
    proc0_print(
        f"Bcrit={this_Bcrit:.4f}, lost={loss_count_global}/{traced_count_global}, "
        f"loss_fraction={loss_fraction:.6f}"
    )

if verbose:
    loss_table = np.column_stack((Bcrit, loss_count_per_Bcrit, loss_fraction_per_Bcrit))
    np.savetxt(
        f"{output_prefix}_loss_fraction_vs_Bcrit.txt",
        loss_table,
        header="Bcrit loss_count loss_fraction",
        fmt="%.8e",
    )

proc0_print(
    f"Total time for tracing and classifying particles: "
    f"{time.time() - time1:.2f} seconds"
)
