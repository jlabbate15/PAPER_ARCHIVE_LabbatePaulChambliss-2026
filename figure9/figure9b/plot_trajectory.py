import time

import numpy as np

from firm3d.field.boozermagneticfield import (
    BoozerRadialInterpolant,
    InterpolatedBoozerField,
)
from firm3d.field.tracing import (
    MaxToroidalFluxStoppingCriterion,
    trace_particles_boozer,
)
from firm3d.plotting.plotting_helpers import (
    plot_trajectory_overhead_cyl,
    plot_trajectory_poloidal,
)
from firm3d.util.constants import (
    ALPHA_PARTICLE_CHARGE,
    ALPHA_PARTICLE_MASS,
    FUSION_ALPHA_PARTICLE_ENERGY,
)
from firm3d.field.trajectory_helpers import TrappedPoincare
from firm3d.util.functions import proc0_print, setup_logging
from firm3d.util.mpi import comm_size, comm_world, verbose

time1 = time.time()

boozmn_filename = "/Users/paullab/codes/firm3d/examples/trapped_map/optimized_poincs/wout_scan62_woutDres.nc"
order = 3  # Order for radial interpolation
reltol = 1e-8  # Relative tolerance for the ODE solver
abstol = 1e-8  # Absolute tolerance for the ODE solver
tmax = 1e-2  # Time for integration
resolution = 48  # Resolution for field interpolation
degree = 3  # Degree for 3d interpolation
ns_interp = resolution
ntheta_interp = resolution
nzeta_interp = resolution

# Setup logging to redirect output to file
setup_logging(f"stdout_trajectory_{resolution}_{comm_size}.txt")

## Setup radial interpolation
bri = BoozerRadialInterpolant(boozmn_filename, order, no_K=True, comm=comm_world)
nfp = bri.nfp
# QI helicity 
# helicity_N = nfp
# helicity_M = 0
# QA helicity
helicity_N = 0
helicity_M = 1

## Setup 3d interpolation
field = InterpolatedBoozerField(
    bri,
    degree,
    ns_interp=ns_interp,
    ntheta_interp=ntheta_interp,
    nzeta_interp=nzeta_interp,
)

Ekin = FUSION_ALPHA_PARTICLE_ENERGY
mass = ALPHA_PARTICLE_MASS
charge = ALPHA_PARTICLE_CHARGE

s_point = 0.4
eta_point = 3

Bcrit = 4.8199

# Use trapped poincare to find initial conditions for trajectory plotting 
poinc = TrappedPoincare(
    field,
    helicity_M,
    helicity_N,
    mass,
    charge,
    Ekin,
    lam = 1/Bcrit,
    s_init = [s_point],
    etas_init = [eta_point],
    Nmaps=1,
    comm=comm_world,
    solver_options={"reltol": 1e-8, "abstol": 1e-8, "axis": 0},
    tmax=1e-2,
)
chi_point = poinc.chis_init[0]

theta_point, zeta_point = poinc.chi_eta_to_theta_zeta(chi_point, eta_point)

# Initialize single trapped particle on s = 0.5 surface with random theta and
# zeta, and zero parallel velocity
vpar0 = np.sqrt(2 * Ekin / mass)
vpar_init = [0]
points = np.zeros((1, 3))
points[0, 0] = s_point  # s = 0.5 surface
points[0, 1] = theta_point  # Random theta
points[0, 2] = zeta_point  # Random zeta

## Trace alpha particle in Boozer coordinates until it hits the s = 1 surface
## Set forget_exact_path=False to save the trajectory information.
## Set the dt_save parameter to the time interval for trajectory data
## to be saved.
traj_booz, res_hits = trace_particles_boozer(
    field,
    points,
    vpar_init,
    tmax=tmax,
    mass=mass,
    charge=charge,
    Ekin=Ekin,
    # stopping_criteria=[MaxToroidalFluxStoppingCriterion(1.0)],
    forget_exact_path=False,
    dt_save=1e-7,
    abstol=abstol,
    reltol=reltol,
)

time2 = time.time()

proc0_print("Elapsed time for tracing: ", time2 - time1)

# ax = plot_trajectory_overhead_cyl(traj_booz[0], field)
# verbose = True
# if verbose:
np.savetxt(f"traj_booz_woutDres_s{s_point}_eta{eta_point}_Bcrit{round(Bcrit, 3)}.txt", traj_booz[0])
#     fig = ax.figure
#     fig.savefig("trajectory_overhead_cyl.png", dpi=300, bbox_inches="tight")

# ax = plot_trajectory_poloidal(traj_booz[0], helicity_N=nfp)

# if verbose:
#     fig = ax.figure
#     fig.savefig("trajectory_poloidal.png", dpi=300, bbox_inches="tight")

#     from firm3d.field.trajectory_helpers import trajectory_to_vtk

#     trajectory_to_vtk(traj_booz[0], field, filename="trajectory")

time3 = time.time()
proc0_print("Elapsed time for saving: ", time3 - time2)