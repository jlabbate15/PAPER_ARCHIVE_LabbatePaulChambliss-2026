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


# COMMON USER INPUTS #
# boozmn_filename = "/Users/paullab/codes/Equilibria/APS 2025 Poster/QI/boozmn2_wout_betaSQuIDhs.nc"
boozmn_filename = ""
Ekin = FUSION_ALPHA_PARTICLE_ENERGY
neta_poinc = 5  # Number of eta initial conditions for poincare
ns_poinc = 100  # Number of s initial conditions for poincare
Nmaps = 1000  # Number of Poincare return maps to compute
modBin = 5.95 # T
call_DESC = False
tmax = 1e-4
axis = 2
#######################


charge = ALPHA_PARTICLE_CHARGE
mass = ALPHA_PARTICLE_MASS

resolution = 48  # Resolution for field interpolation
ns_interp = resolution  # number of radial grid points for interpolation
ntheta_interp = resolution  # number of poloidal grid points for interpolation
nzeta_interp = resolution  # number of toroidal grid points for interpolation
order = 3  # order for interpolation
tol = 1e-8  # Tolerance for ODE solver
s_mirror = 0.2**2  # flux surface for mirroring
theta_mirror = np.pi / 2  # poloidal angle for mirroring
zeta_mirror = 0
helicity_M = 1  # helicity of field strength contours
helicity_N = 0 # nfp for QI, 0 for QA
degree = 3  # Degree for Lagrange interpolation


# Setup logging to redirect output to file
# setup_logging(f"stdout_trapped_map_{resolution}_{comm_size}.txt")

time1 = time.time()

bri_noQSbreak = BoozerRadialInterpolant(boozmn_filename, order, no_K=True, comm=comm_world, helicity_N=helicity_N, helicity_M=helicity_M) # specify helicities to filter QS-breaking modes
bri_QSbreak = BoozerRadialInterpolant(boozmn_filename, order, no_K=True, comm=comm_world)

field_noQSbreak = InterpolatedBoozerField(
    bri_noQSbreak,
    degree,
    ns_interp=ns_interp,
    ntheta_interp=ntheta_interp,
    nzeta_interp=nzeta_interp,
    stellsym=True
)
field_QSbreak = InterpolatedBoozerField(
    bri_QSbreak,
    degree,
    ns_interp=ns_interp,
    ntheta_interp=ntheta_interp,
    nzeta_interp=nzeta_interp,
    stellsym=True
)

print("start tracing")

poinc_noQSbreak = TrappedPoincare(
    field_noQSbreak,
    helicity_M,
    helicity_N,
    s_mirror,
    theta_mirror,
    zeta_mirror,
    mass,
    charge,
    Ekin,
    modBin=modBin,
    ns_poinc=ns_poinc,
    neta_poinc=neta_poinc,
    Nmaps=Nmaps,
    comm=comm_world,
    solver_options={"reltol": tol, "abstol": tol, "axis": axis},
    tmax=tmax,
    # s_init=np.linspace(0,1,ns_interp)
)

poinc_QSbreak = TrappedPoincare(
    field_QSbreak,
    helicity_M,
    helicity_N,
    s_mirror,
    theta_mirror,
    zeta_mirror,
    mass,
    charge,
    Ekin,
    modBin=modBin,
    ns_poinc=ns_poinc,
    neta_poinc=neta_poinc,
    Nmaps=Nmaps,
    comm=comm_world,
    solver_options={"reltol": tol, "abstol": tol, "axis": axis},
    tmax=tmax,
    # s_init=np.linspace(0,1,ns_interp)
)

time2 = time.time()

proc0_print("poincare time: ", time2 - time1)

# Compute frequencies
# omega_eta_prof_nobreak, omega_b_prof_nobreak, s_prof_nobreak = poinc_noQSbreak.compute_frequencies()

# if verbose and not call_DESC:
import matplotlib.pyplot as plt
fig, ax = plt.subplots(nrows=1, ncols=3)
ax[0] = poinc_QSbreak.plot_poincare(ax=ax[0],filename='poinc_QSbreak_test',save_points=True,num=0)
ax[1] = poinc_noQSbreak.plot_poincare(ax=ax[1],filename='poinc_noQSbreak_test',save_points=True,num=1)
# np.save("freq_test",np.array(omega_eta_prof_nobreak/omega_b_prof_nobreak,dtype=object))
# np.save("s_test",np.array(s_prof_nobreak,dtype=object))
# ax[2].plot(omega_eta_prof_nobreak/omega_b_prof_nobreak,s_prof_nobreak)
# ax[2].set_xlabel(r'$\Omega_{\eta}$')
# ax[2].yaxis.set_label_position("right")
# ax[2].yaxis.tick_right()
# ax[2].set_ylabel(r'$\rho$')
# ax[2].set_ylim(0.0,1.0)
# ax[2].set_xlim(-0.0075,0.0001)
plt.savefig("poincare_omega_breaknobreak.png")
