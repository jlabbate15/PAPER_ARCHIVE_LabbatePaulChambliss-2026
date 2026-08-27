import time

import numpy as np

from firm3d.field.boozermagneticfield import (
    InterpolatedBoozerField,
)
from firm3d.field.trajectory_helpers import TrappedPoincare
from firm3d.util.constants import (
    ALPHA_PARTICLE_CHARGE,
    ALPHA_PARTICLE_MASS,
    FUSION_ALPHA_PARTICLE_ENERGY,
)
from firm3d.util.functions import in_github_actions, proc0_print, setup_logging
from firm3d.util.mpi import comm_size, comm_world, verbose

boozmn_filename = "/Users/paullab/codes/firm3d/examples/trapped_map_QI/wout_v20250319_v2_HighShear_000_000000_desc.nc"
charge = ALPHA_PARTICLE_CHARGE
mass = ALPHA_PARTICLE_MASS
Ekin = FUSION_ALPHA_PARTICLE_ENERGY

in_github_actions = False
resolution = 10 if in_github_actions else 48  # Resolution for field interpolation
neta_poinc = 10  # Number of eta initial conditions for poincare
ns_poinc = 5 if in_github_actions else 120  # Number of s initial conditions
Nmaps = 5 if in_github_actions else 1000  # Number of Poincare return maps to compute
ns_interp = resolution  # number of radial grid points for interpolation
ntheta_interp = resolution  # number of poloidal grid points for interpolation
nzeta_interp = resolution  # number of toroidal grid points for interpolation
order = 3  # order for interpolation
tol = 1e-4 if in_github_actions else 1e-8  # Tolerance for ODE solver
s_mirror = 0.5  # flux surface for mirroring
theta_mirror = 0  # poloidal angle for mirroring
helicity_M = 0  # helicity of field strength contours
degree = 3  # Degree for Lagrange interpolation

# Setup logging to redirect output to file
setup_logging(f"stdout_trapped_map_QI_{resolution}_{comm_size}.txt")

time1 = time.time()

Bcrits = [6.5,6.6,6.7,6.8]

field = InterpolatedBoozerField.from_booz_xform(
    boozmn_filename,
    degree=order,
    ns=ns_interp,
    ntheta=ntheta_interp,
    nzeta=nzeta_interp,
    helicity_M=helicity_M,
    helicity_N=-4, # hardcoding nfp=4=N
    comm=comm_world,
)
field_pert = InterpolatedBoozerField.from_booz_xform(
    boozmn_filename,
    degree=order,
    ns=ns_interp,
    ntheta=ntheta_interp,
    nzeta=nzeta_interp,
    # helicity_M=helicity_M,
    # helicity_N=-4, # hardcoding nfp=4=N
    comm=comm_world,
)
nfp = field.nfp
helicity_N = nfp  # helicity of field strength contours
print(f"nfp: {nfp}")
zeta_mirror = np.pi / (2 * nfp)  # poloidal angle for mirroring


for Bcrit in Bcrits:

    poinc = TrappedPoincare(
        field,
        helicity_M,
        helicity_N,
        mass,
        charge,
        Ekin,
        # s_mirror,
        # theta_mirror,
        # zeta_mirror,
        lam=1/Bcrit,
        ns_poinc=ns_poinc,
        neta_poinc=1, # no eta averaging, only eta=0
        Nmaps=Nmaps,
        comm=comm_world,
        solver_options={"reltol": tol, "abstol": tol, "axis": 0},
        tmax=1e-4,
    )

    omega_eta_prof, omega_b_prof, s_prof = poinc.compute_frequencies()

    data = {
        'omega_eta': omega_eta_prof,
        'omega_b': omega_b_prof,
        'Omega_eta': omega_eta_prof / omega_b_prof,
        's': s_prof,
    }
    np.savez(f"freq_unpert_{Bcrit}.npz", **data)


    poinc_pert = TrappedPoincare(
        field_pert,
        helicity_M,
        helicity_N,
        mass,
        charge,
        Ekin,
        # s_mirror,
        # theta_mirror,
        # zeta_mirror,
        lam=1/Bcrit,
        ns_poinc=ns_poinc,
        neta_poinc=neta_poinc,
        Nmaps=Nmaps,
        comm=comm_world,
        solver_options={"reltol": tol, "abstol": tol, "axis": 0},
        tmax=1e-4,
    )

    poinc_pert.plot_poincare(save_data=True, data_filename=f"poinc_pert_{Bcrit}.npz")

    # omega_eta_prof, omega_b_prof, s_prof = poinc_pert.compute_frequencies()

    # data = {
    #     'omega_eta': omega_eta_prof,
    #     'omega_b': omega_b_prof,
    #     'Omega_eta': omega_eta_prof / omega_b_prof,
    #     's': s_prof,
    # }
    # np.savez(f"freq_pert_{Bcrit}.npz", **data)
    
time2 = time.time()

proc0_print("total time: ", time2 - time1)
