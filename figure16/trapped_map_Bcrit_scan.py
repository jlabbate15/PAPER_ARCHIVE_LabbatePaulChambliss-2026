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

output_dir = "data_t1e-1"
n_Bcrit = 32  # Number of Bcrit values to scan

boozmn_filenames = ["wout_scan62_woutDres.nc", "wout_scan60_wDres.nc"] # QA
# boozmn_filenames = ["wout_scan60_wDres.nc"] # QA

charge = ALPHA_PARTICLE_CHARGE
mass = ALPHA_PARTICLE_MASS
Ekin = FUSION_ALPHA_PARTICLE_ENERGY

resolution = 48  # Resolution for field interpolation
neta_poinc = 5  # Number of eta initial conditions for poincare
ns_poinc = 120  # Number of s initial conditions for poincare
Nmaps = 1000  # Number of Poincare return maps to compute
ns_interp = resolution  # number of radial grid points for interpolation
ntheta_interp = resolution  # number of poloidal grid points for interpolation
nzeta_interp = resolution  # number of toroidal grid points for interpolation
order = 3  # order for interpolation
tol = 1e-8  # Tolerance for ODE solver
# s_mirror = 0.5  # flux surface for mirroring
# theta_mirror = np.pi / 2  # poloidal angle for mirroring
# zeta_mirror = 0
helicity_M = 1  # helicity of field strength contours
helicity_N = 0
degree = 3  # Degree for Lagrange interpolation


def bcrit_linspace(field, n=n_Bcrit, ns=32, ntheta=32, nzeta=32):
    """Return linspace of Bcrit between min and max |B| in the equilibrium."""
    nfp = field.nfp
    s = np.linspace(1e-3, 1.0, ns)
    theta = np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False)
    zeta = np.linspace(0.0, 2.0 * np.pi / nfp, nzeta, endpoint=False)
    S, T, Z = np.meshgrid(s, theta, zeta, indexing="ij")
    points = np.column_stack((S.ravel(), T.ravel(), Z.ravel()))
    field.set_points(points)
    modB = field.modB()[:, 0]
    return np.linspace(np.min(modB), np.max(modB), n)


# Setup logging to redirect output to file
setup_logging(f"stdout_trapped_map_{resolution}_{comm_size}.txt")

for boozmn_filename in boozmn_filenames:

    time1 = time.time()

    bri = BoozerRadialInterpolant(
        boozmn_filename, 
        order, 
        no_K=True, 
        comm=comm_world,
        # helicity_M=helicity_M, # uncomment for perfect
        # helicity_N=helicity_N, # uncomment for perfect
    )

    field = InterpolatedBoozerField(
        bri,
        degree,
        ns_interp=ns_interp,
        ntheta_interp=ntheta_interp,
        nzeta_interp=nzeta_interp,
    )

    # Bcrits = bcrit_linspace(field)
    '''
    
    If you want exactly [0.26, 0.29, 0.58, 0.74]: scan62 → [4.8244, 4.8931, 5.5577, 5.9243], scan60 → [4.8289, 4.8852, 5.4291, 5.7292].

    '''
    if boozmn_filename == "wout_scan62_woutDres.nc":
        Bcrits = np.array([4.8244, 4.8931, 5.5577, 5.9243])
    elif boozmn_filename == "wout_scan60_wDres.nc":
        Bcrits = np.array([4.8289, 4.8852, 5.4291, 5.7292])
    print('Bcrits: ', Bcrits)
    for Bcrit in Bcrits:
        print(f"Computing trapped map for Bcrit = {Bcrit}")
        poinc = TrappedPoincare(
            field,
            helicity_M,
            helicity_N,
            # s_mirror,
            # theta_mirror,
            # zeta_mirror,
            mass,
            charge,
            Ekin,
            lam = 1/Bcrit,
            ns_poinc=ns_poinc,
            neta_poinc=neta_poinc,
            Nmaps=Nmaps,
            comm=comm_world,
            solver_options={"reltol": tol, "abstol": tol, "axis": 0},
            tmax=1e-1,
        )

        # if verbose:
        poinc.plot_poincare(filename=f'{output_dir}/trapped_map_{boozmn_filename}_Bcrit_{Bcrit}.png',save_data=True, data_filename=f'{output_dir}/trapped_map_{boozmn_filename}_Bcrit_{Bcrit}.npz')
        omega_eta_prof, omega_b_prof, s_prof = poinc.compute_frequencies()
        np.savez(f'{output_dir}/frequencies_{boozmn_filename}_Bcrit_{Bcrit}.npz', omega_eta_prof=omega_eta_prof, omega_b_prof=omega_b_prof, s_prof=s_prof)

        time2 = time.time()

        proc0_print("elapsed time for equilibrium: ", time2 - time1)