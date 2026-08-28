"""Compare Omega_eta from particle tracing vs bounce averaging in DESC.

For a quasisymmetric or quasi-isodynamic equilibrium (e.g. the precise QA / QH
of Landreman & Paul 2022), compute the trapped-particle precession frequency

    Omega_eta = (d eta / dt) / (2 pi),   eta = Np * alpha / (N - iota * M),

where alpha = theta_PEST - iota * phi is the field line label, Np is the number
of field periods, and (M, N) are the generalized helicity integers of
Landreman & Catto 2012 (|B| = B(psi, M theta - N phi)):

    QA : (M, N) = (1, 0)      -> eta = Np * alpha / (-iota)
    QH : (M, N) = (1, +/-Np)  -> eta = Np * alpha / (N - iota)
    QI : (M, N) = (0, 1)      -> eta = Np * alpha

The two computations are:

1. Guiding-center particle tracing (``desc.particles``): trace a trapped particle
   with the vacuum guiding-center model, unwrap alpha(t) along the trajectory,
   and fit the secular drift d(alpha)/dt.

2. Bounce averaging (``desc.integrals.Bounce1D``): the vacuum guiding-center
   binormal drift is

       d(alpha)/dt = (m/q) (v_par^2 + v_perp^2 / 2) * gbdrift,
       gbdrift = (b x grad|B|) . grad(alpha) / |B|^2,

   which is bounce averaged with weight dl/|v_par| over the magnetic well
   containing the particle's starting point:

       <d(alpha)/dt> = (m v^2 / q)
           * Int dl (1 - lambda B / 2) gbdrift / sqrt(1 - lambda B)
           / Int dl 1 / sqrt(1 - lambda B).

For each flux surface, particles are initialized on ``num_alpha`` field lines

    alpha = np.linspace(0, 2*pi, num_alpha, endpoint=False),
    alpha = theta - iota * zeta,

seeded at zeta0 with theta0 = alpha + iota * zeta0. Particles start at each
field line's |B| minimum. The Bcrit scan uses surface-wide |B| extrema
(min/max of |B| over the flux surface, not per field line):

    Bcrit = Bmin_s + kappa * (Bmax_s - Bmin_s),   kappa in (0, 1),

with pitch lambda = 1/Bcrit (so v_par = 0 where |B| = Bcrit). At each energy the
RMSE over (alpha, Bcrit) of the relative residual is saved and plotted vs energy
fraction (one curve per surface).

Run (from the DESC repo root, in an environment with DESC installed):

    # precise QA (defaults: helicity (1, 0), Np from the equilibrium)
    python omega_eta_tracing_vs_bounce.py
    # precise QH (helicity N = -Np for this configuration)
    python omega_eta_tracing_vs_bounce.py --eq precise_QH --helicity 1 -4
    # QI-like configuration from a file, helicity (0, 1), 5 field lines
    python omega_eta_tracing_vs_bounce.py --eq my_qi.h5 --helicity 0 1 \\
        --zeta0 0.785 --num-alpha 5
    python omega_eta_tracing_vs_bounce.py --fractions 1e-4 1e-2 1 --s 0.5 --n-bcrit 12

Notes
-----
* The bounce-averaged result is exactly linear in energy (zero-orbit-width
  limit), while tracing includes finite-orbit-width effects, so the relative
  difference grows toward full energy.
* Runtime is dominated by tracing at the lowest energies (bounce period scales
  as 1/v), so the default energy grid is modest. Extend ``--fractions`` as
  needed.
"""

import argparse
import gc
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import elementary_charge, proton_mass

import jax

from desc.backend import jnp
from desc.equilibrium.coords import get_rtz_grid
from desc.examples import get
from desc.grid import Grid, LinearGrid
from desc.integrals import Bounce1D
from desc.particles import (
    ManualParticleInitializerFlux,
    VacuumGuidingCenterTrajectory,
    trace_particles,
)
from desc.utils import safediv

# 3.5 MeV alpha particle
E_ALPHA_EV = 3.5e6  # eV
MASS_AMU = 4.001506  # alpha particle mass in proton masses (m/m_p)
CHARGE_E = 2.0  # alpha particle charge in elementary charges


def eval_at_points(eq, nodes, names, chunk=500):
    """Compute quantities pointwise at arbitrary (rho, theta, zeta) nodes.

    Evaluates in chunks to bound memory (the JAX compute graph for thousands of
    arbitrary nodes can exhaust login-node memory).
    """
    nodes = np.atleast_2d(nodes)
    out = {name: [] for name in names}
    for k in range(0, nodes.shape[0], chunk):
        # jitable=False so unique surface indices are assigned; computing
        # "alpha" requires the iota profile, which needs grid.compress.
        grid = Grid(nodes[k : k + chunk], sort=False, jitable=False)
        data = eq.compute(list(names), grid=grid)
        for name in names:
            out[name].append(np.asarray(data[name]))
    return {name: np.concatenate(out[name]) for name in names}


def surface_iota(eq, rho):
    """Rotational transform on the rho surface."""
    grid = LinearGrid(rho=np.array([rho]), M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
    data = eq.compute(["iota"], grid=grid)
    return grid.compress(data["iota"]).item()


def scale_equilibrium(eq, a_target=1.70, B_target=5.86):
    """Rescale an equilibrium to a target minor radius and field strength.

    The DESC example equilibria are normalized to ~1 m major radius and ~1 T
    field, where a 3.5 MeV alpha has a gyroradius larger than the minor radius
    and drifts comparable to parallel streaming, so guiding-center orbits are
    unphysical. Landreman & Paul 2022 scale their configurations to the
    ARIES-CS minor radius (1.70 m) and volume-averaged field (5.86 T) before
    tracing alphas; this does the same.

    For a vacuum equilibrium this is an exact rescaling of the solution:
    lengths scale with R_lmn, Z_lmn and B scales as Psi / length^2.
    """
    data = eq.compute(["a", "<|B|>_vol"])  # default quadrature grid
    L_scale = a_target / float(data["a"])
    B_old = float(data["<|B|>_vol"])
    eq.R_lmn = eq.R_lmn * L_scale
    eq.Z_lmn = eq.Z_lmn * L_scale
    eq.Psi = float(eq.Psi) * L_scale**2 * (B_target / B_old)
    print(
        f"Scaled equilibrium: length x{L_scale:.3f}, "
        f"<|B|>_vol {B_old:.3f} -> {B_target} T, minor radius -> {a_target} m"
    )
    return eq


def eta_denominator(iota, helicity):
    """Denominator N - iota * M of the eta coordinate.

    (M, N) are the generalized helicity integers of Landreman & Catto 2012,
    i.e. |B| is (approximately) a function of (psi, M theta - N phi):
    QA (1, 0), QH (1, +/-Np), QI/omnigenous-poloidal (0, 1).
    """
    M, N = helicity
    den = N - iota * M
    if abs(den) < 1e-12:
        raise ValueError(
            f"eta is singular: N - iota*M = {den} for helicity {helicity} "
            f"and iota = {iota} (resonant surface)."
        )
    return den


def field_line_B_extrema(
    eq,
    rho,
    theta0,
    zeta0,
    num_transit=3,
    knots_per_transit=200,
):
    """|B| min/max on the field line through (rho, theta0, zeta0).

    Returns
    -------
    Bmin, Bmax : float
        Extrema of |B| along the field line segment.
    alpha0 : float
        Field-line label at the seed point.
    zeta_bmin : float
        Toroidal angle of the |B| minimum on the sampled segment (good start
        for deeply trapped particles).
    theta_bmin : float
        Poloidal angle at that |B| minimum.
    """
    seed = eval_at_points(eq, np.array([[rho, theta0, zeta0]]), ["alpha"])
    alpha0 = float(seed["alpha"][0])
    iota = surface_iota(eq, rho)
    zeta = np.linspace(
        zeta0 - num_transit * np.pi,
        zeta0 + num_transit * np.pi,
        num_transit * knots_per_transit + 1,
    )
    grid = get_rtz_grid(eq, rho, alpha0, zeta, coordinates="raz", iota=iota)
    data = eq.compute(["|B|", "theta"], grid=grid)
    B = np.asarray(data["|B|"])
    theta = np.asarray(data["theta"])
    i_min = int(np.argmin(B))
    return float(B.min()), float(B.max()), alpha0, float(zeta[i_min]), float(theta[i_min])


def surface_B_extrema(eq, rho, M=None, N=None):
    """|B| min/max on the flux surface rho (poloidal x toroidal grid)."""
    M = eq.M_grid if M is None else M
    N = eq.N_grid if N is None else N
    grid = LinearGrid(rho=np.array([rho]), M=M, N=N, NFP=eq.NFP)
    data = eq.compute(["|B|"], grid=grid)
    B = np.asarray(data["|B|"])
    return float(B.min()), float(B.max())


def bcrit_to_xi0(B_start, Bcrit):
    """Pitch v_par/v at a point with |B|=B_start for bounce field Bcrit=1/lambda."""
    ratio = B_start / Bcrit
    if ratio >= 1.0:
        raise ValueError(
            f"Particle not trapped at start: B_start={B_start} >= Bcrit={Bcrit}."
        )
    return float(np.sqrt(1.0 - ratio))


def omega_eta_tracing(
    eq,
    rho_values,
    energy_ev,
    xi0,
    tmax,
    n_save,
    Np,
    helicity=(1, 0),
    rtol=1e-7,
    theta0=0.0,
    zeta0=0.0,
):
    """Omega_eta from guiding center tracing, for all particles at once.

    Traces trapped particles (starting at theta0, zeta0 with pitch v_par/v =
    xi0) and extracts the secular drift of the unwrapped field line label
    alpha(t) from a fit through all banana tips. Typically one particle per
    (flux surface, field line) pair.

    ``xi0``, ``theta0``, and ``zeta0`` may be scalars or length-n arrays.

    Returns
    -------
    omega : ndarray, shape (len(rho_values),)
        Omega_eta = Np * <d(alpha)/dt> / ((N - iota M) * 2 pi) for each particle.
    """
    rho_values = np.atleast_1d(rho_values)
    n = rho_values.size
    xi0 = np.broadcast_to(np.atleast_1d(xi0), (n,)).astype(float)
    theta0 = np.broadcast_to(np.atleast_1d(theta0), (n,)).astype(float)
    zeta0 = np.broadcast_to(np.atleast_1d(zeta0), (n,)).astype(float)

    initializer = ManualParticleInitializerFlux(
        rho0=rho_values,
        theta0=theta0,
        zeta0=zeta0,
        xi0=xi0,
        E=np.full(n, energy_ev),
        m=MASS_AMU,
        q=CHARGE_E,
    )
    model = VacuumGuidingCenterTrajectory(frame="flux")
    ts = np.linspace(0, tmax, n_save)
    x, v = trace_particles(
        eq, initializer=initializer, model=model, ts=ts, rtol=rtol, atol=rtol
    )
    x = np.asarray(x)  # (n_particles, n_times, 3) in (rho, theta, zeta)
    v = np.asarray(v)  # (n_particles, n_times, 1), v_parallel

    omega = np.full(n, np.nan)
    for i, rho in enumerate(rho_values):
        traj, vpar = x[i], v[i, :, 0]
        good = np.all(np.isfinite(traj), axis=-1) & np.isfinite(vpar)
        if good.sum() < 10:
            warnings.warn(f"Trajectory at rho={rho} mostly NaN; skipping.")
            continue
        traj, t, vpar = traj[good], ts[good], vpar[good]
        data = eval_at_points(eq, traj, ["alpha"])
        # theta from tracing is wrapped to (-pi, pi], so alpha jumps by 2 pi
        # whenever the poloidal angle wraps; unwrap to recover the secular drift.
        alpha = np.unwrap(np.asarray(data["alpha"]), period=2 * np.pi)

        # alpha(t) = secular drift + bounce-phase oscillation. Sample alpha at
        # banana tips (v_par = 0) so every sample is at the same bounce phase,
        # then the tips of one parity lie on a straight line alpha(t).
        tips = np.flatnonzero(vpar[:-1] * vpar[1:] < 0)

        # Guard against aliasing: need several samples per bounce period for
        # both the tip times and the unwrap of alpha to be trustworthy.
        if tips.size >= 2:
            samples_per_bounce = 2 * np.median(np.diff(tips))
            if samples_per_bounce < 10:
                warnings.warn(
                    f"Only ~{samples_per_bounce:.0f} samples per bounce period at "
                    f"rho={rho}; tip detection may be aliased. Increase n_save."
                )

        if tips.size >= 4:
            # Locate each tip by linear interpolation of v_par through zero,
            # and interpolate alpha to the same instant. This removes the
            # O(dt) jitter of using the nearest saved sample.
            w = -vpar[tips] / (vpar[tips + 1] - vpar[tips])
            t_tip = t[tips] + w * (t[tips + 1] - t[tips])
            a_tip = alpha[tips] + w * (alpha[tips + 1] - alpha[tips])
            # Consecutive tips alternate between the two banana turning points;
            # fit each family separately (same bounce phase) and average.
            slopes = [
                np.polyfit(t_tip[p::2], a_tip[p::2], 1)[0]
                for p in (0, 1)
                if t_tip[p::2].size >= 2
            ]
            dalpha_dt = float(np.mean(slopes))
        else:
            warnings.warn(
                f"Fewer than 2 bounce periods at rho={rho}; using linear fit."
            )
            dalpha_dt = np.polyfit(t, alpha, 1)[0]

        iota = surface_iota(eq, rho)
        omega[i] = Np * dalpha_dt / eta_denominator(iota, helicity) / (2 * np.pi)
    return omega


def _num_integrand(data, B, pitch):
    # (1 - lambda B / 2) / sqrt(1 - lambda B) * gbdrift
    return safediv(
        (1 - 0.5 * pitch * B) * data["gbdrift"], jnp.sqrt(jnp.abs(1 - pitch * B))
    )


def _den_integrand(data, B, pitch):
    # 1 / sqrt(1 - lambda B)  (bounce time weight, up to constant v)
    return safediv(1.0, jnp.sqrt(jnp.abs(1 - pitch * B)))


def omega_eta_bounce(
    eq,
    rho,
    energy_ev,
    Np,
    helicity=(1, 0),
    num_transit=3,
    knots_per_transit=200,
    theta0=0.0,
    zeta0=0.0,
    Bcrit=None,
    xi0=None,
):
    """Omega_eta from bounce averaging the binormal drift with Bounce1D.

    Pitch is set by ``Bcrit`` (lambda = 1/Bcrit) when given; otherwise by
    ``xi0`` through lambda = (1 - xi0^2)/B(start). Averages over the magnetic
    well of the field line alpha0 = alpha(rho, theta0, zeta0) that contains
    zeta0.
    """
    mass = MASS_AMU * proton_mass
    charge = CHARGE_E * elementary_charge
    v2 = 2 * energy_ev * elementary_charge / mass  # v^2 in (m/s)^2

    start = eval_at_points(eq, np.array([[rho, theta0, zeta0]]), ["alpha", "|B|"])
    alpha0 = float(start["alpha"][0])
    B_start = float(start["|B|"][0])
    if Bcrit is not None:
        pitch = 1.0 / float(Bcrit)
    elif xi0 is not None:
        pitch = (1 - xi0**2) / B_start  # lambda = v_perp^2 / (v^2 B), conserved
    else:
        raise ValueError("Provide Bcrit or xi0.")
    pitch_inv = np.array([1 / pitch])

    iota = surface_iota(eq, rho)
    zeta = np.linspace(
        zeta0 - num_transit * np.pi,
        zeta0 + num_transit * np.pi,
        num_transit * knots_per_transit + 1,
    )
    grid = get_rtz_grid(eq, rho, alpha0, zeta, coordinates="raz", iota=iota)
    data = eq.compute(Bounce1D.required_names + ["gbdrift"], grid=grid)

    bounce = Bounce1D(grid.source_grid, data)
    z1, z2 = bounce.points(pitch_inv)
    num, den = bounce.integrate(
        [_num_integrand, _den_integrand],
        pitch_inv,
        {"gbdrift": Bounce1D.reshape(grid.source_grid, data["gbdrift"])},
        points=(z1, z2),
    )

    # Flatten well axis; select the well containing the particle start (zeta0).
    z1, z2 = np.asarray(z1).reshape(-1), np.asarray(z2).reshape(-1)
    num, den = np.asarray(num).reshape(-1), np.asarray(den).reshape(-1)
    valid = z2 > z1  # points() pads missing wells with zeros
    contains_start = valid & (z1 <= zeta0) & (zeta0 <= z2)
    if contains_start.any():
        w = np.flatnonzero(contains_start)[0]
    else:
        warnings.warn(
            f"No magnetic well brackets zeta={zeta0} at rho={rho}; "
            "using the longest well instead."
        )
        if not valid.any():
            return np.nan
        w = np.nanargmax(np.where(valid, den, np.nan))

    dalpha_dt_avg = (mass * v2 / charge) * num[w] / den[w]
    return Np * dalpha_dt_avg / eta_denominator(iota, helicity) / (2 * np.pi)


def main():
    parser = argparse.ArgumentParser(
        description="Omega_eta: tracing vs bounce averaging (QA/QH/QI)."
    )
    parser.add_argument(
        "--eq",
        type=str,
        default="precise_QA",
        help="DESC example name (e.g. precise_QA, precise_QH, W7-X) or path to "
        "a DESC .h5 output file.",
    )
    parser.add_argument(
        "--helicity",
        type=int,
        nargs=2,
        default=[1, 0],
        metavar=("M", "N"),
        help="Generalized helicity integers (M, N) of Landreman & Catto 2012 in "
        "eta = Np*alpha/(N - iota*M). QA: 1 0, QH: 1 +/-Np, QI: 0 1.",
    )
    parser.add_argument(
        "--num-alpha",
        type=int,
        default=1,
        help="Number of field lines. Particles are initialized on "
        "alpha = linspace(0, 2*pi, num_alpha, endpoint=False), where "
        "alpha = theta - iota*zeta. "
        "Each particle starts at the |B| minimum of its field line.",
    )
    parser.add_argument(
        "--zeta0",
        type=float,
        default=0.0,
        help="Toroidal seed angle (radians) used with each alpha to place the "
        "field line via theta0 = alpha + iota*zeta0. For QI, choose a seed that "
        "lies near a magnetic well along the field line.",
    )
    parser.add_argument(
        "--no-scale",
        action="store_true",
        help="Skip rescaling to ARIES-CS size/field (use the equilibrium as is).",
    )
    parser.add_argument(
        "--fractions",
        type=float,
        nargs="+",
        default=list(np.logspace(-7, 0, 8)),
        help="Fractions of 3.5 MeV to scan.",
    )
    parser.add_argument(
        "--s", type=float, nargs="+", default=[0.3, 0.5, 0.7], help="Flux surfaces s."
    )
    parser.add_argument(
        "--n-bcrit",
        type=int,
        default=8,
        help="Number of Bcrit samples between surface Bmin and Bmax "
        "(equally spaced in kappa = (Bcrit-Bmin)/(Bmax-Bmin)).",
    )
    parser.add_argument(
        "--kappa-eps",
        type=float,
        default=0.05,
        help="Exclude the kappa endpoints by this fraction so particles stay "
        "strictly trapped (avoid Bcrit -> Bmin or Bmax).",
    )
    parser.add_argument(
        "--Np",
        type=int,
        default=None,
        help="Number of field periods in eta = Np*alpha/(N - iota*M). "
        "Default: taken from the equilibrium (eq.NFP).",
    )
    parser.add_argument(
        "--n-bounce",
        type=float,
        default=50.0,
        help="Approximate number of bounce periods to trace. More bounces only "
        "help if the sampling rate per bounce is maintained (see "
        "--samples-per-bounce); the tip-fit error decreases like 1/T.",
    )
    parser.add_argument(
        "--samples-per-bounce",
        type=int,
        default=40,
        help="Saved samples per (estimated) bounce period. Must be >~10 or tip "
        "detection aliases and the estimator breaks down.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-7,
        help="Relative/absolute tolerance of the adaptive tracer.",
    )
    parser.add_argument("--output", type=str, default="omega_eta_tracing_vs_bounce")
    args = parser.parse_args()

    fractions = np.sort(np.asarray(args.fractions))
    s_values = list(args.s)
    n_s = len(s_values)
    n_E = len(fractions)
    n_B = args.n_bcrit
    n_a = args.num_alpha
    if n_a < 1:
        raise ValueError("--num-alpha must be >= 1.")
    eps = args.kappa_eps
    if not (0.0 < eps < 0.5):
        raise ValueError("--kappa-eps must lie in (0, 0.5).")
    kappa = np.linspace(eps, 1.0 - eps, n_B)
    # Field-line labels alpha = theta - iota * zeta.
    alphas = np.linspace(0.0, 2.0 * np.pi, n_a, endpoint=False)

    if args.eq.endswith(".h5"):
        print(f"Loading equilibrium from {args.eq}...")
        import desc.io

        eq = desc.io.load(args.eq)
        if hasattr(eq, "__getitem__"):  # EquilibriaFamily
            eq = eq[-1]
    else:
        print(f"Loading example equilibrium {args.eq}...")
        eq = get(args.eq)
    if not args.no_scale:
        # ARIES-CS scale, as in Landreman & Paul 2022 alpha-particle tracing
        eq = scale_equilibrium(eq, a_target=1.70, B_target=5.86)
    if eq.iota is None:
        # Tracing in flux coordinates needs an iota profile assigned to the eq.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            eq.iota = eq.get_profile("iota")

    Np = args.Np if args.Np is not None else int(eq.NFP)
    helicity = tuple(args.helicity)
    print(f"Using eta = {Np}*alpha/({helicity[1]} - iota*{helicity[0]})")
    print(f"Field lines: num_alpha={n_a}, alphas={alphas}")

    rho_values = np.sqrt(np.asarray(s_values))

    # Surface-wide |B| extrema define the Bcrit scan (same for all alphas on
    # that surface). Particles still start at each field line's |B| minimum.
    Bmin = np.full(n_s, np.nan)
    Bmax = np.full(n_s, np.nan)
    Bmin_line = np.full((n_s, n_a), np.nan)
    Bmax_line = np.full((n_s, n_a), np.nan)
    theta_start = np.full((n_s, n_a), np.nan)
    zeta_start = np.full((n_s, n_a), np.nan)
    alpha_start = np.full((n_s, n_a), np.nan)
    Bcrit = np.full((n_s, n_B), np.nan)
    for i, rho in enumerate(rho_values):
        Bmn_s, Bmx_s = surface_B_extrema(eq, rho)
        Bmin[i], Bmax[i] = Bmn_s, Bmx_s
        Bcrit[i] = Bmn_s + kappa * (Bmx_s - Bmn_s)
        print(f"s={s_values[i]}: surface Bmin={Bmn_s:.4f} T, Bmax={Bmx_s:.4f} T")
        iota = surface_iota(eq, rho)
        for a, alpha in enumerate(alphas):
            theta_seed = alpha + iota * args.zeta0
            Bmn, Bmx, alpha0, z_bmin, th_bmin = field_line_B_extrema(
                eq, rho, theta_seed, args.zeta0
            )
            Bmin_line[i, a], Bmax_line[i, a] = Bmn, Bmx
            theta_start[i, a], zeta_start[i, a] = th_bmin, z_bmin
            alpha_start[i, a] = alpha0
            print(
                f"  alpha={alpha:.4f}: line Bmin={Bmn:.4f} T, Bmax={Bmx:.4f} T, "
                f"start (theta,zeta)=({th_bmin:.4f},{z_bmin:.4f}), "
                f"alpha0={alpha0:.4f}"
            )

    # Rough bounce time at full 3.5 MeV energy, used to size the trace length:
    # tau_b ~ (a few connection lengths) / (v sqrt(eps)). The 1/sqrt(fraction)
    # scaling then keeps ~n_bounce bounce periods at every energy.
    v_full = np.sqrt(2 * E_ALPHA_EV * elementary_charge / (MASS_AMU * proton_mass))
    grid0 = LinearGrid(rho=rho_values, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
    data0 = eq.compute(["iota", "R0", "a"], grid=grid0)
    iota_typ = np.mean(np.abs(grid0.compress(data0["iota"])))
    eps_typ = np.mean(rho_values) * data0["a"] / data0["R0"]
    tau_b_full = 4 * np.pi * data0["R0"] / (iota_typ * v_full * np.sqrt(eps_typ))
    print(f"Estimated full-energy bounce period ~ {tau_b_full:.2e} s")

    # Shapes: (surface, alpha, Bcrit, energy)
    omega_tr = np.full((n_s, n_a, n_B, n_E), np.nan)
    omega_ba = np.full((n_s, n_a, n_B, n_E), np.nan)

    # Keep the number of saved samples per bounce period fixed so the banana-tip
    # estimator has the same time resolution (relative to the orbit) at every
    # energy. tmax scales as 1/sqrt(frac), so n_save is energy independent.
    n_save = int(args.n_bounce * args.samples_per_bounce)

    # Flatten (surface, alpha) -> particle index for batched tracing.
    rho_flat = np.repeat(rho_values, n_a)
    theta_flat = theta_start.reshape(-1)
    zeta_flat = zeta_start.reshape(-1)

    for j, frac in enumerate(fractions):
        energy_ev = frac * E_ALPHA_EV
        tmax = args.n_bounce * tau_b_full / np.sqrt(frac)
        print(
            f"[{j + 1}/{n_E}] fraction={frac:.3e} "
            f"(E={energy_ev:.3e} eV, tmax={tmax:.2e} s, n_save={n_save})"
        )
        for k, kap in enumerate(kappa):
            # xi0 from surface Bcrit and |B| at the field-line |B|-min start.
            # Skip (leave NaN) if Bcrit <= B_start (not trapped on that line).
            xi0 = np.full(n_s * n_a, np.nan)
            for i in range(n_s):
                for a in range(n_a):
                    B_start = Bmin_line[i, a]
                    Bc = Bcrit[i, k]
                    idx = i * n_a + a
                    if not np.isfinite(B_start) or not np.isfinite(Bc) or Bc <= B_start:
                        warnings.warn(
                            f"Not trapped at s={s_values[i]}, alpha={alphas[a]:.4f}, "
                            f"kappa={kap:.3f}: B_start={B_start:.4f} >= Bcrit={Bc:.4f}; "
                            "skipping."
                        )
                        continue
                    xi0[idx] = bcrit_to_xi0(B_start, Bc)
            print(
                f"  kappa={kap:.3f}: Bcrit={Bcrit[:, k]}, "
                f"xi0={np.array2string(xi0.reshape(n_s, n_a), precision=3)}"
            )
            # Only trace particles that are trapped for this Bcrit.
            trapped = np.isfinite(xi0)
            if trapped.any():
                try:
                    omega_sub = omega_eta_tracing(
                        eq,
                        rho_flat[trapped],
                        energy_ev,
                        xi0[trapped],
                        tmax,
                        n_save,
                        Np,
                        helicity=helicity,
                        rtol=args.rtol,
                        theta0=theta_flat[trapped],
                        zeta0=zeta_flat[trapped],
                    )
                    omega_flat = np.full(n_s * n_a, np.nan)
                    omega_flat[trapped] = omega_sub
                    omega_tr[:, :, k, j] = omega_flat.reshape(n_s, n_a)
                except Exception as e:  # keep scanning remaining (Bcrit, E)
                    print(f"    tracing failed: {e}")
            for i, rho in enumerate(rho_values):
                for a in range(n_a):
                    if not np.isfinite(xi0[i * n_a + a]):
                        continue
                    try:
                        omega_ba[i, a, k, j] = omega_eta_bounce(
                            eq,
                            rho,
                            energy_ev,
                            Np,
                            helicity=helicity,
                            theta0=theta_start[i, a],
                            zeta0=zeta_start[i, a],
                            Bcrit=Bcrit[i, k],
                        )
                    except Exception as e:
                        print(
                            f"    bounce averaging failed at s={s_values[i]}, "
                            f"alpha={alphas[a]:.4f}, kappa={kap:.3f}: {e}"
                        )
            print(f"    tracing: {omega_tr[:, :, k, j]}")
            print(f"    bounce : {omega_ba[:, :, k, j]}")
            # Drop compiled XLA modules so the (energy x Bcrit) scan does not
            # exhaust host memory (seen as LLVM "Cannot allocate memory").
            jax.clear_caches()
            gc.collect()

    # Relative residual (Omega_tr - Omega_ba) / mean(Omega). Absolute Omega
    # scales with energy, so the dimensionless residual is used for RMSE.
    omega_bar = 0.5 * (omega_tr + omega_ba)
    rel_diff = (omega_tr - omega_ba) / omega_bar

    # RMSE over (alpha, Bcrit) at each energy (shape: n_s x n_E). This is what
    # is saved and plotted vs energy (one curve per surface).
    rmse_over_bcrit = np.full((n_s, n_E), np.nan)
    for i in range(n_s):
        for j in range(n_E):
            r = rel_diff[i, :, :, j]
            if np.isfinite(r).any():
                rmse_over_bcrit[i, j] = float(np.sqrt(np.nanmean(r**2)))

    print("RMSE over (alpha, Bcrit) (relative) vs energy:")
    for i, s in enumerate(s_values):
        print(f"  s={s}: {rmse_over_bcrit[i]}")

    np.savez(
        args.output + ".npz",
        fractions=fractions,
        s_values=np.asarray(s_values),
        alphas=alphas,
        kappa=kappa,
        Bmin=Bmin,
        Bmax=Bmax,
        Bmin_line=Bmin_line,
        Bmax_line=Bmax_line,
        Bcrit=Bcrit,
        theta_start=theta_start,
        zeta_start=zeta_start,
        alpha_start=alpha_start,
        omega_tracing=omega_tr,
        omega_bounce=omega_ba,
        rel_diff=rel_diff,
        rmse_over_bcrit=rmse_over_bcrit,
        Np=Np,
        helicity=np.asarray(helicity),
        eq_name=args.eq,
        num_alpha=n_a,
        zeta0_seed=args.zeta0,
    )
    print(f"Saved results to {args.output}.npz")

    # Plot energy vs RMSE over (alpha, Bcrit) (one curve per surface).
    colors = ["#31688e", "#21918c", "#35b779"]
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, s in enumerate(s_values):
        ax.plot(
            fractions,
            rmse_over_bcrit[i],
            marker="o",
            markersize=5,
            color=colors[i % len(colors)],
            label=rf"$s={s}$",
        )
    ax.set_xscale("log")
    ax.axhline(0, color="gray", linewidth=1, zorder=0)
    ax.set_xlabel(r"Fraction of $\alpha$ Energy", fontsize=16)
    ax.set_ylabel(
        r"RMSE$_{\alpha,\kappa}\left[(\Omega_\eta^{\mathrm{tracing}}"
        r" - \Omega_\eta^{\mathrm{bounce}})"
        r" \, / \, \overline{\Omega_\eta}\right]$",
        fontsize=14,
    )
    ax.legend(fontsize=13, loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.tick_params(axis="both", which="major", labelsize=13)
    fig.tight_layout()
    fig.savefig(args.output + ".png", dpi=300)
    print(f"Saved plot to {args.output}.png")

if __name__ == "__main__":
    main()
