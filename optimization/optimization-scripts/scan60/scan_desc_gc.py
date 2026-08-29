#!/usr/bin/env python3
"""DESC-native GC tracing + f_B postprocessing (no VMEC conversion).

Loads a DESC .h5 equilibrium directly and computes:
  1. f_B = RMS(symmetry-breaking Boozer harmonics) / B00
     via DESC's native Boozer transform — no BOOZ_XFORM or VMEC.
  2. Alpha-particle loss fraction via DESC's VacuumGuidingCenterTrajectory,
     with a fusion birth spatial distribution matching firm3d.

This avoids the DESC→VMEC→BOOZ_XFORM→firm3d pipeline, which changes the
equilibrium through Fourier truncation in the VMEC format.

Usage:
    srun -n1 --gpus-per-task=1 python -u scan_desc_gc.py \
        --job-dir scan_runs/iota_scan22/job_001_nfp2_QA_AR3p5_iota0p56_bump_jcs2 \
        --n-particles 50000 --tmax 1e-2
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Main DESC has both set_device and particles.py; insert first so it wins
# over the DESC_TrappedRes variant installed in the conda env.
# set_device must be called before any other JAX import.
sys.path.insert(0, "/pscratch/sd/e/epaul/codes/DESC")
from desc import set_device
set_device("gpu")

from desc.grid import LinearGrid, Grid
from desc.io import load
from desc.particles import ManualParticleInitializerFlux, VacuumGuidingCenterTrajectory
from desc.batching import vmap_chunked
from scipy.constants import proton_mass, elementary_charge

# diffrax / equinox — use conda-env versions directly
from diffrax import (
    ODETerm, Tsit5, diffeqsolve, SaveAt, PIDController,
    Event, RecursiveCheckpointAdjoint,
)
import jax.numpy as jnp


# ---------- helpers -----------------------------------------------------------

def _resolve_eq(job_dir: Path, eq_name: str) -> Path:
    """Return path to DESC .h5 equilibrium inside job_dir."""
    if eq_name == "auto":
        # prefer after_k_N.h5 (latest multigrid stage)
        ks = []
        for fp in job_dir.glob("after_k_*.h5"):
            try:
                ks.append((int(fp.stem.rsplit("_", 1)[-1]), fp))
            except ValueError:
                continue
        if ks:
            return sorted(ks, key=lambda kv: kv[0])[-1][1]
        fallback = job_dir / "final.h5"
        if fallback.is_file():
            return fallback
        raise FileNotFoundError(f"No after_k_*.h5 or final.h5 in {job_dir}")

    p = job_dir / eq_name
    if not p.is_file():
        raise FileNotFoundError(f"Equilibrium not found: {p}")
    return p


def compute_fB(eq, num_rho=16, M_booz=None, N_booz=None):
    """Compute f_B per flux surface from DESC's native Boozer transform.

    For QA (helicity M=1, N=0) the symmetry-preserving Boozer modes have
    toroidal mode number n=0.  All n≠0 modes break the quasi-axisymmetry.

    Returns
    -------
    dict with fB_max, fB_mean, fB_edge (each in [0, 1], lower = better)
    and fB_per_rho array.
    """
    M_booz = M_booz or 8 * eq.M
    N_booz = N_booz or 8 * eq.N

    rho_vals = np.linspace(1 / (num_rho + 1), 1.0, num_rho)
    grid = LinearGrid(
        rho=rho_vals,
        M=max(eq.M_grid, M_booz),
        N=max(eq.N_grid, N_booz),
        NFP=eq.NFP,
        sym=False,
    )
    data = eq.compute(["|B|_mn_B", "B modes"], grid=grid, M_booz=M_booz, N_booz=N_booz)

    modes = np.array(data["B modes"])   # (n_modes, 3): [l, m, n]
    n_modes = modes.shape[0]
    m_arr = modes[:, 1]
    n_arr = modes[:, 2]

    B_mn = np.array(data["|B|_mn_B"]).reshape((num_rho, n_modes))  # (num_rho, n_modes)

    m0n0 = (m_arr == 0) & (n_arr == 0)
    sb = n_arr != 0  # symmetry-breaking for QA: n ≠ 0

    B00 = B_mn[:, m0n0].squeeze(axis=1)           # (num_rho,)
    rms_sb = np.sqrt(np.sum(B_mn[:, sb] ** 2, axis=1))  # (num_rho,)
    fB_per_rho = rms_sb / np.where(B00 > 0, B00, np.nan)

    return {
        "fB_max": float(np.nanmax(fB_per_rho)),
        "fB_mean": float(np.nanmean(fB_per_rho)),
        "fB_edge": float(fB_per_rho[-1]),
        "fB_per_rho": fB_per_rho.tolist(),
        "fB_rho_vals": rho_vals.tolist(),
    }


def sample_fusion_birth(n_particles, rng):
    """Sample particle positions from the fusion alpha birth distribution.

    Spatial: p(rho) ∝ rho * n_D(rho²)² * <σv>(T(rho²))
    with n_D(s) = 1 - s^5, T_keV(s) = 11.5*(1-s),  σv ∝ T^(-2/3)*exp(-19.94*T^(-1/3))
    (Bader+ 2021, NF 61 116060, same as firm3d).

    theta0, zeta0: uniform.
    xi0 = vpar/v: uniform in [-1, 1].
    """
    def p_vec(rho):
        s = rho ** 2
        n_prof = (1.0 - s ** 5) ** 2
        T = 11.5 * (1.0 - s)
        T = np.where(T > 0, T, np.nan)
        sv = T ** (-2.0 / 3.0) * np.exp(-19.94 * T ** (-1.0 / 3.0))
        sv = np.where(np.isfinite(sv), sv, 0.0)
        return rho * n_prof * sv  # Jacobian rho for sampling over rho ∈ [0,1]

    rho_grid = np.linspace(0.001, 0.999, 10000)
    p_max = float(np.nanmax(p_vec(rho_grid)))

    accepted_rho = []
    while len(accepted_rho) < n_particles:
        batch = max(n_particles * 5, 10000)
        cand = rng.uniform(0.0, 1.0, batch)
        prob = p_vec(cand) / p_max
        mask = rng.uniform(0.0, 1.0, batch) < prob
        accepted_rho.extend(cand[mask].tolist())

    rho0 = np.array(accepted_rho[:n_particles])
    theta0 = rng.uniform(0.0, 2 * np.pi, n_particles)
    zeta0 = rng.uniform(0.0, 2 * np.pi, n_particles)
    xi0 = rng.uniform(-1.0, 1.0, n_particles)

    return rho0, theta0, zeta0, xi0


def sample_uniform_surf(n_particles, rho_surf, eq, rng, ntheta=200, nzeta=200):
    """Sample particle positions uniformly on the flux surface rho = rho_surf.

    Positions are distributed uniformly with respect to the surface area
    element, weighted by the DESC Jacobian sqrt(g) at each (theta, zeta)
    point.  Pitch angle xi0 = vpar/v is uniform in [-1, 1].

    This mirrors firm3d's initialize_position_uniform_surf, adapted for
    DESC flux coordinates (rho, theta, zeta).
    """
    from desc.grid import Grid

    # ------------------------------------------------------------------
    # Step 1: estimate Jacobian max on a coarse grid for normalisation
    # ------------------------------------------------------------------
    theta_g = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
    zeta_g  = np.linspace(0, 2 * np.pi / eq.NFP, nzeta, endpoint=False)
    TT, ZZ  = np.meshgrid(theta_g, zeta_g, indexing="ij")
    nodes   = np.column_stack([
        np.full(TT.size, rho_surf), TT.ravel(), ZZ.ravel()
    ])
    grid     = Grid(nodes, sort=False)
    jac_grid = np.abs(np.array(eq.compute(["sqrt(g)"], grid=grid)["sqrt(g)"]))
    jac_max  = float(jac_grid.max())
    print(f"  Jacobian range on surface: [{jac_grid.min():.4f}, {jac_max:.4f}]",
          flush=True)

    # ------------------------------------------------------------------
    # Step 2: rejection sampling
    # ------------------------------------------------------------------
    theta_acc, zeta_acc = [], []
    batch_size = max(n_particles * 8, 100_000)
    while len(theta_acc) < n_particles:
        theta = rng.uniform(0, 2 * np.pi, batch_size)
        zeta  = rng.uniform(0, 2 * np.pi / eq.NFP, batch_size)
        nodes = np.column_stack([np.full(batch_size, rho_surf), theta, zeta])
        grid  = Grid(nodes, sort=False)
        jac   = np.abs(np.array(eq.compute(["sqrt(g)"], grid=grid)["sqrt(g)"]))
        mask  = rng.uniform(0, 1, batch_size) < jac / jac_max
        theta_acc.extend(theta[mask].tolist())
        zeta_acc.extend(zeta[mask].tolist())

    n = n_particles
    rho0   = np.full(n, rho_surf)
    theta0 = np.array(theta_acc[:n])
    zeta0  = np.array(zeta_acc[:n])
    xi0    = rng.uniform(-1.0, 1.0, n)
    return rho0, theta0, zeta0, xi0


def compute_mu_chunked(initializer, model, eq, chunk_size=1000):
    """Compute y0 and model_args (including mu) in chunks to avoid GPU OOM.

    _compute_modB on all n_particles at once blows up GPU memory via XLA;
    chunking keeps each batch to chunk_size points.

    When eq.iota is None (current-constrained equilibrium), the iota profile
    must be pre-fitted and passed to _compute_modB; otherwise iota is computed
    incorrectly via a single-point surface integral.
    """
    import numpy as np
    from desc.particles import _compute_modB
    from scipy.constants import proton_mass, elementary_charge

    n = len(initializer.rho0)
    x_all = jnp.array([initializer.rho0, initializer.theta0, initializer.zeta0]).T
    params = eq.params_dict

    # Replicate what init_particles() does: pre-compute iota profile if needed.
    modB_kwargs = {}
    if eq.iota is None:
        iota_profile = eq.get_profile("iota")
        params = dict(params)
        params["i_l"] = iota_profile.params
        modB_kwargs["iota"] = iota_profile

    mu_chunks = []
    chunk_size = chunk_size or n
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        x_chunk = x_all[start:end]
        modB_chunk = _compute_modB(x_chunk, eq, params, **modB_kwargs)
        vperp2_chunk = initializer.v0[start:end] ** 2 - initializer.vpar0[start:end] ** 2
        mu_chunk = initializer.m[start:end] * vperp2_chunk / (2.0 * modB_chunk)
        mu_chunks.append(np.array(mu_chunk))

    mu_all = jnp.array(np.concatenate(mu_chunks))
    m_all  = initializer.m
    q_all  = initializer.q

    # y0: (n, 4) = (rho, theta, zeta, vpar)
    vpar_col = initializer.vpar0.reshape(-1, 1)
    y0 = jnp.hstack([x_all, vpar_col])

    # model_args: (n, 3) = (m, q, mu) — same order as model.args = ["m","q","mu"]
    model_args = jnp.stack([m_all, q_all, mu_all], axis=1)

    return y0, model_args


def trace_particles_odeterm(
    eq, y0, model, model_args, ts,
    rtol=1e-6, atol=1e-6, chunk_size=None, throw=False,
):
    """Like desc.particles.trace_particles but wraps model.vf in ODETerm.

    This bypasses the equinox 0.13.4 incompatibility where eqx.field(static=True)
    fields appear as _Missing leaves and break diffrax's term-structure tree_map check.
    ODETerm takes a plain callable, so it has no static fields and works fine.
    """
    params = eq.params_dict

    # If the equilibrium has no stored iota profile (iota=None, only current given),
    # B^theta = (psi_r/sqrt(g)) * (iota * phi_z - lambda_z) requires the iota profile.
    # init_particles() normally handles this; we must replicate it here.
    gc_kwargs = {}
    if eq.iota is None:
        iota_profile = eq.get_profile("iota")
        params = dict(params)           # copy so we don't mutate eq.params_dict
        params["i_l"] = iota_profile.params
        gc_kwargs["iota"] = iota_profile

    # Convert (rho, theta, zeta) → cartesian-like (xp=rho*cos θ, yp=rho*sin θ, zeta)
    # DESC's flux-frame integrator works in these coords to avoid rho<0 issues.
    xp = y0[:, 0] * jnp.cos(y0[:, 1])
    yp = y0[:, 0] * jnp.sin(y0[:, 1])
    y0_cart = y0.at[:, 0].set(xp).at[:, 1].set(yp)

    def exit_event(t, y, args, **kwargs):
        rho = jnp.sqrt(y[0] ** 2 + y[1] ** 2)
        return rho > 1.0

    # model.vf is @jax.jit on a BoundMethod — equinox stores it with _Missing static
    # fields that JAX can't interpret as arrays. Bypass by calling the unbound
    # _compute_flux_coordinates directly; it doesn't use self at all.
    _flux_vf = VacuumGuidingCenterTrajectory._compute_flux_coordinates

    def _plain_vf(t, x, args):
        model_args, eq_or_field, params, kwargs = args
        m, q, mu = model_args
        return _flux_vf(None, x.squeeze(), eq_or_field, params, m, q, mu, **kwargs)

    term = ODETerm(_plain_vf)

    stepsize_controller = PIDController(
        rtol=rtol, atol=atol, dtmin=1e-8, pcoeff=0.3, icoeff=0.3, dcoeff=0,
    )
    max_steps = int(float(ts[-1] - ts[0]) / 1e-8)

    def integrate_one(y0_i, model_args_i):
        return diffeqsolve(
            terms=term,
            solver=Tsit5(),
            y0=y0_i,
            args=[model_args_i, eq, params, gc_kwargs],
            t0=ts[0],
            t1=ts[-1],
            dt0=1e-8,
            saveat=SaveAt(ts=ts),
            max_steps=max_steps,
            stepsize_controller=stepsize_controller,
            adjoint=RecursiveCheckpointAdjoint(),
            event=Event(exit_event),
            throw=throw,
        )

    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="unhashable type")
        out = vmap_chunked(
            integrate_one, in_axes=(0, 0), chunk_size=chunk_size,
        )(y0_cart, model_args)

    yt = jnp.where(jnp.isinf(out.ys), jnp.nan, out.ys)

    # Convert cartesian-like back to flux coords
    rho   = jnp.sqrt(yt[:, :, 0] ** 2 + yt[:, :, 1] ** 2)
    theta = jnp.arctan2(yt[:, :, 1], yt[:, :, 0])
    theta = jnp.where(theta < 0, theta + 2 * jnp.pi, theta)
    x = yt.at[:, :, 0].set(rho).at[:, :, 1].set(theta)

    return x[:, :, :3], x[:, :, 3:]


def compute_loss_fraction(rho_traj, ts):
    """Cumulative loss fraction curve.

    rho_traj : (n_particles, n_ts) array — NaN after particle exits (rho > 1).
    Returns times, loss_frac arrays.
    """
    lost_by_t = np.mean(np.isnan(rho_traj), axis=0)  # non-decreasing
    return np.array(ts), lost_by_t


# ---------- main --------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--job-dir", required=True, type=Path,
                   help="Scan job directory containing the DESC .h5 equilibrium.")
    p.add_argument("--eq-name", default="auto",
                   help="Equilibrium file inside --job-dir. 'auto' picks latest after_k_N.h5.")
    p.add_argument("--n-particles", type=int, default=50000)
    p.add_argument("--tmax", type=float, default=1e-2,
                   help="Max tracing time in seconds (default: 1e-2).")
    p.add_argument("--tmin", type=float, default=1e-5,
                   help="Start of loss-fraction curve time axis (default: 1e-5).")
    p.add_argument("--n-ts", type=int, default=200,
                   help="Number of time output points (log-spaced, default: 200).")
    p.add_argument("--rtol", type=float, default=1e-6)
    p.add_argument("--atol", type=float, default=1e-6)
    p.add_argument("--chunk-size", type=int, default=None,
                   help="Particle batch size for trace_particles (reduces GPU memory).")
    p.add_argument("--num-rho-fB", type=int, default=16,
                   help="Number of flux surfaces for f_B computation (default: 16).")
    p.add_argument("--M-booz", type=int, default=None,
                   help="Max poloidal mode for Boozer transform (default: 2*eq.M).")
    p.add_argument("--N-booz", type=int, default=None,
                   help="Max toroidal mode for Boozer transform (default: 2*eq.N).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--launch-surface", type=float, default=None,
                   help="If set, launch all particles uniformly on the flux surface "
                        "rho=LAUNCH_SURFACE (Jacobian-weighted) instead of the fusion "
                        "birth distribution. Output files are prefixed desc_gc_surfNpM_.")
    p.add_argument("--out-prefix", default=None,
                   help="Override the auto-generated output file prefix "
                        "(default: desc_gc or desc_gc_surfNpM).")
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    job_dir = args.job_dir.resolve()
    if not job_dir.is_dir():
        print(f"ERROR: --job-dir not found: {job_dir}", file=sys.stderr)
        sys.exit(2)

    eq_path = _resolve_eq(job_dir, args.eq_name)
    print(f"DESC-native GC tracing for {job_dir.name}")
    print(f"  equilibrium : {eq_path}")
    print(f"  n_particles : {args.n_particles}")
    print(f"  tmax        : {args.tmax}")

    print("Loading equilibrium...", flush=True)
    eq = load(str(eq_path))
    print(f"  NFP={eq.NFP}  L={eq.L}  M={eq.M}  N={eq.N}", flush=True)

    # ------------------------------------------------------------------
    # 1. f_B via native Boozer transform
    # ------------------------------------------------------------------
    print("Computing f_B (native Boozer transform)...", flush=True)
    t0 = time.time()
    fB_results = compute_fB(eq, num_rho=args.num_rho_fB,
                            M_booz=args.M_booz, N_booz=args.N_booz)
    print(f"  f_B max={fB_results['fB_max']:.4f}  mean={fB_results['fB_mean']:.4f}"
          f"  edge={fB_results['fB_edge']:.4f}  ({time.time()-t0:.1f}s)", flush=True)

    # ------------------------------------------------------------------
    # 2. GC tracing — initial distribution
    # ------------------------------------------------------------------
    rng = np.random.default_rng(args.seed)

    if args.launch_surface is not None:
        rho_surf = args.launch_surface
        eq_tag = Path(args.eq_name).stem if args.eq_name != "auto" else eq_path.stem
        file_prefix = "desc_gc_surf" + f"{rho_surf:.2f}".replace(".", "p") + f"_{eq_tag}"
        print(f"Sampling uniform surface distribution at rho={rho_surf} "
              f"(Jacobian-weighted)...", flush=True)
        rho0, theta0, zeta0, xi0 = sample_uniform_surf(
            args.n_particles, rho_surf, eq, rng
        )
    else:
        file_prefix = "desc_gc"
        print("Sampling fusion birth distribution...", flush=True)
        rho0, theta0, zeta0, xi0 = sample_fusion_birth(args.n_particles, rng)

    if args.out_prefix is not None:
        file_prefix = args.out_prefix

    model = VacuumGuidingCenterTrajectory(frame="flux")
    initializer = ManualParticleInitializerFlux(
        rho0=rho0, theta0=theta0, zeta0=zeta0, xi0=xi0,
        E=3.52e6,  # alpha particle birth energy in eV
        m=4,       # proton masses
        q=2,       # elementary charges
    )

    ts = np.logspace(np.log10(args.tmin), np.log10(args.tmax), args.n_ts)

    # Compute mu = m*vperp²/(2|B|) in chunks to avoid XLA OOM on all 50k particles
    print("Initializing particles (computing mu from |B| in chunks)...", flush=True)
    y0, model_args = compute_mu_chunked(initializer, model, eq, chunk_size=args.chunk_size or 1000)

    mu_arr = np.array(model_args[:, 2])              # (n_particles,) magnetic moment [J/T]
    E_J = 3.52e6 * elementary_charge                 # alpha birth energy [J]
    B_crit_arr = E_J / mu_arr                        # mirror-point field: E/mu [T]

    print("Tracing particles...", flush=True)
    t_trace0 = time.time()
    x, v = trace_particles_odeterm(
        eq=eq,
        y0=y0,
        model=model,
        model_args=model_args,
        ts=ts,
        rtol=args.rtol,
        atol=args.atol,
        chunk_size=args.chunk_size,
        throw=False,
    )
    t_trace1 = time.time()
    print(f"  Tracing done in {t_trace1 - t_trace0:.1f}s", flush=True)

    # ------------------------------------------------------------------
    # 3. Loss fraction
    # ------------------------------------------------------------------
    rho_traj = np.array(x[:, :, 0])  # (n_particles, n_ts)
    ts_out, loss_frac = compute_loss_fraction(rho_traj, ts)

    lost_arr = np.isnan(rho_traj[:, -1])             # (n_particles,) bool
    n_lost = int(lost_arr.sum())

    # Save per-particle data for post-analysis
    npz_path = job_dir / f"{file_prefix}_particles.npz"
    np.savez_compressed(
        npz_path,
        rho0=np.array(initializer.rho0),
        theta0=theta0,
        zeta0=zeta0,
        xi0=xi0,
        mu=mu_arr,
        B_crit=B_crit_arr,
        lost=lost_arr,
    )
    print(f"Wrote {npz_path}  (per-particle rho0, theta0, zeta0, xi0, mu, B_crit, lost)")

    summary = {
        "job_dir": str(job_dir),
        "eq_path": str(eq_path),
        "n_particles": int(args.n_particles),
        "tmax": float(args.tmax),
        "tmin": float(args.tmin),
        "rtol": float(args.rtol),
        "atol": float(args.atol),
        "seed": int(args.seed),
        "trace_seconds": float(t_trace1 - t_trace0),
        "n_lost": n_lost,
        "loss_fraction_total": float(n_lost) / float(args.n_particles),
        "loss_curve_t": ts_out.tolist(),
        "loss_curve_frac": loss_frac.tolist(),
        "loss_at_1ms": float(np.interp(1e-3, ts_out, loss_frac)),
        "loss_at_3ms": float(np.interp(3e-3, ts_out, loss_frac)),
        "loss_at_tmax": float(loss_frac[-1]),
        **fB_results,
    }

    out_json = job_dir / f"{file_prefix}_loss.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {out_json}")
    print(f"  loss_at_tmax = {summary['loss_at_tmax']*100:.2f}%")
    print(f"  loss_at_1ms  = {summary['loss_at_1ms']*100:.2f}%")
    print(f"  loss_at_3ms  = {summary['loss_at_3ms']*100:.2f}%")
    print(f"  fB_max       = {summary['fB_max']:.4f}")

    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(5, 4))
            ax.loglog(ts_out, loss_frac)
            ax.set_xlim([args.tmin, args.tmax])
            ax.set_ylim([1e-4, 1.0])
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Lost fraction")
            ax.set_title(
                f"{job_dir.name}\n"
                f"loss(tmax={args.tmax:g}s) = {summary['loss_at_tmax']:.3f}"
                f"   N={args.n_particles}"
            )
            ax.grid(True, which="both", alpha=0.3)
            fig.tight_layout()
            png = job_dir / f"{file_prefix}_loss.png"
            fig.savefig(png, dpi=120)
            plt.close(fig)
            print(f"Wrote {png}")

            # B_crit vs loss fraction plot — trapped-particle range only
            try:
                from desc.grid import LinearGrid as _LG
                _grid_B = _LG(M=4*eq.M, N=4*eq.N, NFP=eq.NFP,
                               rho=np.linspace(0.05, 1.0, 20), sym=False)
                _Bdata = eq.compute(["|B|"], grid=_grid_B)
                B_min = float(np.array(_Bdata["|B|"]).min())
                B_max = float(np.array(_Bdata["|B|"]).max())
                print(f"  B_min={B_min:.3f} T  B_max={B_max:.3f} T")

                trapped_mask = (B_crit_arr >= B_min) & (B_crit_arr <= B_max)
                B_t    = B_crit_arr[trapped_mask]
                lost_t = lost_arr[trapped_mask]
                print(f"  Trapped: {trapped_mask.sum()} / {len(B_crit_arr)} "
                      f"({trapped_mask.mean()*100:.1f}%)   "
                      f"lost among trapped: {lost_t.mean()*100:.2f}%")

                n_bins = 60
                bins = np.linspace(B_min, B_max, n_bins + 1)
                bin_centers = 0.5 * (bins[:-1] + bins[1:])
                loss_per_bin  = np.full(n_bins, np.nan)
                count_per_bin = np.zeros(n_bins, dtype=int)
                idx = np.clip(np.digitize(B_t, bins) - 1, 0, n_bins - 1)
                for i in range(n_bins):
                    mask = idx == i
                    count_per_bin[i] = mask.sum()
                    if mask.sum() > 10:
                        loss_per_bin[i] = lost_t[mask].mean()

                fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), sharex=True,
                                                 gridspec_kw={"height_ratios": [3, 1]})

                ax1.plot(bin_centers, loss_per_bin * 100, "o-", ms=3)
                ax1.axvline(B_min, color="gray", ls="--", lw=0.8,
                            label=f"$B_{{\\rm min}}$ = {B_min:.2f} T")
                ax1.axvline(B_max, color="gray", ls=":",  lw=0.8,
                            label=f"$B_{{\\rm max}}$ = {B_max:.2f} T")
                ax1.set_ylabel("Loss fraction [%]")
                ax1.set_title(
                    f"{job_dir.name}\n"
                    f"Loss vs $B_{{\\rm crit}} = E/\\mu$  (trapped particles only)"
                )
                ax1.set_ylim(bottom=0)
                ax1.legend(fontsize=8)
                ax1.grid(True, alpha=0.3)

                ax2.bar(bin_centers, count_per_bin,
                        width=bins[1] - bins[0], color="C0", alpha=0.6)
                ax2.set_xlabel("$B_{\\rm crit} = E/\\mu$ [T]")
                ax2.set_ylabel("N particles")
                ax2.grid(True, alpha=0.3)

                fig2.tight_layout()
                png2 = job_dir / f"{file_prefix}_Bcrit.png"
                fig2.savefig(png2, dpi=150)
                plt.close(fig2)
                print(f"Wrote {png2}")
            except Exception as e2:
                print(f"WARN: B_crit plot failed: {e2}")

        except Exception as e:
            print(f"WARN: plot failed: {e}")


if __name__ == "__main__":
    main()
