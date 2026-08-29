#!/usr/bin/env python3
"""Combined diagnostic figure for a scan job directory.

Row 0 — modB contours in Boozer coordinates at 4 flux surfaces
Row 1 — loss vs time | loss vs Bcrit | fB vs rho | Bmin spread vs rho
Row 2 — R-Z cross-sections at 4 toroidal angles (one field period)

Usage:
    python plot_diagnostics.py --job-dir <path/to/job_dir> [--gc-dir <path>]
    python plot_diagnostics.py --job-dir scan_runs/iota_scan37/job_001_...
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/pscratch/sd/e/epaul/codes/DESC_TrappedRes")
from desc.io import load
from desc.grid import LinearGrid
from desc.backend import jnp


# ── helpers ──────────────────────────────────────────────────────────────────

def load_eq(job_dir: Path):
    ks = []
    for fp in job_dir.glob("after_k_*.h5"):
        try:
            ks.append((int(fp.stem.rsplit("_", 1)[-1]), fp))
        except ValueError:
            pass
    if ks:
        return load(str(sorted(ks)[-1][1]))
    fallback = job_dir / "final.h5"
    if fallback.exists():
        return load(str(fallback))
    raise FileNotFoundError(f"No equilibrium in {job_dir}")


def compute_modB_boozer(eq, rho_vals=None, M_booz=None, N_booz=None,
                        n_theta=200, n_zeta=200):
    """Return (theta_B, zeta_B, B_list, iota_val) for each surface in rho_vals."""
    M_booz   = M_booz or 8 * eq.M
    N_booz   = N_booz or 8 * eq.N
    rho_vals = np.array(rho_vals if rho_vals is not None else [0.3, 0.5, 0.7, 1.0])
    grid = LinearGrid(
        rho=rho_vals,
        M=max(eq.M_grid, M_booz),
        N=max(eq.N_grid, N_booz),
        NFP=eq.NFP,
        sym=False,
    )
    data = eq.compute(["|B|_mn_B", "B modes", "iota"], grid=grid,
                      M_booz=M_booz, N_booz=N_booz)

    modes    = np.array(data["B modes"])                          # (n_modes, 3)
    B_mn_all = np.array(data["|B|_mn_B"]).reshape(len(rho_vals), -1)
    iota_val = float(grid.compress(np.array(data["iota"]))[0])    # at first surface

    theta_B = np.linspace(0, 2 * np.pi, n_theta)
    zeta_B  = np.linspace(0, 2 * np.pi / eq.NFP, n_zeta)
    TH, ZT  = np.meshgrid(theta_B, zeta_B, indexing="ij")
    m_arr, n_arr = modes[:, 1], modes[:, 2]

    B_list = []
    for B_mn in B_mn_all:
        B = np.sum(
            B_mn[None, None, :] * np.cos(
                m_arr[None, None, :] * TH[:, :, None]
                - n_arr[None, None, :] * eq.NFP * ZT[:, :, None]
            ),
            axis=2,
        )
        B_list.append(B)
    return theta_B, zeta_B, B_list, rho_vals, iota_val


def compute_fB_profile(eq, num_rho=20, M_booz=None, N_booz=None):
    """Return (rho_vals, fB_per_rho)."""
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
    data = eq.compute(["|B|_mn_B", "B modes"], grid=grid,
                      M_booz=M_booz, N_booz=N_booz)
    modes  = np.array(data["B modes"])
    B_mn   = np.array(data["|B|_mn_B"]).reshape((num_rho, -1))
    n_arr  = modes[:, 2]
    m0n0   = (modes[:, 1] == 0) & (n_arr == 0)
    sb     = n_arr != 0
    B00    = B_mn[:, m0n0].squeeze(axis=1)
    rms_sb = np.sqrt(np.sum(B_mn[:, sb] ** 2, axis=1))
    fB     = rms_sb / np.where(B00 > 0, B00, np.nan)
    return rho_vals, fB


def compute_bmin_profile(eq, num_rho=20, num_field_lines=32,
                          num_transit=2, num_zeta=150,
                          M_booz=None, N_booz=None):
    """Return (rho_vals, bmin_mean, bmin_std, bmin_min, bmin_max) in Tesla."""
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
    data = eq.compute(["|B|_mn_B", "B modes", "iota"], grid=grid,
                      M_booz=M_booz, N_booz=N_booz)
    modes  = np.array(data["B modes"])
    B_mn   = np.array(data["|B|_mn_B"]).reshape((num_rho, -1))
    iota   = grid.compress(np.array(data["iota"]))  # (num_rho,)
    m_arr  = modes[:, 1].astype(float)
    n_arr  = modes[:, 2].astype(float)
    NFP    = float(eq.NFP)

    alpha  = np.linspace(0, 2 * np.pi, num_field_lines, endpoint=False)
    zeta_B = np.linspace(0, 2 * np.pi * num_transit,
                          num_transit * num_zeta, endpoint=False)

    # freq[rho, mode] = m*iota - n*NFP
    freq = iota[:, None] * m_arr[None, :] - n_arr[None, :] * NFP
    # zeta_phase[rho, mode, zeta]
    zeta_phase = freq[:, :, None] * zeta_B[None, None, :]
    # alpha_phase[fl, mode]
    alpha_phase = m_arr[None, :] * alpha[:, None]

    Bc = B_mn[:, :, None] * np.cos(zeta_phase)
    Bs = B_mn[:, :, None] * np.sin(zeta_phase)
    cos_a = np.cos(alpha_phase)
    sin_a = np.sin(alpha_phase)

    # B_along[rho, fl, zeta]
    B_along = (np.einsum("fm,rmz->rfz", cos_a, Bc)
               - np.einsum("fm,rmz->rfz", sin_a, Bs))

    B_min = B_along.min(axis=-1)  # (num_rho, num_fl)
    return (rho_vals,
            B_min.mean(axis=1),
            B_min.std(axis=1),
            B_min.min(axis=1),
            B_min.max(axis=1))


def gc_loss_vs_bcrit(npz_path, eq, n_bins=50):
    """Return (bin_centers, loss_per_bin, count_per_bin) restricted to trapped range."""
    d = np.load(npz_path)
    B_crit = d["B_crit"]
    lost   = d["lost"]

    # B_min / B_max of equilibrium
    grid = LinearGrid(M=4*eq.M, N=4*eq.N, NFP=eq.NFP,
                      rho=np.linspace(0.05, 1.0, 20), sym=False)
    data_B = eq.compute(["|B|"], grid=grid)
    B_vals = np.array(data_B["|B|"])
    B_lo, B_hi = float(B_vals.min()), float(B_vals.max())

    mask = (B_crit >= B_lo) & (B_crit <= B_hi)
    B_t, lost_t = B_crit[mask], lost[mask]

    bins = np.linspace(B_lo, B_hi, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    loss_bin  = np.full(n_bins, np.nan)
    count_bin = np.zeros(n_bins, dtype=int)
    idx = np.clip(np.digitize(B_t, bins) - 1, 0, n_bins - 1)
    for i in range(n_bins):
        m = idx == i
        count_bin[i] = m.sum()
        if m.sum() > 5:
            loss_bin[i] = lost_t[m].mean()
    return centers, loss_bin, count_bin, B_lo, B_hi


def compute_cross_sections(eq, zeta_vals, rho_vals=None, n_theta=200):
    """Compute R-Z cross-sections at given toroidal angles.

    Parameters
    ----------
    eq : DESC Equilibrium
    zeta_vals : array of toroidal angles [rad]
    rho_vals : flux surfaces to show (default: [0.25, 0.5, 0.75, 1.0])
    n_theta : number of poloidal points per surface

    Returns
    -------
    list of dicts, one per zeta:  {"zeta": float, "R": (n_rho, n_theta), "Z": ...}
    """
    from desc.grid import Grid
    rho_vals = np.array(rho_vals if rho_vals is not None else [0.25, 0.5, 0.75, 1.0])
    theta    = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)

    results = []
    for zeta in zeta_vals:
        # Build grid: all (rho, theta) combinations at fixed zeta
        nodes = np.array([
            [r, t, zeta]
            for r in rho_vals
            for t in theta
        ])
        g    = Grid(nodes, sort=False)
        data = eq.compute(["R", "Z"], grid=g)
        R    = np.array(data["R"]).reshape(len(rho_vals), n_theta)
        Z    = np.array(data["Z"]).reshape(len(rho_vals), n_theta)
        results.append({"zeta": zeta, "R": R, "Z": Z})
    return results, rho_vals


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--job-dir", required=True, type=Path)
    p.add_argument("--gc-dir",  type=Path, default=None,
                   help="Directory with desc_gc_particles.npz (defaults to --job-dir)")
    p.add_argument("--rho-surfaces", type=float, nargs="+",
                   default=[0.3, 0.5, 0.7, 1.0],
                   help="Flux surfaces for modB contour plot")
    p.add_argument("--num-rho", type=int, default=20)
    p.add_argument("--num-field-lines", type=int, default=32)
    args = p.parse_args()

    job_dir = args.job_dir.resolve()
    gc_dir  = (args.gc_dir or job_dir).resolve()

    print(f"Loading equilibrium from {job_dir} ...", flush=True)
    eq = load_eq(job_dir)
    print(f"  NFP={eq.NFP}  M={eq.M}  N={eq.N}", flush=True)

    rho_surf = np.array(args.rho_surfaces)
    n_surf   = len(rho_surf)
    label    = job_dir.name[:100]

    # Layout: row 0 = modB contours, row 1 = diagnostics, row 2 = R-Z cross-sections
    # Number of columns = max(n_surf, 4)
    n_cols = max(n_surf, 4)
    fig, axes = plt.subplots(3, n_cols, figsize=(4.5 * n_cols, 15))

    # ── Row 0: modB contours — one panel per surface ─────────────────────────
    print("Computing modB Boozer contours ...", flush=True)
    theta_B, zeta_B, B_list, rho_surf, iota_val = compute_modB_boozer(
        eq, rho_vals=rho_surf)
    zeta_deg  = np.degrees(zeta_B)
    theta_deg = np.degrees(theta_B)

    for col, (B, rho) in enumerate(zip(B_list, rho_surf)):
        ax = axes[0, col]
        levels = np.linspace(B.min(), B.max(), 31)
        cs = ax.contour(zeta_deg, theta_deg, B, levels=levels, cmap="plasma")
        ax.set_xlabel(r"$\zeta_B$ [deg]", fontsize=8)
        ax.set_ylabel(r"$\theta_B$ [deg]", fontsize=8)
        ax.set_title(rf"$|B|$, $\rho={rho:.2f}$"
                     f"\n[{B.min():.2f}–{B.max():.2f} T]", fontsize=9)
        ax.set_xlim([0, 360 / eq.NFP])
        ax.set_ylim([0, 360])
        ax.tick_params(labelsize=7)
        fig.colorbar(cs, ax=ax, pad=0.02, shrink=0.85,
                     format="%.2f").ax.tick_params(labelsize=6)

    # hide unused top-row panels if n_surf < n_cols
    for col in range(n_surf, n_cols):
        axes[0, col].set_visible(False)

    # ── Pre-compute shared data ───────────────────────────────────────────────
    print("Computing f_B profile ...", flush=True)
    gc_json = gc_dir / "desc_gc_loss.json"
    if gc_json.exists():
        with open(gc_json) as f:
            gc_data = json.load(f)
        rho_fB = np.array(gc_data["fB_rho_vals"])
        fB     = np.array(gc_data["fB_per_rho"])
    else:
        rho_fB, fB = compute_fB_profile(eq, num_rho=args.num_rho)
        gc_data = None

    print("Computing Bmin spread vs rho ...", flush=True)
    rho_bm, bmin_mean, bmin_std, bmin_min, bmin_max = compute_bmin_profile(
        eq, num_rho=args.num_rho, num_field_lines=args.num_field_lines,
    )

    # ── Panel (1,0): loss fraction vs time ───────────────────────────────────
    ax = axes[1, 0]
    if gc_data is not None:
        ts   = np.array(gc_data["loss_curve_t"])
        frac = np.array(gc_data["loss_curve_frac"])
        ax.loglog(ts * 1e3, frac * 100)
        for t_ms, key in [(1.0, "loss_at_1ms"), (3.0, "loss_at_3ms")]:
            val = gc_data.get(key)
            if val is not None:
                ax.axvline(t_ms, color="gray", ls="--", lw=0.7, alpha=0.7)
                ax.text(t_ms * 1.08, ax.get_ylim()[0] * 2,
                        f"{val*100:.2f}%", fontsize=7, color="gray")
        ax.set_xlabel("Time [ms]")
        ax.set_ylabel("Lost fraction [%]")
        ax.set_title(f"Alpha loss vs time\n"
                     f"(10 ms: {gc_data['loss_at_tmax']*100:.2f}%)")
        ax.grid(True, which="both", alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No GC data", ha="center", va="center",
                transform=ax.transAxes, fontsize=11)
        ax.set_title("Loss vs time  [no data]")

    # ── Panel (1,1): loss fraction vs B_crit ─────────────────────────────────
    ax = axes[1, 1]
    npz_path = gc_dir / "desc_gc_particles.npz"
    if npz_path.exists():
        print("Computing loss vs B_crit ...", flush=True)
        centers, loss_bin, count_bin, B_lo, B_hi = gc_loss_vs_bcrit(npz_path, eq)
        ax2 = ax.twinx()
        ax2.bar(centers, count_bin, width=centers[1] - centers[0],
                color="C0", alpha=0.2)
        ax2.set_ylabel("N particles", color="C0", fontsize=8)
        ax2.tick_params(axis="y", labelcolor="C0", labelsize=7)
        ax.plot(centers, loss_bin * 100, "o-", ms=3, color="C1", zorder=3)
        ax.axvline(B_lo, color="gray", ls="--", lw=0.8,
                   label=f"$B_{{\\rm min}}$={B_lo:.2f} T")
        ax.axvline(B_hi, color="gray", ls=":", lw=0.8,
                   label=f"$B_{{\\rm max}}$={B_hi:.2f} T")
        ax.set_xlabel(r"$B_{\rm crit} = E/\mu$ [T]")
        ax.set_ylabel("Loss fraction [%]", color="C1")
        ax.tick_params(axis="y", labelcolor="C1")
        ax.set_title("Loss vs $B_{\\rm crit}$ (trapped only)")
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No GC data", ha="center", va="center",
                transform=ax.transAxes, fontsize=11)
        ax.set_title("Loss vs $B_{\\rm crit}$  [no data]")

    # ── Panel (1,2): f_B vs rho ───────────────────────────────────────────────
    ax = axes[1, 2]
    ax.plot(rho_fB, fB, "o-", ms=4, color="C0")
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(r"$f_B$")
    ax.set_title(f"QS error $f_B$ vs $\\rho$\n(edge = {fB[-1]:.4f})")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    # ── Panel (1,3): Bmin spread vs rho ──────────────────────────────────────
    ax = axes[1, 3]
    ax.fill_between(rho_bm, bmin_min, bmin_max, alpha=0.2, color="C2",
                    label="min–max")
    ax.fill_between(rho_bm, bmin_mean - bmin_std, bmin_mean + bmin_std,
                    alpha=0.4, color="C2", label=r"mean $\pm$ std")
    ax.plot(rho_bm, bmin_mean, "o-", ms=4, color="C2",
            label=r"mean $B_\mathrm{min}$")
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(r"$B_{\rm min}(\alpha)$ [T]")
    ax.set_title(r"$B_{\rm min}$ spread vs $\rho$"
                 f"\n(std$_{{\\rm edge}}$={bmin_std[-1]:.3f} T)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ── Row 2: R-Z cross-sections at 4 toroidal angles ───────────────────────
    print("Computing R-Z cross-sections ...", flush=True)
    zeta_vals = np.linspace(0, np.pi / eq.NFP, 4, endpoint=False)  # one field period
    xs, rho_xs = compute_cross_sections(eq, zeta_vals)
    colors_xs  = plt.cm.viridis(np.linspace(0.15, 0.95, len(rho_xs)))

    for col, xd in enumerate(xs):
        ax = axes[2, col]
        zeta_deg = np.degrees(xd["zeta"])
        for j, (R_j, Z_j) in enumerate(zip(xd["R"], rho_xs)):
            R_curve = np.append(xd["R"][j], xd["R"][j][0])
            Z_curve = np.append(xd["Z"][j], xd["Z"][j][0])
            lw  = 1.5 if rho_xs[j] == 1.0 else 0.8
            ax.plot(R_curve, Z_curve, color=colors_xs[j], lw=lw,
                    label=f"$\\rho$={rho_xs[j]:.2f}")
        ax.set_xlabel("R [m]", fontsize=8)
        ax.set_ylabel("Z [m]", fontsize=8)
        ax.set_title(f"$\\zeta = {zeta_deg:.1f}°$", fontsize=9)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
        ax.grid(True, lw=0.4, alpha=0.4)
        if col == n_cols - 1:
            ax.legend(fontsize=6, loc="upper right")

    # hide unused row-2 panels if any
    for col in range(len(zeta_vals), n_cols):
        axes[2, col].set_visible(False)

    # ── save ──────────────────────────────────────────────────────────────────
    fig.suptitle(label, fontsize=8)
    fig.subplots_adjust(left=0.05, right=0.97, top=0.95, bottom=0.05,
                        hspace=0.45, wspace=0.38)
    out = job_dir / "diagnostics.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
