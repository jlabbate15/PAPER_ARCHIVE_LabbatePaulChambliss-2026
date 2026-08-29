#!/usr/bin/env python3
"""Overlay the 2-term QS metric f_C vs rho for scan60 (TR optimized) vs
scan62 (no TR baseline), evaluated directly from each run's final.h5.

Aggregation: max|f_C| over (theta, zeta) at each rho (not DESC's default
flux-surface average -- see desc.plotting.plot_qs_error for that variant).

Normalization: DESC's own B0 convention from plot_qs_error, i.e. a true
flux-surface/volume averaged field strength

    B0 = mean(|B| * sqrt(g)) / mean(sqrt(g))

(NOT compute_scaling_factors()["B"], and NOT divided by psi_r -- no ad-hoc
normalization).

Run from the project root with the `desc` conda env:
    conda activate desc
    python combined_figures/plot_fC_overlay.py
"""
import numpy as np
import matplotlib.pyplot as plt
from desc.equilibrium import Equilibrium
from desc.grid import LinearGrid

plt.rcParams.update({"font.size": 14})

RUNS = [
    (r"Opt with $\Delta_{\rm res}$", "scan60/final.h5", "#1f77b4"),
    (r"Opt without $\Delta_{\rm res}$", "scan62/final.h5", "#d62728"),
]

HELICITY = (1, 0)
RHO = np.linspace(0.05, 1.0, 20)

fig, ax = plt.subplots(figsize=(6, 5))

for label, path, color in RUNS:
    eq = Equilibrium.load(path)

    # DESC's own B0 convention (desc.plotting.plot_qs_error), not
    # compute_scaling_factors()["B"].
    data0 = eq.compute(["R0", "|B|"])
    B0 = np.mean(data0["|B|"] * data0["sqrt(g)"]) / np.mean(data0["sqrt(g)"])

    grid = LinearGrid(M=2 * eq.M_grid, N=2 * eq.N_grid, NFP=eq.NFP, rho=RHO)
    data = eq.compute(["f_C"], grid=grid, helicity=HELICITY)
    f_C = grid.meshgrid_reshape(data["f_C"], "rtz")
    fC_max = np.max(np.abs(f_C), axis=(1, 2)) / B0**3

    ax.plot(RHO, fC_max, "o-", color=color, label=label, markersize=5, linewidth=1.5)
    print(f"{label}: B0={B0:.4f} T  edge max|f_C|/B0^3={fC_max[-1]:.4e}  "
          f"core max|f_C|/B0^3={fC_max[0]:.4e}")

ax.set_xlabel(r"$\rho$")
ax.set_ylabel(r"$\max_{\theta,\zeta}|f_C| \,/\, B_0^3$")
# ax.set_title(r"2-term QS metric $f_C$ vs $\rho$ (max over surface, DESC $B_0^3$ norm.)")
ax.set_xlim(0, 1.05)
ax.set_ylim(bottom=0)
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3)
fig.tight_layout()

out = "combined_figures/fC_max_desc_norm_vs_rho_overlay.png"
fig.savefig(out, dpi=150)
print(f"Wrote {out}")
