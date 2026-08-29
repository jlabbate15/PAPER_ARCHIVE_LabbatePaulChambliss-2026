#!/usr/bin/env python3
"""Overlay trapped-particle bounce/precession frequency ratio omega_alpha/omega_b
vs s for scan60 (TR optimized) vs scan62 (no TR baseline), from each config's
*_trapped_frequencies.npz (produced by trapped_frequencies.py / firm3d
TrappedPoincare).

Each npz holds 10 Bcrit = E/mu levels (lam_values = 1/Bcrit) spanning
[1/B_max, 1/B_min] of that specific equilibrium. Since the two configs have
different B_min/B_max, raw Bcrit values aren't directly comparable -- instead
each level is normalized to a fractional trapping scale

    Bcrit_frac = (Bcrit - Bmin) / (Bmax - Bmin)   in [0, 1]

using the config's own (Bmin, Bmax) (from the trapped_frequencies.py stdout
logs), so a given color means the same relative trapping depth in both
configs. Color encodes Bcrit_frac (shared colormap/scale); linestyle
distinguishes the two configs.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

plt.rcParams.update({"font.size": 14})

RUNS = [
    {
        "label": r"Opt with $\Delta_{\rm res}$",
        "npz": "scan60/scan_60_trapped_frequencies.npz",
        "Bmin": 4.341373833229269,
        "Bmax": 6.2189847761509744,
        "ls": "-",
    },
    {
        "label": r"Opt without $\Delta_{\rm res}$",
        "npz": "scan62/scan_62_trapped_frequencies.npz",
        "Bmin": 4.46608816496712,
        "Bmax": 6.453387929863838,
        "ls": "--",
    },
]

cmap = plt.get_cmap("viridis")
norm = Normalize(vmin=0.0, vmax=1.0)

fig, ax = plt.subplots(figsize=(7, 5.5))

for run in RUNS:
    d = np.load(run["npz"], allow_pickle=True)
    lam_values = d["lam_values"]
    s_prof = d["s_prof"]
    omega_b_prof = d["omega_b_prof"]
    omega_alpha_prof = d["omega_alpha_prof"]

    Bcrit = 1.0 / lam_values
    Bcrit_frac = (Bcrit - run["Bmin"]) / (run["Bmax"] - run["Bmin"])

    for i in range(len(lam_values)):
        s = np.asarray(s_prof[i])
        if len(s) == 0:
            continue
        ratio = np.asarray(omega_alpha_prof[i]) / np.asarray(omega_b_prof[i])
        order = np.argsort(s)
        ax.plot(s[order], ratio[order], ls=run["ls"],
                color=cmap(norm(Bcrit_frac[i])), lw=1.6, marker="o", ms=3)

# Linestyle legend for configs (color is shared/normalized Bcrit, not per-config)
from matplotlib.lines import Line2D
config_handles = [
    Line2D([0], [0], color="k", ls=RUNS[0]["ls"], label=RUNS[0]["label"]),
    Line2D([0], [0], color="k", ls=RUNS[1]["ls"], label=RUNS[1]["label"]),
]
ax.legend(handles=config_handles, loc="upper left", fontsize=14)

sm = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label(r"$(B_{\rm crit}-B_{\rm min})/(B_{\rm max}-B_{\rm min})$")

ax.set_xlabel("s")
ax.set_ylabel(r"$\Omega_\eta$")
# ax.set_title("Trapped-particle precession/bounce frequency ratio vs s")
ax.grid(True, alpha=0.3)
fig.tight_layout()

out = "combined_figures/trapped_frequencies_overlay.png"
fig.savefig(out, dpi=150)
print(f"Wrote {out}")
