#!/usr/bin/env python3
"""Overlay guiding-center alpha-particle loss fraction vs normalized Bcrit for
scan60 (TR optimized) vs scan62 (no TR baseline).

Bcrit = E/mu is renormalized per-config to a fractional trapping scale

    Bcrit_frac = (Bcrit - Bmin) / (Bmax - Bmin)   in [0, 1]

using each device's own (Bmin, Bmax) (same values used in
plot_trapped_frequencies_overlay.py, from the trapped_frequencies.py stdout
logs), so the two configs -- which have different absolute field ranges --
land on a common, comparable x-axis. Binning otherwise follows the same
convention as plot_diagnostics.py::gc_loss_vs_bcrit: 50 bins, loss fraction =
mean(lost) per bin (bins with <=5 particles are dropped).
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 14})

RUNS = [
    {
        "label": r"Opt with $\Delta_{\rm res}$",
        "npz": "scan60/desc_gc_particles.npz",
        "Bmin": 4.341373833229269,
        "Bmax": 6.2189847761509744,
        "color": "#1f77b4",
    },
    {
        "label": r"Opt without $\Delta_{\rm res}$",
        "npz": "scan62/desc_gc_particles.npz",
        "Bmin": 4.46608816496712,
        "Bmax": 6.453387929863838,
        "color": "#d62728",
    },
]


def gc_loss_vs_bcrit_frac(npz_path, Bmin, Bmax, n_bins=50):
    d = np.load(npz_path)
    B_crit = d["B_crit"]
    lost = d["lost"]

    Bcrit_frac = (B_crit - Bmin) / (Bmax - Bmin)
    mask = (Bcrit_frac >= 0) & (Bcrit_frac <= 1)
    frac_t, lost_t = Bcrit_frac[mask], lost[mask]

    bins = np.linspace(0, 1, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    loss_bin = np.full(n_bins, np.nan)
    count_bin = np.zeros(n_bins, dtype=int)
    idx = np.clip(np.digitize(frac_t, bins) - 1, 0, n_bins - 1)
    for i in range(n_bins):
        m = idx == i
        count_bin[i] = m.sum()
        if m.sum() > 5:
            loss_bin[i] = lost_t[m].mean()
    return centers, loss_bin, count_bin


fig, ax = plt.subplots(figsize=(6.5, 5))
ax2 = ax.twinx()

for run in RUNS:
    centers, loss_bin, count_bin = gc_loss_vs_bcrit_frac(
        run["npz"], run["Bmin"], run["Bmax"])
    ax.plot(centers, loss_bin * 100, "o-", ms=3, color=run["color"],
            label=run["label"], zorder=3)
    ax2.bar(centers, count_bin, width=centers[1] - centers[0],
            color=run["color"], alpha=0.12)
    print(f"{run['label']}: Bmin={run['Bmin']:.3f} T  Bmax={run['Bmax']:.3f} T  "
          f"peak loss={np.nanmax(loss_bin)*100:.1f}%")

ax.set_xlabel(r"$(B_{\rm crit}-B_{\rm min})/(B_{\rm max}-B_{\rm min})$")
ax.set_ylabel("Loss fraction [%]")
ax2.set_ylabel("N particles", color="gray")
ax2.tick_params(axis="y", labelcolor="gray")
# ax.set_title(r"Alpha loss vs normalized $B_{\rm crit}$ (trapped particles only)")
ax.set_xlim(0, 1)
ax.set_ylim(bottom=0)
ax.legend(fontsize=9, loc="upper left")
ax.grid(True, alpha=0.3)
fig.tight_layout()

out = "combined_figures/gc_loss_vs_bcrit_overlay.png"
fig.savefig(out, dpi=150)
print(f"Wrote {out}")
