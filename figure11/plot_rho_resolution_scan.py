#!/usr/bin/env python3
"""Rho-resolution scan figure, styled after tr_resolution reference plot.

Blue (left axis):  sum_s Delta_res(s)                 <- stab_sacrifice=False, tr_sum
Green (right axis): sum_s Delta_res(s) * Omega'_h(s)^2 <- stab_sacrifice=True,  tr_sum
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_PATH = "tr_resolution_v14_num_rho.json"
OUT_PATH = "rho_resolution_scan.pdf"

with open(DATA_PATH) as f:
    data = json.load(f)

vals = data["stab"]["num_rho"]["vals"]
blue_y = data["nostab"]["num_rho"]["tr_sum"]
green_y = data["stab"]["num_rho"]["tr_sum"]


blue = "tab:blue"
green = "tab:green"


FontSize = 22
label_kw = {"fontfamily": "Times New Roman", "fontsize": FontSize}
tick_labelsize = FontSize - 2

plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
        "figure.figsize": (6.4, 4.8)
    }
)


# Default axes box in figure coordinates (0–1); tune once, reuse everywhere
AXES_MARGINS = dict(left=0.18, right=0.82, bottom=0.14, top=0.97) # percentages of the figure height/width

plt.rcParams.update(
    {
        "figure.subplot.left": AXES_MARGINS["left"],
        "figure.subplot.right": AXES_MARGINS["right"],
        "figure.subplot.bottom": AXES_MARGINS["bottom"],
        "figure.subplot.top": AXES_MARGINS["top"],
        # optional: spacing for multi-panel figures
        "figure.subplot.wspace": 0.25,
        "figure.subplot.hspace": 0.25,
    }
)


# apply to figures to apply even if using tight_layout
def apply_standard_axes(fig):
    fig.subplots_adjust(**AXES_MARGINS)

fig, ax1 = plt.subplots(figsize=(7, 5.5))

ax1.plot(vals, blue_y, "o-", color=blue, lw=1.8, ms=7)
ax1.set_yscale("log")
ax1.set_xlabel(r"Number of $\rho$ points", **label_kw)
ax1.set_ylabel(r"$\sum_s \Delta_{\rm res}(s)$", color=blue, **label_kw)
ax1.tick_params(axis="y", labelcolor=blue, labelsize=tick_labelsize)
ax1.tick_params(axis="x", labelsize=tick_labelsize)
ax1.grid(True, which="major", lw=0.6, alpha=0.4)

ax2 = ax1.twinx()
ax2.plot(vals, green_y, "o-", color=green, lw=1.8, ms=7)
ax2.set_yscale("log")
ax2.set_ylabel(r"$\sum_s \Delta_{\rm res}(s) \cdot \Omega_\eta'(s)^2$", color=green, rotation=270, labelpad=40, **label_kw)
ax2.tick_params(axis="y", labelcolor=green, labelsize=tick_labelsize)

# Share the same log range on both axes (padded) so gridlines coincide and
# the relative variation of the two curves is directly comparable.
all_y = blue_y + green_y
lo, hi = min(all_y), max(all_y)
ylim = (lo / 1.6, hi * 1.6)
ax1.set_ylim(*ylim)
ax2.set_ylim(*ylim)

for spine in ax1.spines.values():
    spine.set_visible(True)

# fig.tight_layout()
apply_standard_axes(fig)
fig.savefig(OUT_PATH, dpi=300)
print(f"Wrote {OUT_PATH}")
