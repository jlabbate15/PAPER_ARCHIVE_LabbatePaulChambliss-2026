#!/usr/bin/env python3
"""Overlay guiding-center alpha-particle loss fraction vs time for scan60
(TR optimized) vs scan62 (no TR baseline), from each run's desc_gc_loss.json
(loss_curve_t / loss_curve_frac, produced by scan_desc_gc.py).
"""
import json
import matplotlib.pyplot as plt

RUNS = [
    (r"Opt with $\Delta_{\rm res}$", "scan60/desc_gc_loss.json", "#1f77b4"),
    (r"Opt without $\Delta_{\rm res}$", "scan62/desc_gc_loss.json", "#d62728"),
]

fig, ax = plt.subplots(figsize=(6, 5))

for label, path, color in RUNS:
    with open(path) as f:
        d = json.load(f)
    ax.plot(d["loss_curve_t"], d["loss_curve_frac"], "-", color=color,
             label=f"{label}", linewidth=1.8)
    print(f"{label}: loss(tmax={d['tmax']}s) = {d['loss_at_tmax']:.4f}  "
          f"loss(1ms)={d['loss_at_1ms']:.4f}  loss(3ms)={d['loss_at_3ms']:.4f}")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Time [s]")
ax.set_ylabel("Lost fraction")
# ax.set_title("Alpha-particle loss vs time")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()

out = "combined_figures/gc_loss_vs_time_overlay.png"
fig.savefig(out, dpi=150)
print(f"Wrote {out}")
