# precise QA, Bcrit = 0.9794 T, Ekin/E_α = 1e-4 — figure bundle

Self-contained data + scripts for the ρ = 0.35–0.55 three-panel figure at
`Ekin/E_α = 1e-4`, and for the TR-vs-map width comparison at the same energy.
Extracted 2026-07-30 from
`July_2026/20260729_precise_QA_trapped_map_energy_scan/`.

## The figures

| file | what it is |
|---|---|
| `..._TR_three_panel_Efrac_1e-4_hires_ns600_zoom35_55.png` | map / Δ_res / Ω_η vs ρ, windowed to ρ ∈ [0.35, 0.55] |
| `..._zoom35_55_log.png` | same, log x-axis on Δ_res |
| `..._TR_vs_poincare_Efrac_1e-4.png` | predicted vs. map-measured island widths, with the background-floor flag |

Panels: **(a)** firm3d trapped Poincaré map, η vs ρ = √s, in firm3d's own
plotting convention (default colour cycle, `marker="o"`, `s=0.5`); **(b)**
Δ_res(ρ) = Σ_{p,q} Δs(ρ,p,q)⁴ · w(ρ,p,q), with w the objective's `res_weight`,
for both weightings — f_L = `weight_method="linear"`, f_b = `"bump"`;
**(c)** Ω_η(ρ) with the dominant rationals marked.

## Reproduce the figures from the included data

```bash
cd scripts
python plot_tr_three_panel.py --Efrac 1e-4 --tr-suffix _hires \
    --poinc-suffix _ns600 --rho-lim 0.35 0.55 --n-rational 6 \
    --suffix _hires_ns600_zoom35_55
python plot_tr_three_panel.py --Efrac 1e-4 --tr-suffix _hires \
    --poinc-suffix _ns600 --rho-lim 0.35 0.55 --n-rational 6 --log-dres \
    --suffix _hires_ns600_zoom35_55_log
python compare_tr_vs_poincare.py --Efrac 1e-4 --tr-suffix _hires \
    --poinc-suffix _ns600
```

The scripts use paths relative to `scripts/` (`../data`, `../data_tr`,
`../figures`), so run them from that directory. Needs numpy + matplotlib only.

## Regenerate the inputs from scratch

**Poincaré map** (`data/*_ns600.npz`) — firm3d, 128 MPI ranks, ~20 min:

```bash
sbatch scripts/delta_poincare_precise_QA_ns600.sh      # needs inputs/wout_precise_QA_desc.nc
```

`ns_poinc = 600`, `neta_poinc = 20`, `Nmaps = 3000`, `tmax = 1e-1 s`,
`tol = 1e-8`, interpolation resolution 48, helicity (M, N) = (1, 0). Produced
11460 initial conditions, 9474 (82.7%) completing all 3000 returns.
Needs `firm3d` and an MPI build that integrates with `srun` — see the header of
the submit script for the cray-mpich requirement on Delta.

**TR prediction** (`data_tr/tr_*_hires*.npz`) — DESC, ~30 s each on CPU:

```bash
cd data_tr
python ../scripts/run_tr_objective.py --KE-frac 1.0057175533e-04 --converged \
    --weight-method linear --out tr_precise_QA_Bcrit_0.98_Efrac_1e-4_hires.npz
python ../scripts/run_tr_objective.py --KE-frac 1.0057175533e-04 --converged \
    --weight-method bump   --out tr_precise_QA_Bcrit_0.98_Efrac_1e-4_hires_bump.npz
```

`--converged` sets `num_rho=120, num_eta=60, num_quad=40, X=Y=64`, justified by
a one-at-a-time resolution scan at both 1e-3 and 1e-4. Needs DESC from the
`ejp-merge-master` branch of `DESC_TrappedRes`, plus `interpax_fft` and `optax`.

## Two conventions that are easy to get wrong

**Bcrit is 0.9793859573581938 T, not 0.98.** λ is pinned to the exact value of
the pre-existing `Bcrit_0.98` baseline (λ = 1.0210479254750708); the "0.98" in
those filenames is a 2-decimal rounding. Using `1/0.98` puts you on a different
surface.

**DESC normalizes `KE_frac` to 3.5 MeV, firm3d to 3.52 MeV.** Equal `KE_frac`
values are *not* equal energies. The commands above use
`KE_frac = 1e-4 × 3.52/3.5 = 1.0057175533e-04` so the DESC energy matches the
firm3d map's 1e-4.

## Interpreting the width comparison — read this

Every trajectory carries an η-modulation of s from the equilibrium geometry
(nfp = 2, and the mapping angle is η = nfp·ζ), resonance or not. A null test on
a resonance-free band returns dominant mode k = 2 with 100% consensus over 247
librating trajectories — a *stronger* signal than any real island chain gives.
So:

- **Island counts from η-mode analysis are not trustworthy here.** Any chain
  returning k = 2 is reporting the geometric background. Counting islands needs
  that modulation subtracted first; that is not done in these scripts.
- **There is a background floor on the measured widths** of ≈ 0.0027 in s (95th
  percentile of 3680 trajectories more than 0.012 from any predicted resonance).
  `compare_tr_vs_poincare.py` computes it and flags every row below 3× it as
  `BACKGROUND`. Only four chains clear it at 1e-4:

  | p/q | Δs (TR) | width (map) | vs floor | ratio |
  |---|---|---|---|---|
  | 1/1 | 0.01536 | 0.02089 | 7.7× | 1.36 |
  | 1/2 | 0.00734 | 0.01362 | 5.0× | 1.86 |
  | 1/3 | 0.03789 | 0.01308 | 4.8× | 0.35 |
  | 2/3 | 0.01171 | 0.01116 | 4.1× | **0.95** |

  2/3 is the one genuine agreement. The rest of the table is at background level
  and should not be read as a width measurement.

Resolution uncertainty on the DESC side is ~0.3% for the widest chain and 2–6%
for narrower ones, so it does not explain the 1.4–2.9× disagreements above.
