#!/bin/bash
#SBATCH -J poinc-pqa-ns600
#SBATCH -A bhvw-delta-cpu
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --ntasks=128
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH -t 02:00:00
#SBATCH -o poinc-pqa-ns600-%j.out

# Higher Poincare sampling at Ekin/E_alpha = 1e-4 only.
#
# Why: at ns_poinc = 120 the initial surfaces are spaced 1/120 = 0.0083 in s,
# which is WIDER than the 3/4 (Delta_s = 0.0080) and 4/5 (0.0062) island chains
# the TR objective predicts near rho ~ 0.46-0.48. Those chains came back
# unresolved in the island-count test -- at most one initial condition lands
# inside them. ns_poinc = 600 gives 1/600 = 0.00167 spacing, ~4 initial
# conditions across the narrower of the two.
#
# ns_poinc 120 -> 600 at fixed neta_poinc = 20 is 5x the initial conditions:
# the standard 1e-4 run took 152 s on 128 ranks, so expect ~13 min. Radial
# sampling is what resolves island WIDTH, so the refinement goes into ns_poinc
# rather than neta_poinc -- trajectories sweep eta as they map, so eta coverage
# does not come mainly from the initial conditions.
#
# Rank-0 memory: ~11400 trajectories x 3001 returns x 4 arrays x 8 B = 1.1 GB,
# roughly double at the gather peak. Fine on a 256 GB node with --mem=0, and the
# driver gathers to root only (NOT allgather -- see README, third OOM).
#
# cray-mpich MUST be loaded before miniforge3/conda: conda-forge's openmpi does
# not integrate with srun on Delta and silently degrades to 128 singleton MPI
# worlds. Check the trailing number in
# stdout_trapped_map_<tag>_48_<comm_size>.txt -- it must read 128.

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

module load PrgEnv-gnu cray-mpich
module load miniforge3-python
source $(conda info --base)/etc/profile.d/conda.sh
conda activate firm3d

cd /projects/bhvw/epaul/research/20260729_precise_QA_trapped_map_energy_scan

srun -n 128 -c 1 python -u trapped_map_precise_QA_energy_scan.py \
  --equil wout_precise_QA_desc.nc \
  --tag precise_QA_Bcrit_0.98 \
  --Ekin-frac 1e-4 \
  --ns-poinc 600 \
  --neta-poinc 20 \
  --out-suffix _ns600 \
  --skip-existing
