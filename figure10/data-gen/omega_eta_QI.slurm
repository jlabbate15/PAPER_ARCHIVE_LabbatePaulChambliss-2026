#!/bin/bash

#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J omega_eta_QI
#SBATCH -A m4680
#SBATCH -t 09:00:00
#SBATCH -c 32

# Cap threading: a whole Milan node is 256 cores; letting XLA/OpenMP spawn
# that many compilers exhausts address space during long JIT scans.
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false"

# NERSC modules + conda (batch jobs need conda.sh sourced)
module load python
CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/.conda")
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate desc-env

cd /pscratch/sd/j/jlabbate/codes/DESC/DESC || exit 1
mkdir -p comparisondata
python --version

# Faster QI grid vs full 8x8: skip the two lowest energies (tmax ~ 1/sqrt(E)
# dominates runtime) and use 4 Bcrit instead of 8. Still spans FOW growth.
# SQuID is already at reactor scale; do not rescale.
# Five field lines: alpha = linspace(0, 2*pi, 5, endpoint=False); zeta0 seeds each.
python -u omega_eta_tracing_vs_bounce.py \
    --eq desc_eq2_wout_betaSQuIDhs.h5 \
    --helicity 0 1 \
    --zeta0 0.785 \
    --num-alpha 5 \
    --no-scale \
    --fractions 1e-5 1e-4 1e-3 1e-2 1e-1 1 \
    --n-bcrit 4 \
    --n-bounce 25 \
    --samples-per-bounce 40 \
    --output comparisondata/omega_eta_tracing_vs_bounce_QI
