#!/bin/bash

#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J omega_eta_QA
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

python -u omega_eta_tracing_vs_bounce.py \
    --eq precise_QA \
    --helicity 1 0 \
    --num-alpha 5 \
    --output comparisondata/omega_eta_tracing_vs_bounce_QA
