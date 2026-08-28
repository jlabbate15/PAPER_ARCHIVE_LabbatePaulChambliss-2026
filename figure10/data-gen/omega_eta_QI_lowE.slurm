#!/bin/bash

#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J omega_eta_QI_loE
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

# Low-energy complement to omega_eta_QI.slurm (which skips 1e-7 and 1e-6).
# Same Bcrit/trace/alpha settings as the fast QI job so results can be merged later.
python -u omega_eta_tracing_vs_bounce.py \
    --eq desc_eq2_wout_betaSQuIDhs.h5 \
    --helicity 0 1 \
    --zeta0 0.785 \
    --num-alpha 5 \
    --no-scale \
    --fractions 1e-7 1e-6 \
    --n-bcrit 4 \
    --n-bounce 25 \
    --samples-per-bounce 40 \
    --output comparisondata/omega_eta_tracing_vs_bounce_QI_lowE
