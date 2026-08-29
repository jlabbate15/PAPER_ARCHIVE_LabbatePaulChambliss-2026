#!/bin/bash
# GC tracing for scan62 noTR baseline (AR=3.5, iota=0.58, trw=0)
#
#SBATCH --job-name=gc-s62
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --constraint=gpu
#SBATCH --qos=debug_preempt
#SBATCH --account=m4680
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --output=scan_runs/logs/%x-%j.out
#SBATCH --error=scan_runs/logs/%x-%j.err

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${SCRIPT_DIR}"
mkdir -p scan_runs/logs

JOB_DIR="scan_runs/iota_scan62_20260710_nfp2_AR3p5_iota058_noTR_baseline/job_001_nfp2_QA_AR3p5_iota0p58_stab_trw0p0_qsw5p0_pq8_eta40_100it"

module load conda
conda activate /pscratch/sd/e/epaul/envs/desc
export SLURM_CPU_BIND=cores

_jid="${SLURM_JOB_ID}"
trap 'mv -f "scan_runs/logs/${SLURM_JOB_NAME}-${_jid}.out" \
          "${JOB_DIR}/slurm_gc_${_jid}.out" 2>/dev/null || true
      mv -f "scan_runs/logs/${SLURM_JOB_NAME}-${_jid}.err" \
          "${JOB_DIR}/slurm_gc_${_jid}.err" 2>/dev/null || true' EXIT

echo "GC tracing: ${JOB_DIR}"
set -x

srun -n1 --gpus-per-task=1 python -u scan_desc_gc.py \
    --job-dir "${JOB_DIR}" \
    --eq-name final.h5 \
    --n-particles 50000 \
    --tmax 1e-2 \
    --tmin 1e-5 \
    --n-ts 200 \
    --rtol 1e-6 \
    --atol 1e-6 \
    --num-rho-fB 16 \
    --seed 42
