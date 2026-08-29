#!/bin/bash
# GC tracing for scan60 j2-j6 (array tasks 1-5 map to jobs 2-6)
#
#SBATCH --job-name=gc-s60
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --constraint=gpu
#SBATCH --qos=debug
#SBATCH --account=m4680
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --output=scan_runs/logs/%x-%A_%a.out
#SBATCH --error=scan_runs/logs/%x-%A_%a.err

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${SCRIPT_DIR}"
mkdir -p scan_runs/logs

declare -a JOB_DIRS=(
    ""  # placeholder — array is 1-indexed
    "scan_runs/iota_scan60_20260707_nfp2_AR3p5_iota058_hpsweep/job_002_nfp2_QA_AR3p5_iota0p58_stab_trw10p0_qsw5p0_pq8_eta40_100it"
    "scan_runs/iota_scan60_20260707_nfp2_AR3p5_iota058_hpsweep/job_003_nfp2_QA_AR3p5_iota0p58_stab_trw5p0_qsw5p0_pq6_eta40_100it"
    "scan_runs/iota_scan60_20260707_nfp2_AR3p5_iota058_hpsweep/job_004_nfp2_QA_AR3p5_iota0p58_stab_trw5p0_qsw5p0_pq8_eta25_100it"
    "scan_runs/iota_scan60_20260707_nfp2_AR3p5_iota058_hpsweep/job_005_nfp2_QA_AR3p5_iota0p58_stab_trw5p0_qsw3p0_pq8_eta40_100it"
    "scan_runs/iota_scan60_20260707_nfp2_AR3p5_iota058_hpsweep/job_006_nfp2_QA_AR3p5_iota0p58_stab_trw7p0_qsw3p0_pq6_eta25_100it"
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"
JOB_DIR="${JOB_DIRS[$TASK_ID]}"

module load conda
conda activate /pscratch/sd/e/epaul/envs/desc
export SLURM_CPU_BIND=cores

_jid="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"
_tid="${SLURM_ARRAY_TASK_ID:-0}"
trap 'mv -f "scan_runs/logs/${SLURM_JOB_NAME}-${_jid}_${_tid}.out" \
          "${JOB_DIR}/slurm_gc_${_jid}_${_tid}.out" 2>/dev/null || true
      mv -f "scan_runs/logs/${SLURM_JOB_NAME}-${_jid}_${_tid}.err" \
          "${JOB_DIR}/slurm_gc_${_jid}_${_tid}.err" 2>/dev/null || true' EXIT

echo "GC tracing (task ${TASK_ID} -> s60j$((TASK_ID+1))): ${JOB_DIR}"
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
