#!/bin/bash
# SLURM array job for the proximal-lsq hyperparameter scan.
# Like scan_submit.sh but uses --qos=regular with a 1-hour wall time,
# allowing k_max=5 and higher-iota configs that exceed the 30-min debug limit.
#
# Usage:
#   python3 scan_define_iota_scan2.py --out scan_runs/grid_iota_scan2.json
#   N=$(python3 -c "import json; print(len(json.load(open('scan_runs/grid_iota_scan2.json'))))")
#   sbatch --array=1-${N} scan_submit_regular.sh \
#       --grid scan_runs/grid_iota_scan2.json \
#       --scan-dir scan_runs/iota_scan2
#
#SBATCH --job-name=qa-tr-scan
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --account=m4680
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --output=scan_runs/logs/%x-%A_%a.out
#SBATCH --error=scan_runs/logs/%x-%A_%a.err

set -euo pipefail

GRID="scan_runs/grid_iota_scan2.json"
SCAN_DIR="scan_runs/iota_scan2"
REFERENCE="wout_new_QA_aScaling_desc.h5"
DEVICE="gpu"
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --grid)       GRID="$2"; shift 2 ;;
        --scan-dir)   SCAN_DIR="$2"; shift 2 ;;
        --reference)  REFERENCE="$2"; shift 2 ;;
        --device)     DEVICE="$2"; shift 2 ;;
        --extra)      EXTRA_ARGS+=("$2"); shift 2 ;;
        *)            echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${SCRIPT_DIR}"

if [[ ! -f "${GRID}" ]]; then
    echo "ERROR: grid file not found: ${GRID} (cwd=${SCRIPT_DIR})" >&2
    exit 2
fi

mkdir -p "${SCAN_DIR}" "scan_runs/logs"

module load conda
conda activate /pscratch/sd/e/epaul/envs/desc

export SLURM_CPU_BIND=cores

TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"

read -r LABEL JOB_ID CONFIG_JSON < <(python3 - <<PY
import json, sys
with open("${GRID}") as f:
    cfgs = json.load(f)
i = ${TASK_ID} - 1
if i < 0 or i >= len(cfgs):
    print(f"task {i+1} out of range (grid has {len(cfgs)})", file=sys.stderr)
    sys.exit(2)
c = cfgs[i]
print(c["label"], c["job_id"], json.dumps(c["args"]))
PY
)

JOB_DIR="${SCAN_DIR}/job_$(printf '%03d' ${JOB_ID})_${LABEL}"
mkdir -p "${JOB_DIR}"

# Relocate SLURM logs into the job directory. Linux mv preserves open fds so
# all subsequent output (including the srun below) lands in JOB_DIR.
_jid="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"
_tid="${SLURM_ARRAY_TASK_ID:-0}"
mv -f "scan_runs/logs/${SLURM_JOB_NAME}-${_jid}_${_tid}.out" \
      "${JOB_DIR}/slurm_${_jid}_${_tid}.out" 2>/dev/null || true
mv -f "scan_runs/logs/${SLURM_JOB_NAME}-${_jid}_${_tid}.err" \
      "${JOB_DIR}/slurm_${_jid}_${_tid}.err" 2>/dev/null || true

echo "Job ID: ${SLURM_JOB_ID:-?} (array ${SLURM_ARRAY_JOB_ID:-?}_${TASK_ID})"
echo "Host: $(hostname)"
echo "PWD: $(pwd)"
echo "Output dir: ${JOB_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Config: ${CONFIG_JSON}"

echo "${CONFIG_JSON}" | python3 -m json.tool > "${JOB_DIR}/config.json"

mapfile -t CLI_ARGS < <(python3 - <<PY
import json
c = json.loads('''${CONFIG_JSON}''')
def emit(k, v):
    flag = "--" + k.replace("_", "-")
    if isinstance(v, (list, tuple)):
        print(flag)
        for x in v:
            print(x)
    else:
        if isinstance(v, bool):
            v = "true" if v else "false"
        elif v is None:
            v = "none"
        print(f"{flag}={v}")
for k, v in c.items():
    emit(k, v)
PY
)

set -x
srun -n1 --gpus-per-task=1 python -u proximal_lsq_scan_run.py \
    --device "${DEVICE}" \
    --reference "${REFERENCE}" \
    --output-dir "${JOB_DIR}" \
    --run-id "${LABEL}" \
    "${CLI_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"
