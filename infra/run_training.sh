#!/usr/bin/env bash
# ==============================================================================
# open-moe-48b: Preemptible Resilient Distributed Training Launch Controller
# Target Hardware: Cloud TPU v5p Pod Slice
# Execution: Handles preemption recovery, checkpoint resumption, and log streaming
# ==============================================================================

set -euo pipefail

# --- Configuration & Defaults ---
WORKSPACE_ROOT="${HOME}/open-moe-48b"
CONFIG_FILE="${1:-${WORKSPACE_ROOT}/configs/moe_48b_v5p.yaml}"
BUCKET_NAME="${BUCKET_NAME:-open-moe-48b-storage}"
LOGS_DIR="${WORKSPACE_ROOT}/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_LOG="${LOGS_DIR}/training_${TIMESTAMP}.log"

mkdir -p "${LOGS_DIR}"

echo "=================================================================="
echo ">> [open-moe-48b] Initializing Production Training Controller"
echo ">> Config File:  ${CONFIG_FILE}"
echo ">> Target GCS:   gs://${BUCKET_NAME}"
echo ">> Local Log:    ${RUN_LOG}"
echo "=================================================================="

# Activate virtual environment
VENV_DIR="${HOME}/open-moe-venv"
if [ -f "${VENV_DIR}/bin/activate" ]; then
    source "${VENV_DIR}/bin/activate"
else
    echo "WARNING: ${VENV_DIR} not found; proceeding with system Python."
fi

# Multi-host and TPU runtime optimizations
export JAX_PLATFORMS="tpu"
export PYTHONUNBUFFERED=1
export LIBTPU_INIT_ARGS="--xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true"

# Graceful termination handler for TPU preemption events
cleanup() {
    echo ">> [SIGTERM / SIGINT Detected] Initiating graceful shutdown..."
    echo ">> Allowing active Orbax checkpoint flush to complete..."
    sleep 5
    echo ">> Safe shutdown complete. Process terminating."
    exit 0
}
trap cleanup SIGTERM SIGINT

# --- Checkpoint Detection & Automatic Resumption ---
CHECKPOINT_BASE="gs://${BUCKET_NAME}/checkpoints/open-moe-48b"
echo ">> Probing GCS for existing checkpoints at ${CHECKPOINT_BASE}..."

LATEST_CHECKPOINT=""
if command -v gcloud &> /dev/null; then
    LATEST_CHECKPOINT=$(gcloud storage ls "${CHECKPOINT_BASE}/checkpoints/" 2>/dev/null | sort -V | tail -n 1 || true)
elif command -v gsutil &> /dev/null; then
    LATEST_CHECKPOINT=$(gsutil ls "${CHECKPOINT_BASE}/checkpoints/" 2>/dev/null | sort -V | tail -n 1 || true)
fi

LOAD_PARAM_ARG=""
if [ -n "${LATEST_CHECKPOINT}" ]; then
    echo ">> Detected existing checkpoint from prior run: ${LATEST_CHECKPOINT}"
    echo ">> Configuring automatic warm-restart resume..."
    LOAD_PARAM_ARG="load_parameters_path=${LATEST_CHECKPOINT}"
else
    echo ">> No prior checkpoints found. Initializing model weights from scratch (Step 0)."
fi

# --- Launch MaxText Distributed Training ---
MAXTEXT_DIR="${WORKSPACE_ROOT}/maxtext"
if [ ! -d "${MAXTEXT_DIR}" ]; then
    echo "ERROR: MaxText directory not found at ${MAXTEXT_DIR}. Run setup_tpu.sh first."
    exit 1
fi

echo "=================================================================="
echo ">> Launching MaxText Training Loop..."
echo "=================================================================="

cd "${MAXTEXT_DIR}"

# Run training with dual logging (console + persistent file)
set +e
python3 MaxText/train.py \
    "${CONFIG_FILE}" \
    base_output_directory="gs://${BUCKET_NAME}/checkpoints/open-moe-48b" \
    ${LOAD_PARAM_ARG} \
    2>&1 | tee -a "${RUN_LOG}"

EXIT_CODE=$?
set -e

echo ">> MaxText process exited with code ${EXIT_CODE} at $(date)"

# Forward exit status
if [ ${EXIT_CODE} -ne 0 ]; then
    echo "ERROR: Training exited abnormally with status ${EXIT_CODE}."
    echo "Inspect logs at ${RUN_LOG} or GCP Cloud Logging."
    exit ${EXIT_CODE}
fi

echo ">> Training session completed successfully."
