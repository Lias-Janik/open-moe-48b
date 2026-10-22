#!/usr/bin/env bash
# ==============================================================================
# open-moe-48b: Cloud TPU VM Node Provisioning & Environment Bootstrap
# Target Hardware: Google Cloud TPU v4 / v5p (Multi-Host Slice)
# ==============================================================================

set -euo pipefail

echo "=================================================================="
echo ">> [1/6] Initializing Cloud TPU VM Environment Bootstrap..."
echo "=================================================================="

# Export system and hardware environment variables
export DEBIAN_FRONTEND=noninteractive
export JAX_PLATFORMS="tpu"
export TPU_CHIPS_PER_HOST_BOUNDS="2,2,1"
export LIBTPU_INIT_ARGS="--xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_spmd_rng_bit_generator_unsafe=true"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda"
export PYTHONUNBUFFERED=1

echo ">> System Architecture: $(uname -m)"
echo ">> Host Kernel: $(uname -r)"

echo "=================================================================="
echo ">> [2/6] Updating APT Repositories & Base Tooling..."
echo "=================================================================="
sudo apt-get update -y
sudo apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    tmux \
    htop \
    rsync \
    jq \
    python3-dev \
    python3-pip \
    python3-venv \
    google-perftools \
    libgoogle-perftools-dev

echo "=================================================================="
echo ">> [3/6] Setting Up Python 3 Virtual Environment..."
echo "=================================================================="
VENV_DIR="${HOME}/open-moe-venv"
if [ ! -d "${VENV_DIR}" ]; then
    echo ">> Creating clean Python virtual environment at ${VENV_DIR}..."
    python3 -m venv "${VENV_DIR}"
fi

# Activate virtual environment
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip setuptools wheel

echo "=================================================================="
echo ">> [4/6] Installing JAX with TPU Acceleration Support (libtpu)..."
echo "=================================================================="
# Install matching jax[tpu] release wheel from official Google storage
pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

echo "=================================================================="
echo ">> [5/6] Cloning MaxText & Installing Core Dependencies..."
echo "=================================================================="
WORKSPACE_ROOT="${HOME}/open-moe-48b"
if [ ! -d "${WORKSPACE_ROOT}/maxtext" ]; then
    echo ">> Cloning official MaxText repository into ${WORKSPACE_ROOT}/maxtext..."
    git clone https://github.com/google/maxtext.git "${WORKSPACE_ROOT}/maxtext"
fi

# Install open-moe-48b dependencies
if [ -f "${WORKSPACE_ROOT}/requirements.txt" ]; then
    echo ">> Installing open-moe-48b requirements from ${WORKSPACE_ROOT}/requirements.txt..."
    pip install -r "${WORKSPACE_ROOT}/requirements.txt"
fi

# Install MaxText runtime dependencies
if [ -f "${WORKSPACE_ROOT}/maxtext/requirements.txt" ]; then
    echo ">> Installing MaxText requirements..."
    pip install -r "${WORKSPACE_ROOT}/maxtext/requirements.txt" || true
fi

echo "=================================================================="
echo ">> [6/6] Verifying TPU Hardware Availability & Self-Test..."
echo "=================================================================="
python3 - << 'EOF'
import jax

devices = jax.devices()
platform = jax.default_backend()
print(f"JAX Default Backend: {platform}")
print(f"Detected {len(devices)} JAX device(s):")
for idx, dev in enumerate(devices):
    print(f"  [{idx}] {dev.device_kind}")

if platform != "tpu":
    print("WARNING: Default backend is not TPU. Check libtpu installation.")
else:
    print("SUCCESS: Google Cloud TPU drivers initialized and verified.")
EOF

echo "=================================================================="
echo ">> TPU Provisioning Complete! To activate the environment:"
echo ">> source ${VENV_DIR}/bin/activate"
echo "=================================================================="
