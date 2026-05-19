#!/bin/bash
#SBATCH --job-name=build-flashmla
#SBATCH --partition=compute-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=1
#SBATCH --exclusive
#SBATCH --time=01:30:00
#SBATCH --output=/t1data/users/kmswin7/Megatron-LM/logs/%j.%x.log
#SBATCH --error=/t1data/users/kmswin7/Megatron-LM/logs/%j.%x.error

set -exuo pipefail

# Build FlashMLA inside the same OFI+DSA container the training jobs use, so
# the resulting extension is ABI-compatible with our PyTorch / CUDA stack.
# The compiled package is installed into /t1data/users/kmswin7/FlashMLA so
# it's reachable from all subsequent runs via PYTHONPATH (no rebake needed).

srun -l --ntasks=1 --gpus-per-task=1 \
    --container-image="/t1data/users/kmswin7/Megatron-LM/enroot/nemo+26.02+ofi116+dsa.sqsh" \
    --container-mounts="/t1data:/t1data" \
    --no-container-mount-home \
    bash -c '
set -ex
echo "=== Environment check ==="
which python
python --version
which nvcc || true
nvcc --version 2>&1 | head -5 || true
python -c "import torch; print(\"torch\", torch.__version__, \"cuda\", torch.version.cuda)"

cd /t1data/users/kmswin7/FlashMLA

# Build for SM90 (H800) + SM100 (B200). NVCC 12.9+ assumed; this container has CUDA 13.0.
export NVCC_THREADS=16
export TORCH_CUDA_ARCH_LIST="9.0;10.0"

# CUDA 13 layout splits libcudacxx headers under .../include/cccl/cuda/std/, but the
# vendored cutlass expects to find <cuda/std/utility> at a top-level include root.
# Add the cccl/ subdir to NVCC and host-cc include paths so cuda/std/* resolves.
CUDA_CCCL_INC=/usr/local/cuda/targets/x86_64-linux/include/cccl
test -d "$CUDA_CCCL_INC" || CUDA_CCCL_INC=/usr/local/cuda-13.0/targets/x86_64-linux/include/cccl
test -d "$CUDA_CCCL_INC" || { echo "ERROR: cccl include not found"; exit 1; }
export NVCC_PREPEND_FLAGS="-I${CUDA_CCCL_INC} ${NVCC_PREPEND_FLAGS:-}"
export CFLAGS="-I${CUDA_CCCL_INC} ${CFLAGS:-}"
export CXXFLAGS="-I${CUDA_CCCL_INC} ${CXXFLAGS:-}"
echo "Using CCCL include: $CUDA_CCCL_INC"

echo "=== Building FlashMLA (full log) ==="
pip install -v --no-build-isolation -e . > /t1data/users/kmswin7/FlashMLA/build_full.log 2>&1 || {
    echo "=== build FAILED, tail of full log: ==="
    tail -200 /t1data/users/kmswin7/FlashMLA/build_full.log
    echo "=== first compile error in full log: ==="
    grep -B2 -A6 -m1 -E "error:|Error|fatal error" /t1data/users/kmswin7/FlashMLA/build_full.log || true
    exit 1
}
echo "=== build OK ==="

echo "=== Sanity import ==="
python -c "import flash_mla; print(dir(flash_mla))"

echo "=== Sparse prefill test (small) ==="
python tests/test_flash_mla_sparse_prefill.py 2>&1 | tail -30 || echo "test exited with $?"
'
set +x
