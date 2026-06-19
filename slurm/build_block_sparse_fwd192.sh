#!/bin/bash
#SBATCH --job-name=bs-fwd192
#SBATCH --partition=compute-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=1
#SBATCH --time=01:30:00
#SBATCH --output=/t1data/users/pretrain/swa_build_logs/%j.%x.log
#SBATCH --error=/t1data/users/pretrain/swa_build_logs/%j.%x.error
set -exuo pipefail
# JIT-build + validate the 192/128 block-sparse MLA FORWARD (training path).
# Standalone (not in setup.py) -> JIT compile via torch.utils.cpp_extension.load.
CONTAINER=/t1data/container/nemo+26.04.00.sqsh
SRC=/t1data/users/pretrain/FlashMLA
srun -l --ntasks=1 --gpus-per-task=1 \
    --container-image="${CONTAINER}" \
    --container-mounts="/t1data:/t1data" \
    --no-container-mount-home \
    bash -c '
set -ex
cd '"${SRC}"'
export NVCC_THREADS=16
export TORCH_EXTENSIONS_DIR=/t1data/users/pretrain/.torch_ext
export TORCH_CUDA_ARCH_LIST="10.0"
for CAND in /usr/local/cuda/targets/x86_64-linux/include/cccl \
            /usr/local/cuda-13.0/targets/x86_64-linux/include/cccl; do
    if [ -d "$CAND" ]; then
        export NVCC_PREPEND_FLAGS="-I${CAND} ${NVCC_PREPEND_FLAGS:-}"
        export CFLAGS="-I${CAND} ${CFLAGS:-}"
        export CXXFLAGS="-I${CAND} ${CXXFLAGS:-}"
        echo "Using CCCL include: $CAND"
        break
    fi
done
echo "=== JIT build + validate 192/128 block-sparse fwd ==="
cd '"${SRC}"'/tests && python test_block_sparse_fwd_192.py
'
