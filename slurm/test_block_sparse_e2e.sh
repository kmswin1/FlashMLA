#!/bin/bash
#SBATCH --job-name=bs-e2e
#SBATCH --partition=compute-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=/t1data/users/pretrain/swa_build_logs/%j.%x.log
#SBATCH --error=/t1data/users/pretrain/swa_build_logs/%j.%x.error
set -exuo pipefail
# End-to-end 192/128 block-sparse train step: kernel fwd (q_block=256) -> expand
# selection -> kernel bwd (q_block=64), validated vs a single autograd oracle.
# Only a pure-python helper changed since the integrated build -> the editable
# reinstall is a ninja no-op (objects cached); it just restores flash_mla in the
# fresh container, then runs the e2e test.
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
export TORCH_CUDA_ARCH_LIST="9.0;10.0"
for CAND in /usr/local/cuda/targets/x86_64-linux/include/cccl \
            /usr/local/cuda-13.0/targets/x86_64-linux/include/cccl; do
    if [ -d "$CAND" ]; then
        export NVCC_PREPEND_FLAGS="-I${CAND} ${NVCC_PREPEND_FLAGS:-}"
        export CFLAGS="-I${CAND} ${CFLAGS:-}"; export CXXFLAGS="-I${CAND} ${CXXFLAGS:-}"
        break
    fi
done
echo "=== editable reinstall (ninja no-op; restores flash_mla in this container) ==="
pip install -v --no-build-isolation -e . > /t1data/users/pretrain/swa_build_logs/build_e2e.log 2>&1 || {
    echo "=== reinstall FAILED, first error: ==="
    grep -B2 -A12 -m1 -E "error:|fatal error" /t1data/users/pretrain/swa_build_logs/build_e2e.log || true
    tail -40 /t1data/users/pretrain/swa_build_logs/build_e2e.log; exit 1; }
echo "=== end-to-end fwd+bwd validation ==="
cd '"${SRC}"'/tests && python test_block_sparse_e2e.py
'
