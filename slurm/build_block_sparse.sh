#!/bin/bash
#SBATCH --job-name=build-blocksparse
#SBATCH --partition=compute-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=1
#SBATCH --time=02:30:00
#SBATCH --output=/t1data/users/pretrain/swa_build_logs/%j.%x.log
#SBATCH --error=/t1data/users/pretrain/swa_build_logs/%j.%x.error
set -exuo pipefail
# Full build with the block-sparse KV-outer bwd integrated into the main module
# (api.cpp block_sparse_prefill_bwd + setup.py source), then verify the registered
# python API produces correct dQ/dK/dV.
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
        export CFLAGS="-I${CAND} ${CFLAGS:-}"
        export CXXFLAGS="-I${CAND} ${CXXFLAGS:-}"
        echo "Using CCCL include: $CAND"
        break
    fi
done
echo "=== Building FlashMLA + block-sparse bwd ==="
pip install -v --no-build-isolation -e . > /t1data/users/pretrain/swa_build_logs/build_blocksparse.log 2>&1 || {
    echo "=== build FAILED, first compile error: ==="
    grep -B2 -A10 -m1 -E "error:|fatal error" /t1data/users/pretrain/swa_build_logs/build_blocksparse.log || true
    tail -60 /t1data/users/pretrain/swa_build_logs/build_blocksparse.log
    exit 1
}
echo "=== build OK; verifying registration + correctness ==="
python -c "import flash_mla.cuda as c; assert hasattr(c, \"block_sparse_prefill_bwd\"), \"not registered\"; print(\"registered OK\")"
cd '"${SRC}"'/tests && python test_block_sparse_integrated.py
'
