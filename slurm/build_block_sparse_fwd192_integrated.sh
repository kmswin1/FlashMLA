#!/bin/bash
#SBATCH --job-name=bs-fwd192-integ
#SBATCH --partition=compute-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=1
#SBATCH --time=02:30:00
#SBATCH --output=/t1data/users/pretrain/swa_build_logs/%j.%x.log
#SBATCH --error=/t1data/users/pretrain/swa_build_logs/%j.%x.error
set -exuo pipefail
# Full build (sm90+sm100) with the 192/128 block-sparse FWD integrated into the main
# flash_mla.cuda module (api.cpp block_sparse_prefill_fwd + setup.py source), then
# verify the registered python API produces correct O/LSE + run the scaling bench.
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
echo "=== Building FlashMLA + block-sparse fwd/bwd integrated ==="
pip install -v --no-build-isolation -e . > /t1data/users/pretrain/swa_build_logs/build_blocksparse_fwd.log 2>&1 || {
    echo "=== build FAILED, first compile error: ==="
    grep -B2 -A12 -m1 -E "error:|fatal error" /t1data/users/pretrain/swa_build_logs/build_blocksparse_fwd.log || true
    tail -60 /t1data/users/pretrain/swa_build_logs/build_blocksparse_fwd.log
    exit 1
}
echo "=== build OK; verifying registration ==="
python -c "import flash_mla.cuda as c; assert hasattr(c, \"block_sparse_prefill_fwd\"), \"fwd not registered\"; assert hasattr(c, \"block_sparse_prefill_bwd\"); print(\"registered OK: fwd + bwd\")"
python -c "import flash_mla; assert hasattr(flash_mla, \"block_sparse_prefill_fwd\"); print(\"python export OK\")"
echo "=== correctness + scaling bench via the integrated API ==="
cd '"${SRC}"'/tests && python test_block_sparse_fwd_192_integrated.py
echo "=== 3-way bench vs STOCK dense MLA fwd (confirm diagonal-only mask closed the my-full gap) ==="
cd '"${SRC}"'/tests && python bench_block_sparse_fwd_vs_dense.py
echo "=== e2e fwd+bwd (confirm the mask opt did not change fwd O/LSE the bwd consumes) ==="
cd '"${SRC}"'/tests && python test_block_sparse_e2e.py
echo "=== (regression) integrated bwd still OK ==="
cd '"${SRC}"'/tests && python test_block_sparse_integrated.py
'
