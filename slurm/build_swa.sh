#!/bin/bash
#SBATCH --job-name=build-flashmla-swa
#SBATCH --partition=compute-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=1
#SBATCH --time=02:30:00
#SBATCH --output=/t1data/users/pretrain/swa_build_logs/%j.%x.log
#SBATCH --error=/t1data/users/pretrain/swa_build_logs/%j.%x.error

set -exuo pipefail

# Compile-check the sliding-window (SWA) changes against the integrated FlashMLA
# tree at /t1data/users/pretrain/FlashMLA. Builds SM90 (decode) + SM100 (prefill
# fwd+bwd) so BOTH arches that the SWA touches are compiled. Uses an accessible
# container (the kmswin7 nemo+dsa image is not readable from this account).
# cutlass is pre-cloned into csrc/cutlass at the FlashMLA-pinned SHA.

CONTAINER=/t1data/container/nemo+26.04.00.sqsh
SRC=/t1data/users/pretrain/FlashMLA

srun -l --ntasks=1 --gpus-per-task=1 \
    --container-image="${CONTAINER}" \
    --container-mounts="/t1data:/t1data" \
    --no-container-mount-home \
    bash -c '
set -ex
echo "=== Environment check ==="
which python; python --version
which nvcc || true
nvcc --version 2>&1 | head -5 || true
python -c "import torch; print(\"torch\", torch.__version__, \"cuda\", torch.version.cuda)"

cd '"${SRC}"'

# Build SM90 (H100/H800) + SM100 (B200/B300). NVCC 12.9+ required for sm100.
export NVCC_THREADS=16
export TORCH_CUDA_ARCH_LIST="9.0;10.0"

# CUDA 13 splits libcudacxx headers under .../include/cccl/cuda/std/, but vendored
# cutlass expects <cuda/std/utility> at a top-level include root. Add cccl/ to the
# include paths (no-op on CUDA <13 where the dir is absent).
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

echo "=== Building FlashMLA + SWA (full log) ==="
pip install -v --no-build-isolation -e . > /t1data/users/pretrain/swa_build_logs/build_full.log 2>&1 || {
    echo "=== build FAILED, tail of full log: ==="
    tail -200 /t1data/users/pretrain/swa_build_logs/build_full.log
    echo "=== first compile error: ==="
    grep -B2 -A8 -m1 -E "error:|fatal error" /t1data/users/pretrain/swa_build_logs/build_full.log || true
    exit 1
}
echo "=== build OK ==="

echo "=== Sanity import + SWA arg present ==="
python -c "
import flash_mla, inspect
from flash_mla.flash_mla_interface import flash_attn_varlen_func, flash_mla_with_kvcache
assert \"window_size\" in inspect.signature(flash_attn_varlen_func).parameters, \"window_size missing on flash_attn_varlen_func\"
assert \"window_size\" in inspect.signature(flash_mla_with_kvcache).parameters, \"window_size missing on flash_mla_with_kvcache\"
print(\"OK: flash_mla imports; window_size exposed on prefill + decode entrypoints\")
"
'
set +x
