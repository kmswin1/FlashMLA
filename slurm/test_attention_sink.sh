#!/bin/bash
#SBATCH --job-name=attn-sink
#SBATCH --partition=compute-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=1
#SBATCH --time=01:30:00
#SBATCH --output=/t1data/users/pretrain/swa_build_logs/%j.%x.log
#SBATCH --error=/t1data/users/pretrain/swa_build_logs/%j.%x.error
set -exuo pipefail
# gpt-oss attention-sink: validate (training SWA path autograd vs oracle) + w/ vs w/o
# ablation (FWD/BWD training + decode inference). Sink is pure-python on top of the
# kernels' (O, LSE), so the editable reinstall is a ninja no-op.
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
echo "=== editable reinstall (ninja no-op) ==="
pip install --no-build-isolation -e . > /t1data/users/pretrain/swa_build_logs/build_sink.log 2>&1 || {
    tail -40 /t1data/users/pretrain/swa_build_logs/build_sink.log; exit 1; }
echo "=== validate attention sink (training SWA path autograd vs oracle) ==="
cd '"${SRC}"'/tests && python test_attention_sink.py
echo "=== ablation: sink overhead (FWD/BWD training + decode inference) ==="
cd '"${SRC}"'/tests && python bench_attention_sink_ablation.py
'
