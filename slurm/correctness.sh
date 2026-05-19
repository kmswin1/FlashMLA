#!/bin/bash
#SBATCH --job-name=correctness-sparse-bwd
#SBATCH --partition=compute-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=/t1data/users/kmswin7/Megatron-LM/logs/%j.%x.log
#SBATCH --error=/t1data/users/kmswin7/Megatron-LM/logs/%j.%x.error

set -exuo pipefail

# Correctness test for M16 sparse MLA bwd kernel.
# Compares flashmla_bwd output vs PyTorch autograd reference.

srun -l --ntasks=1 --gpus-per-task=1 \
    --container-image="/t1data/users/kmswin7/Megatron-LM/enroot/nemo+26.02+ofi116+dsa.sqsh" \
    --container-mounts="/t1data:/t1data" \
    --no-container-mount-home \
    bash -c '
set -e
export PYTHONPATH=/t1data/users/kmswin7/Megatron-LM:/t1data/users/kmswin7/DeepGEMM:/t1data/users/kmswin7/FlashMLA
export MEGATRON_DSA_USE_FLASHMLA=1
export MEGATRON_DSA_USE_FLASHMLA_BWD=1
export 
export MEGATRON_DSA_BWD_ROUTE_DEBUG=1
export CUDA_LAUNCH_BLOCKING=1
cd /t1data/users/kmswin7/Megatron-LM

echo "=============================================================="
echo "== correctness: dq/dkv vs autograd ref"
echo "=============================================================="
python scripts/test_sparse_mla_bwd_correctness.py
'
set +x
