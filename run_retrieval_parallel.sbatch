#!/bin/bash
#SBATCH --account=xdong
#SBATCH --partition=gpu_standard
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=1:00:00
#SBATCH --job-name=rtr_prj_pt25
#SBATCH --output=logs/ret_%j.out
#SBATCH --error=logs/ret_%j.err
#SBATCH --exclusive

export SLURM_NO_AUDIT=1
export SLURM_DISABLE_FATTR=1

echo "========== JOB STARTED: $(date) =========="
echo "Node list: $SLURM_NODELIST"
echo "Total tasks: $SLURM_NTASKS"
echo "----------------------------------------------------------"

source /home/u20/omidzandi/micromamba/etc/profile.d/mamba.sh
micromamba activate retrieval

cd /xdisk/behrangi/omidzandi/AVHRR-retrieval

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# export REPROJ_THREADS=3
# export MAX_PENDING_REPROJ=4

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=^lo,docker0

nvidia-smi || true
echo "----------------------------------------------------------"

srun --export=ALL python AVHRR_collocation_pipeline/run_retrieval_parallel.py \
    --config config/retrieve_config.toml

echo "=========== JOB ENDED: $(date) ==========="