#!/bin/bash
#SBATCH --job-name=DrBenchmark
#SBATCH --output=logs/drbenchmark_%j.out
#SBATCH --error=logs/drbenchmark_%j.err
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_p6
#SBATCH --cpus-per-task=24
#SBATCH --account=kit@h100
#SBATCH -C h100
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu_h100-t3

# Jean-Zay specific module loading
module purge
module load arch/h100
module load pytorch-gpu/py3/2.7.0

# pip install --user --no-cache-dir -r requirements.txt

# Set environment variables for Jean-Zay
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Create logs directory
mkdir -p logs

# Run the benchmark
echo "Starting DrBenchmark evaluation on Jean-Zay..."
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Account: $SLURM_JOB_ACCOUNT"
echo "Time: $(date)"

# Run the main script
bash run.sh

echo "DrBenchmark completed at $(date)" 