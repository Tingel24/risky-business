#!/bin/bash
#SBATCH --job-name=qwq-dpo-train
#SBATCH --output=results/logs/train-%j.out
#SBATCH --error=results/logs/train-%j.out
#SBATCH --partition=scc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -G A100:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=100GB
#SBATCH --time=12:00:00
#SBATCH -C inet

set -euo pipefail

export HF_HOME="$(ws_find models)"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TRITON_CACHE_DIR="$(ws_find TRITON_CACHE)"

# Ensure output directories exist
mkdir -p results/logs
mkdir -p results/qwq_aligned_model

# Load required modules
module load miniforge3 gcc/13.2.0 cuda/12.6.2

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STARTING DPO FINETUNING FOR QWQ-32B"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Activate your conda environment
source activate fact-llm-env
cd src

# Execute the training script and pipe output to a dedicated log file
accelerate launch --multi_gpu --num_processes 4 finetune_qlora.py 2>&1 | tee "../results/logs/accelerate_train_${SLURM_JOB_ID}.log"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "FINETUNING BATCH COMPLETE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
