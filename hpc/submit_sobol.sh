#!/bin/bash
#SBATCH --job-name=seir-sobol
#SBATCH --output=logs/sobol_%j.out
#SBATCH --error=logs/sobol_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --mail-type=END,FAIL

module load python/3.11 cuda/12.1 2>/dev/null || true
source .venv/bin/activate 2>/dev/null || { python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt scikit-learn; }

mkdir -p logs

echo "=== Sobol Sensitivity Analysis ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_NODELIST"
echo "CPUs:      $SLURM_CPUS_PER_TASK"
echo ""

python3 sensitivity_analysis.py

echo "Sobol complete. PNGs saved."
