#!/bin/bash
#SBATCH --job-name=seir-pipeline
#SBATCH --output=logs/pipeline_%j.out
#SBATCH --error=logs/pipeline_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --mail-type=END,FAIL

# ── Environment setup ────────────────────────────────────────────────────
module load python/3.11 cuda/12.1 2>/dev/null || true

if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt scikit-learn
fi

mkdir -p logs outputs/run_001

echo "=== SEIR Pipeline ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_NODELIST"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Python:    $(python3 --version)"
echo ""

# ── Run simulation pipeline ──────────────────────────────────────────────
python3 run_pipeline.py

echo "Pipeline complete. Parquet saved to outputs/"
