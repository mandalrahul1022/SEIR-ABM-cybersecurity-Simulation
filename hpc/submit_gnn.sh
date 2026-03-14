#!/bin/bash
#SBATCH --job-name=seir-gnn
#SBATCH --output=logs/gnn_%j.out
#SBATCH --error=logs/gnn_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --mail-type=END,FAIL

module load python/3.11 cuda/12.1 2>/dev/null || true
source .venv/bin/activate 2>/dev/null || { python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt scikit-learn; }

mkdir -p logs

echo "=== GNN Predictive Model Training ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_NODELIST"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

# Ensure pipeline ran first (Parquet data must exist)
if [ ! -f "outputs/run_001/simulation_snapshots.parquet" ]; then
    echo "ERROR: No Parquet data found. Run submit_pipeline.sh first."
    exit 1
fi

python3 predictive_model.py

echo "GNN training complete. gnn_performance.png saved."
