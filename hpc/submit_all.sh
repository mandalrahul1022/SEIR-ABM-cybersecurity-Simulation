#!/bin/bash
# submit_all.sh — Submit the full SEIR experiment suite as a Slurm job chain.
# Each job waits for the previous one to finish successfully.
#
# Usage: bash hpc/submit_all.sh

set -e
cd "$(dirname "$0")/.."

echo "Submitting SEIR experiment chain to Slurm..."

JOB1=$(sbatch --parsable hpc/submit_pipeline.sh)
echo "  [1/3] Pipeline submitted: Job $JOB1"

JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 hpc/submit_sobol.sh)
echo "  [2/3] Sobol submitted (depends on $JOB1): Job $JOB2"

JOB3=$(sbatch --parsable --dependency=afterok:$JOB1 hpc/submit_gnn.sh)
echo "  [3/3] GNN submitted (depends on $JOB1): Job $JOB3"

echo ""
echo "Chain submitted: $JOB1 → ($JOB2, $JOB3)"
echo "Monitor with: squeue -u \$USER"
