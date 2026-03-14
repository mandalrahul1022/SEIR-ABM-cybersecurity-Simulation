# ============================================================================
# SEIR-ABM Cybersecurity Simulation — Multi-stage Docker Image
#
# Stages:
#   1. builder  — installs all Python dependencies in a venv
#   2. runtime  — copies only the venv + source code (lean image)
#
# Build:   docker build -t seir-sim .
# Run:     docker run --rm -v $(pwd)/results:/app/outputs seir-sim
#
# GPU:     docker run --rm --gpus all -v $(pwd)/results:/app/outputs seir-sim
#          (requires nvidia-container-toolkit on the host)
# ============================================================================

# ── Stage 1: Builder ────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

COPY requirements.txt .

RUN pip install --no-cache-dir --prefix=/install \
    -r requirements.txt \
    scikit-learn

# ── Stage 2: Runtime ────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="Rajib Mandal"
LABEL description="SEIR-ABM Cybersecurity Simulation with PyTorch tensor engine"

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy project source
COPY . .

# Create output directory
RUN mkdir -p /app/outputs/run_001

# Default: run the full pipeline, then Sobol analysis, then GNN training
ENV PYTHONUNBUFFERED=1

# Entrypoint script that runs all phases sequentially
COPY <<'ENTRY' /app/entrypoint.sh
#!/bin/bash
set -e

echo "============================================"
echo "  SEIR-ABM Cybersecurity Simulation Suite"
echo "============================================"
echo ""

MODE="${RUN_MODE:-all}"

case "$MODE" in
  pipeline)
    echo "[1/1] Running simulation pipeline..."
    python3 run_pipeline.py
    ;;
  sobol)
    echo "[1/1] Running Sobol sensitivity analysis..."
    python3 sensitivity_analysis.py
    ;;
  gnn)
    echo "[1/1] Training GNN predictive model..."
    python3 predictive_model.py
    ;;
  all)
    echo "[1/3] Running simulation pipeline..."
    python3 run_pipeline.py
    echo ""
    echo "[2/3] Running Sobol sensitivity analysis..."
    python3 sensitivity_analysis.py
    echo ""
    echo "[3/3] Training GNN predictive model..."
    python3 predictive_model.py
    ;;
  *)
    echo "Unknown RUN_MODE: $MODE"
    echo "Valid modes: pipeline, sobol, gnn, all"
    exit 1
    ;;
esac

echo ""
echo "============================================"
echo "  All tasks complete."
echo "  Outputs saved to /app/outputs/"
echo "============================================"
ENTRY

RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
