# University Malware SEIR Simulation

Agent-based and tensor-accelerated simulation of a malware outbreak on a university network.
The project uses SEIR dynamics (Susceptible, Exposed, Infected, Patched) on a scale-free
Barabasi-Albert topology, combining a Mesa dashboard for interactive exploration with a
GPU-ready PyTorch engine for high-throughput Monte Carlo analysis.

## Architecture Overview

```
Mesa ABM (N=200)                     TensorCyberSimulation (N=10,000)
models/university_network.py         tensor_engine.py
         │                                    │
    server.py (UI)               ┌────────────┼────────────┐
    run_analysis.ipynb           │            │            │
                          run_pipeline.py  sensitivity_   predictive_
                          (Parquet+PyG)    analysis.py    model.py
                                          (Sobol SA)     (GCN)
```

## Highlights

- **Network-driven contagion** on Barabasi-Albert scale-free graph
- **Two simulation engines**: Mesa ABM (interactive) + PyTorch tensor engine (vectorized)
- **Spectral Graph Theory**: epidemic threshold prediction via adjacency matrix spectral radius
- **Phase III mechanics**: asynchronous patching queue, stochastic edge rewiring, latency-weighted exposure
- **Sobol sensitivity analysis**: variance decomposition across 4 parameters (640 Monte Carlo runs)
- **GNN predictive model**: 2-layer GCN predicts per-node infection 5 ticks ahead (AUC = 0.983)
- **Data pipeline**: Parquet serialization + PyTorch Geometric InMemoryDataset

## Requirements

- Python 3.10+ (tested with Python 3.11)
- Packages listed in `requirements.txt`
- Runs on CPU (Mac/Linux), CUDA, or MPS (auto-detected)

## Setup

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# Interactive Mesa dashboard
python3 server.py          # then open http://127.0.0.1:8521

# Full tensor pipeline (simulation → Parquet → PyG dataset)
python3 run_pipeline.py

# Sobol sensitivity analysis (generates 2 PNGs)
python3 sensitivity_analysis.py

# GNN training (generates gnn_performance.png)
python3 predictive_model.py
```

## Project Structure

| File | Description |
|---|---|
| `tensor_engine.py` | PyTorch vectorized SEIR engine (N=10,000, sparse matrix multiplication) |
| `models/university_network.py` | Mesa ABM with SEIR agents and hub-targeted patching |
| `server.py` | Mesa visualization server with sliders and charts |
| `run_pipeline.py` | End-to-end orchestrator: simulation → Parquet → PyG |
| `sensitivity_analysis.py` | Sobol global sensitivity analysis (SALib) |
| `predictive_model.py` | Graph Convolutional Network for infection prediction |
| `data/parquet_export.py` | zstd-compressed Parquet exporter for simulation snapshots |
| `data/pyg_dataset.py` | PyTorch Geometric InMemoryDataset from Parquet |
| `run_analysis.ipynb` | Mesa model proof experiments (threshold, rate, strategy) |

## Mathematical Architecture & Generalizability

The tensor engine computes stochastic SEIR compartment transitions over a Barabasi-Albert
scale-free topology via sparse matrix-vector multiplication. Transmission probability is
derived as `1 - (1 - β)^k` where `k` is the count of infected neighbors, computed in a
single SpMV operation. The framework is substrate-agnostic: the same engine applies to any
host/pathogen contagion domain where network structure and intervention policy jointly
determine cascade behavior.

### Spectral Graph Theory Calibration

Before simulation, the engine computes the spectral radius ρ(A) of the adjacency matrix
using Lanczos iteration (`scipy.sparse.linalg.eigsh`) and derives the epidemic threshold
λ_c = 1/ρ(A). For BA(10000, m=3), ρ(A) ≈ 22.7 and λ_c ≈ 0.044, confirming that
β = 0.40 exceeds the threshold by ~9x — a mathematically guaranteed pandemic.

## Key Results

### 1. Mesa ABM Experiments (N=200, Monte Carlo, fixed seeds)

- **Percolation threshold**: critical β < 0.020 for BA(200, m=3) under zero patching
- **Patching rate sensitivity**: doubling rate (0.05 → 0.10) reduces mean peak by **32.90%**
- **Strategy comparison**: targeted hub patching reduces peak by **52.24%** vs random

### 2. Tensor Engine (N=10,000, Phase III mechanics)

- **Peak infected**: 9,372 / 10,000 nodes (with 3-tick average queue delay + 5% rewiring)
- **Spectral prediction**: β/λ_c ≈ 9x threshold → pandemic mathematically guaranteed
- **Targeted patching assertion**: ≥ 50% peak reduction vs random (validated)

### 3. Sobol Sensitivity Analysis (640 Saltelli-sampled runs)

| Parameter | S1 (First-order) | ST (Total-order) |
|---|---|---|
| `spread_chance` | 0.494 | **0.528** |
| `patching_rate` | 0.251 | **0.325** |
| `patch_completion_prob` | 0.186 | 0.132 |
| `rewire_rate` | 0.002 | 0.005 |

**Finding**: Viral transmissibility alone explains 52.8% of outbreak variance. The
strongest defensive lever (patching rate) explains 32.5%. Network volatility (rewiring)
is statistically irrelevant (ST = 0.005).

### 4. GNN Predictive Model (2-layer GCN, 705 parameters)

- **Test accuracy**: 93.8%
- **AUC-ROC**: 0.983
- **Hub recall**: 92.8% of future infected nodes correctly identified 5 ticks ahead

## Visual Artifacts

| File | Description |
|---|---|
| `sobol_indices.png` | S1 vs ST bar chart for all 4 parameters |
| `interaction_heatmap.png` | Pairwise S2 parameter interaction matrix |
| `gnn_performance.png` | Training loss/accuracy + ROC curve |
