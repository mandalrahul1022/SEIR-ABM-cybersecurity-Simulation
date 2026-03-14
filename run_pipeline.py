#!/usr/bin/env python3
"""
run_pipeline.py

End-to-end SEIR simulation pipeline:
  1. Build BA graph and initialize node states (tensor_engine.py)
  2. Run 100-tick simulation and stream per-tick snapshots to Parquet (data/parquet_export.py)
  3. Load the Parquet snapshots back as a PyG InMemoryDataset (data/pyg_dataset.py)
  4. Print dataset info and the first Data object for inspection

Requires: torch, torch_geometric, pyarrow, numpy, networkx
Run: python3 run_pipeline.py
"""

from __future__ import annotations

import pathlib
import sys

import torch
from torch import Tensor

# ---- Import the engine and data pipeline --------------------------------
from tensor_engine import (
    N,
    STATE_I,
    STATE_S,
    build_sparse_adj_matrix,
    calculate_epidemic_threshold,
    compute_static_hub_mask,
    device,
    initialize_node_states,
    simulation_step,
)
from data.parquet_export import SimulationExporter
from data.pyg_dataset import SEIRGraphDataset, edge_index_from_sparse_adj

# ---- Run configuration --------------------------------------------------
SPREAD_CHANCE:          float = 0.4
PATCHING_RATE:          float = 0.10
PATCHING_STRATEGY:      str   = "Targeted"
VOLATILITY_RATE:        float = 0.20
PATCH_COMPLETION_PROB:  float = 0.33
REWIRE_RATE:            float = 0.05
NUM_TICKS:              int   = 100
INITIAL_INFECTED:       int   = 5
SEED:                   int   = 42
OUTPUT_DIR:             pathlib.Path = pathlib.Path("outputs/run_001")

# ---- Reproducibility ----------------------------------------------------
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# =========================================================================
# Step 1 — Build graph and static structures
# =========================================================================
print(f"Device: {device}")
print("Building BA graph and sparse adjacency matrix...")
adj_matrix_base: Tensor = build_sparse_adj_matrix()           # shape (N, N), sparse float32

print("Computing static hub mask (top 10% degree nodes)...")
hub_mask: Tensor = compute_static_hub_mask(adj_matrix_base)   # shape (N,), bool

# ---- Pre-simulation spectral calibration --------------------------------
spectral_radius, lambda_c, is_unstable = calculate_epidemic_threshold(adj_matrix_base, SPREAD_CHANCE)

degrees: Tensor = torch.sparse.sum(adj_matrix_base, dim=1).to_dense()  # shape (N,), float32
edge_index: Tensor = edge_index_from_sparse_adj(adj_matrix_base)       # shape (2, E), long, CPU
print(f"  Nodes: {N:,}    Edges (directed): {edge_index.shape[1]:,}")

# =========================================================================
# Step 2 — Initialize state + patch queue
# =========================================================================
state: Tensor = initialize_node_states()                              # shape (N,), int8
patch_queue: Tensor = torch.zeros(N, dtype=torch.bool, device=device) # shape (N,), bool

initial_infected_idx: Tensor = torch.randperm(N, device=device)[:INITIAL_INFECTED]
state[initial_infected_idx] = STATE_I
print(f"  Initial infected nodes: {INITIAL_INFECTED}")

# =========================================================================
# Step 3 — Run simulation and stream snapshots to Parquet
# =========================================================================
print(f"\nRunning {NUM_TICKS}-tick simulation ({PATCHING_STRATEGY} patching, "
      f"queue_prob={PATCH_COMPLETION_PROB}, rewire={REWIRE_RATE})...")
exporter = SimulationExporter(
    output_dir=OUTPUT_DIR,
    degrees=degrees,
    hub_mask=hub_mask,
    filename="simulation_snapshots.parquet",
    buffer_ticks=25,
)

peak_infected: int = 0

for tick in range(NUM_TICKS):
    state, patch_queue, _ = simulation_step(
        state=state,
        adj_matrix=adj_matrix_base,
        spread_chance=SPREAD_CHANCE,
        patching_rate=PATCHING_RATE,
        patching_strategy=PATCHING_STRATEGY,
        hub_mask=hub_mask,
        patch_queue=patch_queue,
        volatility_rate=VOLATILITY_RATE,
        patch_completion_prob=PATCH_COMPLETION_PROB,
        rewire_rate=REWIRE_RATE,
    )
    exporter.record_tick(tick=tick, state=state, patch_queue=patch_queue)

    current_infected: int = int(torch.sum(state == STATE_I).item())
    peak_infected = max(peak_infected, current_infected)

    if tick % 10 == 0 or tick == NUM_TICKS - 1:
        n_susceptible: int = int(torch.sum(state == 0).item())
        n_exposed:     int = int(torch.sum(state == 1).item())
        n_infected:    int = current_infected
        n_patched:     int = int(torch.sum(state == 3).item())
        n_queued:      int = int(torch.sum(patch_queue).item())
        print(
            f"  tick {tick:>3d} | S={n_susceptible:>6,}  E={n_exposed:>5,}  "
            f"I={n_infected:>5,}  P={n_patched:>6,}  Q={n_queued:>5,}"
        )

parquet_path: pathlib.Path = exporter.flush()
print(f"\nParquet snapshot written: {parquet_path}")
print(f"Peak infected across all ticks: {peak_infected:,} / {N:,} nodes")

# =========================================================================
# Step 4 — Load back as PyG InMemoryDataset
# =========================================================================
print("\nBuilding PyG InMemoryDataset from Parquet snapshots...")
dataset = SEIRGraphDataset(
    root=str(OUTPUT_DIR),
    parquet_path=parquet_path,
    edge_index=edge_index,
    label_mode="tick",
)

print(f"\nDataset summary:")
print(f"  Total Data objects (one per tick): {len(dataset)}")
print(f"  First object:  {dataset[0]}")
print(f"  Last  object:  {dataset[-1]}")
print()

first: object = dataset[0]
assert first.x.shape == (N, 4),            f"x shape wrong: {first.x.shape}"
assert first.x.dtype == torch.float32,     f"x dtype wrong: {first.x.dtype}"
assert first.edge_index.shape[0] == 2,     f"edge_index row dim wrong: {first.edge_index.shape}"
assert first.edge_index.dtype == torch.long, f"edge_index dtype wrong: {first.edge_index.dtype}"
assert first.edge_index.shape[1] == edge_index.shape[1], (
    f"edge count mismatch: dataset has {first.edge_index.shape[1]}, "
    f"expected {edge_index.shape[1]}"
)

print("[PASS] x.shape     == (N, 4)      :", first.x.shape)
print("[PASS] x.dtype     == float32     :", first.x.dtype)
print("[PASS] edge_index  == (2, E)      :", first.edge_index.shape)
print("[PASS] edge_index dtype == long   :", first.edge_index.dtype)
print("[PASS] node count consistent across graph and feature matrix")
print("\nPipeline complete.")
