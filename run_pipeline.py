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

# =========================================================================
# Step 1 — Simulation Generator Encapsulation
# =========================================================================

def execute_simulation(seed: int, output_dir: pathlib.Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    print(f"[{output_dir.name}] Building BA graph with seed {seed}...")
    adj_matrix_base: Tensor = build_sparse_adj_matrix()           
    hub_mask: Tensor = compute_static_hub_mask(adj_matrix_base)   

    degrees: Tensor = torch.sparse.sum(adj_matrix_base, dim=1).to_dense()  
    edge_index: Tensor = edge_index_from_sparse_adj(adj_matrix_base)       
    
    # Save the physical topology for the Inductive generalizer
    torch.save(edge_index, output_dir / "edge_index.pt")
    print(f"  Nodes: {N:,}    Edges (directed): {edge_index.shape[1]:,}")

    state: Tensor = initialize_node_states()                              
    patch_queue: Tensor = torch.zeros(N, dtype=torch.bool, device=device) 

    initial_infected_idx: Tensor = torch.randperm(N, device=device)[:INITIAL_INFECTED]
    state[initial_infected_idx] = STATE_I

    exporter = SimulationExporter(
        output_dir=output_dir,
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

    parquet_path: pathlib.Path = exporter.flush()
    print(f"  Peak infected: {peak_infected:,} / {N:,} nodes\n")


# =========================================================================
# Step 2 — Bootstrapper (Multi-Graph)
# =========================================================================

if __name__ == "__main__":
    print(f"Device: {device}")
    
    base_dir = pathlib.Path("outputs/graphs")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate 5 distinct training topologies
    print("\n--- Generating Training Graphs ---")
    for i in range(5):
        run_seed = SEED + i
        run_dir = base_dir / f"train_{i}"
        execute_simulation(run_seed, run_dir)
        
    # Generate 2 distinct testing topologies 
    print("--- Generating Testing Graphs ---")
    for i in range(2):
        run_seed = SEED + 10 + i
        run_dir = base_dir / f"test_{i}"
        execute_simulation(run_seed, run_dir)
        
    print("Multi-Graph Inductive pipeline complete.")