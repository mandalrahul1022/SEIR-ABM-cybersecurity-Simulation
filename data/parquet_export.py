#!/usr/bin/env python3
"""
data/parquet_export.py

Exports per-tick simulation snapshots from TensorCyberSimulation to Apache Parquet.

Schema per row: tick | node_id | state | degree | is_hub | is_in_queue

All tensor inputs originate from tensor_engine.py functions. This module never
modifies tensor_engine.py and introduces no new simulation variables.

Usage
-----
Standalone (CPU-only smoke test with small N):
    python3 data/parquet_export.py

Integrated with tensor_engine.py:
    from data.parquet_export import SimulationExporter
    exporter = SimulationExporter(output_dir="outputs/run_001", degrees=degrees, hub_mask=hub_mask)
    for tick in range(num_ticks):
        state = simulation_step(...)
        exporter.record_tick(tick=tick, state=state)
    exporter.flush()
"""

from __future__ import annotations

import pathlib
from typing import Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Schema — tick, node_id, state, degree, is_hub, is_in_queue
# ---------------------------------------------------------------------------
_PARQUET_SCHEMA: pa.Schema = pa.schema([
    pa.field("tick",        pa.int32(),  nullable=False),
    pa.field("node_id",     pa.int32(),  nullable=False),
    pa.field("state",       pa.int8(),   nullable=False),
    pa.field("degree",      pa.int32(),  nullable=False),
    pa.field("is_hub",      pa.bool_(),  nullable=False),
    pa.field("is_in_queue", pa.bool_(),  nullable=False),
])


class SimulationExporter:
    """
    Buffers per-tick node state snapshots and flushes them to Parquet on disk.

    Parameters
    ----------
    output_dir : str | pathlib.Path
        Directory where the Parquet file will be written.
        Created automatically if it does not exist.
    degrees : Tensor
        1D tensor of shape (N,), dtype float32 or int32.
        Raw node degrees from compute_static_hub_mask() in tensor_engine.py.
        Must remain on CPU or be moved to CPU before passing.
    hub_mask : Tensor
        1D boolean tensor of shape (N,).
        Static hub mask from compute_static_hub_mask() in tensor_engine.py.
    filename : str
        Output Parquet filename (default: "simulation_snapshots.parquet").
    buffer_ticks : int
        Number of ticks to buffer in RAM before writing to disk.
        Lower values reduce peak RAM at the cost of more I/O.
    """

    def __init__(
        self,
        output_dir: str | pathlib.Path,
        degrees: Tensor,          # shape (N,), float32 or int32
        hub_mask: Tensor,         # shape (N,), bool
        filename: str = "simulation_snapshots.parquet",
        buffer_ticks: int = 50,
    ) -> None:
        self._output_dir: pathlib.Path = pathlib.Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._output_path: pathlib.Path = self._output_dir / filename

        # Move static tensors to CPU numpy once; reused every tick
        self._degrees_np: np.ndarray = degrees.cpu().to(torch.int32).numpy()   # shape (N,), int32
        self._hub_mask_np: np.ndarray = hub_mask.cpu().numpy()                 # shape (N,), bool
        self._N: int = self._degrees_np.shape[0]
        self._node_ids_np: np.ndarray = np.arange(self._N, dtype=np.int32)    # shape (N,), int32

        self._buffer_ticks: int = buffer_ticks
        self._tick_buffer: list[dict] = []
        self._writer: Optional[pq.ParquetWriter] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_tick(self, tick: int, state: Tensor, patch_queue: Tensor) -> None:
        """
        Buffer one tick's worth of node states.

        Parameters
        ----------
        tick : int
            Current simulation step index (0-based).
        state : Tensor
            1D tensor of shape (N,), dtype torch.int8.
        patch_queue : Tensor
            1D boolean tensor of shape (N,). True if node is in patching queue.
        """
        state_np: np.ndarray = state.cpu().numpy().astype(np.int8)
        queue_np: np.ndarray = patch_queue.cpu().numpy()
        tick_arr: np.ndarray = np.full(self._N, tick, dtype=np.int32)

        self._tick_buffer.append({
            "tick":        tick_arr,
            "node_id":     self._node_ids_np,
            "state":       state_np,
            "degree":      self._degrees_np,
            "is_hub":      self._hub_mask_np,
            "is_in_queue": queue_np,
        })

        if len(self._tick_buffer) >= self._buffer_ticks:
            self._flush_buffer()

    def flush(self) -> pathlib.Path:
        """
        Write any remaining buffered ticks to Parquet and close the writer.

        Returns
        -------
        pathlib.Path
            Absolute path to the written Parquet file.
        """
        if self._tick_buffer:
            self._flush_buffer()
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        return self._output_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _flush_buffer(self) -> None:
        """Concatenate buffered tick arrays and write as a single row group."""
        tick_col      = np.concatenate([b["tick"]        for b in self._tick_buffer])
        node_col      = np.concatenate([b["node_id"]     for b in self._tick_buffer])
        state_col     = np.concatenate([b["state"]       for b in self._tick_buffer])
        degree_col    = np.concatenate([b["degree"]      for b in self._tick_buffer])
        is_hub_col    = np.concatenate([b["is_hub"]      for b in self._tick_buffer])
        is_queue_col  = np.concatenate([b["is_in_queue"] for b in self._tick_buffer])

        table: pa.Table = pa.table(
            {
                "tick":        pa.array(tick_col,     type=pa.int32()),
                "node_id":     pa.array(node_col,     type=pa.int32()),
                "state":       pa.array(state_col,    type=pa.int8()),
                "degree":      pa.array(degree_col,   type=pa.int32()),
                "is_hub":      pa.array(is_hub_col,   type=pa.bool_()),
                "is_in_queue": pa.array(is_queue_col, type=pa.bool_()),
            },
            schema=_PARQUET_SCHEMA,
        )

        if self._writer is None:
            self._writer = pq.ParquetWriter(
                str(self._output_path),
                schema=_PARQUET_SCHEMA,
                compression="zstd",
            )
        self._writer.write_table(table)
        self._tick_buffer.clear()


# ---------------------------------------------------------------------------
# Convenience function for single-call exports (non-buffered)
# ---------------------------------------------------------------------------

def export_run_to_parquet(
    tick_states: list[Tensor],        # list of (N,) int8 tensors, one per tick
    tick_queues: list[Tensor],        # list of (N,) bool tensors, one per tick
    degrees: Tensor,                  # shape (N,), float32
    hub_mask: Tensor,                 # shape (N,), bool
    output_path: str | pathlib.Path,
) -> pathlib.Path:
    """
    Write a complete simulation run to a single Parquet file in one call.

    Parameters
    ----------
    tick_states : list[Tensor]
        Ordered list of state tensors, one per simulation tick.
    tick_queues : list[Tensor]
        Ordered list of patch_queue tensors, one per simulation tick.
    degrees : Tensor
        Raw node degrees. shape (N,), float32.
    hub_mask : Tensor
        Static hub boolean mask. shape (N,), bool.
    output_path : str | pathlib.Path
        Full file path for the output Parquet file.

    Returns
    -------
    pathlib.Path
        Absolute path to the written Parquet file.
    """
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    N: int = degrees.shape[0]
    degrees_np: np.ndarray  = degrees.cpu().to(torch.int32).numpy()
    hub_mask_np: np.ndarray = hub_mask.cpu().numpy()
    node_ids_np: np.ndarray = np.arange(N, dtype=np.int32)

    all_ticks:     list[np.ndarray] = []
    all_node_ids:  list[np.ndarray] = []
    all_states:    list[np.ndarray] = []
    all_degrees:   list[np.ndarray] = []
    all_is_hubs:   list[np.ndarray] = []
    all_is_queues: list[np.ndarray] = []

    for tick, (state_tensor, queue_tensor) in enumerate(zip(tick_states, tick_queues)):
        all_ticks.append(np.full(N, tick, dtype=np.int32))
        all_node_ids.append(node_ids_np)
        all_states.append(state_tensor.cpu().numpy().astype(np.int8))
        all_degrees.append(degrees_np)
        all_is_hubs.append(hub_mask_np)
        all_is_queues.append(queue_tensor.cpu().numpy())

    table: pa.Table = pa.table(
        {
            "tick":        pa.array(np.concatenate(all_ticks),     type=pa.int32()),
            "node_id":     pa.array(np.concatenate(all_node_ids),  type=pa.int32()),
            "state":       pa.array(np.concatenate(all_states),    type=pa.int8()),
            "degree":      pa.array(np.concatenate(all_degrees),   type=pa.int32()),
            "is_hub":      pa.array(np.concatenate(all_is_hubs),   type=pa.bool_()),
            "is_in_queue": pa.array(np.concatenate(all_is_queues), type=pa.bool_()),
        },
        schema=_PARQUET_SCHEMA,
    )
    pq.write_table(table, str(output_path), compression="zstd")
    return output_path


# ---------------------------------------------------------------------------
# Smoke test — runs without CUDA using tiny N to verify round-trip integrity
# ---------------------------------------------------------------------------

def _smoke_test() -> None:
    """CPU-only round-trip integrity check. Uses N=100 and 5 ticks."""
    print("Running parquet_export smoke test (CPU, N=100, 5 ticks)...")

    _N_SMOKE: int = 100
    _TICKS:   int = 5

    degrees_cpu:  Tensor = torch.randint(low=1, high=20, size=(_N_SMOKE,), dtype=torch.float32)
    hub_mask_cpu: Tensor = degrees_cpu >= torch.quantile(degrees_cpu, 0.90)

    tick_states: list[Tensor] = [
        torch.randint(low=0, high=4, size=(_N_SMOKE,), dtype=torch.int8)
        for _ in range(_TICKS)
    ]
    tick_queues: list[Tensor] = [
        torch.randint(low=0, high=2, size=(_N_SMOKE,), dtype=torch.bool)
        for _ in range(_TICKS)
    ]

    out_path: pathlib.Path = pathlib.Path("/tmp/seir_smoke_test.parquet")
    export_run_to_parquet(tick_states, tick_queues, degrees_cpu, hub_mask_cpu, out_path)

    table_back: pa.Table = pq.read_table(str(out_path))
    assert table_back.schema.equals(_PARQUET_SCHEMA), "Schema mismatch on read-back"
    assert table_back.num_rows == _N_SMOKE * _TICKS, (
        f"Row count mismatch: expected {_N_SMOKE * _TICKS}, got {table_back.num_rows}"
    )
    assert "is_in_queue" in table_back.schema.names, "is_in_queue column missing"

    ticks_read = table_back["tick"].to_pylist()
    assert set(ticks_read) == set(range(_TICKS)), "Tick values missing on read-back"

    states_written = np.concatenate([s.numpy() for s in tick_states])
    states_read    = np.array(table_back["state"].to_pylist(), dtype=np.int8)
    assert np.array_equal(states_written, states_read), "State values corrupted on round-trip"

    print(f"  Rows written:   {table_back.num_rows}")
    print(f"  Schema:         {table_back.schema}")
    print(f"  Columns:        {table_back.schema.names}")
    print(f"  State round-trip: PASS")
    print("parquet_export smoke test: PASSED")


if __name__ == "__main__":
    _smoke_test()
