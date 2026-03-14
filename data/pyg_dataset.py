#!/usr/bin/env python3
"""
data/pyg_dataset.py

torch_geometric.data.InMemoryDataset subclass that reads simulation Parquet
snapshots written by data/parquet_export.py and exposes them as PyG Data objects
ready for GNN training.

One PyG Data object = one simulation tick snapshot.

Node feature matrix x  : shape (N, 4) — columns: [state, degree, is_hub, is_in_queue]
Edge index             : shape (2, E) — directed COO from adj_matrix in tensor_engine.py
Label y                : shape (1,)   — tick index (configurable via label_mode)

This file reads existing Parquet files. It does NOT modify tensor_engine.py or
any other existing file.

Usage
-----
    from data.pyg_dataset import SEIRGraphDataset

    dataset = SEIRGraphDataset(
        root="outputs/run_001",
        parquet_path="outputs/run_001/simulation_snapshots.parquet",
        edge_index=edge_index,  # from build_sparse_adj_matrix() in tensor_engine.py
    )
    print(dataset[0])   # Data(x=[10000, 4], edge_index=[2, E], y=[1])
"""

from __future__ import annotations

import pathlib
from typing import Callable, Optional

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import Tensor

# Conditional PyG import — allows structural/unit tests without torch_geometric installed
try:
    from torch_geometric.data import Data, InMemoryDataset
    _PYG_AVAILABLE: bool = True
except ImportError:  # pragma: no cover
    _PYG_AVAILABLE = False
    # Stub classes so the module can be imported for AST/schema checks
    class InMemoryDataset:  # type: ignore[no-redef]
        pass
    class Data:  # type: ignore[no-redef]
        pass


# ---------------------------------------------------------------------------
# Label modes — controls what y contains per Data object
# ---------------------------------------------------------------------------

LABEL_TICK      = "tick"         # y = tick index (int)
LABEL_PEAK_FRAC = "peak_frac"    # y = fraction of nodes infected at this tick


class SEIRGraphDataset(InMemoryDataset):
    """
    PyG InMemoryDataset built from Parquet simulation snapshots.

    Each item in the dataset corresponds to one (tick, simulation_run) snapshot.

    Parameters
    ----------
    root : str | pathlib.Path
        Root directory used by InMemoryDataset for caching processed data.
    parquet_path : str | pathlib.Path
        Path to the Parquet file written by data/parquet_export.py.
    edge_index : Tensor
        COO edge index of shape (2, E), dtype torch.long.
        Constructed from build_sparse_adj_matrix() in tensor_engine.py via
        adj_matrix.coalesce().indices().
    label_mode : str
        One of LABEL_TICK or LABEL_PEAK_FRAC.
    transform : callable, optional
        PyG transform applied at access time.
    pre_transform : callable, optional
        PyG transform applied at processing time.
    """

    def __init__(
        self,
        root: str | pathlib.Path,
        parquet_path: str | pathlib.Path,
        edge_index: Tensor,          # shape (2, E), long
        label_mode: str = LABEL_TICK,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ) -> None:
        if not _PYG_AVAILABLE:
            raise ImportError(
                "torch_geometric is required for SEIRGraphDataset. "
                "Install it with: pip install torch_geometric"
            )
        self._parquet_path: pathlib.Path = pathlib.Path(parquet_path)
        self._edge_index: Tensor = edge_index.cpu()        # shape (2, E), long
        self._label_mode: str = label_mode

        super().__init__(str(root), transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    # ------------------------------------------------------------------
    # InMemoryDataset interface
    # ------------------------------------------------------------------

    @property
    def raw_file_names(self) -> list[str]:
        return [self._parquet_path.name]

    @property
    def processed_file_names(self) -> list[str]:
        return ["seir_dataset.pt"]

    def download(self) -> None:
        # The Parquet file is written by parquet_export.py; no remote download needed.
        pass

    def process(self) -> None:
        """
        Read Parquet snapshots and convert each tick into a PyG Data object.

        Parquet schema (written by parquet_export.py):
            tick        int32
            node_id     int32
            state       int8
            degree      int32
            is_hub      bool
            is_in_queue bool
        """
        table = pq.read_table(str(self._parquet_path))
        ticks_all: np.ndarray = table["tick"].to_pylist()
        unique_ticks: list[int] = sorted(set(ticks_all))

        tick_col:      np.ndarray = np.array(ticks_all, dtype=np.int32)
        node_col:      np.ndarray = np.array(table["node_id"].to_pylist(),     dtype=np.int32)
        state_col:     np.ndarray = np.array(table["state"].to_pylist(),       dtype=np.int8)
        degree_col:    np.ndarray = np.array(table["degree"].to_pylist(),      dtype=np.int32)
        is_hub_col:    np.ndarray = np.array(table["is_hub"].to_pylist(),      dtype=np.float32)
        is_queue_col:  np.ndarray = np.array(table["is_in_queue"].to_pylist(), dtype=np.float32)

        N: int = int(node_col.max()) + 1

        data_list: list[Data] = []
        for tick in unique_ticks:
            mask: np.ndarray = tick_col == tick
            order: np.ndarray = np.argsort(node_col[mask])

            state_tick:    np.ndarray = state_col[mask][order].astype(np.float32)
            degree_tick:   np.ndarray = degree_col[mask][order].astype(np.float32)
            is_hub_tick:   np.ndarray = is_hub_col[mask][order]
            is_queue_tick: np.ndarray = is_queue_col[mask][order]

            # Node feature matrix: [state, degree, is_hub, is_in_queue], shape (N, 4)
            x: Tensor = torch.from_numpy(
                np.stack([state_tick, degree_tick, is_hub_tick, is_queue_tick], axis=1)
            ).float()  # shape (N, 4), float32

            # Label
            if self._label_mode == LABEL_PEAK_FRAC:
                infected_frac: float = float((state_tick == 2).sum()) / N
                y: Tensor = torch.tensor([infected_frac], dtype=torch.float32)
            else:
                y = torch.tensor([tick], dtype=torch.long)

            data_obj: Data = Data(
                x=x,
                edge_index=self._edge_index,
                y=y,
                num_nodes=N,
            )

            if self.pre_transform is not None:
                data_obj = self.pre_transform(data_obj)

            data_list.append(data_obj)

        data_out, slices_out = self.collate(data_list)
        torch.save((data_out, slices_out), self.processed_paths[0])


# ---------------------------------------------------------------------------
# Helper: extract edge_index from tensor_engine's sparse adj_matrix
# ---------------------------------------------------------------------------

def edge_index_from_sparse_adj(adj_matrix: Tensor) -> Tensor:
    """
    Extract a COO edge index from the sparse adjacency tensor built by
    build_sparse_adj_matrix() in tensor_engine.py.

    Parameters
    ----------
    adj_matrix : Tensor
        Coalesced sparse COO tensor of shape (N, N), float32, on CUDA or CPU.

    Returns
    -------
    Tensor
        Edge index of shape (2, E), dtype torch.long, on CPU.
        Ready to be passed directly to SEIRGraphDataset(edge_index=...).
    """
    coalesced: Tensor = adj_matrix.coalesce()
    return coalesced.indices().cpu()  # shape (2, E), long


# ---------------------------------------------------------------------------
# Smoke test — CPU-only, does not require torch_geometric
# ---------------------------------------------------------------------------

def _smoke_test_schema() -> None:
    """
    Verify that the Parquet reader correctly maps column dtypes to tensors.
    Does not require torch_geometric or CUDA.
    """
    print("Running pyg_dataset schema smoke test (CPU, no PyG required)...")

    import tempfile
    import pyarrow as pa
    import pyarrow.parquet as pq

    _N: int = 50
    _TICKS: int = 3

    # Build a tiny synthetic Parquet matching the schema from parquet_export.py
    node_ids:  np.ndarray = np.arange(_N, dtype=np.int32)
    degrees:   np.ndarray = np.random.randint(1, 15, size=_N, dtype=np.int32)
    is_hubs:   np.ndarray = degrees >= np.percentile(degrees, 90)

    is_queued: np.ndarray = np.random.randint(0, 2, size=_N, dtype=bool)

    rows: dict = {
        "tick":        np.concatenate([np.full(_N, t, dtype=np.int32) for t in range(_TICKS)]),
        "node_id":     np.tile(node_ids, _TICKS),
        "state":       np.random.randint(0, 4, size=_N * _TICKS, dtype=np.int8),
        "degree":      np.tile(degrees, _TICKS),
        "is_hub":      np.tile(is_hubs, _TICKS),
        "is_in_queue": np.tile(is_queued, _TICKS),
    }

    schema = pa.schema([
        pa.field("tick",        pa.int32()),
        pa.field("node_id",     pa.int32()),
        pa.field("state",       pa.int8()),
        pa.field("degree",      pa.int32()),
        pa.field("is_hub",      pa.bool_()),
        pa.field("is_in_queue", pa.bool_()),
    ])
    table = pa.table(
        {k: pa.array(v) for k, v in rows.items()},
        schema=schema,
    )

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        tmp_path = f.name

    pq.write_table(table, tmp_path)

    # Read back and simulate the process() column extraction logic
    t2 = pq.read_table(tmp_path)
    for tick in range(_TICKS):
        mask = np.array(t2["tick"].to_pylist()) == tick
        state_t    = np.array(t2["state"].to_pylist(),       dtype=np.float32)[mask]
        degree_t   = np.array(t2["degree"].to_pylist(),      dtype=np.float32)[mask]
        is_hub_t   = np.array(t2["is_hub"].to_pylist(),      dtype=np.float32)[mask]
        is_queue_t = np.array(t2["is_in_queue"].to_pylist(), dtype=np.float32)[mask]
        x = torch.from_numpy(
            np.stack([state_t, degree_t, is_hub_t, is_queue_t], axis=1)
        )
        assert x.shape == (_N, 4), f"Feature matrix shape mismatch at tick {tick}: {x.shape}"
        assert x.dtype == torch.float32, f"Feature dtype mismatch: {x.dtype}"

    print(f"  Ticks verified:        {_TICKS}")
    print(f"  Feature matrix shape:  ({_N}, 4) per tick — PASS")
    print(f"  Feature dtype:         float32 — PASS")
    print("pyg_dataset schema smoke test: PASSED")


if __name__ == "__main__":
    _smoke_test_schema()
