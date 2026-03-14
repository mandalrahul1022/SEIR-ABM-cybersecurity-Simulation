#!/usr/bin/env python3
"""
predictive_model.py — Phase 5: Predictive Intelligence

Graph Convolutional Network that predicts per-node infection status at a
future tick (T + LOOKAHEAD) given the graph snapshot at tick T.

Architecture:
    Input  → GCNConv(4, 32) → ReLU → Dropout(0.4)
           → GCNConv(32, 16) → ReLU → Dropout(0.4)
           → Linear(16, 1) → Sigmoid
    Loss   : BCEWithLogitsLoss (numerically stable)
    Metric : Accuracy, AUC-ROC, Hub Recall

Reads the Parquet snapshot produced by run_pipeline.py and constructs
(tick_t, tick_{t+k}) training pairs for node-level binary classification.

Generates: gnn_performance.png  (Loss/Accuracy + ROC curve side-by-side)

Run:  python3 predictive_model.py
"""

from __future__ import annotations

import pathlib
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv

from tensor_engine import N, build_sparse_adj_matrix, device as engine_device

# ── Configuration ─────────────────────────────────────────────────────────
PARQUET_PATH: str  = "outputs/run_001/simulation_snapshots.parquet"
LOOKAHEAD: int     = 5       # predict infection state k ticks ahead
HIDDEN_DIM: int    = 32
DROPOUT: float     = 0.4
LR: float          = 0.005
EPOCHS: int        = 60
BATCH_SIZE: int    = 8
TRAIN_FRAC: float  = 0.80
SEED: int          = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

STATE_INFECTED: int = 2


# ── Model ─────────────────────────────────────────────────────────────────

class InfectionPredictor(nn.Module):
    """Two-layer GCN for per-node binary infection prediction."""

    def __init__(self, in_channels: int, hidden: int, dropout: float):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, hidden // 2)
        self.head = nn.Linear(hidden // 2, 1)
        self.dropout = dropout

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Returns raw logits of shape (N,)."""
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.head(h).squeeze(-1)


# ── Data preparation ──────────────────────────────────────────────────────

def load_tick_snapshots(parquet_path: str) -> dict[int, np.ndarray]:
    """
    Read Parquet and return {tick: features_array} where features_array
    has shape (N, 4) — columns [state, degree, is_hub, is_in_queue].
    """
    table = pq.read_table(parquet_path)
    tick_col   = np.array(table["tick"].to_pylist(), dtype=np.int32)
    node_col   = np.array(table["node_id"].to_pylist(), dtype=np.int32)
    state_col  = np.array(table["state"].to_pylist(), dtype=np.float32)
    degree_col = np.array(table["degree"].to_pylist(), dtype=np.float32)
    hub_col    = np.array(table["is_hub"].to_pylist(), dtype=np.float32)
    queue_col  = np.array(table["is_in_queue"].to_pylist(), dtype=np.float32)

    snapshots: dict[int, np.ndarray] = {}
    for tick in sorted(set(tick_col)):
        mask = tick_col == tick
        order = np.argsort(node_col[mask])
        features = np.stack([
            state_col[mask][order],
            degree_col[mask][order],
            hub_col[mask][order],
            queue_col[mask][order],
        ], axis=1)
        snapshots[tick] = features
    return snapshots


def build_paired_dataset(
    snapshots: dict[int, np.ndarray],
    edge_index: Tensor,
    lookahead: int,
) -> list[Data]:
    """
    For each tick T where T+lookahead exists, create a PyG Data:
      x     = node features at tick T           shape (N, 4)
      y     = 1 if node is infected at T+k      shape (N,)
      edge_index = graph topology                shape (2, E)
    """
    ticks = sorted(snapshots.keys())
    data_list: list[Data] = []

    for t in ticks:
        t_future = t + lookahead
        if t_future not in snapshots:
            continue

        x = torch.from_numpy(snapshots[t]).float()
        future_state = snapshots[t_future][:, 0]
        y = torch.from_numpy((future_state == STATE_INFECTED).astype(np.float32))

        data_list.append(Data(x=x, edge_index=edge_index, y=y, num_nodes=x.shape[0]))

    return data_list


# ── Training & Evaluation ────────────────────────────────────────────────

def train_epoch(
    model: InfectionPredictor,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        optimizer.zero_grad()
        logits = model(batch.x, batch.edge_index)
        loss = criterion(logits, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: InfectionPredictor,
    loader: DataLoader,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (accuracy%, auc%, all_probs, all_preds, all_labels)."""
    model.eval()
    all_probs, all_labels = [], []

    for batch in loader:
        logits = model(batch.x, batch.edge_index)
        probs = torch.sigmoid(logits).cpu().numpy()
        labels = batch.y.cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels)

    probs_cat = np.concatenate(all_probs)
    labels_cat = np.concatenate(all_labels)
    preds_cat = (probs_cat >= 0.5).astype(np.int32)

    acc = accuracy_score(labels_cat, preds_cat) * 100.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            auc = roc_auc_score(labels_cat, probs_cat) * 100.0
        except ValueError:
            auc = 50.0

    return acc, auc, probs_cat, preds_cat, labels_cat


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  Phase 5: GNN Predictive Intelligence")
    print("=" * 60)

    # Build edge_index from the engine's adjacency matrix
    print("Building adjacency matrix for edge_index...")
    adj = build_sparse_adj_matrix()
    edge_index: Tensor = adj.coalesce().indices().cpu()
    print(f"  Nodes: {N:,}    Edges: {edge_index.shape[1]:,}")

    # Load Parquet snapshots
    print(f"Loading snapshots from {PARQUET_PATH}...")
    snapshots = load_tick_snapshots(PARQUET_PATH)
    print(f"  Ticks loaded: {len(snapshots)}")

    # Build paired training data
    data_list = build_paired_dataset(snapshots, edge_index, LOOKAHEAD)
    print(f"  Training pairs (T, T+{LOOKAHEAD}): {len(data_list)}")

    # Compute class balance
    total_pos = sum(d.y.sum().item() for d in data_list)
    total_nodes = sum(d.y.shape[0] for d in data_list)
    pos_frac = total_pos / total_nodes
    print(f"  Class balance: {pos_frac*100:.1f}% infected, {(1-pos_frac)*100:.1f}% not infected")

    # Class weight for imbalanced data
    pos_weight = torch.tensor([(1.0 - pos_frac) / max(pos_frac, 1e-6)])

    # Train/test split
    np.random.shuffle(data_list)
    split = int(TRAIN_FRAC * len(data_list))
    train_data = data_list[:split]
    test_data = data_list[split:]
    print(f"  Train: {len(train_data)} graphs    Test: {len(test_data)} graphs")

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    # Model
    model = InfectionPredictor(in_channels=4, hidden=HIDDEN_DIM, dropout=DROPOUT)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print(f"\n  Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"  Optimizer: Adam (lr={LR})")
    print(f"  Loss: BCEWithLogitsLoss (pos_weight={pos_weight.item():.2f})")
    print()

    # Training loop
    history: dict[str, list[float]] = {
        "train_loss": [], "train_acc": [], "test_acc": [], "test_auc": [],
    }

    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion)
        train_acc, _, _, _, _ = evaluate(model, train_loader)
        test_acc, test_auc, _, _, _ = evaluate(model, test_loader)

        history["train_loss"].append(loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        history["test_auc"].append(test_auc)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3d}/{EPOCHS}  loss={loss:.4f}  "
                  f"train_acc={train_acc:.1f}%  test_acc={test_acc:.1f}%  "
                  f"test_auc={test_auc:.1f}%")

    # Final evaluation
    final_acc, final_auc, probs, preds, labels = evaluate(model, test_loader)

    print("\n" + "=" * 60)
    print(f"  GNN Predictive Accuracy: {final_acc:.1f}%")
    print(f"  AUC-ROC: {final_auc:.1f}%")

    # Hub recall: among true infected nodes, what fraction did we catch?
    infected_mask = labels == 1
    if infected_mask.sum() > 0:
        hub_recall = (preds[infected_mask] == 1).sum() / infected_mask.sum() * 100.0
    else:
        hub_recall = 0.0
    print(f"  Model successfully identifies {hub_recall:.1f}% of future propagation hubs.")
    print("=" * 60)

    # ── Visualization ─────────────────────────────────────────────────
    epochs_arr = np.arange(1, EPOCHS + 1)
    fig, (ax_loss, ax_roc) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Loss + Accuracy
    color_loss = "#2196F3"
    color_acc = "#F44336"
    ax_loss.plot(epochs_arr, history["train_loss"], color=color_loss, label="Train Loss")
    ax_loss.set_xlabel("Epoch", fontsize=11)
    ax_loss.set_ylabel("Loss", color=color_loss, fontsize=11)
    ax_loss.tick_params(axis="y", labelcolor=color_loss)

    ax_acc = ax_loss.twinx()
    ax_acc.plot(epochs_arr, history["test_acc"], color=color_acc, label="Test Accuracy")
    ax_acc.set_ylabel("Test Accuracy (%)", color=color_acc, fontsize=11)
    ax_acc.tick_params(axis="y", labelcolor=color_acc)
    ax_loss.set_title("Training Loss & Test Accuracy", fontsize=13)

    lines1, labels1 = ax_loss.get_legend_handles_labels()
    lines2, labels2 = ax_acc.get_legend_handles_labels()
    ax_loss.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    # Right: ROC Curve
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = final_auc / 100.0

    ax_roc.plot(fpr, tpr, color="#FF9800", lw=2, label=f"GCN (AUC = {roc_auc:.3f})")
    ax_roc.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.500)")
    ax_roc.set_xlabel("False Positive Rate", fontsize=11)
    ax_roc.set_ylabel("True Positive Rate", fontsize=11)
    ax_roc.set_title("ROC Curve — Node-Level Infection Prediction", fontsize=13)
    ax_roc.legend(fontsize=10)
    ax_roc.set_xlim(0, 1)
    ax_roc.set_ylim(0, 1.02)

    fig.tight_layout()
    fig.savefig("gnn_performance.png", dpi=150)
    plt.close("all")
    print(f"\n  Saved: gnn_performance.png")
    print("Phase 5 complete.")


if __name__ == "__main__":
    main()
