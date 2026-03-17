#!/usr/bin/env python3
"""
predictive_model.py — Phase 5: Predictive Intelligence

Temporal Graph Convolutional Network (T-GCN) that predicts per-node infection
status at a future tick (T + LOOKAHEAD) given a sliding window of graph
snapshots [T-W+1 .. T].

Architecture:
    Input  → GCNConv(5, 16) → ReLU  (shared across all window ticks)
           → GRU(16, 16)            (accumulates temporal hidden state)
           → Linear(16, 1)          (per-node infection logit)
    Loss   : BCEWithLogitsLoss (numerically stable, class-weighted)
    Metric : Accuracy, AUC-ROC, Hub Recall

Experimental Design:
    Run A — Full Visibility:
        Train on clean data (obs_flag = 1.0 everywhere).
        Evaluate on clean data. Establishes the performance ceiling.

    Run B — Partial Observability (85% masked at eval only):
        Train on the SAME clean data as Run A.
        Evaluate with 85% of infection states replaced by -1 sentinel
        and obs_flag set to 0.0 for masked nodes.
        The delta (Run A AUC - Run B AUC) measures the operational cost
        of partial surveillance.

    This is the standard partial-observability protocol: train the best
    model possible, then stress-test it under degraded input conditions.

Features per node (5 columns):
    [0] infection_state   (0=S, 1=E, 2=I, 3=R; or -1 if masked)
    [1] degree            (normalized node degree)
    [2] is_hub            (1.0 if top-percentile degree)
    [3] is_in_queue       (1.0 if queued for patching)
    [4] obs_flag          (1.0 = infection_state is real, 0.0 = masked)

Reads Parquet snapshots + edge_index.pt produced by run_pipeline.py.
Generates: gnn_performance.png

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

# ── Hardware ──────────────────────────────────────────────────────────────

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Hardware Acceleration: Apple Metal Performance Shaders (MPS) ONLINE")
else:
    device = torch.device("cpu")
    print("Hardware Acceleration: CPU Fallback")

# ── Configuration ─────────────────────────────────────────────────────────

GRAPHS_DIR: str    = "outputs/graphs"
LOOKAHEAD: int     = 5       # predict infection state k ticks ahead
WINDOW_SIZE: int   = 4       # temporal sequence length for GRU
HIDDEN_DIM: int    = 16
LR: float          = 0.005
EPOCHS: int        = 60
BATCH_SIZE: int    = 8
SEED: int          = 42
BLIND_SPOT_RATE: float = 0.85  # fraction of nodes masked in Run B eval

torch.manual_seed(SEED)
np.random.seed(SEED)

STATE_INFECTED: int = 2
STATE_NULLIFIED: float = -1.0  # sentinel for unobserved infection state
NUM_RAW_FEATURES: int = 4      # [state, degree, hub, queue]
NUM_MODEL_FEATURES: int = 5    # [state, degree, hub, queue, obs_flag]


# ── Model ─────────────────────────────────────────────────────────────────

class TemporalGCN(nn.Module):
    """
    T-GCN: GCNConv (spatial) + GRU (temporal).
    
    For each tick in the window, GCNConv aggregates neighbor features.
    The GRU accumulates hidden state across ticks so by tick T the
    representation encodes propagation velocity, not just a snapshot.
    """

    def __init__(self, in_channels: int, hidden: int):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=False)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """
        x shape: (N, W, C) where W=window_size, C=in_channels
        Returns: raw logits of shape (N,)
        """
        N_nodes, W, _ = x.shape

        embeddings = []
        for w in range(W):
            x_w = x[:, w, :]                        # (N, C)
            emb_w = F.relu(self.conv1(x_w, edge_index))  # (N, hidden)
            embeddings.append(emb_w)

        seq = torch.stack(embeddings, dim=0)         # (W, N, hidden)
        _, h_n = self.gru(seq)                       # h_n: (1, N, hidden)
        hidden_state = h_n.squeeze(0)                # (N, hidden)

        return self.head(hidden_state).squeeze(-1)   # (N,)


# ── Data Masking ──────────────────────────────────────────────────────────

def apply_full_visibility_flag(data_list: list[Data]) -> list[Data]:
    """
    Append obs_flag = 1.0 for all nodes across all ticks.
    Expands x from (N, W, 4) to (N, W, 5). No masking applied.
    Used for: Run A (train + test), Run B (train only).
    """
    flagged: list[Data] = []
    for d in data_list:
        x = d.x  # (N, W, 4)
        N_nodes, W, _ = x.shape
        obs_flag = torch.ones(N_nodes, W, 1)  # all visible
        x_new = torch.cat([x, obs_flag], dim=2)  # (N, W, 5)
        flagged.append(Data(
            x=x_new,
            edge_index=d.edge_index,
            y=d.y,
            num_nodes=d.num_nodes,
        ))
    return flagged


def apply_bernoulli_mask(data_list: list[Data], blind_rate: float) -> list[Data]:
    """
    Append obs_flag AND mask infection_state in lockstep.
    
    For each graph:
      1. Sample a Bernoulli mask: each node has `blind_rate` probability
         of being hidden.
      2. For hidden nodes: set infection_state (col 0) to -1 across all
         ticks in the window, and set obs_flag (col 4) to 0.0.
      3. For visible nodes: infection_state unchanged, obs_flag = 1.0.
    
    Expands x from (N, W, 4) to (N, W, 5).
    Used for: Run B (test data only).
    """
    masked: list[Data] = []
    for d in data_list:
        x = d.x.clone()  # (N, W, 4)
        N_nodes, W, _ = x.shape

        # Per-node visibility decision, broadcast across temporal window
        visible = torch.bernoulli(
            torch.full((N_nodes,), 1.0 - blind_rate)
        ).bool()  # True = visible, False = masked

        # Mask infection_state (col 0) for hidden nodes across all ticks
        hidden_mask = ~visible  # (N,)
        hidden_expanded = hidden_mask.unsqueeze(1).expand(-1, W)  # (N, W)
        x[:, :, 0] = torch.where(
            hidden_expanded,
            torch.tensor(STATE_NULLIFIED),
            x[:, :, 0],
        )

        # Build obs_flag in lockstep: 1.0 for visible, 0.0 for masked
        obs_flag = visible.float().unsqueeze(1).unsqueeze(2).expand(
            N_nodes, W, 1
        )  # (N, W, 1)

        x_new = torch.cat([x, obs_flag], dim=2)  # (N, W, 5)
        masked.append(Data(
            x=x_new,
            edge_index=d.edge_index,
            y=d.y,
            num_nodes=d.num_nodes,
        ))
    return masked


# ── Data Loading ──────────────────────────────────────────────────────────

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
    For each valid tick T, creates a PyG Data object:
      x          = node features for ticks [T-W+1 .. T], shape (N, W, 4)
      y          = 1 if node is infected at T+lookahead,  shape (N,)
      edge_index = graph topology                         shape (2, E)
    """
    ticks = sorted(snapshots.keys())
    data_list: list[Data] = []

    for i, t in enumerate(ticks):
        if i < WINDOW_SIZE - 1:
            continue

        t_future = t + lookahead
        if t_future not in snapshots:
            continue

        window_ticks = ticks[i - WINDOW_SIZE + 1 : i + 1]

        # Ensure contiguous window (no gaps)
        if window_ticks[-1] - window_ticks[0] != WINDOW_SIZE - 1:
            continue

        seq = [snapshots[wt] for wt in window_ticks]
        x_numpy = np.stack(seq, axis=1)  # (N, W, 4)
        x = torch.from_numpy(x_numpy).float()

        future_state = snapshots[t_future][:, 0]
        y = torch.from_numpy(
            (future_state == STATE_INFECTED).astype(np.float32)
        )

        data_list.append(Data(
            x=x, edge_index=edge_index, y=y, num_nodes=x.shape[0]
        ))

    return data_list


def load_directory_graphs(target_dir_prefix: str) -> list[Data]:
    """Load all graph directories matching prefix and aggregate Data objects."""
    base_dir = pathlib.Path(GRAPHS_DIR)
    aggregated_data: list[Data] = []

    for graph_dir in sorted(base_dir.glob(f"{target_dir_prefix}_*")):
        edge_index_path = graph_dir / "edge_index.pt"
        parquet_path = graph_dir / "simulation_snapshots.parquet"

        if not edge_index_path.exists() or not parquet_path.exists():
            continue

        edge_index = torch.load(edge_index_path, weights_only=True)
        snapshots = load_tick_snapshots(str(parquet_path))
        data_list = build_paired_dataset(snapshots, edge_index, LOOKAHEAD)
        aggregated_data.extend(data_list)

    return aggregated_data


# ── Training & Evaluation ────────────────────────────────────────────────

def train_epoch(
    model: TemporalGCN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:
    """Standard training step. No masking. Model sees clean data."""
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch.x, batch.edge_index)
        loss = criterion(logits, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: TemporalGCN,
    loader: DataLoader,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate model. No masking applied here -- data is pre-processed."""
    model.eval()
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for batch in loader:
        batch = batch.to(device)
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


# ── Orchestrator ──────────────────────────────────────────────────────────

def _train_and_evaluate(
    train_data: list[Data],
    test_data: list[Data],
    pos_frac: float,
    label: str,
) -> tuple[dict, float, float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Train a fresh T-GCN and return metrics + predictions."""
    pos_weight = torch.tensor(
        [(1.0 - pos_frac) / max(pos_frac, 1e-6)]
    ).to(device)

    np.random.shuffle(train_data)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    model = TemporalGCN(
        in_channels=NUM_MODEL_FEATURES, hidden=HIDDEN_DIM
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"\n  [{label}] Train: {len(train_data)}  Test: {len(test_data)}  "
          f"Params: {param_count:,}")

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
            print(f"    Epoch {epoch:>3d}/{EPOCHS}  loss={loss:.4f}  "
                  f"acc={test_acc:.1f}%  auc={test_auc:.1f}%")

    acc, auc, probs, preds, labels = evaluate(model, test_loader)
    infected_mask = labels == 1
    recall = (
        (preds[infected_mask] == 1).sum() / infected_mask.sum() * 100.0
        if infected_mask.sum() > 0 else 0.0
    )
    return history, acc, auc, recall, probs, preds, labels


def main() -> None:
    print("=" * 60)
    print("  Phase 5: T-GCN Predictive Intelligence")
    print("  Protocol: Train full-visibility, stress-test under masking")
    print("=" * 60)

    # ── Load raw graph data (shape: N, W, 4) ─────────────────────────
    print(f"Loading isolated graphs from {GRAPHS_DIR}...")
    train_data_raw = load_directory_graphs("train")
    test_data_raw = load_directory_graphs("test")

    print(f"  Training pairs (T, T+{LOOKAHEAD}): {len(train_data_raw)}")
    print(f"  Testing pairs  (T, T+{LOOKAHEAD}): {len(test_data_raw)}")

    total_pos = sum(d.y.sum().item() for d in train_data_raw)
    total_nodes = sum(d.y.shape[0] for d in train_data_raw)
    pos_frac = total_pos / max(total_nodes, 1)
    print(f"  Class balance: {pos_frac*100:.1f}% infected, "
          f"{(1-pos_frac)*100:.1f}% not")

    # ── Prepare data for both runs ───────────────────────────────────
    # Both runs train on identical clean data with obs_flag = 1.0
    train_data_clean = apply_full_visibility_flag(train_data_raw)

    # Run A test data: full visibility (obs_flag = 1.0 everywhere)
    test_data_full = apply_full_visibility_flag(test_data_raw)

    # Run B test data: 85% of infection states masked, obs_flag = 0.0
    test_data_masked = apply_bernoulli_mask(test_data_raw, BLIND_SPOT_RATE)

    # ── Run A: Full Visibility ───────────────────────────────────────
    print("\n" + "─" * 60)
    print("  RUN A: Full Visibility (100% node states observed)")
    print("  Train: clean | Eval: clean")
    print("─" * 60)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    hist_full, acc_full, auc_full, rec_full, probs_full, _, labels_full = (
        _train_and_evaluate(
            train_data_clean, test_data_full, pos_frac, "FULL"
        )
    )

    # ── Run B: Partial Observability ─────────────────────────────────
    print("\n" + "─" * 60)
    print(f"  RUN B: Partial Observability "
          f"({BLIND_SPOT_RATE*100:.0f}% node states HIDDEN at eval)")
    print("  Train: clean | Eval: 85% masked")
    print("─" * 60)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    hist_masked, acc_masked, auc_masked, rec_masked, probs_masked, _, labels_masked = (
        _train_and_evaluate(
            train_data_clean, test_data_masked, pos_frac, "MASKED"
        )
    )

    # ── Comparison Summary ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  COMPARATIVE RESULTS")
    print("=" * 60)
    print(f"  {'Metric':<30s} {'Full':>10s} "
          f"{'85% Masked':>12s} {'Delta':>10s}")
    print(f"  {'─'*30}  {'─'*10} {'─'*12} {'─'*10}")
    print(f"  {'Test Accuracy':<30s} {acc_full:>9.1f}% "
          f"{acc_masked:>11.1f}% {acc_masked-acc_full:>+9.1f}%")
    print(f"  {'AUC-ROC':<30s} {auc_full/100:>10.3f} "
          f"{auc_masked/100:>12.3f} {(auc_masked-auc_full)/100:>+10.3f}")
    print(f"  {'Hub Recall':<30s} {rec_full:>9.1f}% "
          f"{rec_masked:>11.1f}% {rec_masked-rec_full:>+9.1f}%")
    print("=" * 60)
    print(f"\n  Protocol: Both runs trained on identical clean data.")
    print(f"  Run B masks 85% of infection states at evaluation only.")
    print(f"  Delta measures the operational cost of partial surveillance.")
    print("=" * 60)

    # ── Visualization (2x2) ──────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    epochs_arr = np.arange(1, EPOCHS + 1)

    # Top-left: Full visibility loss + accuracy
    ax = axes[0, 0]
    ax.plot(epochs_arr, hist_full["train_loss"], color="#2196F3",
            label="Loss")
    ax.set_ylabel("Loss", color="#2196F3", fontsize=10)
    ax.tick_params(axis="y", labelcolor="#2196F3")
    ax2 = ax.twinx()
    ax2.plot(epochs_arr, hist_full["test_acc"], color="#F44336",
             label="Accuracy")
    ax2.set_ylabel("Accuracy (%)", color="#F44336", fontsize=10)
    ax2.tick_params(axis="y", labelcolor="#F44336")
    ax.set_title(f"Full Visibility — Acc {acc_full:.1f}%",
                 fontsize=12, fontweight="bold")
    lines_a, labs_a = ax.get_legend_handles_labels()
    lines_b, labs_b = ax2.get_legend_handles_labels()
    ax.legend(lines_a + lines_b, labs_a + labs_b,
              loc="center right", fontsize=9)

    # Top-right: Masked loss + accuracy
    ax = axes[0, 1]
    ax.plot(epochs_arr, hist_masked["train_loss"], color="#2196F3",
            label="Loss")
    ax.set_ylabel("Loss", color="#2196F3", fontsize=10)
    ax.tick_params(axis="y", labelcolor="#2196F3")
    ax2 = ax.twinx()
    ax2.plot(epochs_arr, hist_masked["test_acc"], color="#F44336",
             label="Accuracy")
    ax2.set_ylabel("Accuracy (%)", color="#F44336", fontsize=10)
    ax2.tick_params(axis="y", labelcolor="#F44336")
    ax.set_title(f"85% Masked — Acc {acc_masked:.1f}%",
                 fontsize=12, fontweight="bold")
    lines_a, labs_a = ax.get_legend_handles_labels()
    lines_b, labs_b = ax2.get_legend_handles_labels()
    ax.legend(lines_a + lines_b, labs_a + labs_b,
              loc="center right", fontsize=9)

    # Bottom-left: Dual ROC curves
    ax = axes[1, 0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fpr_f, tpr_f, _ = roc_curve(labels_full, probs_full)
        fpr_m, tpr_m, _ = roc_curve(labels_masked, probs_masked)
    ax.plot(fpr_f, tpr_f, color="#FF9800", lw=2,
            label=f"Full (AUC = {auc_full/100:.3f})")
    ax.plot(fpr_m, tpr_m, color="#9C27B0", lw=2,
            label=f"85% Masked (AUC = {auc_masked/100:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (0.500)")
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.set_title("ROC: Full vs Partial Observability",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)

    # Bottom-right: Metric comparison bar chart
    ax = axes[1, 1]
    metrics = ["Accuracy", "AUC-ROC", "Hub Recall"]
    full_vals = [acc_full, auc_full, rec_full]
    mask_vals = [acc_masked, auc_masked, rec_masked]
    x_pos = np.arange(len(metrics))
    bar_w = 0.35
    bars1 = ax.bar(x_pos - bar_w / 2, full_vals, bar_w,
                   label="Full Visibility", color="#4CAF50", alpha=0.85)
    bars2 = ax.bar(x_pos + bar_w / 2, mask_vals, bar_w,
                   label="85% Masked", color="#9C27B0", alpha=0.85)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylabel("% Score", fontsize=10)
    ax.set_title("Performance Under Partial Observability",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 105)
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{bar.get_height():.1f}",
                ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{bar.get_height():.1f}",
                ha="center", va="bottom", fontsize=8)

    fig.suptitle(
        "T-GCN Infection Prediction: "
        "Full Visibility vs 85% Partial Observability",
        fontsize=14, fontweight="bold", y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig("gnn_performance.png", dpi=150)
    plt.close("all")
    print(f"\n  Saved: gnn_performance.png")
    print("Phase 5 complete.")


if __name__ == "__main__":
    main()