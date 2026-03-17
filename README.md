# STITCH: Scale-free Temporal InTervention & Contagion Harness

> A GPU-accelerated research platform for modelling network contagion,
> quantifying intervention efficacy, and predicting outbreak propagation
> using Temporal Graph Neural Networks on 10,000-node scale-free networks.
> Containerised for reproducibility. HPC-ready for cluster deployment.

---

## 🔬 What This Project Does

STITCH treats **malware lateral movement** and **biological epidemic spread** as the **same mathematical process**: stochastic diffusion on power-law contact graphs. It builds a complete research pipeline from first principles:

```
Spectral Theory  →  Monte Carlo Sensitivity  →  Tensor Simulation  →  Temporal GNN Prediction
    (Phase 1)            (Phase 2)                  (Phase 3)              (Phase 4)
```

The core insight: if `P(exposure) = 1 − (1 − β)^k` is domain-agnostic, then a model trained on cyber-contagion
learns transferable physics applicable to zoonotic pathogen transmission, ransomware propagation, and any
process governed by stochastic diffusion on networks with power-law degree distributions.

---

## 📊 Technical Specifications

| Dimension | Value |
|---|---|
| **Network scale** | 10⁴ nodes · 6×10⁴ directed edges |
| **Topology** | Barabási-Albert preferential attachment (m = 3) |
| **Simulation engine** | Vectorized SpMV via `torch.sparse.mm` on COO float32 (zero Python loops) |
| **Eigenvalue solver** | ARPACK Lanczos iteration via `scipy.sparse.linalg.eigsh` |
| **Sensitivity analysis** | 640 Saltelli quasi-random samples · SALib Sobol decomposition |
| **GNN architecture** | T-GCN: GCNConv(5,16) + GRU(16,16) + Linear(16,1) · **1,745 parameters** |
| **Temporal window** | 4-tick sliding sequence fed to GRU for propagation velocity encoding |
| **Prediction horizon** | T+5 ticks (per-node binary infection classification) |
| **Evaluation protocol** | **Inductive**: train on 5 BA topologies, test on 2 completely unseen graphs |
| **Partial observability** | Bernoulli masking: 85% of infection states hidden at evaluation only |
| **Hardware targets** | NVIDIA CUDA 12+ · Apple MPS (M-series) · CPU fallback (auto-detected) |
| **Container runtime** | Docker 28+ (multi-stage, `python:3.11-slim`) |
| **HPC scheduler** | Slurm job-array ready (`--dependency=afterok` chaining) |
| **Language / runtime** | Python 3.11 · PyTorch 2.x · torch_geometric 2.7 |

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        STITCH Research Platform                          │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │  tensor_engine.py  ─  Vectorized SEIR Core (N = 10,000)           │  │
│  │  • SpMV transmission: A · x_infected (sparse COO, zero loops)     │  │
│  │  • Phase III: async patch queue · edge rewiring · latency         │  │
│  │  • Spectral calibration: ρ(A) = 21.96 → λ_c = 0.0455            │  │
│  └──────────────┬────────────────────────────────────────────────────┘  │
│                 │                                                        │
│       ┌─────────┴──────────┬──────────────────────┐                     │
│       ▼                    ▼                      ▼                     │
│  run_pipeline.py    sensitivity_             predictive_                │
│  (7 BA graphs →     analysis.py              model.py                   │
│   Parquet + PyG)    640 Saltelli runs         T-GCN (GCNConv + GRU)     │
│       │                    │                  1,745 params              │
│       │                    │                  Inductive AUC = 0.873     │
│       ▼                    ▼                  85% Masked AUC = 0.852    │
│  data/parquet_export.py    sobol_indices.png                            │
│  data/pyg_dataset.py       interaction_heatmap.png                      │
│  zstd columnar snapshots                                                │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │  Reproducible Infrastructure                                       │  │
│  │  Dockerfile (multi-stage) · docker-compose.yml                     │  │
│  │  hpc/submit_pipeline · submit_sobol · submit_gnn (Slurm)          │  │
│  │  app.py (Streamlit interactive dashboard)                          │  │
│  └────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 🔎 Key Findings

### Finding 1 — The Epidemic is Mathematically Guaranteed Before Tick 0

Before the simulation runs a single step, the spectral radius of the adjacency matrix determines the outcome:

| Metric | Value | Meaning |
|---|---|---|
| ρ(A) | 21.96 | Largest eigenvalue via ARPACK Lanczos iteration |
| λ_c = 1/ρ(A) | 0.0455 | Epidemic threshold from spectral graph theory |
| β / λ_c | **8.78×** | Deep supercritical regime at β = 0.40 |

This is not a heuristic. It is a **theorem**: when β > λ_c, the epidemic is guaranteed to reach a macroscopic fraction of nodes. The simulation confirms the prediction: **9,372 / 10,000 nodes (93.7%) infected at peak**, reached at tick 14. The Susceptible population collapses to zero in under 20 ticks despite targeted patching running from tick 0.

![Epidemic Dashboard](epidemic_dashboard.png)

**What the four panels show:**

| Panel | Content |
|---|---|
| **Top-left** | Classic SEIR curve: Susceptible (green) collapses, Infected (red) spikes to 93.7%, Patched (grey) grows slowly. The curve does not flatten. |
| **Top-right** | Hub nodes (purple) ignite the outbreak first. Non-hub nodes (red) carry the bulk. 1,090 hubs exist; patching them buys time but cannot stop the cascade across 8,910 non-hub nodes. |
| **Bottom-left** | Patching queue depth (blue bars) stays shallow; nodes flow through steadily. Cumulative patched (grey line) reaches 1,085 (10.8%) by tick 100. |
| **Bottom-right** | Stacked fate chart: by tick 20 the network is 90%+ infected. It never recovers under β = 0.40. |

---

### Finding 2 — Viral Transmissibility Drives 54.8% of All Outcome Variance

640 Monte Carlo runs with Saltelli quasi-random parameter sampling. Sobol variance decomposition quantifies which parameters determine the epidemic outcome:

![Sobol Sensitivity Indices](sobol_indices.png)

| Parameter | S1 (first-order) | ST (total-order) | S1→ST Gap | Interpretation |
|---|---|---|---|---|
| `spread_chance` (β) | 0.437 | **0.548** | 0.111 | Dominant driver. The attacker's weapon. |
| `patching_rate` | 0.211 | **0.390** | **0.179** | Strongest defensive lever. Largest interaction gap. |
| `patch_completion_prob` | 0.035 | 0.166 | 0.131 | Weak alone; gains power through parameter interactions. |
| `rewire_rate` | -0.012 | 0.005 | 0.017 | Statistically irrelevant. Network volatility does not matter. |

**The critical finding is the S1-to-ST gap for `patching_rate` (0.179)**, the largest in the set. This proves patching does not act in isolation: it amplifies or dampens the effect of every other parameter. The defensive value of patching degrades as the virus gets faster. This is a **policy-relevant insight** with direct analogues in public health intervention strategy.

![Parameter Interaction Heatmap](interaction_heatmap.png)

The heatmap confirms: `spread_chance × patching_rate = 0.204` is the strongest off-diagonal interaction. Network rewiring (rightmost column) is uniformly near zero.

---

### Finding 3 — A Temporal GCN Predicts Infections 5 Ticks Ahead on Unseen Infrastructure

A Temporal Graph Convolutional Network (T-GCN) processes a **4-tick sliding window** of graph state to predict per-node infection status at T+5. The architecture fuses spatial neighbor aggregation (GCNConv) with temporal memory (GRU), encoding **propagation velocity** rather than static snapshots.

**Why temporal matters:** A static GCN sees a photograph. The T-GCN sees a movie. The GRU hidden state accumulates how fast infection is moving through each node's neighborhood over 4 consecutive ticks. By tick T, the representation encodes the *trajectory* of contagion, not just the current snapshot.

```
Architecture: GCNConv(5, 16) → ReLU  [shared across all 4 ticks in window]
              → GRU(16, 16)          [accumulates temporal hidden state]
              → Linear(16, 1)        [per-node infection logit]
              Total: 1,745 parameters
```

**Inductive evaluation protocol:** The model trains on 5 BA graph topologies (seeds 42-46) and is tested on 2 **completely unseen** topologies (seeds 52-53). The test graphs have different hub positions, different edge wiring, and different simulation dynamics. The model cannot memorize the address book of any specific network.

---

### Finding 4 — The Model Retains 97.6% of Ranking Performance Under 85% Telemetry Blackout

**Experimental design:** Both Run A and Run B train on identical clean data with full visibility. The only difference is what the model sees **at evaluation time**. Run B masks 85% of infection states to -1 (sentinel) with a synchronized observability flag set to 0.0 for masked nodes. This is the standard partial-observability protocol: train the best model possible, then stress-test it under degraded conditions.

![GNN Performance](gnn_performance.png)

| Metric | Full Visibility | 85% Masked | Delta | Interpretation |
|---|---|---|---|---|
| **AUC-ROC** | **0.873** | **0.852** | **-0.021** | 97.6% of ranking ability preserved. Graph topology carries the signal. |
| Test Accuracy | 64.8% | 53.8% | -11.0% | Threshold-dependent; misleading under 89.2% class imbalance. |
| Hub Recall | 61.5% | 49.0% | -12.5% | Hub prediction degrades more than overall ranking. Hubs are hardest to predict blind. |

> **Note on accuracy:** Raw accuracy appears low because 89.2% of nodes are infected at the test ticks (deep into the epidemic). A naive "predict everyone infected" classifier scores ~89% accuracy while providing zero actionable intelligence. AUC-ROC is the definitive metric because it evaluates ranking quality across all classification thresholds, independent of class imbalance.

**What the -0.021 AUC delta means operationally:**

In a Security Operations Center, you will never have full telemetry. Endpoints go dark. Agents fail silently. Lateral movement happens in network segments you cannot observe. The -0.021 delta proves that on power-law networks, **graph topology and temporal propagation patterns carry almost all of the predictive signal**. The model does not need to see infection status to predict where the contagion is heading. It infers propagation from structure.

In epidemiology, the parallel is exact: most infections are unreported, most hosts are unsampled. A model that retains 97.6% of its ranking performance under 85% surveillance blindness is operationally useful in both domains.

---

## 📐 Mathematical Core

### Vectorized SEIR Transmission

```
infected_neighbors = A · x_infected        (SpMV, COO sparse float32)
P(exposure)        = 1 − (1 − β)^k         (per-node, vectorized, zero Python loops)
```

This transmission kernel is **domain-agnostic**: `β` is a transmission coefficient, `k` is the count of infected neighbors, and `A` defines the contact topology. Whether nodes represent network endpoints or animal hosts, the math is identical.

### Spectral Threshold (Pre-flight Check)

```
ρ(A) = largest eigenvalue of A     [ARPACK Lanczos, never dense]
λ_c  = 1 / ρ(A)                    [epidemic threshold]
β > λ_c  →  pandemic guaranteed     [spectral graph theory theorem]
```

### Phase III Algorithmic Mechanics

| Mechanism | Detail |
|---|---|
| **Async patching queue** | Geometric drain: each queued node completes with prob `p_drain` per tick |
| **Stochastic edge rewiring** | `rewire_rate` fraction of edges randomly reconnected per tick |
| **Latency-weighted exposure** | Queued nodes transmit at 50% rate during repair |

### T-GCN Forward Pass

```
For each tick w in [T-3, T-2, T-1, T]:
    emb_w = ReLU( GCNConv( x_w, edge_index ) )     # spatial aggregation
    _, h  = GRU( emb_w, h )                          # temporal accumulation

logit = Linear( h )                                   # per-node prediction
```

The GRU hidden state `h` carries forward across ticks, encoding the *velocity* of infection propagation through each node's neighborhood. By tick T, `h` contains temporal momentum information that a static snapshot cannot capture.

---

## ✅ Quantified Research Claims

Every number in this README is reproducible from the codebase. No claim is made without a verifiable source.

| Claim | Value | Verified by |
|---|---|---|
| Spectral radius predicts epidemic a priori | ρ(A) = 21.96, λ_c = 0.0455 | `tensor_engine.py` |
| β exceeds threshold by | **8.78×** | `run_pipeline.py` |
| Peak infection | **9,372 / 10,000 (93.7%)** | `epidemic_dashboard.png` |
| Outbreak peaks at | **tick 14** | `epidemic_dashboard.png` |
| Transmissibility drives outcome variance | **54.8%** (Sobol ST) | `sensitivity_analysis.py` |
| Patching rate S1-to-ST gap (largest interaction) | **0.179** | `sensitivity_analysis.py` |
| Network rewiring contribution | **~0%** | `sobol_indices.png` |
| T-GCN predicts infections 5 ticks ahead (inductive) | **AUC 0.873** | `predictive_model.py` |
| T-GCN AUC under 85% masking (eval only) | **AUC 0.852** | `predictive_model.py` |
| Ranking retention under 85% blindspot | **97.6%** | `gnn_performance.png` |
| Hub recall (full visibility) | **61.5%** | `gnn_performance.png` |
| Hub recall (85% masked) | **49.0%** | `gnn_performance.png` |
| T-GCN parameter count | **1,745** | `predictive_model.py` |
| Training graphs (inductive) | **5 BA topologies** (seeds 42-46) | `run_pipeline.py` |
| Test graphs (unseen) | **2 BA topologies** (seeds 52-53) | `run_pipeline.py` |

---

## 🖥️ Interactive Browser Dashboard

```bash
pip install -r requirements.txt
streamlit run app.py          # opens http://localhost:8501
```

| Panel | What it shows |
|---|---|
| **Sidebar** | Sliders for β, patching rate, patch completion prob, rewire rate, strategy toggle (Random / Targeted), tick count, seed, and a "Run Simulation" button |
| **Tab 1 — Epidemic Simulation** | Runs the full 10,000-node tensor engine live. Displays spectral metrics (ρ, λ_c, β/λ_c), peak infected stats, interactive Plotly SEIR compartment curves, stacked node-fate chart, and patching queue depth |
| **Tab 2 — Spectral Calibration** | Explains the spectral graph theory pre-flight check with the `epidemic_dashboard.png` 4-panel visualization |
| **Tab 3 — Sobol Sensitivity** | Displays `sobol_indices.png` and `interaction_heatmap.png` with narrative interpretation |
| **Tab 4 — GNN Prediction** | Shows T-GCN accuracy / AUC metrics and `gnn_performance.png` with ROC analysis |

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt

# Step 1 — Generate 7 BA graphs (5 train + 2 test), run 100-tick simulations, export Parquet
python3 run_pipeline.py

# Step 2 — Sobol sensitivity analysis (640 Saltelli runs)
python3 sensitivity_analysis.py

# Step 3 — Train T-GCN + evaluate under partial observability
python3 predictive_model.py

# Step 4 — Launch interactive dashboard
streamlit run app.py
```

**Expected output from `predictive_model.py`:**
```
Phase 5: T-GCN Predictive Intelligence
Protocol: Train full-visibility, stress-test under masking

  RUN A: Full Visibility (100% node states observed)
  Train: clean | Eval: clean
    [FULL] Train: 460  Test: 184  Params: 1,745
    ...
    Epoch  60/60  loss=0.0708  acc=64.8%  auc=87.3%

  RUN B: Partial Observability (85% node states HIDDEN at eval)
  Train: clean | Eval: 85% masked
    [MASKED] Train: 460  Test: 184  Params: 1,745
    ...
    Epoch  60/60  loss=0.0904  acc=53.8%  auc=85.2%

  COMPARATIVE RESULTS
    AUC-ROC        0.873       0.852      -0.021
```

---

## 🐳 Reproducible Research Infrastructure

### Docker (one command, zero setup)

```bash
docker build -t stitch .

# Full suite: pipeline → Sobol → GNN (~25 min on CPU)
docker run --rm -v $(pwd)/results:/app/outputs stitch

# GPU-accelerated (NVIDIA host)
docker run --rm --gpus all -v $(pwd)/results:/app/outputs stitch

# Individual phases
docker run --rm -e RUN_MODE=pipeline -v $(pwd)/results:/app/outputs stitch
docker run --rm -e RUN_MODE=sobol    -v $(pwd)/results:/app/outputs stitch
docker run --rm -e RUN_MODE=gnn      -v $(pwd)/results:/app/outputs stitch
```

### HPC / Slurm (university cluster)

```bash
bash hpc/submit_all.sh    # chains pipeline → sobol + gnn via --dependency=afterok
```

---

## 🧬 The Biological Isomorphism

The mathematical framework is domain-agnostic by design. The table below maps every component:

| STITCH Concept | Cybersecurity Domain | Epidemiology Domain |
|---|---|---|
| Node | Network endpoint (server, laptop, IoT device) | Host organism (human, animal, vector) |
| Edge | Network connection (VLAN, subnet link) | Contact event (proximity, bite, aerosol) |
| β (spread_chance) | Exploit success probability | Pathogen transmission coefficient |
| State S → E → I → R | Susceptible → Exposed → Infected → Patched | Susceptible → Exposed → Infectious → Recovered |
| Patching rate | SOC remediation speed | Vaccination or treatment rate |
| Hub node | High-degree server / core switch | Superspreader host |
| ρ(A) | Network spectral radius | Contact network spectral radius |
| 85% masking | Endpoint telemetry blackout | Incomplete epidemiological surveillance |

The T-GCN architecture trained on cyber-contagion learns the **physics of diffusion on power-law graphs**. Replacing the BA adjacency matrix with an empirical host-contact network from USDA or CDC surveillance data makes this a zoonotic pathogen prediction system with no architectural changes.

---

## 📁 Project Structure

| File | Role | Status |
|---|---|---|
| `tensor_engine.py` | Vectorized SEIR core, spectral calibration, Phase III mechanics | Active |
| `run_pipeline.py` | Multi-graph simulation → Parquet → PyG orchestrator (5 train + 2 test BA graphs) | Active |
| `sensitivity_analysis.py` | Sobol global sensitivity, 640 Saltelli runs | Active |
| `predictive_model.py` | T-GCN training + inductive evaluation under partial observability | Active |
| `app.py` | Streamlit browser dashboard for interactive simulation + research visualization | Active |
| `data/parquet_export.py` | zstd Parquet snapshot exporter with per-tick columnar schema | Active |
| `data/pyg_dataset.py` | PyG InMemoryDataset construction from Parquet snapshots | Active |
| `Dockerfile` + `docker-compose.yml` | Multi-stage container image and service orchestration | Active |
| `hpc/submit_*.sh` | Slurm batch scripts for cluster deployment | Active |
| `epidemic_dashboard.png` | 4-panel SEIR epidemic visualization | Artifact |
| `sobol_indices.png` | Sobol S1/ST bar chart | Artifact |
| `interaction_heatmap.png` | Pairwise S2 interaction heatmap | Artifact |
| `gnn_performance.png` | T-GCN training curves + dual ROC (full vs masked) | Artifact |
| `models/university_network.py` | Mesa ABM (N=200, interactive) | Legacy |
| `server.py` | Mesa browser dashboard | Legacy |
| `run_analysis.ipynb` | Mesa proof-of-concept experiments | Legacy |

---

## ⚠️ Honest Limitations & Future Work

**Current limitations:**
- **Single test-seed evaluation.** Results are from test seeds 52-53. A robust publication requires 3-5 test seed combinations with reported mean ± std to confirm stability.
- **Training convergence.** AUC oscillates during training (84.6% → 92.0% → 82.9% → 87.3% across epochs 20-60). A learning rate scheduler (`ReduceLROnPlateau`) or early stopping with best-checkpoint selection would likely push full-visibility AUC above 0.90.
- **Synthetic topology.** BA graphs approximate real-world power-law networks but do not capture community structure, geographic clustering, or heterogeneous edge weights present in empirical network data.
- **Class imbalance.** 89.2% of test nodes are infected, making raw accuracy a near-useless metric. AUC-ROC is the definitive evaluation metric throughout.

**Planned extensions:**
- **Progressive masking sweep.** Evaluate at 50%, 70%, 85%, 95% masking and plot the full AUC degradation curve.
- **Degree-biased masking.** Preferentially mask hub nodes to stress-test the model under worst-case telemetry loss.
- **Spatial cluster masking.** Mask entire subnets to simulate realistic switch/segment failures.
- **Multi-seed stability harness.** Automate 5+ seed combinations and report mean ± std for all metrics.
- **Empirical topology.** Replace BA graphs with real network telemetry or CDC host-contact datasets.

---

## 📝 Requirements

- Python 3.11+ · all packages in `requirements.txt`
- Runs on CPU (Mac/Linux), NVIDIA CUDA 12+, or Apple MPS (auto-detected)

---

## Appendix: Legacy Interactive Dashboard (Mesa)

The original Mesa agent-based model (N=200) provides a live browser UI for interactive exploration. It is self-contained and independent of the tensor engine.

```bash
python3 server.py     # open http://127.0.0.1:8521
```

The dashboard provides real-time sliders for virus spread, patching rate, and outbreak size, with four live charts: SEIR time series, daily events, device-type infection curves, and the live network graph with colour-coded nodes. The Mesa model validated the core mathematical framework before scaling to N=10,000.

---

## License

[MIT](LICENSE)
