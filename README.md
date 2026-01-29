# Community Micro-Transit Dynamic Dispatching (GNN + RL)

This repository is a public, minimal release of a community micro-transit dynamic dispatching system using GNN + RL. It targets a single primary metric: Total Avoided Private Car Travel Time (TACC), while modeling passenger churn and enforcing hard service constraints.

## Goals
- Shift demand from private cars to micro-transit by reducing passenger churn
- Provide a reproducible training/evaluation pipeline
- Separate algorithmic churn from structurally unserviceable demand

## Primary Metric
- **TACC (Total Avoided Private Car Travel Time)**: total private-car travel time avoided by served trips

## Simulation Stack
- **Stage 1**: Event-driven Gym (fast training/eval)
- **Stage 2**: SUMO/TraCI (physical validation; evaluation only)

## Three-Layer Graph
- **Layer 0**: OSM stop features
- **Layer 1**: Drive graph for travel-time computation
- **Layer 2**: Logical dispatch graph (stop-only nodes for GNN)

## Compliance and Audit
- Strict stop-based service: no curbside pickup
- OD requests must map to legal stops
- Structural unreachability is reported separately and excluded from algorithmic churn
- Graph/OD audit reports are generated under `reports/audit/` during runs

## Reproducibility
- Fixed random seeds and run metadata
- Deterministic evaluation with explicit config overrides
- Outputs include audit reports, metrics, and run metadata

## What Include
- Event-driven Gym simulator with multi-vehicle support, hard mask budgets, churn game, CVaR risk, and fairness weighting
- Graph build with audit JSON + SVG, zero-travel-time fixes, and optional zero-in/out pruning
- Mobi-Churn GNN-DQN (PyG TransformerConv) with k-hop subgraph batching
- Curriculum training (L0–L3), evaluator, and baseline adapters (MAPPO/CPO/MOHITO/WU2024)
- Distillation pipeline (student MLP) and sensitivity/ablation evaluation scripts

## Data and Artifacts
This public release does **not** include raw/processed data or run outputs.  
All data artifacts and reports are generated locally under `data/` and `reports/` when you run scripts.

## Quickstart
```bash
# Map OD to legal stops
python scripts/map_od.py --config configs/manhattan.yaml

# Build logical graph
python scripts/build_graph.py --config configs/manhattan.yaml

# Audit graph (JSON + SVG)
python scripts/audit_graph.py --config configs/manhattan.yaml

# Train Node2Vec embeddings (requires external node2vec + gensim)
python scripts/train_node2vec.py --config configs/manhattan.yaml

# Train Mobi-Churn GNN-DQN with curriculum
python scripts/run_curriculum_train.py --config configs/manhattan_curriculum_v13.yaml

# Evaluate baselines
python scripts/run_eval.py --config configs/manhattan_curriculum_v13.yaml --policy edgeq
```

Notes:
- Node2Vec requires a local `node2vec` implementation and `gensim`.
- Audit outputs are written under `reports/audit/`.

## Testing
Core tests are isolated from baseline dependencies via `pytest.ini`.

```bash
pytest -q tests
pytest -v tests
```

## Realtime Visualization (Mobi-Churn GNN-DQN)
Enable publishing in your training config:

```yaml
train:
  viz:
    enabled: true
    zmq_url: "tcp://127.0.0.1:5555"
    publish_every_steps: 5
    publish_on_episode_end: true
    bind: true
    alerts:
      reward_window: 50
      reward_positive_threshold: 0.5
      reward_delta_threshold: 0.2
      low_service_steps: 20
      low_service_threshold: 0.0
      loop_window: 30
      loop_unique_stops_max: 2
      loop_max_served: 1
      entropy_floor: 0.25
      entropy_patience: 20
      epsilon_floor: 0.2
```

Run the dashboard:

```bash
python scripts/run_realtime_viz.py --config configs/manhattan.yaml --port 8050
```

Then start training:

```bash
python scripts/run_edgeq_train.py --config configs/manhattan.yaml
```
