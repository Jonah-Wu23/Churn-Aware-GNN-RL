# Community Micro-transit Dynamic Dispatching (GNN + RL)

This repository provides a maintainable and auditable scaffold for a community micro-transit dynamic dispatching system using GNN + RL. The system targets a single primary metric: Total Avoided Private Car Travel Time (TACC).

## Goals
- Shift demand from private cars to micro-transit by reducing passenger churn
- Provide a reproducible training and evaluation pipeline
- Separate algorithmic churn from structurally unserviceable demand

## Primary Metric
- TACC (Total Avoided Private Car Travel Time): the total time that would have been spent by private cars for trips served by the system

## Two-Stage Simulation
- Stage 1: Event-driven Gym for high-frequency training
- Stage 2: SUMO/TraCI for physical validation and sim-to-real checks

## Three-Layer Graph
- Layer 0: Raw OSM data for physical simulation
- Layer 1: Routing graph for travel time computation
- Layer 2: Logical dispatch graph with stop-only nodes for GNN

## Compliance and Audit
- Strict stop-based service: no curbside pickup
- OD requests must map to legal stops
- Structural unreachability is reported separately and excluded from algorithmic churn
- All data/graph generation must emit JSON audit reports under `reports/audit/`

## Reproducibility
- Fixed random seeds
- All runs log config, code version, and metrics
- Outputs include audit reports, metrics, and run metadata

## NYC Study Area
- NYC experiments use the configured study-area bbox (see `configs/manhattan.yaml`) rather than the full city

## Current Status (Code-Accurate)
- Event-driven Gym simulator with multi-vehicle support, hard mask budgets, churn game, CVaR risk, fairness weight.
- Three-layer graph build with audit JSON + SVG visualization, zero-travel-time fix, and optional zero-in/out pruning.
- Node2Vec embeddings via local `node2vec/` clone (`scripts/train_node2vec.py`).
- Edge-Q ECC model in pure PyTorch; k-hop subgraph batching supported in the env.
- Training runner is still a baseline/diagnostic loop; DQNTrainer exists in `src/train/dqn.py` but is not wired into the CLI yet.
- Curriculum learning, evaluator, and baselines remain pending (see `todos.md`).

## Testing
Core tests are isolated from baseline dependencies. The `baselines/` directory is excluded from pytest discovery via `pytest.ini`.

```bash
# Recommended: run all core tests
pytest -q tests

# Verbose output for debugging
pytest -v tests
```

See `AGENTS.md` for more testing options and baseline setup notes.

## Quickstart (current)
- Filter NYC trips by bbox (optional): `python scripts/filter_taxi_bbox.py --config configs/manhattan.yaml`
- Map OD to legal stops: `python scripts/map_od.py --config configs/manhattan.yaml`
- Build logical graph: `python scripts/build_graph.py --config configs/manhattan.yaml`
- Audit graph (JSON + SVG): `python scripts/audit_graph.py --config configs/manhattan.yaml`
- Train Node2Vec embeddings: `python scripts/train_node2vec.py --config configs/manhattan.yaml`
- Run Gym baseline/diagnostic: `python scripts/run_gym_train.py --config configs/manhattan.yaml`
- Validate in SUMO (skeleton): `python scripts/run_sumo_eval.py --config configs/manhattan.yaml`
- Download Chicago taxi data: `python scripts/download_chicago_taxi.py --start 2025-05-01T00:00:00 --end 2025-05-31T23:45:00 --output data/external/Chicago_5_data.csv`

Notes:
- Node2Vec requires a local `node2vec/` clone and `gensim`. On Windows, use `workers=1`.
- Audit outputs are written under `reports/audit/`.

See `docs/architecture.md`, `docs/requirements.md`, and `todos.md` for details.

## Realtime visualization (EdgeQ)
Enable realtime publishing in your training config:

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

Run the dashboard in a separate process:

```bash
python scripts/run_realtime_viz.py --config configs/manhattan.yaml --port 8050
```

Then start training:

```bash
python scripts/run_edgeq_train.py --config configs/manhattan.yaml
```
