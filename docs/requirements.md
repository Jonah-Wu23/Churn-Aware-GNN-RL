# Requirements

## Functional
- Build three-layer graphs from OSM data
- Map OD requests to legal stops using network Voronoi on a pedestrian graph
- Train a GNN-based edge Q policy in an event-driven Gym
- Support structured curriculum learning stages for training
- Validate trained policies in SUMO/TraCI
- Produce metrics for TACC, churn (waiting vs onboard), waiting time, fairness, and stress tests

## Non-functional
- Maintainable modules with clear I/O contracts
- Auditable data and graph pipelines with JSON reports
- Reproducible experiments with fixed seeds and run metadata
- Scalable to city-scale stop counts with sparse Layer 2 graphs

## Compliance
- Strict stop-based service; no curbside pickup
- OD mapping must be to legal stops only
- Structural unreachability must be reported and excluded from algorithmic churn
- Waiting vs onboard churn must be reported separately

## Audit (hard constraints)
All data generation, OD mapping, and graph building must emit `reports/audit/*.json` with at least:
- euclidean_vs_network_mismatch_rate
- barrier_impact_count (straight line < 50m but walk > 500m)
- structural_unreachability_rate (walk > 10 min)

## Metrics
- TACC (Total Avoided Private Car Travel Time) is the primary system metric
- Churn must be split into algorithmic vs structural
- Report 95th percentile wait and Gini fairness

## Reproducibility
- Fixed random seed per run
- Log full config, git commit, and dataset hashes
- Store metrics and audit outputs alongside model artifacts

## Current Status (Code-Accurate)
- Graph build + audit: implemented (JSON + SVG, zero-travel-time fix, optional zero-in/out pruning)
- OD mapping: implemented with network Voronoi and audit report
- Gym env: event-driven, hard mask budgets, churn game, CVaR, fairness, TACC
- Edge-Q ECC model: implemented in pure PyTorch; k-hop subgraph batching supported
- Node2Vec embeddings: implemented via local `node2vec/` clone + `gensim`
- Training/evaluation: DQNTrainer exists but CLI is still a baseline loop; curriculum, evaluator, and baselines pending

## TODO
- Confirm final field names for OD inputs
- Specify data retention policy for raw sources
- Define minimum acceptable audit thresholds per city
