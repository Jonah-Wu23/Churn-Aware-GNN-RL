# Architecture

## Pipeline Overview
1. Data ingest and OD mapping (src/data)
2. Graph construction (src/graph)
3. Graph audit + embeddings (reports/audit, scripts/train_node2vec.py)
4. Event-driven Gym environment (src/env)
5. GNN policy and value estimation (src/models)
6. Training and curriculum control (src/train)
7. Evaluation metrics and stress tests (src/eval)
8. SUMO/TraCI validation (src/sim_sumo)

## Execution Flow
- Raw data and OSM are stored under `data/raw` and `data/external`
- OD mapping produces `data/processed/od_mapped/*.parquet` and audit reports
- Graph builder produces Layer 1 routing data and Layer 2 dispatch graph, plus audit JSON + SVG
- Node2Vec embeddings are trained from Layer 2 and stored in `data/processed/graph/node2vec_embeddings.parquet`
- Study area filtering keeps only trips where pickup and dropoff zone centroids fall within the bbox
- Gym uses Layer 2 and mapped OD data to generate episodes
- Training logs checkpoints and metrics using structured curriculum stages (pending runner wiring)
- SUMO validation replays policies with physical traffic dynamics (skeleton)

## Module Responsibilities
- `src/data`: download, clean, map OD to stops, emit audit reports
- `src/graph`: build Layer 1 routing graph and Layer 2 logical graph
- `src/env`: event queue, state transitions, multi-vehicle state, reward and mask logic
- `src/models`: Edge-Q ECC model, encoder, policy, inference
- `src/train`: DQN trainer + replay buffer (runner wiring pending)
- `src/sim_sumo`: TraCI adapter and sim-to-real validation
- `src/eval`: metric computation and stress test harness
- `src/utils`: geo/time helpers, logging, reproducibility

## Key Engineering Constraints
- Layer 2 contains only stop nodes
- Layer 1 is used strictly for routing computation
- Masking must enforce passenger time budgets
- Decisions only occur at stop arrivals (event-driven, no mid-edge replanning)
- Structural unreachability must be excluded from algorithmic churn
- NYC bbox filter is applied before OD mapping for this study area
- Barrier impact: straight-line distance < 50 m but pedestrian-walk distance > 500 m
- Unreachability threshold: walk time > 600 sec using walk_speed_mps = 1.4
- Fairness weighting: compute stop weights as 1 + gamma * (dist_to_center / max_dist) using the stop centroid mean as center
- CVaR: compute mean of churn-probability tail above alpha (e.g., 95th percentile) for penalty
- Onboard delay penalty (simplified): apply churn probability to delay over direct time
- Graph build fixes zero travel time edges and can prune zero-in/zero-out nodes

## Fleet-Aware Edge Potential (Optional)

For large-scale fleet scenarios (e.g., 50+ vehicles), we optionally augment edge features with a `fleet_potential` dimension that captures the congestion level at each candidate destination stop. This serves as an observational signal to mitigate herding effects, without altering the decentralized execution architecture or introducing joint action spaces.

- **Computation**: For each stop u, count vehicles whose next target is u (or k-hop neighborhood)
- **Normalization**: `phi(C) = log(1+C) / log(1+num_vehicles)`
- **Injection**: Appended as the 5th dimension of edge_features
- **Default**: Disabled (`use_fleet_potential: false`) to maintain full reproducibility of existing experiments

See `docs/fleet_aware_edge_potential.md` for detailed design and implementation.

## Reward Defaults (Current Implementation)
- churn_tol_sec: 300
- churn_beta: 0.02
- waiting_churn_tol_sec: 300
- waiting_churn_beta: 0.02
- onboard_churn_tol_sec: 300
- onboard_churn_beta: 0.02
- reward_service: 1.0
- reward_waiting_churn_penalty: 1.0
- reward_onboard_churn_penalty: 1.0
- reward_travel_cost_per_sec: 0.0
- reward_tacc_weight: 1.0
- reward_onboard_delay_weight: 0.1
- reward_cvar_penalty: 1.0
- reward_fairness_weight: 1.0
- cvar_alpha: 0.95
- fairness_gamma: 1.0

## TODO
- Define interface between Gym and SUMO policy wrappers
- Specify curriculum trigger metrics and logging schema
- Wire DQNTrainer into CLI runner and evaluator
