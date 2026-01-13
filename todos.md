# Project TODOs (Strict Spec + Acceptance Criteria)

## Non-Negotiables
- All items below must be implemented to spec (no partial implementations, no "minimal" shortcuts, no missing sub-items).
- All acceptance criteria are executable or objectively verifiable (tests, logs, artifacts, deterministic replay).
- Stage naming: Stage 1 = event-driven Gym simulator; Stage 2 = SUMO/TraCI overrides travel time dynamics.

## Collaboration Rules
- Chat with the user in Chinese.
- Write repository artifacts in English (code, docs, TODOs, configs, logs).

---

# A. Stage 1 (Event-Driven Gym) - Core Simulation, Controllability, Reproducibility
- [x] Implement a true event-driven simulator with a priority queue.
  - Acceptance: events include at least `Order`, `VehicleArrival`, `VehicleDeparture`, `ChurnCheck`.
  - Acceptance: simulation time always jumps to the next event time (no fixed timestep drift).
  - Acceptance: deterministic replay for a fixed seed: identical event trace, metrics, and logs.
- [x] Support "re-routing at stops" only.
  - Acceptance: policy decisions are triggered only at stop arrivals (no mid-edge replanning).
  - Acceptance: action space is defined as stop-to-stop edges on Layer-2.
- [x] Model mixed demand: real-time + scheduled/reservation requests.
  - Acceptance: request generator supports both streams and preserves timestamps.
  - Acceptance: scheduled requests enter the system at their scheduled time; real-time requests arrive stochastically/streamed.
- [x] Implement multi-vehicle support.
  - Acceptance: `num_vehicles >= 1`.
  - Acceptance: each vehicle has its own state (location, available time, onboard manifest, route/next decision).
  - Acceptance: evaluator can run episodes with multiple vehicles and produces the same metric schema.
- [x] Implement vehicle capacity constraints.
  - Acceptance: boarding never exceeds capacity.
  - Acceptance: capacity affects feasible actions and metrics (denials, overflow attempts impossible by design).
- [x] Enforce strict stop-based service (No Curbside Pickup).
  - Acceptance: boarding/alighting occurs only at legal stops; serving outside stops is impossible by design.
- [x] Define passenger lifecycle states and transitions.
  - Acceptance: every request is in exactly one of:
    - `waiting`
    - `onboard`
    - `served`
    - `churned_waiting`
    - `churned_onboard`
    - `structurally_unserviceable`
  - Acceptance: state transitions are logged and validated (no illegal transitions; no double-counting).

---

# B. Churn Game - Behavioral Model for Waiting and Detour (Consistent)
- [x] Implement a sigmoid/logit churn probability model with a tolerance threshold.
  - Acceptance: `P_churn(t) = sigmoid(beta * (t - T_tol))`.
  - Acceptance: unit tests verify behavior at `t < T_tol`, `t ~= T_tol`, `t >> T_tol`.
  - Acceptance: waiting and onboard churn can use distinct `(beta, T_tol)` parameters.
- [x] Apply churn model to waiting time.
  - Acceptance: waiting passengers churn stochastically based on waiting time.
  - Acceptance: churn events are logged and counted; churn randomness is reproducible for fixed seed.
- [x] Apply churn model to onboard detour/delay consistently.
  - Acceptance: onboard "extra delay vs direct" produces churn risk using the same sigmoid family.
  - Acceptance: churn-onboard is tracked separately from churn-waiting.
- [x] Separate "algorithmic churn" vs "structural unreachability".
  - Acceptance: structurally unserviceable OD (e.g., walk time > threshold) is excluded from algorithmic churn metrics.
  - Acceptance: metrics reports explicitly include both: structural unreachability rate and algorithmic churn rates.

---

# C. Risk Aggregation - CVaR + Fairness Weight (W_fair)
- [x] Implement per-stop risk aggregation for waiting queues.
  - Acceptance: for each stop `v`, compute `Risk_mean` and `Risk_CVaR` over passenger churn probabilities in the waiting queue.
- [x] Implement CVaR tail risk exactly as "Top 5% emphasis".
  - Acceptance: configurable `cvar_alpha` defaults to `0.95`.
  - Acceptance: CVaR computed from the upper tail as defined (validated by tests with known distributions).
- [x] Implement geographic fairness weight `W_fair = 1 + gamma * Dist(stop, center)`.
  - Acceptance: distance-to-center computed from stop coordinates.
  - Acceptance: weights stored and used in risk/fairness accounting; reproducible across runs.
- [x] Ensure fairness is used as described.
  - Acceptance: fairness weighting influences decision pressure toward edge stops.
  - Acceptance: metric reports include per-stop service and Gini.

---

# D. Graph Abstraction - Three-Layer Graph Alignment
- [x] Ensure Stage 1 uses Layer-2 logical stop graph for decision actions.
  - Acceptance: actions are edges between stops (Layer-2).
  - Acceptance: travel times are derived from Layer-1 routing-derived edges as stored in Layer-2.
- [x] Ensure travel time `T_travel` is the edge attribute used by policy and features.
  - Acceptance: each candidate action uses a defined travel time source.
  - Acceptance: Stage 1 uses static `T_travel`; Stage 2 may override with real-time SUMO.

---

# E. State/Feature Spec - Must Match Doc
- [x] Implement node feature vector per stop exactly (dimension = 5).
  - Acceptance: `h_v = [Risk_mean, Risk_CVaR, Count_pax, W_fair, Emb_geo]`.
  - Acceptance: definitions match doc; `Emb_geo` is not a dummy constant.
- [x] Implement `Emb_geo` as offline-pretrained Node2Vec embedding.
  - Acceptance: Node2Vec training pipeline exists (script + config).
  - Acceptance: embeddings are saved and loaded.
  - Acceptance: consistent `node_id` mapping across runs (stable join keys, no reindex drift).
- [x] Implement edge feature vector for each candidate (v->u) exactly (dimension = 4).
  - Acceptance: `e_{v->u} = [delta_eta_max, delta_cvar_onboard, count_violation, T_travel]`.
  - Acceptance: computed per candidate action from current onboard + graph times.
- [x] Define and implement `delta_eta_max` precisely.
  - Acceptance: "max incremental ETA among onboard passengers" (worst-case increase vs staying on best route from current stop).
  - Acceptance: validated by unit tests with synthetic routes.
- [x] Define and implement `delta_cvar_onboard` precisely.
  - Acceptance: "CVaR increase of onboard churn-risk distribution".
  - Acceptance: compute onboard risk distribution before/after action, then take CVaR delta; tests cover correctness.
- [x] Define and implement `count_violation` precisely.
  - Acceptance: "number of onboard passengers that would violate hard constraint if action executed".
  - Acceptance: computed consistently with the hard mask rule; tests cover edge cases (0, 1, many violations).

---

# F. Hard Mask - Dynamic Budget Commitment (Explainable)
- [x] Implement budget commitment at boarding/pickup time.
  - Acceptance: `T_max_i = alpha * T_direct_i` with `alpha ~= 1.5` (configurable).
  - Acceptance: `T_direct_i` is computed and stored per passenger at pickup/boarding.
  - Acceptance: budget values remain immutable per passenger once committed.
- [x] Implement hard mask rule.
  - Acceptance: if an action `u` leads to any onboard passenger ETA > `T_max_i`, mask the action and treat `Q(action) = -inf`.
  - Acceptance: masked actions are never selected; training targets respect masking.
- [x] Add explainable mask debugging.
  - Acceptance: for any masked action, output the passenger(s) responsible, their ETA, `T_max_i`, and margin.
  - Acceptance: logs are deterministic for fixed seed and can be toggled by config.

---

# G. Policy Network - Edge-Q GNN (ECC/GAT) + k-hop Subgraph
- [x] Implement Edge-Q GNN using ECC (Edge-Conditioned Convolution).
  - Acceptance: message passing uses edge features to condition convolution (not a simple MLP edge scorer).
  - Acceptance: produces per-edge Q-values aligned with candidate edges.
- [ ] Provide an alternative GAT variant if required by the writeup.
  - Acceptance: switchable architecture via config.
  - Acceptance: both architectures produce edge Q-values for the same evaluator interface.
- [x] Implement k-hop subgraph extraction at inference (`k=3`).
  - Acceptance: given current vehicle stop, extract k-hop subgraph, run GNN, score outgoing edges.
  - Acceptance: latency consistent with "lightweight inference" goal; benchmark script included.
- [x] Implement "static/dynamic separation" optimization.
  - Acceptance: topology and static embeddings are precomputed.
  - Acceptance: online updates only recompute dynamic `h_v` and `e_{v->u}` terms.
- [ ] Optional (per doc): teacher-student distillation for deployment.
  - Acceptance: teacher deeper model trains; student shallow model matches Q outputs within tolerance; evaluation script reports divergence.

---

# H. Reward + North Star Metric (TACC) - Doc Alignment
- [ ] Implement reward terms exactly as described in the doc.
  - Acceptance: includes base service reward, travel cost, churn probability penalty, onboard delay penalty, CVaR penalty, fairness weighting, and TACC weighting.
  - Acceptance: each term logged separately; config controls weights.
- [x] Implement TACC ("Total Avoided Private Car Travel Time") as the north-star metric.
  - Acceptance: TACC computed and accumulated.
  - Acceptance: definition is documented and consistent with OD data source/proxy (explicit formula + units).

---

# I. Curriculum Learning (Structured Curriculum) - L0-L3 (+ L4 Stress)
- [ ] Implement structured scenario generator for curriculum phases.
  - Acceptance: L0-L3 match the doc; optional L4 stress; each phase changes OD spatial density/pressure exactly as described.
- [ ] Implement "stuckness" as constraint saturation.
  - Acceptance: stuckness computed as masked-action ratio per step/episode; logged.
- [ ] Implement trigger signal `rho = service_rate / (1 + gamma * stuckness)`.
  - Acceptance: rho computed and used to advance curriculum; transitions are logged with thresholds.
- [ ] Implement specific "bait" and "surge" stress tests.
  - Acceptance: scenarios reproduce center-short-trip bait vs edge-long-trip scarcity; surge overload; evaluator outputs metrics.

---

# J. Evaluation + Baselines - Comparable and Paper-Ready
- [ ] Implement a unified evaluator producing paper metrics (same code path for all policies).
  - Acceptance: metrics include:
    - System Churn Rate (explicit split waiting vs onboard, plus algorithmic vs structural)
    - TACC
    - 95th percentile wait time
    - Fairness (Gini of stop service rate)
  - Acceptance: reproducible across seeds; outputs saved with config + seed + dataset hashes.
- [ ] Implement baseline policies from doc.
  - Acceptance: `Greedy`, `Insertion Heuristic`, `Standard RL` (DQN optimizing total wait time; no churn/CVaR/edge features).
  - Acceptance: all baselines run end-to-end in the same evaluator and produce identical metric schema.
- [ ] Implement HCRide baseline.
  - Acceptance: matches referenced HCRide rule/parameters; documented (English) with citations in `docs/` or `src/eval/README.md`.
  - Acceptance: produces evaluator metrics.
- [ ] Ensure fairness metric definition is explicit and reproducible.
  - Acceptance: define "stop service rate" denominator and stop set.
  - Acceptance: compute Gini reproducibly; include unit tests.

---

# Required Tests (Do Not Skip)
- [x] Unit tests for churn sigmoid shape (waiting + onboard) with deterministic RNG.
- [x] Unit tests for CVaR (`cvar_alpha=0.95`) with known distributions.
- [x] Unit tests for hard mask correctness and debug output determinism.
- [ ] Unit tests for passenger state machine (exactly-one-state invariant and valid transitions).
- [ ] Integration test: fixed seed episode reproduces identical event trace + key metrics (multi-vehicle + capacity).
