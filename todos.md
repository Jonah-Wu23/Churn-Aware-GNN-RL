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
- [x] Implement reward terms exactly as described in the doc.
  - Acceptance: includes base service reward, travel cost, churn probability penalty, onboard delay penalty, CVaR penalty, fairness weighting, and TACC weighting.
  - Acceptance: each term logged separately; config controls weights; logged in `src/env/gym_env.py` step info and `reward_terms.jsonl` from training runs.
- [x] Implement TACC ("Total Avoided Private Car Travel Time") as the north-star metric.
  - Acceptance: TACC computed and accumulated.
  - Acceptance: definition is documented and consistent with OD data source/proxy (explicit formula + units).

---

# I. Curriculum Learning (Structured Curriculum) - L0-L3 (+ L4 Stress)
- [x] Implement structured scenario generator for curriculum phases.
  - Acceptance: L0-L3 match the doc; optional L4 stress; each phase changes OD spatial density/pressure exactly as described; stage OD + audit JSON emitted per stage via `src/train/curriculum.py`.
- [x] Implement "stuckness" as constraint saturation.
  - Acceptance: stuckness computed as masked-action ratio per step/episode; logged in `reward_terms.jsonl` (step) and `train_log.jsonl` (episode).
- [x] Implement trigger signal `rho = service_rate / (1 + gamma * stuckness)`.
  - Acceptance: rho computed and used to advance curriculum; transitions are logged with thresholds in `curriculum_log.jsonl`.
- [x] Implement specific "bait" and "surge" stress tests.
  - Acceptance: scenarios reproduce center-short-trip bait vs edge-long-trip scarcity; surge overload; evaluator outputs metrics to `stress_metrics.json` + `stress_metrics.csv`.

---

# J. Evaluation + Baselines - Comparable and Paper-Ready
- [x] Implement a unified evaluator producing paper metrics (same code path for all policies).
  - Acceptance: metrics include:
    - System Churn Rate (explicit split waiting vs onboard, plus algorithmic vs structural) via `src/eval/evaluator.py`.
    - TACC via `tacc_total` in `eval_results.json`.
    - 95th percentile wait time via `wait_time_p95_sec`.
    - Fairness (Gini of stop service rate) via `service_gini`.
  - Acceptance: reproducible across seeds; outputs saved with config + seed + dataset hashes in `eval_results.json` (plus `eval_episodes.csv`).
- [x] Implement HCRide baseline.
  - Acceptance: matches referenced HCRide rule/parameters; documented (English) with citations in `docs/` or `src/eval/README.md`.
  - Acceptance: produces evaluator metrics.
- [x] Implement MAPPO baseline.
  - Acceptance: matches upstream MAPPO implementation/params (baselines/on-policy); documented (English) with citations in `docs/` or `src/eval/README.md`.
  - Acceptance: produces evaluator metrics.
- [x] Implement safety-starter-agents baseline (CPO).
  - Acceptance: matches upstream CPO implementation/params (safety-starter-agents); documented (English) with citations in `docs/` or `src/eval/README.md`.
  - Acceptance: produces evaluator metrics.
- [x] Implement MOHITO (UAI 2025) baseline.
  - Acceptance: matches upstream MOHITO implementation/params (baselines/mohito-public, PettingZoo env integration); documented (English) with citations in `docs/` or `src/eval/README.md`.
  - Acceptance: produces evaluator metrics.
- [x] Implement Wu et al. (C&IE 2024) multi-agent microtransit baseline.
  - Acceptance: matches upstream implementation/params (baselines/transportation_sparse/Multi-Agent Deep Reinforcement Learning based Real-time Planning Approach for Responsive Customized Bus Routes); documented (English) with citations in `docs/` or `src/eval/README.md`.
  - Acceptance: produces evaluator metrics.
- [x] Ensure fairness metric definition is explicit and reproducible.
  - Acceptance: define "stop service rate" denominator and stop set.
  - Acceptance: compute Gini reproducibly; include unit tests.
  - **Done**: Unified Gini via `src/utils/fairness.py` (RMD algorithm), stop set = all Layer-2 stops (including zero-service), unit tests in `tests/test_fairness_gini.py`.

---

# K. Doc vs Implementation Drift (v5 merged doc) - Fix Inconsistencies End-to-End
Goal: align runnable entrypoints with `docs/流失驱动的社区微公交动态调度架构设计_合并版.md` so a user can reproduce "Stage 1 (Gym) training -> Stage 1 eval -> Stage 2 (SUMO) validation" without touching baselines.

## K1. Provide a single "new system" run path (no baselines)
- [x] Add a dedicated CLI that runs "our new thing" end-to-end (train + eval) without importing any baseline code.
  - Step 1: add `scripts/run_edgeq_train.py` that trains via `DQNTrainer` (optionally curriculum stages).
  - Step 2: add `scripts/run_edgeq_eval.py` that evaluates `policy=edgeq` by loading the trained model artifact.
  - Acceptance: both scripts work with `configs/manhattan.yaml` only; no additional required flags except `--config` and `--run-dir`.
  - Acceptance: evaluator produces `reports/eval/.../eval_results.json` including hashes + metrics.

## K2. Make Stage-1 training match the Edge-Q ECC design (global graph + action edges)
- [x] Deprecate `scripts/run_gym_train.py` as "diagnostic only" and ensure it cannot be confused with the paper-aligned training.
  - [x] Step 1: rename its log banner to "diagnostic baseline" and document limitations.
  - [x] Step 2: ensure the paper-aligned training path always uses:
    - `graph_edge_index` + `graph_edge_features` for message passing, and
    - `action_edge_index` + `edge_features` for candidate actions (current-stop -> candidate-stop).
  - [x] Acceptance: training loop never falls back to "star-edge-only message passing".
  - [x] Acceptance: add a small unit/integration test that asserts the training data dict includes `graph_edge_index` for forward passes.

## K3. Ensure training produces loadable model artifacts for `policy=edgeq`
- [x] Persist trained policy checkpoints (for evaluator consumption).
  - Done: DQNTrainer writes `edgeq_model_latest.pt`/`edgeq_model_final.pt` plus `checkpoint_*.pt` into the run directory.
  - Acceptance: `src/eval/evaluator.py` can load `eval.model_path` and run `policy=edgeq` without code changes.
- [x] Make model artifact path discoverable without manual copy-paste.
  - [x] Step 1: emit a machine-readable `run_meta.json` in the run dir with `model_path_final`/`model_path_latest`.
  - [x] Step 2: allow `scripts/run_eval.py` to accept `--model-path` overriding YAML config.
  - [x] Acceptance: user can run eval immediately after training without editing YAML.
- [x] Wire Stage-2 validation to consume the Stage-1 trained policy.
  - Step 1: define the interface: how SUMO step provides observations and how policy emits stop decisions.
  - Step 2: implement a minimal-but-complete TraCI loop that:
    - loads the same Layer-2 stop graph and travel-time priors,
    - replaces travel time with TraCI-measured dynamics,
    - logs identical metric schema as Stage 1 evaluator (plus sim-to-real deltas).
  - Acceptance: `python scripts/run_sumo_eval.py --config ... --model-path ...` runs and writes outputs under `reports/sumo_eval/...`.
  - Implementation: `src/sim_sumo/sumo_env.py`, `src/sim_sumo/traci_adapter.py`, `src/sim_sumo/sumo_evaluator.py`

## K5. Make tests runnable without baseline dependencies
- [x] Prevent baseline folders from breaking `pytest` in the core repo.
  - Step 1: restrict test discovery to `tests/` (or add `--ignore baselines` in config).
  - Step 2: document the recommended commands: `pytest -q tests` for core checks.
  - Acceptance: `pytest -q tests` passes on a clean environment without extra baseline deps.

---

# Required Tests (Do Not Skip)
- [x] Unit tests for churn sigmoid shape (waiting + onboard) with deterministic RNG.
- [x] Unit tests for CVaR (`cvar_alpha=0.95`) with known distributions.
- [x] Unit tests for hard mask correctness and debug output determinism.
- [x] Unit tests for passenger state machine (exactly-one-state invariant and valid transitions).
- [x] Integration test: fixed seed episode reproduces identical event trace + key metrics (multi-vehicle + capacity).

---

# L. Risk Mitigation for MOHITO and Wu2024 Baselines - Ensure Fair Comparison

## Objective
Eliminate reviewers' concerns about "apples-to-oranges" comparisons by ensuring MOHITO and Wu2024 baselines are trained under the same conditions as other baselines (MAPPO, CPO, HCRide). The goal is not to make them "win", but to ensure they learn the basic task so fairness metrics (Gini, service rate) are interpretable and comparable.

## L1. Risk Points to Address
- [ ] **Training Inconsistency**: Other baselines are in-domain trained, but MOHITO/Wu2024 may be zero-shot or random-init.
  - Acceptance: Document current training status for MOHITO and Wu2024.
  - Acceptance: Identify if they use pre-trained weights, random init, or zero-shot cross-domain inference.
- [ ] **Domain Gap Impact on Fairness**: Models that fail to learn the task will have low service rates, leading to unstable or meaningless Gini coefficients.
  - Acceptance: Verify that current MOHITO/Wu2024 results show reasonable service rates (not extremely low).
  - Acceptance: If service rates are too low, flag this as a risk to fairness metric validity.

## L2. Minimum Requirements for "Complete Risk Elimination"
### 2.1 In-Domain Training
- [ ] **Train both MOHITO and Wu2024 in the same domain** (Manhattan microtransit environment).
  - Acceptance: Training uses the same `EventDrivenEnv` and `EnvConfig` as MAPPO/CPO/HCRide.
  - Acceptance: Training mode is either:
    - **Option 1 (Strongly Recommended)**: Train from scratch (random init) to convergence.
    - **Option 2 (Also Acceptable)**: Fine-tune from pre-trained weights (e.g., rideshare domain) to convergence on our task.
  - Acceptance: Document the chosen approach clearly in paper/README.

### 2.2 Training Budget Alignment
- [ ] **Use the same training budget** (env steps, episodes, or decision steps) for MOHITO/Wu2024 as for other baselines.
  - Acceptance: Training budget is explicitly defined and documented in README or paper experimental setup.
  - Acceptance: Training curves (service rate, churn rate) are logged and saved for verification.
  - Acceptance: Convergence is defined as "service rate and churn rate stabilize on validation set" (not necessarily best performance).

### 2.3 Unified Reward, Mask, and Episode Rules
- [ ] **Ensure identical training conditions** across all baselines:
  - Acceptance: Same reward function (no special fairness regularization for MOHITO/Wu2024).
  - Acceptance: Same feasibility mask, capacity constraints, time windows, timeout rules.
  - Acceptance: Same structural unreachability handling.
  - Acceptance: Config files for MOHITO/Wu2024 training match other baselines in all critical parameters.

### 2.4 Unified Data Split and Evaluation Protocol
- [ ] **Use the same train/test split**:
  - Acceptance: Training seeds (or training demand samples) are strictly separated from test seeds.
  - Acceptance: Test seeds are fixed and identical across all baselines.
- [ ] **Use the same evaluator**:
  - Acceptance: All baselines (including MOHITO/Wu2024) are evaluated via the unified `src/eval/evaluator.py`.
  - Acceptance: Same `_compute_metrics()` for all baselines.
  - Acceptance: Metrics output format is identical (same JSON schema).

## L3. Training Protocol (Strong Immunity - Required)
- [ ] **Train MOHITO from scratch** in the Manhattan domain.
  - Acceptance: Random initialization (no pre-trained weights).
  - Acceptance: Train to convergence using the unified training protocol.
  - Acceptance: Save training logs, curves, and final checkpoint.
- [ ] **Train Wu2024 from scratch** in the Manhattan domain.
  - Acceptance: Random initialization (same as MOHITO).
  - Acceptance: Train to convergence using the unified training protocol.
  - Acceptance: Save training logs, curves, and final checkpoint.
- **Rationale**: This approach is the most transparent and least open to criticism. Both baselines must be trained from random initialization to convergence, ensuring complete comparability with MAPPO/CPO/HCRide.

## L4. Unified Re-Training Protocol Template (Paper-Ready Checklist)
### 4.1 Training Scenario Distribution
- [ ] **Define training scenario mix**:
  - Acceptance: Use "Normal" scenario as primary distribution.
  - Acceptance: Optionally mix small amounts of curriculum scenarios (Bait/Surge) for robustness.
  - Acceptance: Fixed mixing ratio across all baselines.
  - Acceptance: Document scenario distribution in config and README.

### 4.2 Training Budget
- [ ] **Fix training budget** (same for all baselines):
  - Acceptance: Define budget in terms of decision steps or episodes.
  - Acceptance: Document budget in README/paper.
  - Example: "All baselines trained for 1M decision steps" or "500 episodes".

### 4.3 Randomness Control
- [ ] **Use multiple training seeds**:
  - Acceptance: At least 3 training seeds per baseline.
  - Acceptance: Training seeds are distinct from test seeds.
  - Acceptance: Document all seeds used.
- [ ] **Fix test seeds**:
  - Acceptance: All baselines evaluated on the same fixed set of test seeds.
  - Acceptance: Test seeds never used during training.

### 4.4 Hyperparameter Strategy
- [ ] **Choose one hyperparameter approach**:
  - **Option A (Most Stable)**: Small grid search (e.g., 2 learning rates × 2 entropy coefficients) applied equally to all baselines.
    - Acceptance: All baselines use the same grid.
    - Acceptance: Document grid search results.
  - **Option B (Simpler)**: Use default hyperparameters from original papers + necessary scaling adjustments (e.g., normalization).
    - Acceptance: Document all hyperparameters in config files.
    - Acceptance: No per-baseline special tuning.
- [ ] **Document chosen strategy** in README/paper.

### 4.5 Convergence Criteria
- [ ] **Define convergence** as "service rate and algorithmic churn rate stabilize on validation set":
  - Acceptance: Not required to reach optimal performance, only to "learn the task".
  - Acceptance: Training stops when metrics plateau for N consecutive validation checks.
  - Acceptance: Document convergence criteria.

### 4.6 Evidence Chain (Reproducibility)
- [ ] **Save all training artifacts**:
  - Acceptance: Config hash, seed, training curves, final checkpoint, git commit SHA.
  - Acceptance: All artifacts saved in a structured run directory.
  - Acceptance: Run directory includes `run_meta.json` with metadata.
- [ ] **Make training reproducible**:
  - Acceptance: Provide training script + config that can reproduce results.
  - Acceptance: Document training commands in README.

### 4.7 Final Reporting
- [ ] **Report in-domain trained results** in main comparison table:
  - Acceptance: Main table includes only in-domain trained MOHITO/Wu2024 results.
  - Acceptance: Clearly label as "in-domain trained" or "trained to convergence".
- [ ] **Optionally include zero-shot results** as supplementary:
  - Acceptance: Add appendix row showing "zero-shot (pre-trained)" results as reference.
  - Acceptance: This demonstrates transparency and will not reduce paper quality.

## L5. Fairness Metric Immunity Patch (Strongly Recommended)
- [ ] **Bundle key metrics together** in result tables to prevent "Gini meaningless at low service" criticism:
  - Acceptance: For each baseline, report:
    - `served` (or `total_boardings`)
    - `service_rate`
    - `service_gini`
  - Acceptance: Ensure service volume is not "extremely low" before interpreting Gini.
  - Acceptance: If service rate is very low, flag this in the table or text.
- [ ] **Add validation check**:
  - Acceptance: If `service_rate < threshold` (e.g., 30%), add a note that fairness metrics may be unstable.
  - Acceptance: Document this threshold and rationale.

## L6. Implementation Tasks
### 6.1 MOHITO Training Integration
- [ ] **Implement MOHITO training adapter**:
  - Acceptance: Create `scripts/run_mohito_train.py` or integrate into unified training script.
  - Acceptance: Adapter converts `EventDrivenEnv` observations to MOHITO input format.
  - Acceptance: Training loop saves checkpoints compatible with evaluator.
- [ ] **Run MOHITO training**:
  - Acceptance: Train for defined budget to convergence.
  - Acceptance: Save all artifacts (logs, curves, checkpoints).
- [ ] **Validate MOHITO convergence**:
  - Acceptance: Service rate stabilizes on validation set.
  - Acceptance: Churn rate shows improvement trend.

### 6.2 Wu2024 Training Integration
- [ ] **Implement Wu2024 training adapter**:
  - Acceptance: Create `scripts/run_wu2024_train.py` or integrate into unified training script.
  - Acceptance: Adapter converts `EventDrivenEnv` observations to Wu2024 pointer-net input format.
  - Acceptance: Training loop saves checkpoints compatible with evaluator.
- [ ] **Run Wu2024 training**:
  - Acceptance: Train for defined budget to convergence.
  - Acceptance: Save all artifacts (logs, curves, checkpoints).
- [ ] **Validate Wu2024 convergence**:
  - Acceptance: Service rate stabilizes on validation set.
  - Acceptance: Churn rate shows improvement trend.

### 6.3 Unified Evaluation After Training
- [ ] **Re-run evaluator with trained MOHITO/Wu2024**:
  - Acceptance: Use `src/eval/evaluator.py` with `policy=mohito` and `policy=wu2024`.
  - Acceptance: Load trained checkpoints (not random or zero-shot).
  - Acceptance: Evaluate on fixed test seeds (same as other baselines).
- [ ] **Verify metric comparability**:
  - Acceptance: Service rates for all baselines are in reasonable range (e.g., >30%).
  - Acceptance: Gini coefficients are stable (not degenerate due to low service).
  - Acceptance: All metrics logged in same format.

### 6.4 Documentation Updates
- [ ] **Update README/paper experimental setup**:
  - Acceptance: Document training protocol for MOHITO/Wu2024 (from-scratch or fine-tuned).
  - Acceptance: Document training budget, seeds, hyperparameters.
  - Acceptance: State clearly: "All baselines trained in-domain to convergence using unified protocol".
- [ ] **Update baseline documentation**:
  - Acceptance: `src/eval/README.md` includes training details for MOHITO/Wu2024.
  - Acceptance: Include references to original papers + any modifications.
- [ ] **Add training evidence**:
  - Acceptance: Include training curves (service rate, churn rate) in supplementary materials or appendix.
  - Acceptance: Reference run directories with saved artifacts.

## L7. Acceptance Criteria for "Complete Risk Elimination"
- [ ] **Final checklist**:
  - [ ] MOHITO and Wu2024 are trained in-domain (Manhattan EventDrivenEnv) to convergence.
  - [ ] Training uses the same reward, mask, episode rules, and budget as MAPPO/CPO/HCRide.
  - [ ] Train/test seeds are strictly separated and identical across all baselines.
  - [ ] All baselines evaluated via unified `src/eval/evaluator.py` producing identical metric schema.
  - [ ] Service rates are reasonable (>30%) for all baselines, ensuring Gini interpretability.
  - [ ] All training artifacts (configs, seeds, curves, checkpoints, git SHAs) are saved and reproducible.
  - [ ] Documentation (README + paper) clearly states "in-domain trained" and provides full experimental protocol.
  - [ ] Main comparison table shows only in-domain trained results; zero-shot results (if any) are in appendix as reference.

## Summary Statement (For Paper/README)
> **"MOHITO and Wu2024 baselines are trained in-domain to convergence using the same environment, reward function, training budget, and data split as other baselines (MAPPO, CPO, HCRide). All baselines are evaluated via a unified evaluator producing identical metrics, ensuring fair and interpretable comparison, especially for fairness (Gini) metrics."**

---

