# Evaluation Policies

## HCRide baseline (policy = "hcride")

This baseline adapts HCRide (Jiang et al., IJCAI 2025) to the stop-to-stop routing setting.

Implementation summary:
- Reward uses waiting time + fairness term: r = -WT + alpha * (-abs(WT - meanWT) / 3) (from the HCRide simulator).
- WT is the mean predicted pickup waiting time (minutes) for riders that would be boarded at the candidate stop after travel.
- meanWT is the historical mean waiting time at that stop (minutes), computed from boarded riders.
- Driver preference uses visitation frequency V_k(u) from the active vehicle's stop history.
  H+_k = {u | V_k(u) > d} with d = preference_threshold.
  H0_k includes stops within radius V_k(u1) * preference_radius_scale_m of any u1 in H+.
  H-_k is the remaining set.
- Preference cost follows the HCRide paper: if the destination is in H-_k, cost equals the distance to the nearest H+ stop;
  otherwise cost is 0.
- Action score = reward - lagrange_lambda * cost, select the max score among valid actions.
- If a stop has no waiting passengers, empty_stop_penalty discourages idle moves.

Config keys:
- eval.policy: "hcride"
- eval.hcride.alpha (default 1.5)
- eval.hcride.lagrange_lambda (default 1.0)
- eval.hcride.preference_threshold (default 0.1)
- eval.hcride.preference_radius_scale_m (default 1000.0)
- eval.hcride.empty_stop_penalty (default 1e6)

Citations:
- Lin Jiang, Yu Yang, Guang Wang. "HCRide: Harmonizing Passenger Fairness and Driver Preference for Human-Centered
  Ride-Hailing." IJCAI 2025. Local PDF: baselines/HCRide/2508.04811v1.pdf
- Official code: https://github.com/LinJiang18/HCRide

---

## MAPPO baseline (policy = "mappo")

This baseline implements Multi-Agent PPO (MAPPO) from the on-policy repository.

Implementation summary:
- Uses parameter sharing across all vehicles (single shared policy).
- Centralized Training with Decentralized Execution (CTDE): centralized critic, decentralized actors.
- Actor: MLP with GRU recurrent layer, outputs action logits over discrete stop choices.
- Critic: MLP with GRU, takes global state summary, outputs value estimate.
- Observation: flattened node features (5-dim) + edge features (4-dim × neighbor_k) + onboard summary + position embedding.
- Action space: Discrete(neighbor_k + 1), where the last action is NOOP (no operation).
- Async decision handling: NOOP + active_masks for synchronous interface compatibility.
- Team reward: all agents receive the same global reward signal.

Environment wrapper (`src/env/mappo_env_wrapper.py`):
- Wraps EventDrivenEnv for on-policy compatibility.
- Fixed action_dim = neighbor_k + 1.
- available_actions mask: inactive agents only have NOOP available.

Config keys:
- eval.policy: "mappo"
- eval.model_path: path to trained actor.pt
- eval.mappo.neighbor_k (default 8)
- eval.mappo.hidden_size (default 64)
- eval.mappo.recurrent_N (default 1)

Training:
```bash
python scripts/run_mappo_train.py --config configs/manhattan.yaml \
    --num_env_steps 100000 --hidden_size 64
```

Evaluation:
```bash
python scripts/run_mappo_eval.py --config configs/manhattan.yaml \
    --model-path reports/mappo_train/run_xxx/models/actor.pt
```

Hyperparameters (from MAPPO paper appendix):
- lr = 5e-4, critic_lr = 5e-4
- ppo_epoch = 10, num_mini_batch = 1
- clip_param = 0.2, entropy_coef = 0.01
- gamma = 0.99, gae_lambda = 0.95
- hidden_size = 64, recurrent_N = 1 (GRU)

Fair comparison notes:
- Observation uses similar feature sources as Edge-Q (node/edge features from k-hop subgraph).
- Action space uses neighbor_k candidates + NOOP, matching graph structure.
- Training and evaluation use the same EnvConfig parameters.

Citations:
- Chao Yu, Akash Velu, Eugene Vinitsky, Jiaxuan Gao, Yu Wang, Alexandre Bayen, Yi Wu.
  "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games."
  NeurIPS 2022. arXiv:2103.01955. Local PDF: baselines/on-policy/2103.01955v4.pdf
- Official code: https://github.com/marlbenchmark/on-policy

---

## CPO baseline (policy = "cpo")

Constrained Policy Optimization (CPO) using trust-region updates with safety constraints.

Implementation summary:
- Trust-region policy update based on TRPO with cost constraint handling.
- Uses conjugate gradient descent with Fisher vector product for policy updates.
- Separate value functions for reward (V) and cost (Vc).
- Feasibility branches: handles both feasible and infeasible constraint scenarios.
- Single shared policy, asynchronous vehicle turns (same as other baselines).

Cost definition (enters CPO constraint):
- Onboard churn events (delay-induced passenger loss)
- Capacity overflow attempts (trying to board beyond capacity)
- Service commitment violations (action masked by budget constraint)

NOT in cost (structural, not policy fault):
- Structurally unreachable OD pairs (graph/data issue)
- Initial corridor frozen failures

Action masking:
- Hard feasibility rule applied at logits level (before softmax).
- Masking is applied during BOTH training and evaluation.
- Audit metric: `masked_probability_mass` tracks invalid action probability.

Environment wrapper (`src/env/cpo_env_wrapper.py`):
- Wraps EventDrivenEnv for PyTorch-CPO compatibility.
- Fixed action_dim = neighbor_k + 1 (NOOP).
- Returns `info['cost']` for CPO constraint learning.

Config keys:
- eval.policy: "cpo"
- eval.model_path: path to trained model.pkl
- eval.cpo.neighbor_k (default 8)

Training:
```bash
python scripts/run_cpo_train.py --config configs/manhattan.yaml \
    --max-iter 500 --max-constraint 10.0
```

Evaluation:
```bash
python scripts/run_eval.py --config configs/manhattan.yaml \
    --policy cpo --model-path reports/cpo_train/run_xxx/best_model.pkl
```

Hyperparameters (aligned with Achiam et al., 2017):
- max_kl = 0.01 (trust region)
- damping = 0.1 (CG numerical stability)
- gamma = 0.99 (reward discount)
- tau = 0.95 (GAE lambda)
- max_constraint = 10-25 (cost limit, calibrate via baseline runs)
- hidden_size = (64, 64)

Citations:
- Joshua Achiam, David Held, Aviv Tamar, Pieter Abbeel.
  "Constrained Policy Optimization." ICML 2017. arXiv:1705.10528
- Alex Ray, Joshua Achiam, Dario Amodei.
  "Benchmarking Safe Exploration in Deep Reinforcement Learning." 2019.
- PyTorch implementation: baselines/PyTorch-CPO (adapted from Khrylx/PyTorch-RL)

---

## MOHITO baseline (policy = "mohito")

MOHITO (Multi-Agent Reinforcement Learning using Hypergraphs for Task-Open Systems) from UAI 2025.

**Important**: This is a **zero-shot cross-domain baseline**. The models are pretrained on the upstream rideshare domain and evaluated on our microtransit domain without retraining. This provides a baseline comparison point but does NOT represent a MOHITO agent trained on our domain.

Implementation summary:
- Multi-vehicle parameter sharing: all vehicles share one actor network.
- Minimal hypergraph adapter: constructs MOHITO-compatible graph from EventDrivenEnv features.
- Heuristic feature mapping: our 5-dim node features mapped to MOHITO's schema.
- Zero exploration: epsilon=0 for deterministic reproducibility.

Graph construction (minimal hypergraph):
- Agent nodes: vehicle state (location, onboard count, accepted count)
- Task nodes: waiting passengers at stops with non-empty queues
- Action nodes: candidate destination stops
- Edge nodes: hyperedge connectors linking agent-task-action

Feature mapping (heuristic, not semantic equivalence):

| MOHITO Field | Our Mapping |
|--------------|-------------|
| node_type | Fixed constant per node category |
| location | Stop index mod grid_size² |
| accepted_count | Waiting queue length at stop |
| riding_count | Onboard passenger count |

Config keys:
- eval.policy: "mohito"
- eval.model_path: path to trained actor.pth
- eval.mohito.feature_len (default 5)
- eval.mohito.num_layers_actor (default 20)
- eval.mohito.hidden_dim (default 50)
- eval.mohito.heads (default 2)
- eval.mohito.grid_size (default 10)
- eval.mohito.epsilon (default 0.0, fixed for deterministic eval)

Evaluation:
```bash
python scripts/run_eval.py --config configs/manhattan.yaml \
    --policy mohito --model-path baselines/mohito-public/rideshare/results/xxx/policy_agent0.pth
```

Upstream code structure (`baselines/mohito-public/rideshare/`):
- `mohitoR/gat.py`: ActorNetwork and CriticNetwork with GAT layers
- `mohitoR/trainer.py`: Training loop with MADDPG-style updates
- `mohitoR/evaluator.py`: Evaluation script
- `rideshare/ride.py`: Rideshare environment

Note on pretrained models:
- You can train your own models using the upstream trainer, or
- Use the provided test episode checkpoints for zero-shot evaluation

Citations:
- Gayathri Anil, Prashant Doshi, Daniel Redder, Adam Eck, Leen-Kiat Soh.
  "MOHITO: Multi-Agent Reinforcement Learning using Hypergraphs for Task-Open Systems."
  UAI 2025. Local PDF: baselines/mohito-public/adresUAI25_cameraready.pdf
- Official code: baselines/mohito-public (submodule)

---

## Wu2024 baseline (policy = "wu2024")

Wu et al. "Multi-Agent Deep Reinforcement Learning based Real-time Planning Approach for Responsive Customized Bus Routes" from Computers & Industrial Engineering 2024.

> [!CAUTION]
> **CRITICAL DECLARATIONS (READ BEFORE USE)**
>
> **1. Weight Status (pretrained=false):**
> The baseline repository contains NO pretrained `.pt` files. This implementation uses `random_init` or `uniform_logits` mode. Results do NOT represent the paper's method performance ceiling.
>
> **2. Domain Transfer:**
> EventDrivenEnv uses dynamic k-hop subgraph; original paper uses fixed station topology. This is heuristic mapping (Strategy S1), not retrained.
>
> **3. Reproducibility:**
> All randomness controlled by seed. Evaluation JSON records: `seed`, `config_hash`, `weights_mode`.

### Implementation Summary

This is an **architecture placeholder baseline** for unified evaluator alignment:

- **Core Architecture**: PointerNetwork with 1D-Conv encoder + GRU + Attention decoder
- **Strategy S1**: Fixed Kmax candidates + padding (constant input dimension)
- **V0 Features**: Minimal closed-loop (coordinates, travel_time, waiting_count, load, time)
- **Multi-Agent**: Parameter sharing (single model for all vehicles)

### Candidate Selection (Strategy S1)

Each decision step:
1. Get candidate actions from env (filtered by action_mask)
2. Sort by travel time (ascending)
3. Take first Kmax candidates; pad with dummy if fewer
4. Apply mask: dummy and infeasible positions = 0

### Config Keys

```yaml
eval:
  policy: "wu2024"
  model_path: ""  # Optional, leave empty for random_init
  wu2024:
    kmax: 64          # Max candidates per decision (default 64)
    weights_mode: "random_init"  # "random_init" or "uniform_logits"
    hidden_size: 128  # GRU hidden size (default 128)
    num_layers: 1     # GRU layers (default 1)
    dropout: 0.1      # Dropout rate (default 0.1)
```

### Weights Mode

| Mode | Description |
|------|-------------|
| `random_init` | Use model forward pass with Xavier-initialized weights |
| `uniform_logits` | Skip model, select first valid action (deterministic) |

### Evaluation

```powershell
python scripts/run_eval.py --config configs/manhattan.yaml --policy wu2024 --episodes 5
```

### Hyperparameters (from paper)

| Parameter | Value | Source |
|-----------|-------|--------|
| hidden_size | 128 | Section 4.2 |
| num_layers | 1 | Section 4.2 |
| dropout | 0.1 | Section 4.2 |
| actor_lr | 5e-4 | Table 2 |
| batch_size | 128 | Table 2 |
| max_load | 40 | Section 4.1 |

### Citations

```bibtex
@article{wu2024multiagent,
  title={Multi-Agent Deep Reinforcement Learning based Real-time Planning 
         Approach for Responsive Customized Bus Routes},
  author={Wu, Xiaoyang and others},
  journal={Computers \& Industrial Engineering},
  year={2024},
  volume={186},
  pages={109764},
  publisher={Elsevier}
}
```

Local PDF: `baselines/transportation_sparse/Multi-Agent Deep Reinforcement Learning.../1-s2.0-S0360835223008641-main.pdf`
