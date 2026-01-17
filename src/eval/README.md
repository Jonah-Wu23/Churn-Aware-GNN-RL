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

### Evaluation Modes

| Mode | Command | Description |
|------|---------|-------------|
| **In-Domain Trained** (Recommended) | `--model-path reports/mohito_train/*/mohito_actor_final.pth` | A2C trained from scratch on Manhattan domain |
| Zero-Shot | `--model-path baselines/mohito-public/rideshare/.../policy_agent0.pth` | Pretrained on rideshare, cross-domain inference |

### In-Domain Training (L2 Unified Protocol)

Trains MOHITO from random initialization using the same EventDrivenEnv and reward/mask as other baselines:

```powershell
# Full training (200k steps)
python scripts/run_mohito_train.py --config configs/manhattan.yaml

# Smoke test (1k steps)
python scripts/run_mohito_train.py --config configs/manhattan.yaml --total-steps 1000
```

Training features:
- **Algorithm**: A2C (Actor-Critic) with GAT-based actor + MLP critic
- **Reward**: Same as MAPPO/CPO/EdgeQ (env.step() reward only, no custom shaping)
- **Mask**: Hard mask from env.get_action_mask()
- **Checkpoint schema**: Unified format (model_state_dict, config_hash, git_commit)
- **Convergence**: service_rate stabilizes on eval split

### Evaluation

```powershell
# With trained model
python scripts/run_eval.py --config configs/manhattan.yaml \
    --policy mohito --model-path reports/mohito_train/run_xxx/mohito_actor_final.pth
```

### Config Keys

```yaml
eval:
  mohito:
    feature_len: 5
    num_layers_actor: 20
    hidden_dim: 50
    heads: 2
    grid_size: 10
    epsilon: 0.0

mohito_train:
  total_steps: 200000
  learning_rate: 0.001
  entropy_coef: 0.01
```

### Citations

- Gayathri Anil, Prashant Doshi, Daniel Redder, Adam Eck, Leen-Kiat Soh.
  "MOHITO: Multi-Agent Reinforcement Learning using Hypergraphs for Task-Open Systems."
  UAI 2025. Local PDF: baselines/mohito-public/adresUAI25_cameraready.pdf

---

## Wu2024 baseline (policy = "wu2024")

Wu et al. "Multi-Agent Deep Reinforcement Learning based Real-time Planning Approach for Responsive Customized Bus Routes" from Computers & Industrial Engineering 2024.

### Evaluation Modes

| Mode | Command | Description |
|------|---------|-------------|
| **In-Domain Trained** (Recommended) | `--model-path reports/wu2024_train/*/wu2024_model_final.pt` | A2C trained from scratch on Manhattan domain |
| Random Init | `weights_mode: random_init`, no model_path | Xavier-initialized weights |
| Uniform | `weights_mode: uniform_logits` | Select first valid action |

### In-Domain Training (L2 Unified Protocol)

Trains Wu2024 from random initialization using the same EventDrivenEnv and reward/mask as other baselines:

```powershell
# Full training (200k steps)
python scripts/run_wu2024_train.py --config configs/manhattan.yaml

# Smoke test (1k steps)
python scripts/run_wu2024_train.py --config configs/manhattan.yaml --total-steps 1000
```

Training features:
- **Algorithm**: A2C with PointerNet actor + shared-encoder MLP critic
- **Mask**: Hard logits mask (invalid actions = -inf, entropy only over valid)
- **Reward**: Same as MAPPO/CPO/EdgeQ (env.step() reward only)
- **Checkpoint schema**: Unified format (model_state_dict, config_hash, git_commit)
- **Convergence**: service_rate stabilizes on eval split

### Evaluation

```powershell
# With trained model
python scripts/run_eval.py --config configs/manhattan.yaml \
    --policy wu2024 --model-path reports/wu2024_train/run_xxx/wu2024_model_final.pt
```

### Config Keys

```yaml
eval:
  wu2024:
    kmax: 32
    hidden_size: 128
    weights_mode: "trained"  # "trained", "random_init", or "uniform_logits"

wu2024_train:
  total_steps: 200000
  learning_rate: 0.0005
  entropy_coef: 0.01
```

### Architecture

- **Core**: PointerNetwork with 1D-Conv encoder + GRU + Attention decoder
- **Strategy S1**: Fixed Kmax candidates + padding (constant input dimension)
- **V0 Features**: coordinates, travel_time, waiting_count, load, time

### Citations

```bibtex
@article{wu2024multiagent,
  title={Multi-Agent Deep Reinforcement Learning based Real-time Planning 
         Approach for Responsive Customized Bus Routes},
  author={Wu, Xiaoyang and others},
  journal={Computers \& Industrial Engineering},
  year={2024},
  volume={186},
  pages={109764}
}
```

Local PDF: `baselines/transportation_sparse/.../1-s2.0-S0360835223008641-main.pdf`

