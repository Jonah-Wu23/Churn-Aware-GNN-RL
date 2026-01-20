# EdgeQ Training Protocol

This document defines the training protocol for EdgeQ curriculum learning. It serves as a binding specification that ensures reproducibility and prevents training failures.

## Core Principles

The protocol is built on 6 non-negotiable axioms:

1. **Rho-Gated Transitions**: Stage transitions are controlled by `rho` threshold, not step timeouts
2. **Epsilon Continuity**: Epsilon depends solely on `global_step`, continuous across all boundaries
3. **Reward Stability**: Phase3 rewards never regress to defaults; linear interpolation from phase2
4. **Evaluation-Based Selection**: Best model selected by fixed-seed evaluation, not training extremes
5. **State Continuity**: Training state (replay/optimizer/RNG) is continuous across phases
6. **Audit Trail**: All performance regressions are explainable from logs

---

## Configuration Reference

```yaml
curriculum:
  # Basic
  trigger_rho: 0.5           # Must reach this to transition
  stage_max_steps: 50000     # Base training budget per stage
  stage_min_episodes: 3      # Minimum episodes before transition check
  
  # Rho-Gated Transitions
  rho_window_size: 5         # Episodes to average for transition check
  require_rho_transition: true
  stage_extension_steps: 30000
  max_stage_extensions: 2
  fail_policy: "fail_fast"   # "fail_fast" or "forced"
  rho_warning_threshold: 0.35
  
  # Collapse Protection
  collapse_drop_delta: 0.10
  collapse_min_rho: 0.15
  collapse_patience: 2
  epsilon_cap_on_collapse: 0.3
  
  # Evaluation
  eval_enabled: true
  eval_seeds: [42, 123, 456, 789, 1000]
  eval_interval_steps: 5000
  
  # Reward Ramp
  reward_ramp_steps: 10000
```

---

## Key Mechanisms

### 1. Rho-Gated Transitions

**Transition condition**: `episodes >= min_episodes AND rho_window_mean >= trigger_rho`

- Uses **sliding window mean** (last N episodes), not single-episode values
- Stage continues training if threshold not met
- Extensions allowed up to `max_stage_extensions`
- After exhausting extensions: **FAIL_FAST** (terminate run) or **FORCED** (log warning and proceed)

### 2. Epsilon Continuity

Epsilon is computed from a single global counter:
```python
epsilon = linear_schedule(start=1.0, end=0.05, decay_steps=100000, step=global_step)
```

- `global_step` increments on every env interaction
- Saved in checkpoints and restored on resume
- Never resets between phases or stages

### 3. Reward Ramp (Phase3 Linear Interpolation)

During phase3, reward weights transition smoothly from phase2:
```python
alpha = min(1.0, phase_step / reward_ramp_steps)
weight(t) = (1 - alpha) * W2 + alpha * W3_target
```

- Prevents sudden reward changes that cause policy collapse
- If phase3_overrides is empty, W3_target = W2 (no change)

### 4. Fixed-Seed Evaluation

Model selection uses deterministic evaluation:
- **Seeds**: 5 fixed seeds for reproducibility
- **Epsilon**: 0 (greedy policy)
- **Metric**: `mean_rho` as primary, `service_rate` as tiebreaker
- **Frequency**: Every `eval_interval_steps`

### 5. Collapse Protection

Triggers when:
- `mean_rho < collapse_min_rho` (absolute floor)
- `best_rho - mean_rho > collapse_drop_delta` (relative drop)

Response:
1. Rollback to best checkpoint
2. Set epsilon cap (prevent high exploration)
3. Reduce learning rate

---

## Verification Checklist

### Before AutoDL Upload

- [ ] Unit tests pass: `pytest tests/test_training_protocol.py -v`
- [ ] Config validated: phase3_env_overrides not empty
- [ ] Smoke test completed (local, minimal steps)

### AutoDL 5-Minute Check

After training starts, verify in curriculum_log.jsonl:

1. **Epsilon continuity**: No jumps back to ~1.0 at phase transitions
2. **Reward weights**: phase3 shows same values as phase2 initially
3. **Buffer size**: Not reset to 0 between phases
4. **Global step**: Monotonically increasing

### Success Criteria

- L1/L2 transitions only occur when `rho_window_mean >= trigger_rho`
- Phase3 epsilon continues from phase2 (no reset)
- Final `best_rho` checkpoint achieves `rho >= 0.5` on fixed-seed eval
- Training curve shows no phase3 regression

---

## File Reference

| Component | File |
|-----------|------|
| DQN Trainer | `src/train/dqn.py` |
| Curriculum Runner | `src/train/runner.py` |
| Reward Ramp | `src/train/reward_ramp.py` |
| Evaluation | `src/train/evaluation_checkpointer.py` |
| Replay Buffer | `src/train/replay_buffer.py` |
| Config | `configs/manhattan_curriculum_v13.yaml` |
| Tests | `tests/test_training_protocol.py` |
