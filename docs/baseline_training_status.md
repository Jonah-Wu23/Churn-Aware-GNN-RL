# Baseline Training Status Report

This document addresses the risk points identified in `todos.md` Section L1 regarding training inconsistency and domain gap impact on fairness metrics.

---

## Executive Summary

| Baseline | Training Status | Weights Source | Service Rate | Fairness Metric Validity |
|----------|----------------|----------------|--------------|-------------------------|
| MAPPO | ✅ In-domain trained | `reports/mappo_train/*/actor.pt` | TBD | Valid |
| CPO | ✅ In-domain trained | `reports/cpo_train/*/best_model.pkl` | TBD | Valid |
| HCRide | ✅ Rule-based | N/A (heuristic) | TBD | Valid |
| MOHITO | ⚠️ Zero-shot cross-domain | `baselines/mohito-public/rideshare/` | TBD | **Potentially Invalid** |
| Wu2024 | ❌ Random init / uniform_logits | None | **0.0%** | **Invalid** |

---

## Detailed Analysis

### MOHITO (UAI 2025)

**Current Training Status**: Zero-shot cross-domain inference

| Property | Value |
|----------|-------|
| Pretrained Domain | Rideshare (ride-hailing) |
| Evaluation Domain | Microtransit (community bus) |
| Weight Source | `baselines/mohito-public/rideshare/results/*/policy_agent0.pth` |
| In-Domain Training | ❌ None |
| Fine-tuning | ❌ None |
| Inference Mode | Deterministic (epsilon=0) |

**Configuration (from `configs/manhattan.yaml`)**:
```yaml
mohito:
  feature_len: 5
  num_layers_actor: 20
  hidden_dim: 50
  heads: 2
  epsilon: 0.0  # Fixed for deterministic evaluation
```

**Domain Gap Analysis**:
- MOHITO was trained on rideshare (task-open, dynamic matching)
- Our task is microtransit (fixed-route, stop-based, capacity-constrained)
- Feature mapping is heuristic, not semantically equivalent

**Risk Level**: ⚠️ Medium
- Model may not have learned microtransit-specific routing patterns
- Zero-shot transfer may result in suboptimal service rates
- Fairness metrics may be unstable if service rate is low

---

### Wu2024 (C&IE 2024)

**Current Training Status**: Random initialization / Uniform logits (no training)

| Property | Value |
|----------|-------|
| Pretrained Weights | ❌ None available (repository contains no `.pt` files) |
| weights_mode | `uniform_logits` (selects first valid action) |
| In-Domain Training | ❌ None |
| Service Rate | **0.0%** (verified on 2026-01-16) |

**Configuration (from `configs/manhattan.yaml`)**:
```yaml
wu2024:
  kmax: 32
  weights_mode: "uniform_logits"
  # Note: weights_mode can be "random_init" or "uniform_logits"
```

**Verified Evaluation Results** (from `reports/eval/wu2024_20260116_215111/eval_results.json`):
```json
{
  "served_mean": 0.0,
  "service_rate_mean": 0.0,
  "waiting_churned_mean": 1.0,
  "service_gini_mean": 0.0
}
```

**Risk Level**: ❌ Critical
- Service rate of 0.0% means **no passengers were served**
- Gini coefficient of 0.0 is **meaningless** (cannot compute fairness for zero service)
- This baseline cannot be fairly compared with in-domain trained baselines

---

## Impact on Fairness Metrics

### Gini Coefficient Validity Threshold

Per the risk mitigation strategy in `docs/免除风险方法.md`:

> **Fairness metrics are only interpretable when service rate exceeds a minimum threshold (recommended: 30%)**

| Baseline | Service Rate | Gini Validity |
|----------|-------------|---------------|
| MOHITO | TBD (need evaluation) | Depends on service rate |
| Wu2024 | 0.0% | ❌ **Invalid** |

### Recommended Service Rate Threshold

```python
SERVICE_RATE_THRESHOLD = 0.30  # 30%

def is_gini_valid(service_rate: float) -> bool:
    """Check if Gini coefficient is interpretable."""
    return service_rate >= SERVICE_RATE_THRESHOLD
```

---

## Acceptance Criteria Verification

### L1.1: Training Inconsistency

> **Acceptance: Document current training status for MOHITO and Wu2024.**

✅ **Met**: This document details training status for both baselines.

> **Acceptance: Identify if they use pre-trained weights, random init, or zero-shot cross-domain inference.**

✅ **Met**:
- MOHITO: Uses pre-trained rideshare weights for zero-shot cross-domain inference
- Wu2024: Uses random initialization (no pretrained weights available)

---

### L1.2: Domain Gap Impact on Fairness

> **Acceptance: Verify that current MOHITO/Wu2024 results show reasonable service rates (not extremely low).**

⚠️ **Partially Met**:
- Wu2024: ❌ Service rate = 0.0% (extremely low)
- MOHITO: Pending evaluation run

> **Acceptance: If service rates are too low, flag this as a risk to fairness metric validity.**

✅ **Met**: Wu2024 is flagged as critical risk (Gini invalid at 0.0% service rate).

---

## Recommendations

### Immediate Actions (Documentation Only)
1. ✅ Created this risk documentation
2. ⏳ Run MOHITO evaluation to obtain service rate data
3. ⏳ Update comparison tables to include fairness validity flags

### Future Actions (Requires Implementation - See L2-L7)
To completely eliminate reviewer concerns, the following is recommended:

1. **In-Domain Training**: Train both MOHITO and Wu2024 from scratch in the Manhattan microtransit environment
2. **Budget Alignment**: Use same training budget as MAPPO/CPO (e.g., 1M decision steps)
3. **Unified Protocol**: Same reward, masks, episode rules across all baselines
4. **Convergence Criterion**: Service rate stabilizes above 30%

> [!IMPORTANT]
> This document addresses **L1 only** (investigation and documentation). Complete risk elimination requires implementing L2-L7 training protocols.

---

## References

- Risk mitigation strategy: [`docs/免除风险方法.md`](./免除风险方法.md)
- MOHITO adapter: [`src/baselines/mohito_adapter.py`](../src/baselines/mohito_adapter.py)
- Wu2024 adapter: [`src/baselines/wu2024_adapter.py`](../src/baselines/wu2024_adapter.py)
- Evaluation README: [`src/eval/README.md`](../src/eval/README.md)
- Wu2024 evaluation results: `reports/eval/wu2024_20260116_215111/eval_results.json`

---

## Change Log

| Date | Author | Change |
|------|--------|--------|
| 2026-01-17 | Auto-generated | Initial version addressing L1 risk points |
