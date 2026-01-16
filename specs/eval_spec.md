# Evaluation Specification

## Core Metrics
- TACC: Total Avoided Private Car Travel Time
- Algorithmic churn rate (excludes structural_unreachable)
- 95th percentile wait time
- Gini coefficient of stop service volume (boardings)

## Fairness Metric Definition

### Stop Service Volume Gini

The Gini coefficient measures inequality in service distribution across stops.

**Input**: `service_count_by_stop` — number of passengers **boarded** at each stop.

**Stop Set**: ALL Layer-2 stops in the logical graph, including zero-service stops. This is critical for reproducible cross-baseline comparability.

**Algorithm**: Relative Mean Difference (RMD), the original Gini definition:

```
G = (Σᵢ Σⱼ |xᵢ - xⱼ|) / (2 * n² * μ)
```

Where n = number of stops, μ = mean service volume.

**Boundary Convention**: When μ=0 (no service), returns 0.0 (no inequality).

**Range**: [0, 1] where 0 = perfect equality, 1 = maximum inequality.

**Implementation**: `src/utils/fairness.py` provides the canonical implementation.

### Distinction: W_fair vs service_gini

| Metric | Phase | Purpose |
|--------|-------|---------|
| **W_fair** | Training | Geographic fairness weight, used to bias dispatch decisions toward peripheral stops (incentive) |
| **service_gini** | Evaluation | Outcome metric measuring actual service distribution inequality (result) |

W_fair ≠ service_gini. W_fair is a training-time guidance signal; service_gini is a post-hoc result metric.

## Structural Unreachability
- Report separately and exclude from algorithmic churn

## Stress Tests
- Surge: demand spike 1.5x
- Bait: center short trips vs edge long trips
- Conflict: onboard passengers vs hotspot demand

## Reporting
- Metrics CSV per run
- JSON summaries under reports/metrics/
- Include config hash and git commit

## TODO
- Specify tolerance bounds for Gym vs SUMO metrics
- Define minimum stress test repetitions per scenario

