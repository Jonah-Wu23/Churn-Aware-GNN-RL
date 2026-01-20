# Fleet-Aware Edge Potential (FAEP)

## Overview

FAEP is an **observational signal** (state augmentation), NOT a scheduler or coordinator. It captures fleet-level congestion information to mitigate herding effects in large-scale scenarios (50+ vehicles).

> **Key Invariant**: FAEP does not introduce joint actions, centralized critics, or CTDE mechanisms. The system remains fully decentralized execution with a shared policy.

## Design Principles

1. **Pure Observation**: FAEP is injected as the 5th dimension of `edge_features`
2. **Default Off**: `use_fleet_potential: false` preserves all existing experiments
3. **Backward Compatible**: Old checkpoints (edge_dim=4) work when FAEP is disabled

## Indexing Contract

> [!IMPORTANT]
> **Single Source of Truth for Stop Identification**

FAEP uses **Layer-2 stop_id** (same as `gnn_node_id`) as the canonical key for all density computations:

```
density_map: Dict[int, float]  # key = stop_id (NOT node_index)
```

When computing fleet potential for candidate actions:
1. `actions` list contains stop_ids (destination stops)
2. `density_map.get(action_stop_id, 0.0)` retrieves the density
3. No additional index mapping required

## Density Computation: `next_stop` Mode

### Definition

```
C(u) = number of vehicles whose next_target_stop == u
```

### Boundary Cases

| Vehicle State | `next_target_stop` Value |
|---------------|-------------------------|
| In transit to stop X | X (the decided destination) |
| **Waiting for decision at stop Y** | **Y (current_stop)** |
| Just arrived, processing boarding | current_stop |

> [!NOTE]
> Vehicles waiting for decisions are counted at their current stop. This is more conservative and better suppresses herding.

### Normalization (φ function)

Two supported modes:

1. **log1p_norm** (default, recommended):
   ```
   φ(C) = log(1 + C) / log(1 + num_vehicles)
   ```
   - Range: [0, 1]
   - Smooth sensitivity to fleet scale

2. **linear_norm**:
   ```
   φ(C) = C / num_vehicles
   ```
   - Range: [0, 1]
   - Linear proportionality

## Optional: `k_hop` Mode

For k-hop diffusion, density spreads to neighboring stops:

```python
# Precomputed at env initialization
k_hop_neighbors: Dict[int, Set[int]]  # stop_id -> set of k-hop neighbor stop_ids

# At runtime
for vehicle in vehicles:
    target = vehicle.next_target_stop
    for neighbor in k_hop_neighbors[target]:
        density_map[neighbor] += 1.0 / len(k_hop_neighbors[target])
```

> **Performance Note**: k-hop neighborhoods MUST be precomputed during env initialization, not computed per `get_feature_batch()` call.

## Edge Features Schema

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | delta_eta_max | Max ETA increase for onboard passengers |
| 1 | delta_cvar | CVaR change of onboard churn risk |
| 2 | count_violation | Number of hard constraint violations |
| 3 | travel_time | Travel time to destination (seconds) |
| **4** | **fleet_potential** | φ(C(u)) - congestion at destination (FAEP only) |

## Configuration

```yaml
env:
  use_fleet_potential: false      # Default: disabled
  fleet_potential_mode: "next_stop"  # Options: "next_stop", "k_hop"
  fleet_potential_k: 1            # Only used when mode="k_hop"
  fleet_potential_phi: "log1p_norm"  # Options: "log1p_norm", "linear_norm"
```

## Logging Contract

When `use_fleet_potential=true`, `step()` returns `info` with:

```python
info["fleet_density_summary"] = {
    "max": float,           # Maximum density across all stops
    "mean": float,          # Average density
    "top_5_congested_stops": List[Tuple[int, float]]  # [(stop_id, density), ...]
}
```

All values are native Python types (int, float, list, tuple) for JSON serialization.

## Checkpoint Compatibility

| Checkpoint edge_dim | use_fleet_potential | Result |
|---------------------|---------------------|--------|
| 4 | false | ✅ Compatible |
| 4 | true | ❌ Error: "Checkpoint edge_dim=4 incompatible with env edge_dim=5" |
| 5 | false | ❌ Error: dimension mismatch |
| 5 | true | ✅ Compatible |

## Determinism Guarantee

FAEP introduces **no additional randomness**:
- Density is computed purely from current vehicle states
- `event_trace_digest` excludes `info` fields (only events and state transitions)
- Fixed seed produces identical FAEP values across runs

## Paper Narrative

> For large-scale fleet scenarios, we optionally augment edge features with a fleet congestion potential φ(C(u)) that captures how many vehicles are currently targeting each candidate stop. This serves as an observational signal to mitigate herding effects, without altering the decentralized execution architecture or introducing joint action spaces.
