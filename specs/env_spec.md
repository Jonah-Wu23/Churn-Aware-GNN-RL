# Environment Specification (Event-driven Gym)

## Event Types (Current Implementation)
- Order
- VehicleArrival
- VehicleDeparture
- ChurnCheck

## Step Semantics
- A step occurs at a decision point when a vehicle reaches a stop
- Action is selecting the next stop from the neighbor set
- Time advances to the next event (request arrival or vehicle arrival)
- Vehicles make decisions only when they are ready at a stop

## Reward
- Base service reward per served passenger
- Waiting churn penalty based on logit churn probability
- Onboard churn penalty based on the same logit model
- TACC contribution for successfully served trips
- Fairness penalty via stop weight 1 + gamma * (dist_to_center / max_dist)
- CVaR penalty: mean of churn-probability tail above alpha (e.g., 95th percentile)
- Onboard delay penalty: sum churn-probability equivalents of delay over direct time

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

## Reward Logging (Current Implementation)
- step_served
- step_waiting_churn_prob_sum
- step_waiting_churn_prob_weighted_sum
- step_waiting_churn_cvar
- step_onboard_delay_prob_sum
- step_onboard_churn_prob_sum
- step_tacc_gain

## Action Masking (Hard Constraints)
- Budget commitment at boarding: T_max = alpha * T_direct (per passenger)
- Action is infeasible if any onboard passenger ETA exceeds T_max
- Capacity constraint: if capacity is full and waiting exists at the next stop, action is masked
- Debug mode records which passengers triggered the mask

## Direct Travel Time and Structural Unserviceable
- direct_time_sec is computed via Layer-2 shortest path at load time
- Requests with non-finite direct_time_sec or mapping structural_unreachable are marked structural_unserviceable
- Structural unserviceable requests are excluded from algorithmic churn metrics

## Observation/Feature Batch
- get_feature_batch() returns:
  - node_features: [num_nodes, 5]
  - edge_features: [num_actions, 4]
  - action_mask, actions, action_node_indices
  - graph_edge_index, graph_edge_features (for ECC message passing)
  - current_stop, current_node_index
- get_feature_batch(k_hop) returns a subgraph batch with remapped node indices

## Capacity and Fleet
- num_vehicles controls parallel vehicles, each with independent stop/time/onboard
- vehicle_capacity limits boarding per vehicle at each stop

## Request Lifecycle
- request arrives -> waiting -> onboard -> served or canceled
- waiting cancel reasons: timeout or probabilistic churn
- onboard cancel reason: probabilistic churn based on delay over direct time

## Termination
- Horizon reached
- No active vehicles and no pending requests

## Reproducibility Hooks
- event_trace_digest and event_trace_length are reported at episode end
- mask_debug entries are deterministic for a fixed seed

## TODO
- Define exact churn probability parameters and defaults
