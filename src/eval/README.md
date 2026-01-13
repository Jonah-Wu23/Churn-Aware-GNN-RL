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
