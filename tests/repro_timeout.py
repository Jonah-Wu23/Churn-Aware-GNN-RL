
import sys
import os
import unittest
sys.path.insert(0, os.getcwd())

import numpy as np
from src.env.gym_env import EventDrivenEnv, EnvConfig, VehicleState

class MockEnv(EventDrivenEnv):
    def __init__(self, config: EnvConfig):
        # Skip super init to avoid loading files
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        # Manually init needed structures
        self.stop_ids = [0, 1]
        self.stop_index = {0: 0, 1: 1}
        self.neighbors = {0: [(1, 10.0)], 1: [(0, 10.0)]}
        self.shortest_time_sec = np.array([[0, 10], [10, 0]])
        self.fairness_weight = {0: 1.0, 1: 1.0}
        self.requests = []
        self.reset()
        
    def _load_graph(self):
        pass
        
    def _load_requests(self):
        pass

def test_timeout_clamping():
    # Setup config with 900s timeout
    cfg = EnvConfig(
        max_sim_time_sec=900,
        num_vehicles=1,
        vehicle_capacity=10,
        reward_scale=1.0,
        reward_potential_alpha=0.0
    )
    
    env = MockEnv(cfg)
    
    # Inject a vehicle at stop 0
    env.vehicles = [VehicleState(vehicle_id=0, current_stop=0, available_time=0.0)]
    env.ready_vehicle_ids = [0]
    env.active_vehicle_id = 0
    
    # Inject a mock request that arrives at 900.163
    # This ensures next event is past timeout
    # Use EVENT_ORDER = "Order"
    from src.env.gym_env import EVENT_ORDER
    
    # We bypass `_schedule_initial_events` and directly fill event_queue 
    # OR we can just add to env.requests and check if _advance_until_ready works
    # But reset() clears event_queue.
    
    # Let's manually push an event to event_queue
    # event format: (time, priority, seq, type, payload)
    env.event_queue = []
    env._schedule_event(900.163, EVENT_ORDER, {"request_id": 100})
    
    print(f"Start time: {env.current_time}")
    
    # We want to perform a WAIT action (action = current_stop = 0).
    # travel_time calculation for wait action:
    # if wait_action and event_queue:
    #    next_time = event_queue[0][0]
    #    travel_time = max(0, next_time - current_time)
    
    # Here next_time is 900.163
    # current_time is 0
    # travel_time = 900.163
    # arrival_time = 0 + 900.163 = 900.163
    
    # Logic in step():
    # if max_sim_time_sec is set and arrival_time >= max:
    #    force clamp and done
    
    _, _, done, info = env.step(0)  # Action 0 = Wait at stop 0
    
    print(f"End time: {env.current_time}")
    print(f"Done: {done}")
    print(f"Done reason: {info.get('done_reason')}")
    
    if env.current_time > 900.000001:
        print("FAIL: Time exceeded 900s")
        sys.exit(1)
    elif env.current_time == 900.0:
        print("SUCCESS: Time clamped to 900s")
    else:
        print(f"WARNING: Ended at {env.current_time}")

if __name__ == "__main__":
    test_timeout_clamping()
