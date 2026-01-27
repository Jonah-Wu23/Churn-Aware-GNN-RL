from dataclasses import dataclass

from src.eval.evaluator import _compute_metrics


@dataclass
class _Vehicle:
    onboard: list


class _StubEnv:
    def __init__(self) -> None:
        self.requests = [
            {"request_time_sec": 0, "pickup_time_sec": 10},
            {"request_time_sec": 0, "pickup_time_sec": 20},
        ]
        self.structurally_unserviceable = 0
        self.waiting_churned = 0
        self.waiting_timeouts = 0
        self.onboard_churned = 0
        self.served = 2
        self.waiting = {1: [], 2: []}
        self.vehicles = [_Vehicle(onboard=[{"id": 1}]), _Vehicle(onboard=[])]
        self.service_count_by_stop = {1: 1, 2: 1}
        self.stop_ids = [1, 2]


def test_compute_metrics_uses_vehicles_when_no_fleet():
    env = _StubEnv()
    metrics = _compute_metrics(env, total_tacc=0.0)
    assert metrics["onboard_remaining"] == 1.0
