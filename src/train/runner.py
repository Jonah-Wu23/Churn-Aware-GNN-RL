"""Training runner skeleton with curriculum hooks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainConfig:
    seed: int
    max_steps: int


def run_training(config: TrainConfig) -> None:
    """Run training with fixed seeds and structured curriculum."""
    raise NotImplementedError
