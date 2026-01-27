"""Hard-mask helpers.

These utilities implement a key safety rule for hard constraints:
if a passenger's hard deadline is already violated even under the *best possible*
route from the current stop, then the hard mask should not collapse the action
space to a singleton. In that case the constraint is unrecoverable and should be
handled by penalties / churn dynamics rather than forbidding all actions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


DEFAULT_MAX_TIME_SEC = 36000.0


def sanitize_time_sec(value: float, max_time_sec: float = DEFAULT_MAX_TIME_SEC) -> float:
    """Clamp travel time to a finite, non-negative number for downstream logic."""
    if value is None:
        return float(max_time_sec)
    v = float(value)
    if not np.isfinite(v):
        return float(max_time_sec)
    return float(max(0.0, min(v, float(max_time_sec))))


def hard_deadline_over_by_sec(
    eta_total_sec: float,
    pickup_time_sec: float,
    t_max_sec: float,
) -> float:
    """Positive value indicates deadline violation."""
    return float(eta_total_sec - pickup_time_sec - t_max_sec)


@dataclass(frozen=True)
class HardMaskGate:
    """Decision about whether a passenger's hard mask is recoverable."""

    enforce: bool
    baseline_over_by_sec: float
    baseline_eta_sec: float


def compute_hard_mask_gate(
    *,
    pickup_time_sec: Optional[float],
    t_max_sec: Optional[float],
    current_time_sec: float,
    best_remaining_sec: float,
    slack_sec: float = 0.0,
    max_time_sec: float = DEFAULT_MAX_TIME_SEC,
    skip_unrecoverable: bool = True,
) -> HardMaskGate:
    """Compute whether to enforce hard mask for a passenger.

    If `skip_unrecoverable` is True and the baseline route from the current stop
    already violates the deadline by more than `slack_sec`, we mark it as
    unrecoverable and return enforce=False.
    """
    if pickup_time_sec is None or t_max_sec is None:
        return HardMaskGate(enforce=True, baseline_over_by_sec=0.0, baseline_eta_sec=float(current_time_sec))

    best_remaining_sec = sanitize_time_sec(best_remaining_sec, max_time_sec=max_time_sec)
    baseline_eta_sec = float(current_time_sec) + float(best_remaining_sec)
    baseline_over_by = hard_deadline_over_by_sec(baseline_eta_sec, float(pickup_time_sec), float(t_max_sec))

    enforce = True
    if skip_unrecoverable and baseline_over_by > float(slack_sec):
        enforce = False

    return HardMaskGate(
        enforce=bool(enforce),
        baseline_over_by_sec=float(baseline_over_by),
        baseline_eta_sec=float(baseline_eta_sec),
    )

