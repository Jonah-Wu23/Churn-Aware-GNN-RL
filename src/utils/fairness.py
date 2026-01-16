"""
Fairness metrics for stop service volume distribution.

This module provides a standardized Gini coefficient implementation
for measuring service fairness across stops.

Terminology:
- "Stop Service Volume" = number of passengers boarded at each stop
- NOT "Stop Service Rate" (rate implies a denominator which is undefined here)

Design Decisions (approved 2026-01-16):
1. Algorithm: Relative Mean Difference (RMD) - the original Gini definition
2. Stop set: All Layer-2 stops (including zero-service stops)
3. μ=0 convention: Returns 0.0 (no inequality when nothing is served)

Mathematical Definition:
    G = (Σᵢ Σⱼ |xᵢ - xⱼ|) / (2 * n² * μ)
    
    where:
        n = number of stops (full Layer-2 stop set)
        μ = mean of service volumes
        xᵢ = service volume at stop i

Range: [0, 1]
    0 = perfect equality (all stops served equally)
    1 = perfect inequality (all service at one stop)

Note: This metric measures OUTCOME fairness (service distribution).
      It is distinct from W_fair (fairness weight), which is a TRAINING-TIME
      incentive used to bias dispatch decisions toward peripheral stops.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def gini_coefficient(values: List[float]) -> float:
    """
    Compute Gini coefficient using Relative Mean Difference (RMD).
    
    Formula:
        G = (Σᵢ Σⱼ |xᵢ - xⱼ|) / (2 * n² * μ)
    
    Boundary cases:
        - Empty array: returns 0.0
        - Single element: returns 0.0 (no inequality possible)
        - All zeros (μ=0): returns 0.0 (convention: no inequality)
        - All equal non-zero: returns 0.0 (perfect equality)
    
    Properties:
        - Permutation invariant (order of values doesn't matter)
        - Range: [0, 1]
    
    Args:
        values: List of service volumes (must be non-negative).
                Should include ALL Layer-2 stops with zero-fill for unvisited.
    
    Returns:
        Gini coefficient in range [0, 1].
        
    Examples:
        >>> gini_coefficient([10, 10, 10])
        0.0
        >>> gini_coefficient([0, 0, 100])
        0.6666666666666666
        >>> gini_coefficient([1, 2, 3, 4, 5])
        0.26666666666666666
    """
    if not values:
        return 0.0
    
    arr = np.array(values, dtype=np.float64)
    
    # Single element: no inequality possible
    if len(arr) == 1:
        return 0.0
    
    # All zeros (μ=0): convention returns 0.0
    mean = float(np.mean(arr))
    if mean == 0.0:
        return 0.0
    
    # Compute pairwise absolute differences
    # diff_sum = Σᵢ Σⱼ |xᵢ - xⱼ|
    diff_sum = float(np.abs(arr[:, None] - arr[None, :]).sum())
    
    # G = diff_sum / (2 * n² * μ)
    n = len(arr)
    gini = diff_sum / (2.0 * n * n * mean)
    
    return float(gini)


def compute_service_volume_gini(
    service_count_by_stop: Dict[int, int],
    all_stop_ids: List[int],
) -> float:
    """
    Compute Gini coefficient for service volume distribution across ALL stops.
    
    This function ensures proper vector alignment: the service volume vector
    includes ALL Layer-2 stops in a fixed order, with zero-fill for stops
    that were not visited during the episode.
    
    IMPORTANT: This is critical for cross-baseline comparability.
    All baselines (HCRide, MAPPO, CPO, MOHITO, Wu2024) must use the same
    stop set to avoid systematic bias in Gini values.
    
    Args:
        service_count_by_stop: Dict mapping stop_id -> boardings count.
                               May be sparse (missing stops = 0).
        all_stop_ids: Complete list of Layer-2 stop IDs (the reference set).
                      Order doesn't matter due to permutation invariance,
                      but a consistent order ensures reproducibility.
    
    Returns:
        Gini coefficient in range [0, 1].
        
    Example:
        >>> service = {0: 10, 1: 0, 2: 5}
        >>> all_stops = [0, 1, 2, 3]  # stop 3 not in service dict
        >>> compute_service_volume_gini(service, all_stops)
        # values = [10, 0, 5, 0] -> Gini computed
    """
    # Build aligned vector: all stops, zero-fill for missing
    values = [float(service_count_by_stop.get(stop_id, 0)) for stop_id in all_stop_ids]
    return gini_coefficient(values)
