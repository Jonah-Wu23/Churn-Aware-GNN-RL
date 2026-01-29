"""SUMO/TraCI integration for Stage 2 validation.

This package provides:
- SUMOEnv: SUMO-based environment mirroring Stage 1 interface
- TraCIConnection: TraCI connection management
- TraCISimulationRunner: Full episode runner with TraCI
- SUMOEvaluator: Unified evaluator producing Stage 1-compatible metrics

Usage:
    from src.sim_sumo import SUMOEnv, SUMOEnvConfig, evaluate_sumo
"""

from .sumo_env import SUMOEnv, SUMOEnvConfig, SUMOVehicleState
from .traci_adapter import (
    TraCIConfig,
    TraCIConnection,
    TraCISimulationRunner,
    StopRouteManager,
    SUMOVehicleController,
    run_traci_validation,
    TRACI_AVAILABLE,
)
from .sumo_evaluator import (
    SUMOEvalConfig,
    SUMOEpisodeRunner,
    evaluate_sumo,
)

__all__ = [
    "SUMOEnv",
    "SUMOEnvConfig",
    "SUMOVehicleState",
    "TraCIConfig",
    "TraCIConnection",
    "TraCISimulationRunner",
    "StopRouteManager",
    "SUMOVehicleController",
    "run_traci_validation",
    "TRACI_AVAILABLE",
    "SUMOEvalConfig",
    "SUMOEpisodeRunner",
    "evaluate_sumo",
]
