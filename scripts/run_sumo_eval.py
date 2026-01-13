"""CLI entrypoint for Stage 2 SUMO/TraCI validation.

This script runs a trained policy in SUMO simulation and produces metrics
compatible with the Stage 1 evaluator, plus sim-to-real delta statistics.

Usage:
    python scripts/run_sumo_eval.py --config configs/manhattan.yaml --model-path runs/.../edgeq_model_final.pt

The script supports two modes:
1. Simulation mode (default): Uses prior travel times from Layer-2 graph
2. TraCI mode (--use-traci): Connects to actual SUMO simulation

Outputs are written to reports/sumo_eval/<policy>_<timestamp>/:
- sumo_eval_results.json: Full results with metrics and sim-to-real deltas
- sumo_eval_episodes.csv: Per-episode metrics
- sim_to_real_deltas.csv: Travel time delta statistics
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

LOG = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Stage 2 SUMO/TraCI validation for trained policies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with trained Edge-Q model (simulation mode)
  python scripts/run_sumo_eval.py --config configs/manhattan.yaml --model-path runs/train_123/edgeq_model_final.pt

  # Run with TraCI (requires SUMO installation)
  python scripts/run_sumo_eval.py --config configs/manhattan.yaml --model-path model.pt --use-traci

  # Run with greedy baseline
  python scripts/run_sumo_eval.py --config configs/manhattan.yaml --policy greedy

  # Specify output directory
  python scripts/run_sumo_eval.py --config configs/manhattan.yaml --model-path model.pt --run-dir reports/sumo_eval/my_run
        """,
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file (e.g., configs/manhattan.yaml)",
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to trained model checkpoint (.pt file). Required for edgeq policy.",
    )
    
    parser.add_argument(
        "--policy",
        type=str,
        choices=["edgeq", "greedy", "random"],
        default=None,
        help="Policy to evaluate (overrides config). Default: edgeq",
    )
    
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Number of evaluation episodes (overrides config)",
    )
    
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum steps per episode (overrides config)",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)",
    )
    
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Output directory for results. Default: reports/sumo_eval/<policy>_<timestamp>",
    )
    
    parser.add_argument(
        "--use-traci",
        action="store_true",
        help="Use actual SUMO/TraCI connection (requires SUMO installation and sumo_cfg_path in config)",
    )
    
    parser.add_argument(
        "--sumo-gui",
        action="store_true",
        help="Use SUMO GUI for visualization (only with --use-traci)",
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default=None,
        help="Device for model inference (overrides config)",
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    
    return parser.parse_args()


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the script."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)
    
    LOG.info("=" * 60)
    LOG.info("Stage 2 SUMO/TraCI Validation")
    LOG.info("=" * 60)
    
    from src.utils.config import load_config
    from src.sim_sumo.sumo_evaluator import evaluate_sumo
    
    config_path = Path(args.config)
    if not config_path.exists():
        LOG.error("Configuration file not found: %s", config_path)
        return 1
    
    LOG.info("Loading configuration from %s", config_path)
    config = load_config(str(config_path))
    
    if args.policy is not None:
        config.setdefault("eval", {})["policy"] = args.policy
    
    if args.model_path is not None:
        config.setdefault("eval", {})["model_path"] = args.model_path
    
    if args.episodes is not None:
        config.setdefault("eval", {})["episodes"] = args.episodes
    
    if args.max_steps is not None:
        config.setdefault("eval", {})["max_steps"] = args.max_steps
    
    if args.seed is not None:
        config.setdefault("eval", {})["seed"] = args.seed
        config.setdefault("env", {})["seed"] = args.seed
    
    if args.device is not None:
        config.setdefault("eval", {})["device"] = args.device
    
    if args.sumo_gui:
        config.setdefault("sumo", {})["sumo_gui"] = True
    
    policy = config.get("eval", {}).get("policy", "edgeq")
    model_path = config.get("eval", {}).get("model_path")
    
    if policy == "edgeq" and not model_path:
        LOG.error("--model-path is required for edgeq policy")
        LOG.error("Usage: python scripts/run_sumo_eval.py --config ... --model-path ...")
        return 1
    
    if model_path and not Path(model_path).exists():
        LOG.error("Model file not found: %s", model_path)
        return 1
    
    run_dir = Path(args.run_dir) if args.run_dir else None
    
    LOG.info("Configuration:")
    LOG.info("  Policy: %s", policy)
    LOG.info("  Model path: %s", model_path or "(not required)")
    LOG.info("  Episodes: %s", config.get("eval", {}).get("episodes", 5))
    LOG.info("  Max steps: %s", config.get("eval", {}).get("max_steps", "from config"))
    LOG.info("  Use TraCI: %s", args.use_traci)
    LOG.info("  Output dir: %s", run_dir or "(auto-generated)")
    
    if args.use_traci:
        sumo_cfg = config.get("sumo", {}).get("sumo_cfg_path", "")
        if not sumo_cfg:
            LOG.warning("No sumo_cfg_path in config; TraCI mode may fail")
            LOG.warning("Add 'sumo: sumo_cfg_path: path/to/network.sumocfg' to your config")
        
        try:
            from src.sim_sumo.traci_adapter import TRACI_AVAILABLE
            if not TRACI_AVAILABLE:
                LOG.warning("TraCI not available. Install SUMO and set SUMO_HOME environment variable.")
                LOG.warning("Falling back to simulation mode.")
        except ImportError:
            LOG.warning("TraCI import failed, falling back to simulation mode")
    
    try:
        output_path = evaluate_sumo(
            config=config,
            config_path=config_path,
            run_dir=run_dir,
            model_path_override=args.model_path,
            use_traci=args.use_traci,
        )
        
        LOG.info("=" * 60)
        LOG.info("Evaluation complete!")
        LOG.info("Results: %s", output_path)
        LOG.info("=" * 60)
        
        import json
        results = json.loads(output_path.read_text(encoding="utf-8"))
        agg = results.get("aggregate", {})
        
        LOG.info("Summary:")
        LOG.info("  Episodes: %.0f", agg.get("episodes", 0))
        LOG.info("  Service rate: %.3f +/- %.3f", 
                 agg.get("service_rate_mean", 0), agg.get("service_rate_std", 0))
        LOG.info("  Algorithmic churn rate: %.3f +/- %.3f",
                 agg.get("algorithmic_churn_rate_mean", 0), agg.get("algorithmic_churn_rate_std", 0))
        LOG.info("  Wait time P95: %.1f +/- %.1f sec",
                 agg.get("wait_time_p95_sec_mean", 0), agg.get("wait_time_p95_sec_std", 0))
        LOG.info("  Service Gini: %.3f +/- %.3f",
                 agg.get("service_gini_mean", 0), agg.get("service_gini_std", 0))
        
        sim_to_real = results.get("sim_to_real_summary", {})
        if sim_to_real.get("num_episodes", 0) > 0:
            LOG.info("  Sim-to-Real mean delta: %.2f sec (ratio: %.3f)",
                     sim_to_real.get("mean_delta_sec_across_episodes", 0),
                     sim_to_real.get("mean_delta_ratio_across_episodes", 0))
        
        return 0
        
    except FileNotFoundError as e:
        LOG.error("File not found: %s", e)
        return 1
    except ValueError as e:
        LOG.error("Configuration error: %s", e)
        return 1
    except Exception as e:
        LOG.exception("Evaluation failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
