#!/usr/bin/env python
"""MAPPO evaluation script for micro-transit dispatch.

This script evaluates a trained MAPPO actor on the micro-transit dispatch
environment using the unified evaluator.

Usage:
    python scripts/run_mappo_eval.py --config configs/manhattan.yaml \
        --model-path reports/mappo_train/run_xxx/models/actor.pt

Reference:
    Chao Yu et al. "The Surprising Effectiveness of PPO in Cooperative 
    Multi-Agent Games." NeurIPS 2022. arXiv:2103.01955
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="MAPPO evaluation for micro-transit")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to trained actor.pt model")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of evaluation episodes")
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Output directory for results")
    parser.add_argument("--neighbor-k", type=int, default=8,
                        help="Number of candidate actions")
    parser.add_argument("--hidden-size", type=int, default=64,
                        help="Actor hidden size (must match training)")
    parser.add_argument("--recurrent-N", type=int, default=1,
                        help="Number of recurrent layers (must match training)")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # Override eval settings for MAPPO
    if "eval" not in config:
        config["eval"] = {}
    config["eval"]["policy"] = "mappo"
    config["eval"]["model_path"] = args.model_path
    config["eval"]["episodes"] = args.episodes
    
    if "mappo" not in config["eval"]:
        config["eval"]["mappo"] = {}
    config["eval"]["mappo"]["neighbor_k"] = args.neighbor_k
    config["eval"]["mappo"]["hidden_size"] = args.hidden_size
    config["eval"]["mappo"]["recurrent_N"] = args.recurrent_N
    
    # Run evaluation
    from src.eval.evaluator import evaluate
    
    run_dir = Path(args.run_dir) if args.run_dir else None
    result_path = evaluate(config, args.config, run_dir)
    
    print(f"Evaluation complete. Results saved to: {result_path}")


if __name__ == "__main__":
    main()
