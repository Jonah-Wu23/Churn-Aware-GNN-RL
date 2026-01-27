"""CLI entrypoint for unified evaluator."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.evaluator import evaluate
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--policy", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    cfg = load_config(args.config)
    eval_cfg = dict(cfg.get("eval", {}))
    if args.policy:
        eval_cfg["policy"] = args.policy
    if args.model_path:
        model_path = Path(args.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")
        eval_cfg["model_path"] = str(model_path)
    if eval_cfg:
        cfg["eval"] = eval_cfg
    run_dir = Path(args.run_dir) if args.run_dir else None
    output_path = evaluate(cfg, config_path=args.config, run_dir=run_dir)
    logging.info("Eval results written to %s", output_path)


if __name__ == "__main__":
    main()
