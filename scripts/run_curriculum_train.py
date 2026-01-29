"""CLI entrypoint for curriculum training."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.train.runner import run_curriculum_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--start-stage", default=None)
    parser.add_argument("--init-model-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    run_dir = Path(args.run_dir) if args.run_dir else None
    log_path = run_curriculum_training(
        args.config,
        run_dir=run_dir,
        start_stage=args.start_stage,
        init_model_path=args.init_model_path,
    )
    logging.info("Curriculum log written to %s", log_path)


if __name__ == "__main__":
    main()
