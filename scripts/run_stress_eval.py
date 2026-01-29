"""CLI entrypoint for bait/surge stress tests."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.train.runner import run_stress_tests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    run_dir = Path(args.run_dir) if args.run_dir else None
    metrics_path = run_stress_tests(args.config, run_dir=run_dir)
    logging.info("Stress metrics written to %s", metrics_path)


if __name__ == "__main__":
    main()
