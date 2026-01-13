"""CLI entrypoint for SUMO validation."""

from __future__ import annotations

import argparse

from src.utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    _cfg = load_config(args.config)
    raise NotImplementedError


if __name__ == "__main__":
    main()
