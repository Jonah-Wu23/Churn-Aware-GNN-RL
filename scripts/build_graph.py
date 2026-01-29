"""CLI entrypoint for building the logical graph."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.graph.build_logical_graph import build_logical_graph
from src.utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to YAML config file")
    parser.add_argument("--place-name")
    parser.add_argument("--north", type=float)
    parser.add_argument("--south", type=float)
    parser.add_argument("--east", type=float)
    parser.add_argument("--west", type=float)
    parser.add_argument("--cutoff-sec", type=int)
    parser.add_argument("--neighbor-k", type=int)
    parser.add_argument("--out-dir")
    parser.add_argument("--audit-path")
    parser.add_argument("--ensure-strong-connectivity", action="store_true", help="Add bridging edges to ensure strong connectivity")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config) if args.config else {}
    graph_cfg = cfg.get("graph", {})
    paths_cfg = cfg.get("paths", {})

    place_name = args.place_name or graph_cfg.get("place_name")
    cutoff_sec = args.cutoff_sec or graph_cfg.get("cutoff_sec", 900)
    neighbor_k = args.neighbor_k or graph_cfg.get("neighbor_k", 20)
    out_dir = args.out_dir or paths_cfg.get("processed_graph_dir", "data/processed/graph")
    audit_path = args.audit_path or paths_cfg.get("graph_audit_path")
    min_travel_time_sec = graph_cfg.get("min_travel_time_sec", 1.0)
    prune_zero_in = bool(graph_cfg.get("prune_zero_in", False))
    prune_zero_out = bool(graph_cfg.get("prune_zero_out", False))
    ensure_strong_connectivity = args.ensure_strong_connectivity or bool(graph_cfg.get("ensure_strong_connectivity", False))

    bbox = None
    if all(value is not None for value in (args.north, args.south, args.east, args.west)):
        bbox = {
            "north": float(args.north),
            "south": float(args.south),
            "east": float(args.east),
            "west": float(args.west),
        }
    elif isinstance(graph_cfg.get("bbox"), dict):
        cfg_bbox = graph_cfg.get("bbox", {})
        if all(key in cfg_bbox for key in ("north", "south", "east", "west")):
            bbox = {
                "north": float(cfg_bbox["north"]),
                "south": float(cfg_bbox["south"]),
                "east": float(cfg_bbox["east"]),
                "west": float(cfg_bbox["west"]),
            }

    if not place_name and not bbox:
        raise SystemExit("Missing place_name or bbox (args or config)")

    build_logical_graph(
        place_name=place_name,
        bbox=bbox,
        cutoff_sec=int(cutoff_sec),
        neighbor_k=int(neighbor_k),
        out_dir=Path(out_dir),
        audit_path=Path(audit_path) if audit_path else None,
        min_travel_time_sec=float(min_travel_time_sec),
        prune_zero_in=prune_zero_in,
        prune_zero_out=prune_zero_out,
        ensure_strong_connectivity=ensure_strong_connectivity,
    )


if __name__ == "__main__":
    main()
