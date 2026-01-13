"""Map OD requests to legal stops and write audit report."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.od_mapping import MappingConfig, map_od_to_stops
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to YAML config file", required=True)
    parser.add_argument("--input-glob")
    parser.add_argument("--stops-path")
    parser.add_argument("--out-dir")
    parser.add_argument("--audit-path")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    cfg = load_config(args.config)
    mapping_cfg = cfg.get("mapping", {})
    graph_cfg = cfg.get("graph", {})
    paths_cfg = cfg.get("paths", {})
    bbox = graph_cfg.get("bbox", {})

    config = MappingConfig(
        euclidean_knn=int(mapping_cfg.get("euclidean_knn", 12)),
        walk_threshold_sec=int(mapping_cfg.get("walk_threshold_sec", 600)),
        soft_assignment_delta_sec=int(mapping_cfg.get("soft_assignment_delta_sec", 30)),
    )

    input_glob = args.input_glob or paths_cfg.get("od_input_glob", "data/processed/nyc_bbox/*.parquet")
    stops_path = args.stops_path or paths_cfg.get("graph_nodes_path", "data/processed/graph/layer2_nodes.parquet")
    out_dir = args.out_dir or paths_cfg.get("od_output_dir", "data/processed/od_mapped")
    audit_path = args.audit_path or paths_cfg.get("od_audit_path", "reports/audit/od_mapping.json")

    stops_df = pd.read_parquet(stops_path)

    input_paths = list(Path().glob(input_glob))
    if not input_paths:
        raise SystemExit(f"No files match {input_glob}")

    audits = []
    for path in input_paths:
        od_df = pd.read_parquet(path)
        out_path = Path(out_dir) / path.name
        _, audit = map_od_to_stops(od_df, stops_df, config, out_path, bbox)
        audit["input_path"] = str(path)
        audit["output_path"] = str(out_path)
        audits.append(audit)

    audit_path = Path(audit_path)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"files": audits}
    audit_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("Wrote audit to %s", audit_path)


if __name__ == "__main__":
    main()
