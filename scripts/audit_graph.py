"""Generate Layer-2 graph audit report and visualization."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import load_config
from src.graph.build_logical_graph import write_audit_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Layer-2 graph")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    paths_cfg = cfg.get("paths", {})

    nodes_path = Path(paths_cfg.get("graph_nodes_path", "data/processed/graph/layer2_nodes.parquet"))
    edges_path = Path(paths_cfg.get("graph_edges_path", "data/processed/graph/layer2_edges.parquet"))
    audit_path = Path(paths_cfg.get("graph_audit_path", "reports/audit/graph_build.json"))
    viz_path = Path(paths_cfg.get("graph_viz_path", "reports/audit/graph_build.svg"))

    nodes = pd.read_parquet(nodes_path)
    edges = pd.read_parquet(edges_path)
    write_audit_report(
        out_dir=nodes_path.parent,
        nodes=nodes,
        edges=edges,
        audit_path=audit_path,
        viz_path=viz_path,
        build_params={"source": "audit_graph"},
    )


if __name__ == "__main__":
    main()
