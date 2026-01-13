"""Train Node2Vec embeddings for Layer-2 stop graph."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Iterable

import networkx as nx
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import load_config

LOG = logging.getLogger(__name__)


def _import_node2vec() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    local_pkg = repo_root / "node2vec"
    if local_pkg.exists():
        sys.path.insert(0, str(local_pkg))
    try:
        from node2vec import Node2Vec  # noqa: F401
        from gensim.models import Word2Vec  # noqa: F401
        return
    except ImportError as exc:
        if "gensim" in str(exc).lower():
            raise ImportError("gensim is required; install it to use node2vec") from exc
        missing = exc

    try:
        import importlib.metadata as metadata
        from importlib.metadata import PackageNotFoundError

        original_version = metadata.version

        def _version(name: str) -> str:
            if name == "node2vec":
                return "0.0.0"
            return original_version(name)

        metadata.version = _version
        from node2vec import Node2Vec  # noqa: F401
        from gensim.models import Word2Vec  # noqa: F401
    except PackageNotFoundError as exc:
        raise ImportError("node2vec metadata missing; install package or use editable install") from exc
    except ImportError as exc:
        if "gensim" in str(exc).lower():
            raise ImportError("gensim is required; install it to use node2vec") from exc
        raise ImportError("node2vec package not found; check the local clone at node2vec/") from exc


def _edge_weight(travel_time: float, mode: str, epsilon: float) -> float:
    if mode == "inverse_travel_time":
        return float(1.0 / max(travel_time, epsilon))
    if mode == "travel_time":
        return float(travel_time)
    raise ValueError(f"Unknown weight_mode: {mode}")


def _load_graph(nodes_path: Path, edges_path: Path, weight_mode: str) -> tuple[nx.DiGraph, pd.DataFrame]:
    nodes = pd.read_parquet(nodes_path)
    edges = pd.read_parquet(edges_path)

    required_nodes = {"gnn_node_id"}
    required_edges = {"source", "target", "travel_time_sec"}
    if not required_nodes.issubset(nodes.columns):
        raise ValueError("Layer 2 nodes missing gnn_node_id")
    if not required_edges.issubset(edges.columns):
        raise ValueError("Layer 2 edges missing source/target/travel_time_sec")
    if edges["travel_time_sec"].isna().any():
        raise ValueError("Layer 2 edges contain null travel_time_sec values")

    graph = nx.DiGraph()
    node_ids = nodes["gnn_node_id"].astype(int).tolist()
    graph.add_nodes_from(node_ids)
    for src, dst, travel in edges[["source", "target", "travel_time_sec"]].itertuples(
        index=False, name=None
    ):
        weight = _edge_weight(float(travel), weight_mode, epsilon=1e-6)
        graph.add_edge(int(src), int(dst), weight=weight)
    return graph, nodes[["gnn_node_id"]].copy()


def _write_embeddings(
    nodes: pd.DataFrame,
    model,
    out_path: Path,
) -> Dict[str, int]:
    node_ids = nodes["gnn_node_id"].astype(int).tolist()
    dim = int(model.vector_size)
    missing = []
    rows = []
    for node_id in node_ids:
        key = str(int(node_id))
        if key in model.wv:
            vector = model.wv[key].astype(float).tolist()
        else:
            vector = [0.0] * dim
            missing.append(int(node_id))
        row = {"gnn_node_id": int(node_id)}
        row.update({f"emb_geo_{i}": float(vector[i]) for i in range(dim)})
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("gnn_node_id")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return {"missing_nodes": len(missing), "dim": dim}


def _write_audit(audit_path: Path, payload: dict) -> None:
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="ascii")


def train_node2vec(cfg: Dict[str, object]) -> None:
    _import_node2vec()
    from node2vec import Node2Vec

    nodes_path = Path(cfg["graph_nodes_path"])
    edges_path = Path(cfg["graph_edges_path"])
    output_path = Path(cfg["output_path"])
    audit_path = Path(cfg.get("audit_path", "reports/audit/node2vec_embeddings.json"))
    weight_mode = str(cfg.get("weight_mode", "inverse_travel_time"))

    graph, nodes = _load_graph(nodes_path, edges_path, weight_mode)
    node2vec = Node2Vec(
        graph,
        dimensions=int(cfg.get("dimensions", 32)),
        walk_length=int(cfg.get("walk_length", 40)),
        num_walks=int(cfg.get("num_walks", 200)),
        p=float(cfg.get("p", 1.0)),
        q=float(cfg.get("q", 1.0)),
        weight_key="weight",
        workers=int(cfg.get("workers", 1)),
        quiet=bool(cfg.get("quiet", True)),
        seed=int(cfg.get("seed", 7)),
    )
    model = node2vec.fit(
        window=int(cfg.get("window", 10)),
        min_count=int(cfg.get("min_count", 1)),
        batch_words=int(cfg.get("batch_words", 4)),
    )

    stats = _write_embeddings(nodes, model, output_path)
    audit = {
        "graph_nodes_path": str(nodes_path),
        "graph_edges_path": str(edges_path),
        "output_path": str(output_path),
        "dimensions": int(cfg.get("dimensions", 32)),
        "walk_length": int(cfg.get("walk_length", 40)),
        "num_walks": int(cfg.get("num_walks", 200)),
        "p": float(cfg.get("p", 1.0)),
        "q": float(cfg.get("q", 1.0)),
        "weight_mode": weight_mode,
        "seed": int(cfg.get("seed", 7)),
        "missing_nodes": stats["missing_nodes"],
    }
    _write_audit(audit_path, audit)
    LOG.info("Saved embeddings to %s", output_path)
    LOG.info("Saved audit to %s", audit_path)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Node2Vec embeddings")
    parser.add_argument("--config", required=True)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = load_config(args.config)
    embed_cfg = cfg.get("embeddings")
    if not isinstance(embed_cfg, dict):
        raise SystemExit("Missing embeddings config section")

    logging.basicConfig(level=logging.INFO)
    train_node2vec(embed_cfg)


if __name__ == "__main__":
    main()
