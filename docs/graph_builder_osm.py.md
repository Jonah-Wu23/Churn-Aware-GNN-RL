# Graph Builder Notes

This repository builds the logical dispatch graph using `src/graph/build_logical_graph.py`
and the CLI entrypoint `scripts/build_graph.py`.

Current behavior
- Downloads Layer 1 drivable graph and stop points via OSMnx.
- Snaps stops to the nearest drivable nodes.
- Runs truncated Dijkstra to build Layer 2 edges (k-nearest by travel time).
- If no neighbors are found within cutoff, falls back to full Dijkstra and undirected routing.
- Fixes zero travel_time_sec edges using haversine estimates and min_travel_time_sec.
- Optional pruning removes zero-in/zero-out nodes iteratively.
- Writes artifacts to parquet:
  - `data/processed/graph/layer2_nodes.parquet`
  - `data/processed/graph/layer2_edges.parquet`
  - `data/processed/graph/stop_map.parquet`
- Emits audit JSON at `reports/audit/graph_build.json`.
- Emits SVG visualization at `reports/audit/graph_build.svg`.

Configuration
- Use `configs/manhattan.yaml` for bbox/place, cutoff, neighbor_k, and output paths.
- CLI: `python scripts/build_graph.py --config configs/manhattan.yaml`
- Audit-only CLI: `python scripts/audit_graph.py --config configs/manhattan.yaml`
