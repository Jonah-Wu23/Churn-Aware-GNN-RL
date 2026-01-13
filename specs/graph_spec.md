# Graph Specification

## Layer Definitions
- Layer 0: Raw OSM data for physical simulation
- Layer 1: Routing graph (road junctions, drive network) for travel time
- Layer 2: Logical dispatch graph (stop nodes only)

## Layer 2 Construction
- Nodes: stop_id, geometry, optional embeddings
- Edges: stop_id -> stop_id with travel_time_sec
- Build edges via truncated Dijkstra on Layer 1
  - cutoff_sec: default 1200 (20 min)
  - neighbor_k: default 30
  - weight: travel_time
- If no neighbors exist within cutoff, fall back to full Dijkstra
- If still unreachable, attempt undirected routing; finally fall back to haversine estimates
- Enforce min_travel_time_sec (default 1.0) to avoid zero-cost edges
- Optional iterative pruning of zero-in/zero-out nodes (config: prune_zero_in/out)

## Sparsification
- Retain only top K nearest neighbors by travel time
- Ensure directed edges; symmetrize only if explicitly configured

## Outputs
- data/processed/graph/layer2_nodes.parquet
- data/processed/graph/layer2_edges.parquet
- data/processed/graph/stop_map.parquet
- data/processed/graph/node2vec_embeddings.parquet

## Audit Outputs
- reports/audit/graph_build.json
- Include connectivity stats, zero-in/zero-out counts, travel_time stats
- Include fallback counts, zero-travel-time fixes, prune stats
- Visualization saved to reports/audit/graph_build.svg
- Optional audit-only run via scripts/audit_graph.py

## TODO
- Define minimum connectivity thresholds per city
