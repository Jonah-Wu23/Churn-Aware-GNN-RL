"""
Logical dispatch graph builder.

Layer 1 (routing) is used only for shortest path computation.
Layer 2 contains stop nodes only.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd

try:
    import osmnx as ox
except ImportError as exc:  # pragma: no cover
    raise ImportError("osmnx is required for graph building") from exc

LOG = logging.getLogger(__name__)


def _haversine_meters(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    rad = np.pi / 180.0
    dlon = (lon2 - lon1) * rad
    dlat = (lat2 - lat1) * rad
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1 * rad) * np.cos(lat2 * rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return float(6371000.0 * c)


def load_drive_graph(
    place_name: str | None,
    bbox: Dict[str, float] | None,
) -> nx.MultiDiGraph:
    """Download and enrich the drivable routing graph (Layer 1)."""
    if bbox:
        LOG.info(
            "Downloading drive network for bbox north=%s south=%s east=%s west=%s",
            bbox["north"],
            bbox["south"],
            bbox["east"],
            bbox["west"],
        )
        graph = ox.graph_from_bbox(
            (bbox["west"], bbox["south"], bbox["east"], bbox["north"]),
            network_type="drive",
        )
    else:
        LOG.info("Downloading drive network for %s", place_name)
        graph = ox.graph_from_place(place_name, network_type="drive")
    graph = ox.add_edge_speeds(graph)
    graph = ox.add_edge_travel_times(graph)
    return graph


def load_stop_points(place_name: str | None, bbox: Dict[str, float] | None) -> pd.DataFrame:
    """Download stop points from OSM (Layer 0) and return as DataFrame."""
    tags = {"highway": "bus_stop"}
    if bbox:
        gdf = ox.features_from_bbox(
            (bbox["west"], bbox["south"], bbox["east"], bbox["north"]),
            tags,
        )
    else:
        gdf = ox.features_from_place(place_name, tags)
    gdf = gdf[gdf.geom_type == "Point"]
    gdf = gdf.reset_index()
    if "osmid" not in gdf.columns:
        if "id" in gdf.columns:
            gdf["osmid"] = gdf["id"]
        else:
            gdf["osmid"] = gdf.index
    gdf["lon"] = gdf.geometry.x
    gdf["lat"] = gdf.geometry.y
    gdf["gnn_node_id"] = range(len(gdf))
    return gdf


def snap_stops_to_graph(gdf_stops: pd.DataFrame, graph: nx.MultiDiGraph) -> pd.DataFrame:
    """Snap stops to nearest nodes in Layer 1 and keep mapping columns."""
    xs = gdf_stops.geometry.x.values
    ys = gdf_stops.geometry.y.values
    nearest_nodes = ox.distance.nearest_nodes(graph, xs, ys)
    gdf_stops = gdf_stops.copy()
    gdf_stops["phys_node_id"] = nearest_nodes
    return gdf_stops


def build_layer2_edges(
    graph: nx.MultiDiGraph,
    stop_map: Dict[int, int],
    cutoff_sec: int,
    neighbor_k: int,
    stop_coords: Dict[int, Tuple[float, float]],
    fallback_speed_mps: float,
) -> List[Dict[str, float]]:
    """Build Layer 2 edges via truncated Dijkstra on Layer 1."""
    edges: List[Dict[str, float]] = []
    total = len(stop_map)
    fallback_stats = {
        "directed_full": 0,
        "undirected_cutoff": 0,
        "undirected_full": 0,
        "haversine": 0,
    }

    def _neighbors_from_lengths(lengths: Dict[int, float], src_id: int) -> List[Tuple[int, float]]:
        candidates: List[Tuple[int, float]] = []
        for dst_gnn_id, dst_phys_id in stop_map.items():
            if src_id == dst_gnn_id:
                continue
            if dst_phys_id in lengths:
                candidates.append((dst_gnn_id, float(lengths[dst_phys_id])))
        return candidates

    def _haversine_meters(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
        rad = np.pi / 180.0
        dlon = (lon2 - lon1) * rad
        dlat = (lat2 - lat1) * rad
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1 * rad) * np.cos(lat2 * rad) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        return float(6371000.0 * c)

    for i, (src_gnn_id, src_phys_id) in enumerate(stop_map.items()):
        if i % 100 == 0:
            LOG.info("Processing stop %d/%d", i, total)

        lengths = nx.single_source_dijkstra_path_length(
            graph,
            src_phys_id,
            cutoff=cutoff_sec,
            weight="travel_time",
        )
        neighbors = _neighbors_from_lengths(lengths, src_gnn_id)
        if not neighbors:
            LOG.warning(
                "No neighbors within cutoff for stop %s; falling back to full Dijkstra",
                src_gnn_id,
            )
            lengths = nx.single_source_dijkstra_path_length(
                graph,
                src_phys_id,
                weight="travel_time",
            )
            neighbors = _neighbors_from_lengths(lengths, src_gnn_id)
            if neighbors:
                fallback_stats["directed_full"] += 1
            if not neighbors:
                undirected = graph.to_undirected(as_view=True)
                lengths = nx.single_source_dijkstra_path_length(
                    undirected,
                    src_phys_id,
                    cutoff=cutoff_sec,
                    weight="travel_time",
                )
                neighbors = _neighbors_from_lengths(lengths, src_gnn_id)
                if neighbors:
                    fallback_stats["undirected_cutoff"] += 1
                if not neighbors:
                    lengths = nx.single_source_dijkstra_path_length(
                        undirected,
                        src_phys_id,
                        weight="travel_time",
                    )
                    neighbors = _neighbors_from_lengths(lengths, src_gnn_id)
                    if neighbors:
                        fallback_stats["undirected_full"] += 1
                if not neighbors:
                    src_coord = stop_coords.get(int(src_gnn_id))
                    if src_coord is None:
                        LOG.warning("No coordinates for stop %s; skipping edges", src_gnn_id)
                        continue
                    distances: List[Tuple[int, float]] = []
                    for dst_gnn_id, dst_coord in stop_coords.items():
                        if dst_gnn_id == src_gnn_id:
                            continue
                        dist = _haversine_meters(
                            src_coord[0],
                            src_coord[1],
                            dst_coord[0],
                            dst_coord[1],
                        )
                        distances.append((dst_gnn_id, dist))
                    if not distances:
                        LOG.warning("No coordinate neighbors for stop %s; skipping edges", src_gnn_id)
                        continue
                    distances.sort(key=lambda item: item[1])
                    neighbors = [
                        (dst_id, dist / max(fallback_speed_mps, 0.1))
                        for dst_id, dist in distances[:neighbor_k]
                    ]
                    fallback_stats["haversine"] += 1

        neighbors.sort(key=lambda item: item[1])
        neighbors = neighbors[:neighbor_k]

        for dst_gnn_id, travel_time in neighbors:
            edges.append(
                {
                    "source": int(src_gnn_id),
                    "target": int(dst_gnn_id),
                    "travel_time_sec": float(travel_time),
                }
            )

    return edges, fallback_stats


def _fix_zero_travel_times(
    edges: pd.DataFrame,
    stop_coords: Dict[int, Tuple[float, float]],
    fallback_speed_mps: float,
    min_travel_time_sec: float,
) -> Dict[str, int]:
    zero_mask = edges["travel_time_sec"].astype(float) <= 0
    zero_count = int(zero_mask.sum())
    if zero_count == 0:
        return {"zero_travel_time_edges": 0, "fixed_with_coords": 0, "fixed_with_min": 0}

    fixed_coords = 0
    fixed_min = 0
    for idx in edges[zero_mask].index:
        src = int(edges.at[idx, "source"])
        dst = int(edges.at[idx, "target"])
        src_coord = stop_coords.get(src)
        dst_coord = stop_coords.get(dst)
        if src_coord and dst_coord:
            dist = _haversine_meters(src_coord[0], src_coord[1], dst_coord[0], dst_coord[1])
            travel = dist / max(fallback_speed_mps, 0.1)
            travel = max(float(travel), float(min_travel_time_sec))
            edges.at[idx, "travel_time_sec"] = travel
            fixed_coords += 1
        else:
            edges.at[idx, "travel_time_sec"] = float(min_travel_time_sec)
            fixed_min += 1
    return {
        "zero_travel_time_edges": zero_count,
        "fixed_with_coords": fixed_coords,
        "fixed_with_min": fixed_min,
    }


def _compute_scc_stats(
    graph: nx.DiGraph,
    coords: Dict[int, Tuple[float, float]],
) -> Dict[str, object]:
    """Compute strongly connected component statistics."""
    sccs = list(nx.strongly_connected_components(graph))
    if len(sccs) <= 1:
        return {
            "scc_count": len(sccs),
            "scc_sizes": [len(scc) for scc in sccs],
            "sources": [],
            "sinks": [],
            "strongly_connected": len(sccs) == 1,
        }

    condensation = nx.condensation(graph)
    in_deg = dict(condensation.in_degree())
    out_deg = dict(condensation.out_degree())
    sources = [i for i in condensation.nodes if in_deg.get(i, 0) == 0]
    sinks = [i for i in condensation.nodes if out_deg.get(i, 0) == 0]

    scc_sizes = []
    for i in sorted(condensation.nodes):
        members = condensation.nodes[i]["members"]
        scc_sizes.append(len(members))

    return {
        "scc_count": len(sccs),
        "scc_sizes": scc_sizes,
        "sources": sources,
        "sinks": sinks,
        "strongly_connected": False,
    }


def _ensure_strong_connectivity(
    edges_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    coords: Dict[int, Tuple[float, float]],
    fallback_speed_mps: float,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Add minimal bridging edges to make the graph strongly connected.

    Uses SCC condensation to identify sources and sinks, then adds edges
    from sinks back to sources in a cyclic pattern.
    """
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes_df["gnn_node_id"].astype(int).tolist())
    graph.add_edges_from(
        edges_df[["source", "target"]].astype(int).itertuples(index=False, name=None)
    )

    if nx.is_strongly_connected(graph):
        return edges_df, {
            "bridging_edges_added": 0,
            "scc_count_before": 1,
            "scc_count_after": 1,
            "strongly_connected_before": True,
            "strongly_connected_after": True,
        }

    condensation = nx.condensation(graph)
    scc_count_before = condensation.number_of_nodes()
    in_deg = dict(condensation.in_degree())
    out_deg = dict(condensation.out_degree())
    sources = sorted([i for i in condensation.nodes if in_deg.get(i, 0) == 0])
    sinks = sorted([i for i in condensation.nodes if out_deg.get(i, 0) == 0])

    if not sources or not sinks:
        LOG.warning("No sources or sinks found in condensation; skipping SCC stitching")
        return edges_df, {
            "bridging_edges_added": 0,
            "scc_count_before": scc_count_before,
            "scc_count_after": scc_count_before,
            "strongly_connected_before": False,
            "strongly_connected_after": False,
            "error": "no_sources_or_sinks",
        }

    def _scc_centroid(scc_id: int) -> Tuple[float, float]:
        members = condensation.nodes[scc_id]["members"]
        lons = [coords[m][0] for m in members if m in coords]
        lats = [coords[m][1] for m in members if m in coords]
        if not lons:
            return (0.0, 0.0)
        return (float(np.mean(lons)), float(np.mean(lats)))

    def _nearest_node_to_centroid(scc_id: int, centroid: Tuple[float, float]) -> int:
        members = list(condensation.nodes[scc_id]["members"])
        if not members:
            raise ValueError(f"SCC {scc_id} has no members")
        best_node = members[0]
        best_dist = float("inf")
        for node in members:
            if node not in coords:
                continue
            dist = _haversine_meters(coords[node][0], coords[node][1], centroid[0], centroid[1])
            if dist < best_dist:
                best_dist = dist
                best_node = node
        return int(best_node)

    k = max(len(sources), len(sinks))
    while len(sources) < k:
        sources.append(sources[len(sources) % len(sources)])
    while len(sinks) < k:
        sinks.append(sinks[len(sinks) % len(sinks)])

    new_edges = []
    existing_edges = set(
        zip(edges_df["source"].astype(int).tolist(), edges_df["target"].astype(int).tolist())
    )

    for i in range(k):
        sink_id = sinks[i]
        source_id = sources[(i + 1) % len(sources)]
        if sink_id == source_id:
            continue

        sink_centroid = _scc_centroid(sink_id)
        source_centroid = _scc_centroid(source_id)
        sink_node = _nearest_node_to_centroid(sink_id, sink_centroid)
        source_node = _nearest_node_to_centroid(source_id, source_centroid)

        if (sink_node, source_node) in existing_edges:
            continue

        if sink_node in coords and source_node in coords:
            dist = _haversine_meters(
                coords[sink_node][0], coords[sink_node][1],
                coords[source_node][0], coords[source_node][1],
            )
            travel_time = dist / max(fallback_speed_mps, 0.1)
        else:
            travel_time = 300.0

        new_edges.append({
            "source": int(sink_node),
            "target": int(source_node),
            "travel_time_sec": float(travel_time),
        })
        existing_edges.add((sink_node, source_node))
        LOG.info(
            "Added bridging edge: %d -> %d (SCC %d -> SCC %d, %.1f sec)",
            sink_node, source_node, sink_id, source_id, travel_time,
        )

    if new_edges:
        new_edges_df = pd.DataFrame(new_edges)
        edges_df = pd.concat([edges_df, new_edges_df], ignore_index=True)

    graph_after = nx.DiGraph()
    graph_after.add_nodes_from(nodes_df["gnn_node_id"].astype(int).tolist())
    graph_after.add_edges_from(
        edges_df[["source", "target"]].astype(int).itertuples(index=False, name=None)
    )
    strongly_connected_after = nx.is_strongly_connected(graph_after)
    scc_count_after = nx.number_strongly_connected_components(graph_after)

    return edges_df, {
        "bridging_edges_added": len(new_edges),
        "bridging_edges": new_edges,
        "scc_count_before": scc_count_before,
        "scc_count_after": scc_count_after,
        "strongly_connected_before": False,
        "strongly_connected_after": strongly_connected_after,
    }


def _prune_graph(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    prune_zero_in: bool,
    prune_zero_out: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    if not (prune_zero_in or prune_zero_out):
        return nodes, edges, {
            "pruned_nodes": 0,
            "pruned_zero_in": 0,
            "pruned_zero_out": 0,
            "prune_rounds": 0,
        }

    total_removed = set()
    removed_zero_in = 0
    removed_zero_out = 0
    rounds = 0

    while True:
        graph = nx.DiGraph()
        graph.add_nodes_from(nodes["gnn_node_id"].astype(int).tolist())
        graph.add_edges_from(edges[["source", "target"]].astype(int).itertuples(index=False, name=None))

        in_deg = dict(graph.in_degree())
        out_deg = dict(graph.out_degree())
        zero_in = [n for n in graph.nodes if in_deg.get(n, 0) == 0]
        zero_out = [n for n in graph.nodes if out_deg.get(n, 0) == 0]

        to_remove = set()
        if prune_zero_in:
            to_remove.update(zero_in)
        if prune_zero_out:
            to_remove.update(zero_out)

        if not to_remove:
            break

        total_removed.update(to_remove)
        removed_zero_in += int(len([n for n in zero_in if n in to_remove]))
        removed_zero_out += int(len([n for n in zero_out if n in to_remove]))
        keep_nodes = [n for n in graph.nodes if n not in to_remove]
        nodes = nodes[nodes["gnn_node_id"].astype(int).isin(keep_nodes)].copy()
        edges = edges[
            ~edges["source"].astype(int).isin(to_remove)
            & ~edges["target"].astype(int).isin(to_remove)
        ].copy()
        rounds += 1

    return nodes, edges, {
        "pruned_nodes": int(len(total_removed)),
        "pruned_zero_in": int(removed_zero_in),
        "pruned_zero_out": int(removed_zero_out),
        "prune_rounds": int(rounds),
    }


def save_graph_artifacts(
    out_dir: Path,
    gdf_stops: pd.DataFrame,
    edges: List[Dict[str, float]],
) -> None:
    """Save Layer 2 nodes/edges and stop map artifacts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    nodes_path = out_dir / "layer2_nodes.parquet"
    edges_path = out_dir / "layer2_edges.parquet"
    map_path = out_dir / "stop_map.parquet"

    nodes = gdf_stops[["gnn_node_id", "osmid", "lon", "lat", "geometry"]].copy()
    nodes["geometry_wkt"] = nodes["geometry"].astype(str)
    nodes = nodes.drop(columns=["geometry"])
    stop_map = gdf_stops[["gnn_node_id", "phys_node_id"]]

    pd.DataFrame(nodes).to_parquet(nodes_path, index=False)
    pd.DataFrame(edges).to_parquet(edges_path, index=False)
    pd.DataFrame(stop_map).to_parquet(map_path, index=False)

    LOG.info("Saved nodes to %s", nodes_path)
    LOG.info("Saved edges to %s", edges_path)
    LOG.info("Saved stop map to %s", map_path)


def _parse_point_wkt(value: str) -> Tuple[float, float] | None:
    if not value:
        return None
    text = str(value).strip()
    if not text.upper().startswith("POINT"):
        return None
    try:
        inner = text.split("(", 1)[1].split(")")[0].strip()
        parts = inner.replace(",", " ").split()
        if len(parts) < 2:
            return None
        return float(parts[0]), float(parts[1])
    except (ValueError, IndexError):
        return None


def _extract_coords(nodes: pd.DataFrame) -> Dict[int, Tuple[float, float]]:
    coords: Dict[int, Tuple[float, float]] = {}
    if {"lon", "lat"}.issubset(nodes.columns):
        for node_id, lon, lat in nodes[["gnn_node_id", "lon", "lat"]].itertuples(index=False, name=None):
            coords[int(node_id)] = (float(lon), float(lat))
        return coords
    if "geometry_wkt" in nodes.columns:
        for node_id, wkt in nodes[["gnn_node_id", "geometry_wkt"]].itertuples(index=False, name=None):
            parsed = _parse_point_wkt(wkt)
            if parsed is not None:
                coords[int(node_id)] = parsed
    return coords


def _build_svg(
    graph: nx.DiGraph,
    coords: Dict[int, Tuple[float, float]],
    out_path: Path,
    title: str,
) -> None:
    width = 1200
    height = 1200
    padding = 40
    nodes = list(graph.nodes())
    if not coords:
        layout = nx.spring_layout(graph, seed=7)
        coords = {int(node): (float(x), float(y)) for node, (x, y) in layout.items()}
    xs = [coords[node][0] for node in nodes if node in coords]
    ys = [coords[node][1] for node in nodes if node in coords]
    if not xs or not ys:
        raise ValueError("No coordinates available for visualization")

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1e-6)
    span_y = max(max_y - min_y, 1e-6)

    def _scale(point: Tuple[float, float]) -> Tuple[float, float]:
        x, y = point
        sx = padding + (x - min_x) / span_x * (width - 2 * padding)
        sy = padding + (max_y - y) / span_y * (height - 2 * padding)
        return sx, sy

    lines = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='#ffffff'/>",
        f"<text x='{padding}' y='{padding - 12}' font-family='Arial' font-size='18' fill='#111111'>{title}</text>",
        "<g stroke='#4b6b88' stroke-opacity='0.15' stroke-width='0.7'>",
    ]

    for src, dst in graph.edges():
        if src not in coords or dst not in coords:
            continue
        x1, y1 = _scale(coords[int(src)])
        x2, y2 = _scale(coords[int(dst)])
        lines.append(f"<line x1='{x1:.2f}' y1='{y1:.2f}' x2='{x2:.2f}' y2='{y2:.2f}'/>")

    lines.append("</g>")
    lines.append("<g fill='#0a2a43' fill-opacity='0.9'>")
    for node in nodes:
        if node not in coords:
            continue
        x, y = _scale(coords[int(node)])
        lines.append(f"<circle cx='{x:.2f}' cy='{y:.2f}' r='2.2'/>")
    lines.append("</g>")
    lines.append("</svg>")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="ascii")


def write_audit_report(
    out_dir: Path,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    audit_path: Path | None = None,
    viz_path: Path | None = None,
    build_params: Dict[str, object] | None = None,
) -> None:
    """Write audit JSON and SVG visualization for Layer 2 graph."""
    resolved_path = audit_path or Path("reports") / "audit" / "graph_build.json"
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    viz_path = viz_path or Path("reports") / "audit" / "graph_build.svg"

    graph = nx.DiGraph()
    graph.add_nodes_from(nodes["gnn_node_id"].astype(int).tolist())
    graph.add_edges_from(edges[["source", "target"]].astype(int).itertuples(index=False, name=None))

    in_deg = dict(graph.in_degree())
    out_deg = dict(graph.out_degree())
    isolated = [n for n in graph.nodes if in_deg.get(n, 0) == 0 and out_deg.get(n, 0) == 0]
    zero_out = [n for n in graph.nodes if out_deg.get(n, 0) == 0]
    zero_in = [n for n in graph.nodes if in_deg.get(n, 0) == 0]

    travel = edges["travel_time_sec"].astype(float).to_numpy()
    travel_stats = {
        "min": float(np.min(travel)) if len(travel) else 0.0,
        "p50": float(np.percentile(travel, 50)) if len(travel) else 0.0,
        "p95": float(np.percentile(travel, 95)) if len(travel) else 0.0,
        "mean": float(np.mean(travel)) if len(travel) else 0.0,
        "max": float(np.max(travel)) if len(travel) else 0.0,
    }

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(out_dir),
        "node_count": int(graph.number_of_nodes()),
        "edge_count": int(graph.number_of_edges()),
        "isolated_nodes": int(len(isolated)),
        "isolated_node_ids_sample": [int(n) for n in isolated[:20]],
        "zero_out_degree": int(len(zero_out)),
        "zero_out_degree_sample": [int(n) for n in zero_out[:20]],
        "zero_in_degree": int(len(zero_in)),
        "zero_in_degree_sample": [int(n) for n in zero_in[:20]],
        "weakly_connected": bool(nx.is_weakly_connected(graph)) if graph.number_of_nodes() else False,
        "strongly_connected": bool(nx.is_strongly_connected(graph)) if graph.number_of_nodes() else False,
        "weak_components": int(nx.number_weakly_connected_components(graph)) if graph.number_of_nodes() else 0,
        "largest_weak_component": int(
            max((len(c) for c in nx.weakly_connected_components(graph)), default=0)
        ),
        "degree_out": {
            "min": int(min(out_deg.values())) if out_deg else 0,
            "mean": float(np.mean(list(out_deg.values()))) if out_deg else 0.0,
            "max": int(max(out_deg.values())) if out_deg else 0,
        },
        "degree_in": {
            "min": int(min(in_deg.values())) if in_deg else 0,
            "mean": float(np.mean(list(in_deg.values()))) if in_deg else 0.0,
            "max": int(max(in_deg.values())) if in_deg else 0,
        },
        "travel_time_sec": travel_stats,
        "visualization_path": str(viz_path),
        "build_params": build_params or {},
    }
    resolved_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="ascii")
    LOG.info("Wrote audit report to %s", resolved_path)

    coords = _extract_coords(nodes)
    _build_svg(graph, coords, viz_path, title="Layer 2 Graph")
    LOG.info("Wrote visualization to %s", viz_path)


def build_logical_graph(
    place_name: str | None,
    bbox: Dict[str, float] | None,
    cutoff_sec: int,
    neighbor_k: int,
    out_dir: Path,
    audit_path: Path | None = None,
    min_travel_time_sec: float = 1.0,
    prune_zero_in: bool = False,
    prune_zero_out: bool = False,
    ensure_strong_connectivity: bool = False,
) -> None:
    """Entry point for building Layer 2 logical dispatch graph."""
    if not place_name and not bbox:
        raise ValueError("Either place_name or bbox must be provided")

    graph = load_drive_graph(place_name, bbox)
    gdf_stops = load_stop_points(place_name, bbox)
    gdf_stops = snap_stops_to_graph(gdf_stops, graph)
    stop_map = dict(zip(gdf_stops["gnn_node_id"], gdf_stops["phys_node_id"]))

    coords = {
        int(stop_id): (float(point.x), float(point.y))
        for stop_id, point in gdf_stops[["gnn_node_id", "geometry"]].itertuples(index=False, name=None)
    }
    speed_samples = []
    for _, _, data in graph.edges(data=True):
        travel_time = data.get("travel_time")
        length = data.get("length")
        if travel_time and length and travel_time > 0:
            speed_samples.append(float(length) / float(travel_time))
    fallback_speed = float(np.median(speed_samples)) if speed_samples else 10.0

    edges, fallback_stats = build_layer2_edges(
        graph,
        stop_map,
        cutoff_sec,
        neighbor_k,
        coords,
        fallback_speed,
    )
    edges_df = pd.DataFrame(edges)
    zero_fix_stats = _fix_zero_travel_times(edges_df, coords, fallback_speed, min_travel_time_sec)
    nodes_df = gdf_stops[["gnn_node_id", "osmid", "lon", "lat", "geometry"]].copy()
    nodes_df["geometry_wkt"] = nodes_df["geometry"].astype(str)
    nodes_df = nodes_df.drop(columns=["geometry"])
    nodes_df, edges_df, prune_stats = _prune_graph(
        nodes_df,
        edges_df,
        prune_zero_in=prune_zero_in,
        prune_zero_out=prune_zero_out,
    )

    scc_stitch_stats = {}
    if ensure_strong_connectivity:
        edges_df, scc_stitch_stats = _ensure_strong_connectivity(
            edges_df, nodes_df, coords, fallback_speed,
        )

    keep_nodes = set(nodes_df["gnn_node_id"].astype(int).tolist())
    gdf_stops = gdf_stops[gdf_stops["gnn_node_id"].astype(int).isin(keep_nodes)].copy()
    save_graph_artifacts(out_dir, gdf_stops, edges_df.to_dict("records"))
    nodes_path = out_dir / "layer2_nodes.parquet"
    edges_path = out_dir / "layer2_edges.parquet"
    nodes_df = pd.read_parquet(nodes_path)
    edges_df = pd.read_parquet(edges_path)
    build_params = {
        "place_name": place_name,
        "bbox": bbox,
        "cutoff_sec": int(cutoff_sec),
        "neighbor_k": int(neighbor_k),
        "fallback_speed_mps": float(fallback_speed),
        "fallback_counts": fallback_stats,
        "min_travel_time_sec": float(min_travel_time_sec),
        "zero_travel_time_fix": zero_fix_stats,
        "prune_zero_in": bool(prune_zero_in),
        "prune_zero_out": bool(prune_zero_out),
        "prune_stats": prune_stats,
        "ensure_strong_connectivity": bool(ensure_strong_connectivity),
        "scc_stitch_stats": scc_stitch_stats,
    }
    write_audit_report(
        out_dir,
        nodes_df,
        edges_df,
        audit_path=audit_path,
        viz_path=Path("reports") / "audit" / "graph_build.svg",
        build_params=build_params,
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build logical dispatch graph")
    parser.add_argument("--place-name")
    parser.add_argument("--north", type=float)
    parser.add_argument("--south", type=float)
    parser.add_argument("--east", type=float)
    parser.add_argument("--west", type=float)
    parser.add_argument("--cutoff-sec", type=int, default=900)
    parser.add_argument("--neighbor-k", type=int, default=20)
    parser.add_argument("--out-dir", default="data/processed/graph")
    parser.add_argument("--ensure-strong-connectivity", action="store_true", help="Add bridging edges to ensure strong connectivity")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    bbox = None
    if all(value is not None for value in (args.north, args.south, args.east, args.west)):
        bbox = {
            "north": float(args.north),
            "south": float(args.south),
            "east": float(args.east),
            "west": float(args.west),
        }
    build_logical_graph(
        place_name=args.place_name,
        bbox=bbox,
        cutoff_sec=args.cutoff_sec,
        neighbor_k=args.neighbor_k,
        out_dir=Path(args.out_dir),
        ensure_strong_connectivity=args.ensure_strong_connectivity,
    )


if __name__ == "__main__":
    main()
