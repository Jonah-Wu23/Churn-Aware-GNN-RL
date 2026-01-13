"""OD to stop mapping using network Voronoi on pedestrian graph."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import networkx as nx
import pandas as pd

try:
    import osmnx as ox
except ImportError as exc:  # pragma: no cover
    raise ImportError("osmnx is required for OD mapping") from exc


@dataclass
class MappingConfig:
    euclidean_knn: int
    walk_threshold_sec: int
    soft_assignment_delta_sec: int
    walk_speed_mps: float = 1.4


def _haversine_meters(lon1: np.ndarray, lat1: np.ndarray, lon2: np.ndarray, lat2: np.ndarray) -> np.ndarray:
    rad = np.pi / 180.0
    dlon = (lon2 - lon1) * rad
    dlat = (lat2 - lat1) * rad
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1 * rad) * np.cos(lat2 * rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371000.0 * c


def _load_pedestrian_graph(bbox: Dict[str, float]) -> nx.MultiDiGraph:
    graph = ox.graph_from_bbox(
        (bbox["west"], bbox["south"], bbox["east"], bbox["north"]),
        network_type="walk",
    )
    return graph


def _prepare_stops(stops_df: pd.DataFrame) -> pd.DataFrame:
    if "lon" not in stops_df.columns or "lat" not in stops_df.columns:
        raise ValueError("stops_df must include lon/lat columns")
    return stops_df[["gnn_node_id", "lon", "lat"]].copy()


def _build_stop_kdtree(stops_df: pd.DataFrame):
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:  # pragma: no cover
        raise ImportError("scipy is required for KD-tree nearest neighbors") from exc
    coords = np.column_stack([stops_df["lon"].to_numpy(), stops_df["lat"].to_numpy()])
    return cKDTree(coords)


def _map_points(
    points: pd.DataFrame,
    lon_col: str,
    lat_col: str,
    stops_df: pd.DataFrame,
    stop_ped_nodes: np.ndarray,
    ped_graph: nx.MultiDiGraph,
    config: MappingConfig,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    xs = points[lon_col].to_numpy()
    ys = points[lat_col].to_numpy()

    stop_tree = _build_stop_kdtree(stops_df)
    _, candidate_idx = stop_tree.query(np.column_stack([xs, ys]), k=config.euclidean_knn)
    if candidate_idx.ndim == 1:
        candidate_idx = candidate_idx[:, None]

    origin_nodes = ox.distance.nearest_nodes(ped_graph, xs, ys)

    best_stop_ids = np.full(len(points), -1, dtype=int)
    best_walk_m = np.full(len(points), np.inf, dtype=float)
    euclid_stop_ids = np.full(len(points), -1, dtype=int)
    euclid_walk_m = np.full(len(points), np.inf, dtype=float)
    euclid_dist_m = np.full(len(points), np.inf, dtype=float)

    for node_id in pd.unique(origin_nodes):
        idxs = np.where(origin_nodes == node_id)[0]
        lengths = nx.single_source_dijkstra_path_length(
            ped_graph,
            node_id,
            cutoff=config.walk_threshold_sec * config.walk_speed_mps * 5,
            weight="length",
        )

        for idx in idxs:
            candidates = candidate_idx[idx]
            candidate_stop_nodes = stop_ped_nodes[candidates]
            candidate_walk = [lengths.get(int(stop_node), np.inf) for stop_node in candidate_stop_nodes]
            best_local = int(np.argmin(candidate_walk))
            best_stop_ids[idx] = int(stops_df.iloc[candidates[best_local]]["gnn_node_id"])
            best_walk_m[idx] = float(candidate_walk[best_local])

            euclid_idx = int(candidates[0])
            euclid_stop_ids[idx] = int(stops_df.iloc[euclid_idx]["gnn_node_id"])
            euclid_stop_node = int(stop_ped_nodes[euclid_idx])
            euclid_walk_m[idx] = float(lengths.get(euclid_stop_node, np.inf))
            euclid_dist_m[idx] = float(
                _haversine_meters(
                    np.array([xs[idx]]),
                    np.array([ys[idx]]),
                    np.array([stops_df.iloc[euclid_idx]["lon"]]),
                    np.array([stops_df.iloc[euclid_idx]["lat"]]),
                )[0]
            )

    best_walk_sec = best_walk_m / config.walk_speed_mps
    euclid_walk_sec = euclid_walk_m / config.walk_speed_mps

    return (
        pd.Series(best_stop_ids),
        pd.Series(best_walk_sec),
        pd.Series(euclid_stop_ids),
        pd.Series(euclid_walk_sec),
        pd.Series(euclid_dist_m),
        pd.Series(euclid_walk_m),
        pd.Series(origin_nodes),
    )


def map_od_to_stops(
    od_df: pd.DataFrame,
    stops_df: pd.DataFrame,
    config: MappingConfig,
    out_path: Path,
    bbox: Dict[str, float],
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Map OD points to legal stops and return mapped frame + audit stats."""
    stops = _prepare_stops(stops_df)
    ped_graph = _load_pedestrian_graph(bbox)
    stop_ped_nodes = ox.distance.nearest_nodes(
        ped_graph,
        stops["lon"].to_numpy(),
        stops["lat"].to_numpy(),
    )

    required_cols = {"pu_lon", "pu_lat", "do_lon", "do_lat"}
    if not required_cols.issubset(set(od_df.columns)):
        raise ValueError("od_df must include pu_lon/pu_lat/do_lon/do_lat columns")

    (
        pu_stop,
        pu_walk_sec,
        pu_euclid_stop,
        pu_euclid_walk_sec,
        pu_euclid_dist_m,
        pu_euclid_walk_m,
        _,
    ) = _map_points(od_df, "pu_lon", "pu_lat", stops, stop_ped_nodes, ped_graph, config)
    (
        do_stop,
        do_walk_sec,
        do_euclid_stop,
        do_euclid_walk_sec,
        do_euclid_dist_m,
        do_euclid_walk_m,
        _,
    ) = _map_points(od_df, "do_lon", "do_lat", stops, stop_ped_nodes, ped_graph, config)

    mapped = od_df.copy()
    mapped["pickup_stop_id"] = pu_stop
    mapped["dropoff_stop_id"] = do_stop
    mapped["pickup_walk_time_sec"] = pu_walk_sec
    mapped["dropoff_walk_time_sec"] = do_walk_sec

    structural_unreachable = (pu_walk_sec > config.walk_threshold_sec) | (
        do_walk_sec > config.walk_threshold_sec
    )
    mapped["structural_unreachable"] = structural_unreachable

    mismatch_rate = float((pu_stop != pu_euclid_stop).mean())
    pickup_barrier = (pu_euclid_dist_m < 50) & (pu_euclid_walk_m > 500)
    dropoff_barrier = (do_euclid_dist_m < 50) & (do_euclid_walk_m > 500)
    barrier_impact_any = pickup_barrier | dropoff_barrier
    structural_unreach_rate = float(structural_unreachable.mean())
    pickup_structural_rate = float((pu_walk_sec > config.walk_threshold_sec).mean())
    dropoff_structural_rate = float((do_walk_sec > config.walk_threshold_sec).mean())

    audit = {
        "rows": int(len(mapped)),
        "mismatch_rate": mismatch_rate,
        "barrier_impact_count": int(barrier_impact_any.sum()),
        "pickup_barrier_impact_count": int(pickup_barrier.sum()),
        "dropoff_barrier_impact_count": int(dropoff_barrier.sum()),
        "barrier_impact_rate": float(barrier_impact_any.mean()),
        "pickup_barrier_impact_rate": float(pickup_barrier.mean()),
        "dropoff_barrier_impact_rate": float(dropoff_barrier.mean()),
        "structural_unreachability_rate": structural_unreach_rate,
        "structural_unreachability_count": int(structural_unreachable.sum()),
        "pickup_structural_unreachability_rate": pickup_structural_rate,
        "dropoff_structural_unreachability_rate": dropoff_structural_rate,
        "pickup_structural_unreachability_count": int((pu_walk_sec > config.walk_threshold_sec).sum()),
        "dropoff_structural_unreachability_count": int((do_walk_sec > config.walk_threshold_sec).sum()),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mapped.to_parquet(out_path, index=False)
    return mapped, audit
