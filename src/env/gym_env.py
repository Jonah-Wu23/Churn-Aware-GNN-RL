"""Event-driven Gym environment skeleton."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import heapq
import json
import hashlib
import logging

import numpy as np
import networkx as nx
import pandas as pd

from src.utils.fairness import gini_coefficient, compute_service_volume_gini
from src.utils.hard_mask import (
    DEFAULT_MAX_TIME_SEC,
    HardMaskGate,
    compute_hard_mask_gate,
    hard_deadline_over_by_sec,
    sanitize_time_sec,
)


def _haversine_meters(lon1: np.ndarray, lat1: np.ndarray, lon2: np.ndarray, lat2: np.ndarray) -> np.ndarray:
    rad = np.pi / 180.0
    dlon = (lon2 - lon1) * rad
    dlat = (lat2 - lat1) * rad
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1 * rad) * np.cos(lat2 * rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371000.0 * c


EVENT_ORDER = "Order"
EVENT_VEHICLE_ARRIVAL = "VehicleArrival"
EVENT_VEHICLE_DEPARTURE = "VehicleDeparture"
EVENT_CHURN_CHECK = "ChurnCheck"

EVENT_PRIORITY = {
    EVENT_ORDER: 0,
    EVENT_VEHICLE_ARRIVAL: 1,
    EVENT_VEHICLE_DEPARTURE: 2,
    EVENT_CHURN_CHECK: 3,
}

LOG = logging.getLogger(__name__)


@dataclass
class EnvConfig:
    max_horizon_steps: int = 200
    max_sim_time_sec: Optional[float] = None
    allow_stop_when_actions_exist: bool = False
    mask_alpha: float = 1.5
    walk_threshold_sec: int = 600
    max_requests: int = 2000
    seed: int = 7
    num_vehicles: int = 1
    vehicle_capacity: int = 6
    request_timeout_sec: int = 600
    realtime_request_rate_per_sec: float = 0.0
    realtime_request_count: int = 0
    realtime_request_end_sec: float = 0.0
    churn_tol_sec: int = 300
    churn_beta: float = 0.02
    waiting_churn_tol_sec: Optional[int] = None
    waiting_churn_beta: Optional[float] = None
    onboard_churn_tol_sec: Optional[int] = None
    onboard_churn_beta: Optional[float] = None
    reward_service: float = 1.0
    reward_service_transform: str = "none"
    reward_service_transform_scale: float = 1.0
    reward_churn_penalty: Optional[float] = None
    reward_waiting_churn_penalty: float = 1.0
    reward_onboard_churn_penalty: float = 1.0
    reward_travel_cost_per_sec: float = 0.0
    reward_tacc_weight: float = 1.0
    reward_tacc_transform: str = "none"
    reward_tacc_transform_scale: float = 1.0
    reward_onboard_delay_weight: float = 0.1
    reward_cvar_penalty: float = 1.0
    reward_fairness_weight: float = 1.0
    reward_terminal_backlog_penalty: float = 0.0  # Normalized terminal backlog penalty
    reward_congestion_penalty: float = 0.0
    reward_scale: float = 1.0
    reward_step_backlog_penalty: float = 0.0  # Per-step penalty per waiting/onboard pax
    reward_waiting_time_penalty_per_sec: float = 0.0  # Per-step penalty per waiting sec
    reward_potential_alpha: float = 0.0
    reward_potential_alpha_source: str = "env_default"
    reward_potential_lost_weight: float = 0.0
    reward_potential_scale_with_reward_scale: bool = True
    debug_abort_on_alert: bool = True
    debug_dump_dir: str = "reports/debug/potential_alerts"
    demand_exhausted_min_time_sec: float = 300.0  # Grace period before demand-exhausted termination
    allow_demand_exhausted_termination: bool = True
    cvar_alpha: float = 0.95
    fairness_gamma: float = 1.0
    travel_time_multiplier: float = 1.0
    debug_mask: bool = False
    od_glob: str = "data/processed/od_mapped/*.parquet"
    graph_nodes_path: str = "data/processed/graph/layer2_nodes.parquet"
    graph_edges_path: str = "data/processed/graph/layer2_edges.parquet"
    graph_embeddings_path: str = "data/processed/graph/node2vec_embeddings.parquet"
    # Time-based train/eval split: "train" uses first ratio, "eval" uses rest, None disables
    time_split_mode: Optional[str] = None  # "train", "eval", or None
    time_split_ratio: float = 0.3  # fraction of time for training (default 30%)
    # Fleet-Aware Edge Potential (FAEP) - default disabled for backward compatibility
    use_fleet_potential: bool = False
    fleet_potential_mode: str = "next_stop"  # Options: "next_stop", "k_hop", "hybrid"
    fleet_potential_k: int = 1  # Only used when mode="k_hop"
    fleet_potential_hybrid_center_weight: float = 0.5
    fleet_potential_hybrid_neighbor_weight: float = 0.5
    fleet_potential_phi: str = "log1p_norm"  # Options: "log1p_norm", "linear_norm"
    # Hard-mask robustness: avoid action-space collapse when deadlines are already missed.
    hard_mask_skip_unrecoverable: bool = False
    hard_mask_slack_sec: float = 0.0

    def __post_init__(self) -> None:
        if self.reward_churn_penalty is not None:
            self.reward_waiting_churn_penalty = float(self.reward_churn_penalty)
        if self.waiting_churn_tol_sec is None:
            self.waiting_churn_tol_sec = int(self.churn_tol_sec)
        if self.waiting_churn_beta is None:
            self.waiting_churn_beta = float(self.churn_beta)
        if self.onboard_churn_tol_sec is None:
            self.onboard_churn_tol_sec = int(self.churn_tol_sec)
        if self.onboard_churn_beta is None:
            self.onboard_churn_beta = float(self.churn_beta)
        if self.travel_time_multiplier <= 0:
            raise ValueError("travel_time_multiplier must be positive")
        if self.max_sim_time_sec is not None and float(self.max_sim_time_sec) <= 0:
            raise ValueError("max_sim_time_sec must be positive when set")
        for field_name in ("reward_service_transform", "reward_tacc_transform"):
            transform = getattr(self, field_name)
            if transform not in {"none", "log1p"}:
                raise ValueError(f"{field_name} must be 'none' or 'log1p'")
        if self.reward_service_transform == "log1p" and self.reward_service_transform_scale <= 0:
            raise ValueError("reward_service_transform_scale must be positive for log1p")
        if self.reward_tacc_transform == "log1p" and self.reward_tacc_transform_scale <= 0:
            raise ValueError("reward_tacc_transform_scale must be positive for log1p")
        if self.reward_potential_lost_weight < 0:
            raise ValueError("reward_potential_lost_weight must be >= 0")
        if self.hard_mask_slack_sec < 0:
            raise ValueError("hard_mask_slack_sec must be >= 0")


@dataclass
class VehicleState:
    vehicle_id: int
    current_stop: int
    available_time: float = 0.0
    onboard: List[dict] = field(default_factory=list)
    visit_counts: Dict[int, int] = field(default_factory=dict)


class EventDrivenEnv:
    """Minimal event-driven environment that can run a basic episode."""

    def __init__(self, config: EnvConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self._load_graph()
        self._precompute_k_hop_neighbors()  # Precompute for k_hop mode
        self._load_requests()
        self.reset()

    def seed(self, seed: int) -> None:
        """Reset RNG seed for deterministic evaluation runs."""
        self.config.seed = int(seed)
        self.rng = np.random.default_rng(self.config.seed)

    def _load_graph(self) -> None:
        nodes = pd.read_parquet(self.config.graph_nodes_path)
        edges = pd.read_parquet(self.config.graph_edges_path)
        self._validate_layer2_graph(nodes, edges)
        self._load_embeddings(nodes)
        edges = edges.copy()
        edges["source"] = edges["source"].astype(int)
        edges["target"] = edges["target"].astype(int)
        edges["travel_time_sec"] = edges["travel_time_sec"].astype(float)
        if self.config.travel_time_multiplier != 1.0:
            edges["travel_time_sec"] = edges["travel_time_sec"] * float(self.config.travel_time_multiplier)
        self.stop_ids = nodes["gnn_node_id"].astype(int).tolist()
        if "lon" in nodes.columns and "lat" in nodes.columns:
            self.stop_coords = {
                int(stop_id): (float(lon), float(lat))
                for stop_id, lon, lat in nodes[["gnn_node_id", "lon", "lat"]].itertuples(index=False, name=None)
            }
        else:
            self.stop_coords = {int(stop_id): (0.0, 0.0) for stop_id in self.stop_ids}
        self.stop_index = {int(stop_id): idx for idx, stop_id in enumerate(self.stop_ids)}
        src_idx = edges["source"].map(self.stop_index).astype(int).to_numpy()
        dst_idx = edges["target"].map(self.stop_index).astype(int).to_numpy()
        self.graph_edge_index = np.stack([src_idx, dst_idx], axis=0).astype(np.int64)
        edge_dim = 5 if self.config.use_fleet_potential else 4
        graph_edge_features = np.zeros((len(edges), edge_dim), dtype=np.float32)
        graph_edge_features[:, 3] = edges["travel_time_sec"].to_numpy(dtype=np.float32)
        self.graph_edge_features = graph_edge_features
        if "lon" in nodes.columns and "lat" in nodes.columns:
            center_lon = float(nodes["lon"].mean())
            center_lat = float(nodes["lat"].mean())
            dists = _haversine_meters(
                nodes["lon"].to_numpy(),
                nodes["lat"].to_numpy(),
                np.full(len(nodes), center_lon),
                np.full(len(nodes), center_lat),
            )
            max_dist = float(dists.max() if len(dists) else 1.0)
            weights = 1.0 + self.config.fairness_gamma * (dists / max_dist)
            self.fairness_weight = {
                int(stop_id): float(weight)
                for stop_id, weight in zip(nodes["gnn_node_id"].tolist(), weights)
            }
        else:
            self.fairness_weight = {int(stop_id): 1.0 for stop_id in self.stop_ids}
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(self.stop_ids)
        self.graph.add_weighted_edges_from(
            edges[["source", "target", "travel_time_sec"]].itertuples(index=False, name=None),
            weight="travel_time_sec",
        )
        self._build_shortest_time_cache()
        self.neighbors: Dict[int, List[Tuple[int, float]]] = {}
        for src, dst, travel in edges.itertuples(index=False, name=None):
            self.neighbors.setdefault(int(src), []).append((int(dst), float(travel)))
        adj: Dict[int, set[int]] = {int(stop_id): set() for stop_id in self.stop_ids}
        for src, dst, _travel in edges.itertuples(index=False, name=None):
            adj[int(src)].add(int(dst))
            adj[int(dst)].add(int(src))
        self.adj_undirected = {key: sorted(list(value)) for key, value in adj.items()}
        isolated = [stop_id for stop_id in self.stop_ids if stop_id not in self.neighbors]
        if isolated:
            LOG.warning("Layer 2 graph has %d stop(s) with no outgoing edges", len(isolated))

    def _precompute_k_hop_neighbors(self) -> None:
        """Precompute k-hop neighborhood for each stop (used by k_hop mode).
        
        This is called once during __init__ to avoid BFS in get_feature_batch.
        """
        k = max(1, self.config.fleet_potential_k)
        self._k_hop_neighbor_cache: Dict[int, List[int]] = {}
        for stop_id in self.stop_ids:
            self._k_hop_neighbor_cache[int(stop_id)] = self._k_hop_nodes(int(stop_id), k)

    def _build_shortest_time_cache(self) -> None:
        n = len(self.stop_ids)
        shortest = np.full((n, n), np.inf, dtype=np.float32)
        np.fill_diagonal(shortest, 0.0)
        for src_stop_id in self.stop_ids:
            src_idx = self.stop_index[int(src_stop_id)]
            lengths = nx.single_source_dijkstra_path_length(self.graph, int(src_stop_id), weight="travel_time_sec")
            for dst_stop_id, dist in lengths.items():
                dst_idx = self.stop_index.get(int(dst_stop_id))
                if dst_idx is None:
                    continue
                shortest[src_idx, dst_idx] = float(dist)
        self.shortest_time_sec = shortest

    def _apply_positive_reward_transform(self, value: float, transform: str, scale: float) -> float:
        if transform == "none":
            return float(value)
        if value <= 0:
            return float(value)
        if transform == "log1p":
            scale = max(float(scale), 1e-6)
            return float(scale * np.log1p(value / scale))
        raise ValueError(f"Unknown reward transform: {transform}")

    def _compute_waiting_remaining(self) -> float:
        return float(sum(len(q) for q in self.waiting.values()))

    def _compute_onboard_remaining(self) -> float:
        return float(sum(len(v.onboard) for v in self.vehicles))

    def _compute_backlog_scalar(self) -> float:
        return self._compute_waiting_remaining() + self._compute_onboard_remaining()

    def _compute_lost_total(self) -> float:
        return float(
            self.waiting_churned
            + self.onboard_churned
            + self.structurally_unserviceable
        )

    def _compute_phi(self, backlog_scalar: float) -> float:
        lost_total = self._compute_lost_total()
        return float(backlog_scalar + self.config.reward_potential_lost_weight * lost_total)

    def _snapshot_potential(self) -> Dict[str, float]:
        waiting_remaining = self._compute_waiting_remaining()
        onboard_remaining = self._compute_onboard_remaining()
        backlog_scalar = waiting_remaining + onboard_remaining
        lost_total = self._compute_lost_total()
        phi = self._compute_phi(backlog_scalar)
        return {
            "waiting_remaining": float(waiting_remaining),
            "onboard_remaining": float(onboard_remaining),
            "phi_backlog": float(backlog_scalar),
            "phi": float(phi),
            "lost_total": float(lost_total),
            "waiting_churned": float(self.waiting_churned),
            "onboard_churned": float(self.onboard_churned),
            "structural_unserviceable": float(self.structurally_unserviceable),
        }

    def get_potential_debug(self) -> Dict[str, float]:
        """Return a consistent potential snapshot for debug/fallback paths."""
        snap = self._snapshot_potential()
        return {
            "reward_potential_alpha": float(self.config.reward_potential_alpha),
            "reward_potential_alpha_source": str(self.config.reward_potential_alpha_source),
            "reward_potential_lost_weight": float(self.config.reward_potential_lost_weight),
            "reward_potential_scale_with_reward_scale": bool(self.config.reward_potential_scale_with_reward_scale),
            "phi_before": float(snap["phi"]),
            "phi_after": float(snap["phi"]),
            "phi_delta": 0.0,
            "phi_backlog_before": float(snap["phi_backlog"]),
            "phi_backlog_after": float(snap["phi_backlog"]),
            "lost_total_before": float(snap["lost_total"]),
            "lost_total_after": float(snap["lost_total"]),
            "waiting_churned_before": float(snap["waiting_churned"]),
            "waiting_churned_after": float(snap["waiting_churned"]),
            "onboard_churned_before": float(snap["onboard_churned"]),
            "onboard_churned_after": float(snap["onboard_churned"]),
            "structural_unserviceable_before": float(snap["structural_unserviceable"]),
            "structural_unserviceable_after": float(snap["structural_unserviceable"]),
            "waiting_remaining_before": float(snap["waiting_remaining"]),
            "waiting_remaining_after": float(snap["waiting_remaining"]),
            "onboard_remaining_before": float(snap["onboard_remaining"]),
            "onboard_remaining_after": float(snap["onboard_remaining"]),
            "reward_potential_shaping": 0.0,
            "reward_potential_shaping_raw": 0.0,
        }

    @staticmethod
    def compute_potential_shaping(
        reward_potential_alpha: float,
        phi_before: float,
        phi_after: float,
        reward_scale: float,
        scale_with_reward_scale: bool,
    ) -> Dict[str, float]:
        phi_delta = float(phi_before - phi_after)
        shaping_raw = float(reward_potential_alpha) * phi_delta
        shaping_scaled = float(shaping_raw) * float(reward_scale) if scale_with_reward_scale else float(shaping_raw)
        return {
            "phi_delta": phi_delta,
            "reward_potential_shaping_raw": shaping_raw,
            "reward_potential_shaping": shaping_scaled,
        }

    def _k_hop_nodes(self, center: int, k: int) -> List[int]:
        if k <= 0:
            return [int(center)]
        visited = {int(center)}
        frontier = {int(center)}
        for _ in range(int(k)):
            next_frontier: set[int] = set()
            for node in frontier:
                next_frontier.update(self.adj_undirected.get(int(node), []))
            next_frontier -= visited
            visited |= next_frontier
            frontier = next_frontier
            if not frontier:
                break
        return sorted(visited)

    def _load_embeddings(self, nodes: pd.DataFrame) -> None:
        emb_cols = [col for col in nodes.columns if col.startswith("emb_geo_")]
        if emb_cols:
            emb_df = nodes[["gnn_node_id"] + emb_cols].copy()
        else:
            emb_path = Path(self.config.graph_embeddings_path)
            if not emb_path.exists():
                raise FileNotFoundError(f"Missing embeddings file: {emb_path}")
            emb_df = pd.read_parquet(emb_path)
            if "gnn_node_id" not in emb_df.columns:
                raise ValueError("Embeddings file missing gnn_node_id column")
            emb_cols = [col for col in emb_df.columns if col.startswith("emb_geo_")]
            if not emb_cols:
                raise ValueError("Embeddings file missing emb_geo_* columns")
            emb_df = emb_df[["gnn_node_id"] + emb_cols].copy()

        if emb_df["gnn_node_id"].duplicated().any():
            raise ValueError("Embeddings contain duplicate gnn_node_id values")
        if emb_df["gnn_node_id"].isna().any():
            raise ValueError("Embeddings contain null gnn_node_id values")
        if emb_df[emb_cols].isna().any().any():
            raise ValueError("Embeddings contain null emb_geo values")

        node_ids = set(nodes["gnn_node_id"].astype(int).tolist())
        emb_ids = set(emb_df["gnn_node_id"].astype(int).tolist())
        missing = node_ids - emb_ids
        extra = emb_ids - node_ids
        if missing:
            raise ValueError("Embeddings missing node ids from Layer 2")
        if extra:
            raise ValueError("Embeddings include unknown node ids")

        emb_df = emb_df.set_index("gnn_node_id").loc[sorted(node_ids)]
        emb_matrix = emb_df[emb_cols].to_numpy(dtype=float)
        if emb_matrix.shape[1] < 1:
            raise ValueError("Embeddings must include at least one dimension")
        if not np.isfinite(emb_matrix).all():
            raise ValueError("Embeddings contain non-finite values")

        emb_scalar = emb_matrix[:, 0]
        self.geo_embedding_scalar = {
            int(node_id): float(value)
            for node_id, value in zip(sorted(node_ids), emb_scalar)
        }

    def _validate_layer2_graph(self, nodes: pd.DataFrame, edges: pd.DataFrame) -> None:
        required_nodes = {"gnn_node_id"}
        missing_nodes = required_nodes - set(nodes.columns)
        if missing_nodes:
            raise ValueError(f"Layer 2 nodes missing columns: {sorted(missing_nodes)}")

        required_edges = {"source", "target", "travel_time_sec"}
        missing_edges = required_edges - set(edges.columns)
        if missing_edges:
            raise ValueError(f"Layer 2 edges missing columns: {sorted(missing_edges)}")

        if nodes["gnn_node_id"].isna().any():
            raise ValueError("Layer 2 nodes contain null gnn_node_id values")
        if nodes["gnn_node_id"].duplicated().any():
            raise ValueError("Layer 2 nodes contain duplicate gnn_node_id values")

        for col in ("source", "target", "travel_time_sec"):
            if edges[col].isna().any():
                raise ValueError(f"Layer 2 edges contain null {col} values")

        src = pd.to_numeric(edges["source"], errors="coerce")
        dst = pd.to_numeric(edges["target"], errors="coerce")
        travel = pd.to_numeric(edges["travel_time_sec"], errors="coerce")
        if not np.all(np.isfinite(src)) or not np.all(np.isfinite(dst)):
            raise ValueError("Layer 2 edges contain non-numeric source/target values")
        if not np.all(np.isfinite(travel)):
            raise ValueError("Layer 2 edges contain non-finite travel_time_sec values")
        if (travel < 0).any():
            raise ValueError("Layer 2 edges contain negative travel_time_sec values")

        node_ids = set(nodes["gnn_node_id"].astype(int).tolist())
        invalid_src = set(src.astype(int).tolist()) - node_ids
        invalid_dst = set(dst.astype(int).tolist()) - node_ids
        if invalid_src or invalid_dst:
            raise ValueError("Layer 2 edges reference unknown stop ids")

    def _load_requests(self) -> None:
        glob_pattern = self.config.od_glob
        if Path(glob_pattern).is_absolute():
            paths = [Path(glob_pattern)]
        else:
            paths = list(Path().glob(glob_pattern))
        if not paths:
            raise FileNotFoundError(f"No OD files match {self.config.od_glob}")
        frames = [pd.read_parquet(path) for path in paths]
        od = pd.concat(frames, ignore_index=True)
        required = {"pickup_stop_id", "dropoff_stop_id", "tpep_pickup_datetime"}
        if not required.issubset(set(od.columns)):
            raise ValueError("OD data must include pickup_stop_id/dropoff_stop_id/tpep_pickup_datetime")
        od = od.sort_values("tpep_pickup_datetime").reset_index(drop=True)
        
        # Time-based train/eval split
        if self.config.time_split_mode is not None:
            t_min = od["tpep_pickup_datetime"].min()
            t_max = od["tpep_pickup_datetime"].max()
            duration = (t_max - t_min).total_seconds()
            cutoff_time = t_min + pd.Timedelta(seconds=duration * self.config.time_split_ratio)
            
            if self.config.time_split_mode == "train":
                od = od[od["tpep_pickup_datetime"] <= cutoff_time].copy()
                LOG.info(f"Time split [train]: using {len(od):,} records before {cutoff_time}")
            elif self.config.time_split_mode == "eval":
                od = od[od["tpep_pickup_datetime"] > cutoff_time].copy()
                LOG.info(f"Time split [eval]: using {len(od):,} records after {cutoff_time}")
            else:
                raise ValueError(f"Invalid time_split_mode: {self.config.time_split_mode}")
            od = od.reset_index(drop=True)
        
        if self.config.max_requests:
            od = od.iloc[: self.config.max_requests].copy()
        t0 = od["tpep_pickup_datetime"].iloc[0]
        od["request_time_sec"] = (od["tpep_pickup_datetime"] - t0).dt.total_seconds()
        scheduled = od[
            ["pickup_stop_id", "dropoff_stop_id", "request_time_sec"]
        ].to_dict("records")
        for idx, req in enumerate(scheduled):
            req["request_id"] = int(idx)
            req["source"] = "scheduled"
            req["structural_unreachable"] = bool(od.iloc[idx].get("structural_unreachable", False))

        realtime = self._build_realtime_requests(
            base_requests=scheduled,
            start_id=len(scheduled),
        )
        combined = scheduled + realtime
        combined.sort(key=lambda item: (item["request_time_sec"], item["request_id"]))
        self.requests = []
        self.request_index: Dict[int, dict] = {}
        for req in combined:
            req = dict(req)
            req["pickup_stop_id"] = int(req["pickup_stop_id"])
            req["dropoff_stop_id"] = int(req["dropoff_stop_id"])
            req["direct_time_sec"] = self._shortest_time(req["pickup_stop_id"], req["dropoff_stop_id"])
            req["structural_unserviceable"] = bool(
                (not np.isfinite(req["direct_time_sec"])) or req.get("structural_unreachable", False)
            )
            req["status"] = None
            self.requests.append(req)
            self.request_index[int(req["request_id"])] = req

    def _build_realtime_requests(self, base_requests: List[dict], start_id: int) -> List[dict]:
        if self.config.realtime_request_rate_per_sec <= 0 or self.config.realtime_request_count <= 0:
            return []
        if base_requests:
            end_time = base_requests[-1]["request_time_sec"]
        else:
            end_time = float(self.config.max_horizon_steps)
        if self.config.realtime_request_end_sec > 0:
            end_time = float(self.config.realtime_request_end_sec)

        realtime: List[dict] = []
        current_time = 0.0
        req_id = start_id
        rate = float(self.config.realtime_request_rate_per_sec)
        while len(realtime) < self.config.realtime_request_count and current_time <= end_time:
            delta = float(self.rng.exponential(1.0 / rate))
            current_time += delta
            if current_time > end_time:
                break
            if base_requests:
                base = base_requests[int(self.rng.integers(0, len(base_requests)))]
                pickup = base["pickup_stop_id"]
                dropoff = base["dropoff_stop_id"]
            else:
                pickup = int(self.rng.choice(self.stop_ids))
                dropoff = int(self.rng.choice(self.stop_ids))
                if pickup == dropoff and len(self.stop_ids) > 1:
                    while dropoff == pickup:
                        dropoff = int(self.rng.choice(self.stop_ids))
            realtime.append(
                {
                    "request_id": int(req_id),
                    "pickup_stop_id": int(pickup),
                    "dropoff_stop_id": int(dropoff),
                    "request_time_sec": float(current_time),
                    "source": "realtime",
                    "structural_unreachable": False,
                }
            )
            req_id += 1
        return realtime

    def reset(self) -> Dict[str, float]:
        self.current_time = 0.0
        if self.config.num_vehicles < 1:
            raise ValueError("num_vehicles must be >= 1")
        if self.config.vehicle_capacity < 1:
            raise ValueError("vehicle_capacity must be >= 1")
        self.vehicles: List[VehicleState] = [
            VehicleState(vehicle_id=idx, current_stop=int(self.rng.choice(self.stop_ids)))
            for idx in range(self.config.num_vehicles)
        ]
        self._last_fleet_density_map: Dict[int, float] = {}
        self._fleet_density_max_history: List[float] = []
        self._visited_stops = {int(vehicle.current_stop) for vehicle in self.vehicles}
        self.event_queue: List[Tuple[float, int, int, str, dict]] = []
        self.event_seq = 0
        self.event_log: List[dict] = []
        self.state_log: List[dict] = []
        self.ready_vehicle_ids: List[int] = []
        self.active_vehicle_id: int | None = None
        self.waiting: Dict[int, List[dict]] = {stop_id: [] for stop_id in self.stop_ids}
        self.done = False
        self.steps = 0
        self.served = 0
        self.waiting_churned = 0
        self.waiting_timeouts = 0
        self.onboard_churned = 0
        self.structurally_unserviceable = 0
        self.service_count_by_stop = {int(stop_id): 0 for stop_id in self.stop_ids}
        self.dropoff_count_by_stop = {int(stop_id): 0 for stop_id in self.stop_ids}
        self.acc_wait_time_by_stop = {int(stop_id): 0.0 for stop_id in self.stop_ids}
        self.canceled_requests: List[dict] = []
        self.last_mask_debug: List[dict] = []
        self._step_stats = self._init_step_stats()
        for req in self.requests:
            req["status"] = None
            req["pickup_time_sec"] = None
            req.pop("cancel_reason", None)
            req.pop("t_max_sec", None)
        self._schedule_initial_events()
        self._prepare_initial_arrivals()
        self._step_stats = self._init_step_stats()
        self._advance_until_ready()
        return self._observe()

    def _observe(self) -> Dict[str, float]:
        active_vehicle = self._get_active_vehicle()
        return {
            "time_sec": self.current_time,
            "active_vehicle_id": float(self.active_vehicle_id) if self.active_vehicle_id is not None else -1.0,
            "current_stop": float(active_vehicle.current_stop if active_vehicle else -1),
            "onboard": float(len(active_vehicle.onboard) if active_vehicle else 0),
            "waiting": float(sum(len(v) for v in self.waiting.values())),
            "ready_vehicles": float(len(self.ready_vehicle_ids)),
        }

    @property
    def current_stop(self) -> int:
        vehicle = self._get_active_vehicle()
        if vehicle is None and self.vehicles:
            vehicle = self.vehicles[0]
        return int(vehicle.current_stop) if vehicle else -1

    @current_stop.setter
    def current_stop(self, value: int) -> None:
        vehicle = self._get_active_vehicle()
        if vehicle is None and self.vehicles:
            self.active_vehicle_id = self.vehicles[0].vehicle_id
            vehicle = self.vehicles[0]
        if vehicle is None:
            raise RuntimeError("No vehicle available to set current_stop")
        vehicle.current_stop = int(value)

    @property
    def onboard(self) -> List[dict]:
        vehicle = self._get_active_vehicle()
        if vehicle is None and self.vehicles:
            vehicle = self.vehicles[0]
        return vehicle.onboard if vehicle else []

    @onboard.setter
    def onboard(self, value: List[dict]) -> None:
        vehicle = self._get_active_vehicle()
        if vehicle is None and self.vehicles:
            self.active_vehicle_id = self.vehicles[0].vehicle_id
            vehicle = self.vehicles[0]
        if vehicle is None:
            raise RuntimeError("No vehicle available to set onboard")
        vehicle.onboard = value

    def _churn_prob(self, wait_sec: float, beta: float, tol_sec: float) -> float:
        x = float(beta * (wait_sec - tol_sec))
        x = float(np.clip(x, -60.0, 60.0))
        return float(1.0 / (1.0 + np.exp(-x)))

    def _waiting_churn_prob(self, wait_sec: float) -> float:
        return self._churn_prob(
            wait_sec,
            beta=float(self.config.waiting_churn_beta),
            tol_sec=float(self.config.waiting_churn_tol_sec),
        )

    def _onboard_churn_prob(self, delay_sec: float) -> float:
        return self._churn_prob(
            delay_sec,
            beta=float(self.config.onboard_churn_beta),
            tol_sec=float(self.config.onboard_churn_tol_sec),
        )

    def _cvar(self, probs: List[float]) -> float:
        if not probs:
            return 0.0
        probs = sorted(probs)
        tail_start = int(max(0, np.floor(self.config.cvar_alpha * len(probs))))
        tail = probs[tail_start:]
        return float(np.mean(tail)) if tail else 0.0

    def _gini(self, values: List[float]) -> float:
        """Compute Gini coefficient using standardized algorithm.
        
        See src/utils/fairness.py for algorithm details and documentation.
        """
        return gini_coefficient(values)

    def _compute_waiting_risks(self) -> Dict[int, Tuple[float, float, int]]:
        risks: Dict[int, Tuple[float, float, int]] = {}
        for stop_id, queue in self.waiting.items():
            if not queue:
                risks[int(stop_id)] = (0.0, 0.0, 0)
                continue
            waits = [self.current_time - req["request_time_sec"] for req in queue]
            probs = [self._waiting_churn_prob(wait_sec) for wait_sec in waits]
            risks[int(stop_id)] = (float(np.mean(probs)), self._cvar(probs), len(queue))
        return risks

    def _compute_fleet_density_by_stop(self) -> Dict[int, float]:
        """Compute fleet density C(u) for each stop.
        
        Uses stop_id as the canonical key (Indexing Contract).
        
        Modes:
        - "next_stop": C(u) = #vehicles whose next_target_stop == u
        - "k_hop": Spreads density to k-hop neighborhood
        - "hybrid": Center weight + k-hop neighbor diffusion
        
        Vehicle state handling:
        - available_time <= current_time: waiting for decision at current_stop
        - available_time > current_time: in transit (current_stop stores destination)
        
        Returns:
            Dict mapping stop_id -> density count (raw, not normalized)
        """
        density: Dict[int, float] = {int(stop_id): 0.0 for stop_id in self.stop_ids}

        waiting_targets: List[int] = []
        in_transit_targets: List[int] = []
        for vehicle in self.vehicles:
            if vehicle.available_time <= self.current_time:
                waiting_targets.append(int(vehicle.current_stop))
            else:
                in_transit_targets.append(int(vehicle.current_stop))

        mode = str(self.config.fleet_potential_mode)
        center_weight = float(self.config.fleet_potential_hybrid_center_weight)
        neighbor_weight = float(self.config.fleet_potential_hybrid_neighbor_weight)

        def _apply_target(target_stop: int) -> None:
            if mode == "k_hop":
                neighbors = self._k_hop_neighbor_cache.get(target_stop, [target_stop])
                spread_value = 1.0 / max(1, len(neighbors))
                for neighbor in neighbors:
                    density[int(neighbor)] = density.get(int(neighbor), 0.0) + spread_value
                return
            if mode == "hybrid":
                density[target_stop] = density.get(target_stop, 0.0) + center_weight
                if neighbor_weight <= 0.0:
                    return
                neighbors = self._k_hop_neighbor_cache.get(target_stop, [target_stop])
                neighbor_nodes = [int(neighbor) for neighbor in neighbors if int(neighbor) != target_stop]
                if not neighbor_nodes:
                    return
                spread_value = neighbor_weight / len(neighbor_nodes)
                for neighbor in neighbor_nodes:
                    density[int(neighbor)] = density.get(int(neighbor), 0.0) + spread_value
                return
            density[target_stop] = density.get(target_stop, 0.0) + 1.0

        for target_stop in waiting_targets:
            _apply_target(target_stop)
        for target_stop in in_transit_targets:
            _apply_target(target_stop)

        return density

    def _refresh_fleet_density_map(self, record_history: bool = False) -> Dict[int, float]:
        if not (self.config.use_fleet_potential or self.config.reward_congestion_penalty > 0.0):
            self._last_fleet_density_map = {}
            return {}
        density_map = self._compute_fleet_density_by_stop()
        self._last_fleet_density_map = density_map
        if record_history:
            density_values = list(density_map.values())
            density_max = float(max(density_values)) if density_values else 0.0
            self._fleet_density_max_history.append(density_max)
        return density_map

    def _append_fleet_density_summary(self, info: Dict[str, float]) -> None:
        if not self.config.use_fleet_potential:
            return
        density_map = getattr(self, "_last_fleet_density_map", {})
        if density_map:
            density_values = list(density_map.values())
            info["fleet_density_summary"] = {
                "max": float(max(density_values)) if density_values else 0.0,
                "mean": float(sum(density_values) / len(density_values)) if density_values else 0.0,
                "top_5_congested_stops": [
                    (int(stop_id), float(density))
                    for stop_id, density in sorted(
                        density_map.items(), key=lambda x: x[1], reverse=True
                    )[:5]
                ],
            }
        else:
            info["fleet_density_summary"] = {"max": 0.0, "mean": 0.0, "top_5_congested_stops": []}

    def _append_episode_density_metrics(self, info: Dict[str, float]) -> None:
        history = self._fleet_density_max_history
        if history:
            info["fleet_density_max_mean"] = float(np.mean(history))
            info["fleet_density_max_p95"] = float(np.percentile(history, 95))
        else:
            info["fleet_density_max_mean"] = 0.0
            info["fleet_density_max_p95"] = 0.0
        total_stops = len(self.stop_ids)
        if total_stops > 0:
            info["stop_coverage_ratio"] = float(len(self._visited_stops) / total_stops)
        else:
            info["stop_coverage_ratio"] = 0.0
    
    def _apply_fleet_potential_phi(self, density: float) -> float:
        """Apply normalization function phi to density value.
        
        Args:
            density: Raw density count C(u)
        
        Returns:
            Normalized fleet potential in range [0, 1]
        """
        num_vehicles = max(1, self.config.num_vehicles)
        
        if self.config.fleet_potential_phi == "linear_norm":
            value = float(density / num_vehicles)
        else:
            # Default: log1p_norm
            # phi(C) = log(1 + C) / log(1 + num_vehicles)
            value = float(np.log1p(density) / np.log1p(num_vehicles))
        return float(min(1.0, max(0.0, value)))

    def _schedule_event(self, time_sec: float, event_type: str, payload: dict) -> None:
        priority = EVENT_PRIORITY.get(event_type, 99)
        heapq.heappush(self.event_queue, (float(time_sec), int(priority), int(self.event_seq), event_type, payload))
        self.event_seq += 1

    def _log_event(self, time_sec: float, event_type: str, payload: dict) -> None:
        entry = {
            "time_sec": float(time_sec),
            "event_type": event_type,
            "payload": payload,
        }
        self.event_log.append(entry)

    def _transition(self, req: dict, new_state: str, reason: str | None = None, time_sec: float | None = None) -> None:
        old_state = req.get("status")
        allowed = {
            None: {"waiting", "structurally_unserviceable"},
            "waiting": {"onboard", "churned_waiting"},
            "onboard": {"served", "churned_onboard"},
            "structurally_unserviceable": set(),
            "served": set(),
            "churned_waiting": set(),
            "churned_onboard": set(),
        }
        if new_state not in allowed.get(old_state, set()):
            raise ValueError(f"Illegal transition {old_state} -> {new_state}")
        req["status"] = new_state
        log_time = float(self.current_time if time_sec is None else time_sec)
        self.state_log.append(
            {
                "time_sec": log_time,
                "request_id": int(req["request_id"]),
                "from_state": old_state,
                "to_state": new_state,
                "reason": reason,
            }
        )

    def _schedule_initial_events(self) -> None:
        for req in self.requests:
            req["pickup_time_sec"] = None
            if req.get("structural_unserviceable", False):
                self.structurally_unserviceable += 1
                self._transition(
                    req,
                    "structurally_unserviceable",
                    reason="structural_unserviceable",
                    time_sec=float(req["request_time_sec"]),
                )
                continue
            self._schedule_event(req["request_time_sec"], EVENT_ORDER, {"request_id": int(req["request_id"])})

    def _event_trace_digest(self) -> str:
        payload = json.dumps(self.event_log, sort_keys=True, ensure_ascii=True).encode("ascii")
        return hashlib.sha256(payload).hexdigest()

    def _ingest_requests(self) -> None:
        """Compatibility shim for legacy tests; events are now scheduled upfront."""
        return None

    def _shortest_time(self, src: int, dst: int) -> float:
        if src == dst:
            return 0.0
        src_idx = self.stop_index.get(int(src))
        dst_idx = self.stop_index.get(int(dst))
        if src_idx is None or dst_idx is None:
            return float("inf")
        return float(self.shortest_time_sec[src_idx, dst_idx])

    def _apply_churn(self) -> Tuple[int, int, float, float, float, List[float], List[int], List[int], int]:
        churned = 0
        timeout = 0
        churn_prob_sum = 0.0
        churn_prob_weighted_sum = 0.0
        churn_probs: List[float] = []
        churned_ids: List[int] = []
        timeout_ids: List[int] = []
        prob_count = 0
        for stop_id, queue in self.waiting.items():
            remain = []
            for req in queue:
                wait_sec = self.current_time - req["request_time_sec"]
                if wait_sec > self.config.request_timeout_sec:
                    timeout += 1
                    req["cancel_reason"] = "timeout"
                    self._transition(req, "churned_waiting", reason="timeout")
                    timeout_ids.append(int(req["request_id"]))
                    self.canceled_requests.append(req)
                    continue

                prob = self._waiting_churn_prob(wait_sec)
                churn_prob_sum += float(prob)
                churn_probs.append(float(prob))
                churn_prob_weighted_sum += float(prob) * self.fairness_weight.get(int(stop_id), 1.0)
                prob_count += 1
                if self.rng.random() < prob:
                    churned += 1
                    req["cancel_reason"] = "probabilistic_churn"
                    self._transition(req, "churned_waiting", reason="probabilistic_churn")
                    churned_ids.append(int(req["request_id"]))
                    self.canceled_requests.append(req)
                else:
                    remain.append(req)
            self.waiting[stop_id] = remain
        cvar = self._cvar(churn_probs)
        return churned, timeout, churn_prob_sum, churn_prob_weighted_sum, cvar, churn_probs, churned_ids, timeout_ids, prob_count

    def _apply_onboard_churn(self, vehicle: VehicleState) -> Tuple[int, float, List[float], List[int], int]:
        churned = 0
        prob_sum = 0.0
        churn_probs: List[float] = []
        churned_ids: List[int] = []
        prob_count = 0
        remain = []
        for pax in vehicle.onboard:
            elapsed = self.current_time - pax["pickup_time_sec"]
            delay_over_direct = max(0.0, elapsed - pax.get("direct_time_sec", elapsed))
            prob = self._onboard_churn_prob(delay_over_direct)
            prob_sum += float(prob)
            churn_probs.append(float(prob))
            prob_count += 1
            if self.rng.random() < prob:
                churned += 1
                pax["cancel_reason"] = "onboard_churn"
                self._transition(pax, "churned_onboard", reason="onboard_churn")
                churned_ids.append(int(pax["request_id"]))
                self.canceled_requests.append(pax)
            else:
                remain.append(pax)
        vehicle.onboard = remain
        return churned, prob_sum, churn_probs, churned_ids, prob_count

    def _handle_vehicle_arrival(
        self, vehicle: VehicleState
    ) -> Tuple[int, float, float, float, float, List[int]]:
        vehicle.visit_counts[int(vehicle.current_stop)] = (
            vehicle.visit_counts.get(int(vehicle.current_stop), 0) + 1
        )
        self._visited_stops.add(int(vehicle.current_stop))
        dropped = [p for p in vehicle.onboard if p["dropoff_stop_id"] == vehicle.current_stop]
        kept = [p for p in vehicle.onboard if p["dropoff_stop_id"] != vehicle.current_stop]
        vehicle.onboard = kept
        dropped_count = len(dropped)
        tacc_gain = float(sum(pax.get("direct_time_sec", 0.0) for pax in dropped))
        delay_prob_sum = 0.0
        for pax in dropped:
            elapsed = self.current_time - pax["pickup_time_sec"]
            delay_over_direct = max(0.0, elapsed - pax.get("direct_time_sec", elapsed))
            delay_prob_sum += self._onboard_churn_prob(delay_over_direct)
            self._transition(pax, "served", reason="dropoff")
            self.dropoff_count_by_stop[int(vehicle.current_stop)] += 1

        (
            onboard_churned,
            onboard_prob_sum,
            onboard_probs,
            onboard_churned_ids,
            onboard_prob_count,
        ) = self._apply_onboard_churn(vehicle)
        self.onboard_churned += onboard_churned
        self._step_stats["onboard_churned"] += onboard_churned
        self._step_stats["onboard_churn_prob_sum"] += onboard_prob_sum
        self._step_stats["onboard_churn_count"] += onboard_prob_count
        self._step_stats["onboard_churn_probs"].extend(onboard_probs)

        capacity_left = max(0, self.config.vehicle_capacity - len(vehicle.onboard))
        queue = self.waiting[vehicle.current_stop]
        denied = max(0, len(queue) - capacity_left)
        boarded = queue[:capacity_left]
        self.waiting[vehicle.current_stop] = queue[capacity_left:]
        for req in boarded:
            req["pickup_time_sec"] = self.current_time
            req["t_max_sec"] = (
                float("inf")
                if not np.isfinite(req["direct_time_sec"])
                else self.config.mask_alpha * req["direct_time_sec"]
            )
            self._transition(req, "onboard", reason="boarded")
            vehicle.onboard.append(req)
            wait_sec = max(0.0, float(self.current_time) - float(req["request_time_sec"]))
            self.acc_wait_time_by_stop[int(vehicle.current_stop)] += wait_sec
        if boarded:
            self.service_count_by_stop[int(vehicle.current_stop)] += len(boarded)

        return dropped_count, tacc_gain, delay_prob_sum, float(len(boarded)), float(denied), onboard_churned_ids

    def _prepare_initial_arrivals(self) -> None:
        for vehicle in self.vehicles:
            vehicle.visit_counts = {int(vehicle.current_stop): 1}
            self._schedule_event(
                0.0,
                EVENT_VEHICLE_ARRIVAL,
                {"vehicle_id": int(vehicle.vehicle_id)},
            )

    def _advance_until_ready(self) -> None:
        # If there are already ready vehicles queued (e.g., multiple arrivals at the same sim_time),
        # we must pick one immediately. Otherwise the simulator can "stall" at a fixed time while
        # the trainer keeps issuing actions, producing 0 served/churn and flat metrics.
        if self.config.max_sim_time_sec is not None and self.current_time >= float(self.config.max_sim_time_sec):
            self.current_time = float(self.config.max_sim_time_sec)
            self.done = True
            return
        if self.active_vehicle_id is None and self.ready_vehicle_ids:
            self.active_vehicle_id = self.ready_vehicle_ids.pop(0)
            return
        while not self.ready_vehicle_ids and not self.done:
            if not self.event_queue:
                self.done = True
                return
            next_time = self.event_queue[0][0]
            if self.config.max_sim_time_sec is not None and float(next_time) >= float(self.config.max_sim_time_sec):
                self.current_time = float(self.config.max_sim_time_sec)
                self.done = True
                return
            self.current_time = float(next_time)
            events_at_time: List[Tuple[float, int, int, str, dict]] = []
            while self.event_queue and self.event_queue[0][0] == next_time:
                events_at_time.append(heapq.heappop(self.event_queue))
            for _, _, _, event_type, payload in events_at_time:
                if event_type == EVENT_ORDER:
                    req = self.request_index[int(payload["request_id"])]
                    self._transition(req, "waiting", reason="request_arrival", time_sec=self.current_time)
                    self.waiting[int(req["pickup_stop_id"])].append(req)
                    self._log_event(self.current_time, EVENT_ORDER, {"request_id": int(req["request_id"])})
                elif event_type == EVENT_VEHICLE_ARRIVAL:
                    vehicle_id = int(payload["vehicle_id"])
                    vehicle = self.vehicles[vehicle_id]
                    (
                        dropped,
                        tacc_gain,
                        delay_prob_sum,
                        boarded,
                        denied,
                        onboard_churned_ids,
                    ) = self._handle_vehicle_arrival(vehicle)
                    self.served += dropped
                    self._step_stats["served"] += dropped
                    self._step_stats["tacc_gain"] += tacc_gain
                    self._step_stats["onboard_delay_prob_sum"] += delay_prob_sum
                    self._step_stats["boarded"] += boarded
                    self._step_stats["denied_boarding"] += denied
                    self._log_event(
                        self.current_time,
                        EVENT_VEHICLE_ARRIVAL,
                        {
                            "vehicle_id": vehicle_id,
                            "dropped": int(dropped),
                            "boarded": int(boarded),
                            "denied": int(denied),
                            "onboard_churned_ids": onboard_churned_ids,
                        },
                    )
                    self.ready_vehicle_ids.append(vehicle_id)
            (
                churned,
                timeout,
                churn_prob_sum,
                churn_prob_weighted_sum,
                churn_cvar,
                churn_probs,
                churned_ids,
                timeout_ids,
                churn_prob_count,
            ) = self._apply_churn()
            self.waiting_churned += churned
            self.waiting_timeouts += timeout
            self._step_stats["waiting_churned"] += churned
            self._step_stats["waiting_timeouts"] += timeout
            self._step_stats["waiting_churn_prob_sum"] += churn_prob_sum
            self._step_stats["waiting_churn_prob_weighted_sum"] += churn_prob_weighted_sum
            self._step_stats["waiting_churn_count"] += churn_prob_count
            self._step_stats["waiting_churn_probs"].extend(churn_probs)
            self._step_stats["waiting_churn_cvar"] = self._cvar(self._step_stats["waiting_churn_probs"])
            self._log_event(
                self.current_time,
                EVENT_CHURN_CHECK,
                {
                    "waiting_churned": int(churned),
                    "waiting_timeouts": int(timeout),
                    "churned_ids": churned_ids,
                    "timeout_ids": timeout_ids,
                },
            )
            if self.active_vehicle_id is None and self.ready_vehicle_ids:
                self.active_vehicle_id = self.ready_vehicle_ids.pop(0)
                return

        if self.active_vehicle_id is None and self.ready_vehicle_ids:
            self.active_vehicle_id = self.ready_vehicle_ids.pop(0)

    def _has_future_orders(self) -> bool:
        return any(event_type == EVENT_ORDER for _, _, _, event_type, _ in self.event_queue)

    def _no_pending_requests(self) -> bool:
        waiting_remaining = sum(len(q) for q in self.waiting.values())
        onboard_remaining = sum(len(v.onboard) for v in self.vehicles)
        return waiting_remaining == 0 and onboard_remaining == 0

    def _get_active_vehicle(self) -> VehicleState | None:
        if self.active_vehicle_id is None:
            return None
        return self.vehicles[self.active_vehicle_id]

    def get_action_mask(self, debug: bool = False) -> Tuple[List[int], List[bool]]:
        vehicle = self._get_active_vehicle()
        if vehicle is None:
            return [], []
        candidates = self.neighbors.get(vehicle.current_stop, [])
        if not candidates:
            return [], []

        actions = [dst for dst, _ in candidates]
        dropoff_stops = {int(pax["dropoff_stop_id"]) for pax in vehicle.onboard}
        has_onboard = len(vehicle.onboard) > 0
        is_full = len(vehicle.onboard) >= int(self.config.vehicle_capacity)
        has_dropoff_candidate = any(int(dst) in dropoff_stops for dst, _ in candidates)
        mask = []
        debug_entries = []
        hard_mask_slack_sec = float(getattr(self.config, "hard_mask_slack_sec", 0.0))
        hard_mask_skip_unrecoverable = bool(getattr(self.config, "hard_mask_skip_unrecoverable", False))
        hard_mask_skipped = []
        hard_mask_gate_by_id: Dict[int, HardMaskGate] = {}
        if has_onboard:
            for pax in vehicle.onboard:
                pickup_time_sec = pax.get("pickup_time_sec")
                t_max_sec = pax.get("t_max_sec")
                dropoff_stop_id = pax.get("dropoff_stop_id")
                if dropoff_stop_id is None:
                    continue
                gate = compute_hard_mask_gate(
                    pickup_time_sec=pickup_time_sec,
                    t_max_sec=t_max_sec,
                    current_time_sec=float(self.current_time),
                    best_remaining_sec=self._shortest_time(vehicle.current_stop, int(dropoff_stop_id)),
                    slack_sec=hard_mask_slack_sec,
                    max_time_sec=DEFAULT_MAX_TIME_SEC,
                    skip_unrecoverable=hard_mask_skip_unrecoverable,
                )
                hard_mask_gate_by_id[id(pax)] = gate
                if not gate.enforce:
                    hard_mask_skipped.append(
                        {
                            "request_id": pax.get("request_id"),
                            "dropoff_stop_id": int(dropoff_stop_id),
                            "baseline_eta_sec": float(gate.baseline_eta_sec),
                            "baseline_over_by_sec": float(gate.baseline_over_by_sec),
                        }
                    )
        for dst, travel in candidates:
            feasible = True
            violations = []
            if is_full and has_dropoff_candidate and int(dst) not in dropoff_stops:
                feasible = False
                violations.append(
                    {
                        "type": "onboard_priority_full",
                        "vehicle_id": int(vehicle.vehicle_id),
                        "dropoff_count": int(len(dropoff_stops)),
                    }
                )
            dropoffs_at_dst = sum(1 for pax in vehicle.onboard if pax["dropoff_stop_id"] == dst)
            projected_onboard = len(vehicle.onboard) - dropoffs_at_dst
            capacity_left = self.config.vehicle_capacity - projected_onboard
            waiting_at_dst = len(self.waiting.get(int(dst), []))
            if capacity_left <= 0 and waiting_at_dst > 0:
                feasible = False
                violations.append(
                    {
                        "type": "capacity",
                        "vehicle_id": int(vehicle.vehicle_id),
                        "projected_onboard": int(projected_onboard),
                        "capacity": int(self.config.vehicle_capacity),
                        "waiting_at_dst": int(waiting_at_dst),
                    }
                )
            for pax in vehicle.onboard:
                gate = hard_mask_gate_by_id.get(id(pax))
                if gate is not None and not gate.enforce:
                    continue
                remaining = sanitize_time_sec(
                    self._shortest_time(dst, pax["dropoff_stop_id"]),
                    max_time_sec=DEFAULT_MAX_TIME_SEC,
                )
                eta_total = float(self.current_time) + float(travel) + float(remaining)
                pickup_time_sec = pax.get("pickup_time_sec")
                t_max_sec = pax.get("t_max_sec")
                if pickup_time_sec is None or t_max_sec is None:
                    continue
                over_by = hard_deadline_over_by_sec(eta_total, float(pickup_time_sec), float(t_max_sec))
                if over_by > hard_mask_slack_sec:
                    feasible = False
                    violations.append(
                        {
                            "type": "hard_mask",
                            "request_id": pax.get("request_id"),
                            "dropoff_stop_id": pax["dropoff_stop_id"],
                            "eta_sec": float(eta_total),
                            "t_max_sec": float(pax["t_max_sec"]),
                            "over_by_sec": float(over_by),
                            "travel_time_sec": float(travel),
                            "remaining_sec": float(remaining),
                        }
                    )
                    break
            mask.append(feasible)
            if violations:
                debug_entries.append(
                    {
                        "action": int(dst),
                        "vehicle_id": int(vehicle.vehicle_id),
                        "violations": violations,
                    }
                )
        if hard_mask_skip_unrecoverable and hard_mask_skipped and (debug or self.config.debug_mask):
            debug_entries.insert(
                0,
                {
                    "type": "hard_mask_skip_unrecoverable",
                    "vehicle_id": int(vehicle.vehicle_id),
                    "slack_sec": hard_mask_slack_sec,
                    "skipped": hard_mask_skipped,
                },
            )
        if not self.config.allow_stop_when_actions_exist:
            stop_indices = [idx for idx, dst in enumerate(actions) if int(dst) == int(vehicle.current_stop)]
            if stop_indices:
                other_feasible = any(
                    bool(mask[idx]) for idx in range(len(mask)) if idx not in stop_indices
                )
                if other_feasible:
                    for idx in stop_indices:
                        if mask[idx]:
                            mask[idx] = False
                            debug_entries.append(
                                {
                                    "action": int(actions[idx]),
                                    "vehicle_id": int(vehicle.vehicle_id),
                                    "violations": [{"type": "noop_disallowed"}],
                                }
                            )
        if actions and not any(mask):
            if int(vehicle.current_stop) in actions:
                stop_idx = actions.index(int(vehicle.current_stop))
                mask[stop_idx] = True
                debug_entries.append(
                    {
                        "action": int(actions[stop_idx]),
                        "vehicle_id": int(vehicle.vehicle_id),
                        "violations": [{"type": "noop_fallback"}],
                    }
                )
            else:
                actions.append(int(vehicle.current_stop))
                mask.append(True)
                debug_entries.append(
                    {
                        "action": int(vehicle.current_stop),
                        "vehicle_id": int(vehicle.vehicle_id),
                        "violations": [{"type": "noop_fallback"}],
                    }
                )
        if debug or self.config.debug_mask:
            self.last_mask_debug = debug_entries
        return actions, mask

    def get_feature_batch(self, k_hop: int | None = None) -> Dict[str, np.ndarray]:
        vehicle = self._get_active_vehicle()
        if vehicle is None:
            self._advance_until_ready()
            vehicle = self._get_active_vehicle()
        if vehicle is None:
            node_features = np.zeros((len(self.stop_ids), 5), dtype=np.float32)
            stop_index = {stop_id: idx for idx, stop_id in enumerate(self.stop_ids)}
            risks = self._compute_waiting_risks()
            for stop_id, (risk_mean, risk_cvar, count) in risks.items():
                idx = stop_index[stop_id]
                node_features[idx, 0] = float(risk_mean)
                node_features[idx, 1] = float(risk_cvar)
                node_features[idx, 2] = float(min(count, 500.0))
                node_features[idx, 3] = self.fairness_weight.get(int(stop_id), 1.0)
                if int(stop_id) not in self.geo_embedding_scalar:
                    raise ValueError(f"Missing geo embedding for stop {stop_id}")
                node_features[idx, 4] = self.geo_embedding_scalar[int(stop_id)]
            return {
                "node_features": node_features,
                "edge_features": np.zeros((0, 5 if self.config.use_fleet_potential else 4), dtype=np.float32),
                "action_mask": np.zeros((0,), dtype=bool),
                "actions": np.zeros((0,), dtype=np.int64),
                "action_node_indices": np.zeros((0,), dtype=np.int64),
                "node_ids": np.array(self.stop_ids, dtype=np.int64),
                "graph_edge_index": self.graph_edge_index.copy(),
                "graph_edge_features": self.graph_edge_features.copy(),
                "current_stop": np.array([-1], dtype=np.int64),
                "current_node_index": np.array([-1], dtype=np.int64),
            }
        node_features = np.zeros((len(self.stop_ids), 5), dtype=np.float32)
        stop_index = {stop_id: idx for idx, stop_id in enumerate(self.stop_ids)}

        risks = self._compute_waiting_risks()
        for stop_id, (risk_mean, risk_cvar, count) in risks.items():
            idx = stop_index[stop_id]
            node_features[idx, 0] = float(risk_mean)
            node_features[idx, 1] = float(risk_cvar)
            node_features[idx, 2] = float(min(count, 500.0))
            node_features[idx, 3] = self.fairness_weight.get(int(stop_id), 1.0)
            if int(stop_id) not in self.geo_embedding_scalar:
                raise ValueError(f"Missing geo embedding for stop {stop_id}")
            node_features[idx, 4] = self.geo_embedding_scalar[int(stop_id)]

        actions, mask = self.get_action_mask(debug=self.config.debug_mask)
        base_edge_dim = 4
        edge_dim = 5 if self.config.use_fleet_potential else base_edge_dim
        edge_features = np.zeros((len(actions), edge_dim), dtype=np.float32)
        
        # Compute fleet density if FAEP enabled
        fleet_density_map: Dict[int, float] = {}
        if self.config.use_fleet_potential:
            fleet_density_map = self._refresh_fleet_density_map(record_history=False)
        for i, action in enumerate(actions):
            travel_time = dict(self.neighbors.get(vehicle.current_stop, [])).get(action, 0.0)
            delta_eta_max = 0.0
            onboard_risks_before: List[float] = []
            onboard_risks_after: List[float] = []
            count_violation = 0.0
            
            # Sanitization constants
            MAX_TIME_VAL = 36000.0
            
            for pax in vehicle.onboard:
                curr_eta = self._shortest_time(vehicle.current_stop, pax["dropoff_stop_id"])
                new_eta_leg = self._shortest_time(action, pax["dropoff_stop_id"])
                
                # Handle infinite ETAs (unreachable traps)
                if not np.isfinite(curr_eta): curr_eta = MAX_TIME_VAL
                if not np.isfinite(new_eta_leg): new_eta_leg = MAX_TIME_VAL
                
                new_eta = travel_time + new_eta_leg
                delta_eta_max = max(delta_eta_max, max(0.0, new_eta - curr_eta))
                
                delay_before = max(0.0, curr_eta - pax.get("direct_time_sec", curr_eta))
                delay_after = max(0.0, new_eta - pax.get("direct_time_sec", new_eta))
                
                onboard_risks_before.append(self._onboard_churn_prob(delay_before))
                onboard_risks_after.append(self._onboard_churn_prob(delay_after))
                
                if pax.get("pickup_time_sec") is not None:
                    eta_total = self.current_time + new_eta
                    if eta_total - pax["pickup_time_sec"] > pax["t_max_sec"]:
                        count_violation += 1.0
            
            delta_cvar = self._cvar(onboard_risks_after) - self._cvar(onboard_risks_before)
            edge_features[i, 0] = float(min(delta_eta_max, MAX_TIME_VAL))
            edge_features[i, 1] = float(delta_cvar)
            edge_features[i, 2] = float(min(count_violation, 100.0))
            edge_features[i, 3] = float(min(travel_time, MAX_TIME_VAL))
            
            # Append fleet potential as 5th dimension if FAEP enabled
            if self.config.use_fleet_potential:
                # action is stop_id (Indexing Contract)
                raw_density = fleet_density_map.get(int(action), 0.0)
                edge_features[i, 4] = self._apply_fleet_potential_phi(raw_density)

        full_batch = {
            "node_features": node_features,
            "edge_features": edge_features,
            "action_mask": np.array(mask, dtype=bool),
            "actions": np.array(actions, dtype=np.int64),
            "action_node_indices": np.array([stop_index[a] for a in actions], dtype=np.int64),
            "node_ids": np.array(self.stop_ids, dtype=np.int64),
            "graph_edge_index": self.graph_edge_index.copy(),
            "graph_edge_features": self.graph_edge_features.copy(),
            "current_stop": np.array([vehicle.current_stop], dtype=np.int64),
            "current_node_index": np.array([stop_index[int(vehicle.current_stop)]], dtype=np.int64),
        }
        if k_hop is None:
            return full_batch

        center = int(vehicle.current_stop)
        sub_nodes = set(self._k_hop_nodes(center, int(k_hop)))
        sub_nodes.add(center)
        sub_nodes.update(int(a) for a in actions)
        sub_node_ids = sorted(sub_nodes)
        local_index = {stop_id: idx for idx, stop_id in enumerate(sub_node_ids)}
        full_indices = [stop_index[int(stop_id)] for stop_id in sub_node_ids]

        sub_node_features = node_features[full_indices]
        src_list: List[int] = []
        dst_list: List[int] = []
        edge_attr_list: List[List[float]] = []
        for src_stop_id in sub_node_ids:
            for dst_stop_id, travel_time in self.neighbors.get(int(src_stop_id), []):
                if int(dst_stop_id) not in sub_nodes:
                    continue
                src_list.append(local_index[int(src_stop_id)])
                dst_list.append(local_index[int(dst_stop_id)])
                if self.config.use_fleet_potential:
                    edge_attr_list.append([0.0, 0.0, 0.0, float(travel_time), 0.0])
                else:
                    edge_attr_list.append([0.0, 0.0, 0.0, float(travel_time)])

        if src_list:
            sub_edge_index = np.array([src_list, dst_list], dtype=np.int64)
            sub_edge_features = np.array(edge_attr_list, dtype=np.float32)
        else:
            sub_edge_index = np.zeros((2, 0), dtype=np.int64)
            edge_dim = 5 if self.config.use_fleet_potential else 4
            sub_edge_features = np.zeros((0, edge_dim), dtype=np.float32)

        return {
            "node_features": sub_node_features,
            "edge_features": edge_features,
            "action_mask": np.array(mask, dtype=bool),
            "actions": np.array(actions, dtype=np.int64),
            "action_node_indices": np.array([local_index[int(a)] for a in actions], dtype=np.int64),
            "node_ids": np.array(sub_node_ids, dtype=np.int64),
            "graph_edge_index": sub_edge_index,
            "graph_edge_features": sub_edge_features,
            "current_stop": np.array([center], dtype=np.int64),
            "current_node_index": np.array([local_index[center]], dtype=np.int64),
        }

    def step(self, action: int) -> Tuple[Dict[str, float], float, bool, Dict[str, float]]:
        if self.done:
            raise RuntimeError("Episode is done, call reset()")
        self._step_stats = self._init_step_stats()
        before_snapshot = self._snapshot_potential()
        waiting_remaining_before = before_snapshot["waiting_remaining"]
        onboard_remaining_before = before_snapshot["onboard_remaining"]
        phi_backlog_before = before_snapshot["phi_backlog"]
        phi_before = before_snapshot["phi"]
        lost_total_before = before_snapshot["lost_total"]
        waiting_churned_before = before_snapshot["waiting_churned"]
        onboard_churned_before = before_snapshot["onboard_churned"]
        structural_before = before_snapshot["structural_unserviceable"]
        if self.config.max_sim_time_sec is not None and self.current_time > float(self.config.max_sim_time_sec):
            self.current_time = float(self.config.max_sim_time_sec)
        if self.config.max_sim_time_sec is not None and self.current_time >= float(self.config.max_sim_time_sec):
            self.done = True
            after_snapshot = self._snapshot_potential()
            waiting_remaining = after_snapshot["waiting_remaining"]
            onboard_remaining = after_snapshot["onboard_remaining"]
            info = {
                "done_reason": "max_sim_time",
                "waiting_remaining": waiting_remaining,
                "onboard_remaining": onboard_remaining,
                "reward_potential_alpha": float(self.config.reward_potential_alpha),
                "reward_potential_alpha_source": str(self.config.reward_potential_alpha_source),
                "reward_potential_lost_weight": float(self.config.reward_potential_lost_weight),
                "reward_potential_scale_with_reward_scale": bool(
                    self.config.reward_potential_scale_with_reward_scale
                ),
                "phi_before": float(phi_before),
                "phi_after": float(after_snapshot["phi"]),
                "phi_delta": 0.0,
                "phi_backlog_before": float(phi_backlog_before),
                "phi_backlog_after": float(after_snapshot["phi_backlog"]),
                "lost_total_before": float(lost_total_before),
                "lost_total_after": float(after_snapshot["lost_total"]),
                "waiting_churned_before": float(waiting_churned_before),
                "waiting_churned_after": float(after_snapshot["waiting_churned"]),
                "onboard_churned_before": float(onboard_churned_before),
                "onboard_churned_after": float(after_snapshot["onboard_churned"]),
                "structural_unserviceable_before": float(structural_before),
                "structural_unserviceable_after": float(after_snapshot["structural_unserviceable"]),
                "waiting_remaining_before": float(waiting_remaining_before),
                "onboard_remaining_before": float(onboard_remaining_before),
                "waiting_remaining_after": float(waiting_remaining),
                "onboard_remaining_after": float(onboard_remaining),
                "reward_potential_shaping": 0.0,
                "reward_potential_shaping_raw": 0.0,
                "reward_components": {
                    "reward_base_service": 0.0,
                    "reward_waiting_churn_penalty": 0.0,
                    "reward_fairness_penalty": 0.0,
                    "reward_cvar_penalty": 0.0,
                    "reward_travel_cost": 0.0,
                    "reward_onboard_delay_penalty": 0.0,
                    "reward_onboard_churn_penalty": 0.0,
                    "reward_backlog_penalty": 0.0,
                    "reward_waiting_time_penalty": 0.0,
                    "reward_potential_shaping": 0.0,
                    "reward_congestion_penalty": 0.0,
                    "reward_tacc_bonus": 0.0,
                },
                "reward_components_raw": {"reward_potential_shaping_raw": 0.0},
            }
            return self._observe(), 0.0, True, info
        if (
            self.config.allow_demand_exhausted_termination
            and self.current_time >= float(self.config.demand_exhausted_min_time_sec)
            and self._no_pending_requests()
            and not self._has_future_orders()
        ):
            self.done = True
            info = {
                "done_reason": "demand_exhausted",
                "waiting_remaining": 0.0,
                "onboard_remaining": 0.0,
                "reward_potential_alpha": float(self.config.reward_potential_alpha),
                "reward_potential_alpha_source": str(self.config.reward_potential_alpha_source),
                "reward_potential_lost_weight": float(self.config.reward_potential_lost_weight),
                "reward_potential_scale_with_reward_scale": bool(
                    self.config.reward_potential_scale_with_reward_scale
                ),
                "phi_before": float(phi_before),
                "phi_after": 0.0,
                "phi_delta": float(phi_before),
                "phi_backlog_before": float(phi_backlog_before),
                "phi_backlog_after": 0.0,
                "lost_total_before": float(lost_total_before),
                "lost_total_after": 0.0,
                "waiting_churned_before": float(waiting_churned_before),
                "waiting_churned_after": 0.0,
                "onboard_churned_before": float(onboard_churned_before),
                "onboard_churned_after": 0.0,
                "structural_unserviceable_before": float(structural_before),
                "structural_unserviceable_after": 0.0,
                "waiting_remaining_before": float(waiting_remaining_before),
                "onboard_remaining_before": float(onboard_remaining_before),
                "waiting_remaining_after": 0.0,
                "onboard_remaining_after": 0.0,
                "reward_potential_shaping": 0.0,
                "reward_potential_shaping_raw": 0.0,
                "reward_components": {
                    "reward_base_service": 0.0,
                    "reward_waiting_churn_penalty": 0.0,
                    "reward_fairness_penalty": 0.0,
                    "reward_cvar_penalty": 0.0,
                    "reward_travel_cost": 0.0,
                    "reward_onboard_delay_penalty": 0.0,
                    "reward_onboard_churn_penalty": 0.0,
                    "reward_backlog_penalty": 0.0,
                    "reward_waiting_time_penalty": 0.0,
                    "reward_potential_shaping": 0.0,
                    "reward_congestion_penalty": 0.0,
                    "reward_tacc_bonus": 0.0,
                },
                "reward_components_raw": {"reward_potential_shaping_raw": 0.0},
            }
            return self._observe(), 0.0, True, info
        vehicle = self._get_active_vehicle()
        if vehicle is None:
            self._advance_until_ready()
            vehicle = self._get_active_vehicle()
            if vehicle is None:
                if self.config.use_fleet_potential or self.config.reward_congestion_penalty > 0.0:
                    self._refresh_fleet_density_map(record_history=True)
                self.done = True
                # Populate info for early-return path
                after_snapshot = self._snapshot_potential()
                waiting_remaining = after_snapshot["waiting_remaining"]
                onboard_remaining = after_snapshot["onboard_remaining"]
                info = {
                    "done_reason": "no_valid_action",
                    "waiting_remaining": waiting_remaining,
                    "onboard_remaining": onboard_remaining,
                    "waiting_timeouts": float(self.waiting_timeouts),
                    "reward_congestion_penalty": 0.0,
                    "dst_density_raw": 0.0,
                    "fleet_potential_dst": 0.0,
                    "reward_potential_alpha": float(self.config.reward_potential_alpha),
                    "reward_potential_alpha_source": str(self.config.reward_potential_alpha_source),
                    "reward_potential_lost_weight": float(self.config.reward_potential_lost_weight),
                    "reward_potential_scale_with_reward_scale": bool(
                        self.config.reward_potential_scale_with_reward_scale
                    ),
                    "phi_before": float(phi_before),
                    "phi_after": float(after_snapshot["phi"]),
                    "phi_delta": float(phi_before - after_snapshot["phi"]),
                    "phi_backlog_before": float(phi_backlog_before),
                    "phi_backlog_after": float(after_snapshot["phi_backlog"]),
                    "lost_total_before": float(lost_total_before),
                    "lost_total_after": float(after_snapshot["lost_total"]),
                    "waiting_churned_before": float(waiting_churned_before),
                    "waiting_churned_after": float(after_snapshot["waiting_churned"]),
                    "onboard_churned_before": float(onboard_churned_before),
                    "onboard_churned_after": float(after_snapshot["onboard_churned"]),
                    "structural_unserviceable_before": float(structural_before),
                    "structural_unserviceable_after": float(after_snapshot["structural_unserviceable"]),
                    "waiting_remaining_before": float(waiting_remaining_before),
                    "onboard_remaining_before": float(onboard_remaining_before),
                    "waiting_remaining_after": float(waiting_remaining),
                    "onboard_remaining_after": float(onboard_remaining),
                    "reward_potential_shaping": 0.0,
                    "reward_potential_shaping_raw": 0.0,
                    "reward_components": {
                        "reward_base_service": 0.0,
                        "reward_waiting_churn_penalty": 0.0,
                        "reward_fairness_penalty": 0.0,
                        "reward_cvar_penalty": 0.0,
                        "reward_travel_cost": 0.0,
                        "reward_onboard_delay_penalty": 0.0,
                        "reward_onboard_churn_penalty": 0.0,
                        "reward_backlog_penalty": 0.0,
                        "reward_waiting_time_penalty": 0.0,
                        "reward_potential_shaping": 0.0,
                        "reward_congestion_penalty": 0.0,
                        "reward_tacc_bonus": 0.0,
                    },
                    "reward_components_raw": {"reward_potential_shaping_raw": 0.0},
                }
                self._append_fleet_density_summary(info)
                self._append_episode_density_metrics(info)
                return self._observe(), 0.0, True, info
        
        if self.config.use_fleet_potential or self.config.reward_congestion_penalty > 0.0:
            self._refresh_fleet_density_map(record_history=True)

        neighbor_actions = [dst for dst, _ in self.neighbors.get(vehicle.current_stop, [])]
        wait_action = False
        if action not in neighbor_actions:
            if int(action) == int(vehicle.current_stop):
                wait_action = True
            else:
                raise ValueError(f"Action {action} not in neighbor set for stop {vehicle.current_stop}")
        if not self.config.allow_stop_when_actions_exist and int(action) == int(vehicle.current_stop):
            actions, mask = self.get_action_mask(debug=self.config.debug_mask)
            if actions:
                try:
                    stop_idx = actions.index(int(action))
                except ValueError:
                    stop_idx = None
                if stop_idx is not None:
                    other_feasible = any(
                        bool(mask[idx]) for idx in range(len(mask)) if idx != stop_idx
                    )
                    if other_feasible:
                        raise ValueError("NOOP action is disallowed when feasible actions exist")
        
        # Debug模式：验证动作是否违反mask约束
        if self.config.debug_mask:
            actions, mask = self.get_action_mask(debug=True)
            try:
                action_idx = actions.index(action)
                if not mask[action_idx]:
                    raise ValueError(
                        f"Action violates hard mask constraints: action={action}, "
                        f"mask_debug={self.last_mask_debug}"
                    )
            except ValueError as e:
                if "not in list" in str(e):
                    raise ValueError(f"Action {action} not in action list {actions}")
                raise

        travel_time = dict(self.neighbors.get(vehicle.current_stop, [])).get(action)
        if travel_time is None:
            if wait_action and self.event_queue:
                next_time = float(self.event_queue[0][0])
                travel_time = max(0.0, next_time - float(self.current_time))
            elif wait_action:
                travel_time = 0.0
            else:
                raise ValueError("Missing travel time for action")

        arrival_time = self.current_time + travel_time
        if self.config.max_sim_time_sec is not None and arrival_time >= float(self.config.max_sim_time_sec):
            self.current_time = float(self.config.max_sim_time_sec)
            self.done = True
            after_snapshot = self._snapshot_potential()
            waiting_remaining = after_snapshot["waiting_remaining"]
            onboard_remaining = after_snapshot["onboard_remaining"]
            info = {
                "done_reason": "max_sim_time",
                "waiting_remaining": waiting_remaining,
                "onboard_remaining": onboard_remaining,
            }
            return self._observe(), 0.0, True, info
        self._log_event(
            self.current_time,
            EVENT_VEHICLE_DEPARTURE,
            {
                "vehicle_id": int(vehicle.vehicle_id),
                "from_stop": int(vehicle.current_stop),
                "to_stop": int(action),
                "travel_time_sec": float(travel_time),
            },
        )
        vehicle.current_stop = int(action)
        vehicle.available_time = arrival_time
        self._schedule_event(arrival_time, EVENT_VEHICLE_ARRIVAL, {"vehicle_id": int(vehicle.vehicle_id)})
        self.steps += 1
        # Consume the current decision. The next decision must come from the ready-queue or a future event.
        self.active_vehicle_id = None
        self._advance_until_ready()

        reward_base_service = self.config.reward_service * self._step_stats["served"]
        reward_waiting_churn = self.config.reward_waiting_churn_penalty * self._step_stats["waiting_churn_prob_sum"]
        reward_fairness = (
            self.config.reward_fairness_weight * self._step_stats["waiting_churn_prob_weighted_sum"]
        )
        reward_cvar = self.config.reward_cvar_penalty * self._step_stats["waiting_churn_cvar"]
        reward_travel_cost = self.config.reward_travel_cost_per_sec * travel_time
        reward_onboard_delay = self.config.reward_onboard_delay_weight * self._step_stats["onboard_delay_prob_sum"]
        reward_onboard_churn = self.config.reward_onboard_churn_penalty * self._step_stats["onboard_churn_prob_sum"]
        reward_tacc = self.config.reward_tacc_weight * self._step_stats["tacc_gain"]
        reward_base_service = self._apply_positive_reward_transform(
            reward_base_service,
            self.config.reward_service_transform,
            self.config.reward_service_transform_scale,
        )
        reward_tacc = self._apply_positive_reward_transform(
            reward_tacc,
            self.config.reward_tacc_transform,
            self.config.reward_tacc_transform_scale,
        )

        after_snapshot = self._snapshot_potential()
        waiting_remaining = after_snapshot["waiting_remaining"]
        onboard_remaining = after_snapshot["onboard_remaining"]
        phi_backlog_after = after_snapshot["phi_backlog"]
        phi_after = after_snapshot["phi"]
        phi_delta = float(phi_before - phi_after)
        lost_total_after = after_snapshot["lost_total"]
        waiting_churned_after = after_snapshot["waiting_churned"]
        onboard_churned_after = after_snapshot["onboard_churned"]
        structural_after = after_snapshot["structural_unserviceable"]
        reward_backlog = self.config.reward_step_backlog_penalty * (waiting_remaining + onboard_remaining)
        waiting_time_sec = 0.0
        if self.config.reward_waiting_time_penalty_per_sec > 0.0:
            for queue in self.waiting.values():
                for req in queue:
                    waiting_time_sec += max(0.0, self.current_time - float(req["request_time_sec"]))
        reward_waiting_time = self.config.reward_waiting_time_penalty_per_sec * waiting_time_sec
        potential = self.compute_potential_shaping(
            self.config.reward_potential_alpha,
            phi_before,
            phi_after,
            float(self.config.reward_scale),
            bool(self.config.reward_potential_scale_with_reward_scale),
        )
        reward_potential_shaping_raw = potential["reward_potential_shaping_raw"]
        reward_potential_shaping = potential["reward_potential_shaping"]

        scale = float(self.config.reward_scale)
        if scale != 1.0:
            reward_base_service *= scale
            reward_waiting_churn *= scale
            reward_fairness *= scale
            reward_cvar *= scale
            reward_travel_cost *= scale
            reward_onboard_delay *= scale
            reward_onboard_churn *= scale
            reward_tacc *= scale
            reward_backlog *= scale
            reward_waiting_time *= scale
            if not self.config.reward_potential_scale_with_reward_scale:
                reward_potential_shaping *= 1.0

        dst_density_raw = 0.0
        fleet_potential_dst = 0.0
        reward_congestion = 0.0
        if self.config.use_fleet_potential or self.config.reward_congestion_penalty > 0.0:
            density_map = getattr(self, "_last_fleet_density_map", {})
            dst_density_raw = float(density_map.get(int(action), 0.0))
            fleet_potential_dst = float(self._apply_fleet_potential_phi(dst_density_raw))
            reward_congestion = float(self.config.reward_congestion_penalty * fleet_potential_dst)

        reward_components = {
            "reward_base_service": float(reward_base_service),
            "reward_waiting_churn_penalty": float(-reward_waiting_churn),
            "reward_fairness_penalty": float(-reward_fairness),
            "reward_cvar_penalty": float(-reward_cvar),
            "reward_travel_cost": float(-reward_travel_cost),
            "reward_onboard_delay_penalty": float(-reward_onboard_delay),
            "reward_onboard_churn_penalty": float(-reward_onboard_churn),
            "reward_backlog_penalty": float(-reward_backlog),
            "reward_waiting_time_penalty": float(-reward_waiting_time),
            "reward_potential_shaping": float(reward_potential_shaping),
            "reward_congestion_penalty": float(-reward_congestion),
            "reward_tacc_bonus": float(reward_tacc),
        }
        reward_components_raw = {
            "reward_potential_shaping_raw": float(reward_potential_shaping_raw),
        }
        reward = float(sum(reward_components.values()))
        if self.steps >= self.config.max_horizon_steps:
            self.done = True
        if self.config.max_sim_time_sec is not None and self.current_time >= float(self.config.max_sim_time_sec):
            self.done = True

        waiting_churn_count = float(self._step_stats["waiting_churn_count"])
        onboard_churn_count = float(self._step_stats["onboard_churn_count"])
        waiting_churn_prob_mean = float(
            self._step_stats["waiting_churn_prob_sum"] / max(1.0, waiting_churn_count)
        )
        onboard_churn_prob_mean = float(
            self._step_stats["onboard_churn_prob_sum"] / max(1.0, onboard_churn_count)
        )

        info = {
            "served": float(self.served),
            "waiting_churned": float(self.waiting_churned),
            "waiting_timeouts": float(self.waiting_timeouts),
            "onboard_churned": float(self.onboard_churned),
            "structural_unserviceable": float(self.structurally_unserviceable),
            "total_requests": float(len(self.requests)),
            "structural_unserviceable_rate": float(
                self.structurally_unserviceable / max(1, len(self.requests))
            ),
            "algorithmic_churned": float(self.waiting_churned + self.onboard_churned),
            "step_served": float(self._step_stats["served"]),
            "step_waiting_churned": float(self._step_stats["waiting_churned"]),
            "step_waiting_timeouts": float(self._step_stats["waiting_timeouts"]),
            "step_onboard_churned": float(self._step_stats["onboard_churned"]),
            "step_waiting_churn_prob_sum": float(self._step_stats["waiting_churn_prob_sum"]),
            "step_waiting_churn_prob_weighted_sum": float(self._step_stats["waiting_churn_prob_weighted_sum"]),
            "step_waiting_churn_cvar": float(self._step_stats["waiting_churn_cvar"]),
            "step_waiting_churn_count": float(waiting_churn_count),
            "step_waiting_churn_prob_mean": float(waiting_churn_prob_mean),
            "step_onboard_delay_prob_sum": float(self._step_stats["onboard_delay_prob_sum"]),
            "step_onboard_churn_prob_sum": float(self._step_stats["onboard_churn_prob_sum"]),
            "step_onboard_churn_count": float(onboard_churn_count),
            "step_onboard_churn_prob_mean": float(onboard_churn_prob_mean),
            "step_tacc_gain": float(self._step_stats["tacc_gain"]),
            "step_boarded": float(self._step_stats["boarded"]),
            "step_denied_boarding": float(self._step_stats["denied_boarding"]),
            "step_travel_time_sec": float(travel_time),
            "reward_base_service": float(reward_base_service),
            "reward_waiting_churn_penalty": float(reward_waiting_churn),
            "reward_fairness_penalty": float(reward_fairness),
            "reward_cvar_penalty": float(reward_cvar),
            "reward_travel_cost": float(reward_travel_cost),
            "reward_onboard_delay_penalty": float(reward_onboard_delay),
            "reward_onboard_churn_penalty": float(reward_onboard_churn),
            "reward_backlog_penalty": float(reward_backlog),
            "reward_waiting_time_penalty": float(reward_waiting_time),
            "reward_potential_shaping": float(reward_potential_shaping),
            "reward_potential_shaping_raw": float(reward_potential_shaping_raw),
            "reward_congestion_penalty": float(reward_congestion),
            "reward_tacc_bonus": float(reward_tacc),
            "reward_scale": float(self.config.reward_scale),
            "reward_potential_alpha": float(self.config.reward_potential_alpha),
            "reward_potential_alpha_source": str(self.config.reward_potential_alpha_source),
            "reward_potential_lost_weight": float(self.config.reward_potential_lost_weight),
            "reward_potential_scale_with_reward_scale": bool(self.config.reward_potential_scale_with_reward_scale),
            "phi_before": float(phi_before),
            "phi_after": float(phi_after),
            "phi_delta": float(phi_delta),
            "phi_backlog_before": float(phi_backlog_before),
            "phi_backlog_after": float(phi_backlog_after),
            "lost_total_before": float(lost_total_before),
            "lost_total_after": float(lost_total_after),
            "waiting_churned_before": float(waiting_churned_before),
            "waiting_churned_after": float(waiting_churned_after),
            "onboard_churned_before": float(onboard_churned_before),
            "onboard_churned_after": float(onboard_churned_after),
            "structural_unserviceable_before": float(structural_before),
            "structural_unserviceable_after": float(structural_after),
            "waiting_remaining_before": float(waiting_remaining_before),
            "onboard_remaining_before": float(onboard_remaining_before),
            "waiting_remaining_after": float(waiting_remaining),
            "onboard_remaining_after": float(onboard_remaining),
            "dst_density_raw": float(dst_density_raw),
            "fleet_potential_dst": float(fleet_potential_dst),
            "reward_total": float(reward),
            "reward_components": dict(reward_components),
            "reward_components_raw": dict(reward_components_raw),
        }
        
        # Add backlog fields (always available)
        info["step_waiting_time_sec"] = float(waiting_time_sec)
        info["waiting_remaining"] = waiting_remaining
        info["onboard_remaining"] = onboard_remaining
        
        if self.done:
            # Determine done_reason
            if self.config.max_sim_time_sec is not None and self.current_time >= float(self.config.max_sim_time_sec):
                done_reason = "max_sim_time"
            elif self.steps >= self.config.max_horizon_steps:
                done_reason = "max_horizon"
            elif not self.event_queue:
                done_reason = "event_queue_empty"
            else:
                done_reason = "no_valid_action"
            info["done_reason"] = done_reason
            
            # Apply terminal backlog penalty (normalized)
            if self.config.reward_terminal_backlog_penalty > 0:
                norm_backlog = (waiting_remaining + onboard_remaining) / max(1, self.config.max_requests)
                terminal_penalty = self.config.reward_terminal_backlog_penalty * norm_backlog
                reward -= terminal_penalty
                info["terminal_backlog_penalty_applied"] = float(terminal_penalty)
            
            info["event_trace_digest"] = self._event_trace_digest()
            info["event_trace_length"] = float(len(self.event_log))
            info["service_by_stop"] = dict(self.service_count_by_stop)
            info["dropoffs_by_stop"] = dict(self.dropoff_count_by_stop)
            # Use aligned vector (all Layer-2 stops) for reproducible cross-baseline Gini
            info["service_gini"] = float(compute_service_volume_gini(
                self.service_count_by_stop, self.stop_ids
            ))
            self._append_episode_density_metrics(info)
        if self.config.debug_mask:
            info["mask_debug"] = self.last_mask_debug
        
        # Fleet density summary (FAEP logging - required when enabled)
        self._append_fleet_density_summary(info)
        
        return self._observe(), reward, self.done, info

    def _init_step_stats(self) -> Dict[str, float | List[float]]:
        return {
            "served": 0.0,
            "waiting_churned": 0.0,
            "waiting_timeouts": 0.0,
            "onboard_churned": 0.0,
            "waiting_churn_prob_sum": 0.0,
            "waiting_churn_prob_weighted_sum": 0.0,
            "waiting_churn_cvar": 0.0,
            "waiting_churn_count": 0.0,
            "waiting_churn_probs": [],
            "onboard_delay_prob_sum": 0.0,
            "onboard_churn_prob_sum": 0.0,
            "onboard_churn_count": 0.0,
            "onboard_churn_probs": [],
            "tacc_gain": 0.0,
            "boarded": 0.0,
            "denied_boarding": 0.0,
        }
