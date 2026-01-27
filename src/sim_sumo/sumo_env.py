"""SUMO-based environment for Stage 2 validation.

This module provides a SUMO/TraCI-based environment that mirrors the Stage 1
EventDrivenEnv interface but replaces static travel times with real SUMO dynamics.
It enables Sim-to-Real validation of trained policies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging
import os
import sys

import numpy as np
import networkx as nx
import pandas as pd

from src.utils.fairness import gini_coefficient
from src.utils.hard_mask import (
    DEFAULT_MAX_TIME_SEC,
    HardMaskGate,
    compute_hard_mask_gate,
    hard_deadline_over_by_sec,
    sanitize_time_sec,
)

LOG = logging.getLogger(__name__)


def _haversine_meters(lon1: np.ndarray, lat1: np.ndarray, lon2: np.ndarray, lat2: np.ndarray) -> np.ndarray:
    rad = np.pi / 180.0
    dlon = (lon2 - lon1) * rad
    dlat = (lat2 - lat1) * rad
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1 * rad) * np.cos(lat2 * rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371000.0 * c


@dataclass
class SUMOEnvConfig:
    """Configuration for SUMO-based environment."""
    
    sumo_cfg_path: str = ""
    sumo_binary: str = "sumo"
    sumo_gui: bool = False
    sumo_step_length: float = 1.0
    sumo_port: int = 8813
    sumo_seed: int = 7
    sumo_start_time: float = 0.0
    sumo_end_time: float = 36000.0
    sumo_warmup_steps: int = 100
    
    vehicle_type_id: str = "minibus"
    vehicle_speed_mps: float = 8.33
    
    mask_alpha: float = 1.5
    walk_threshold_sec: int = 600
    max_requests: int = 2000
    seed: int = 7
    num_vehicles: int = 1
    vehicle_capacity: int = 6
    request_timeout_sec: int = 600
    
    churn_tol_sec: int = 300
    churn_beta: float = 0.02
    waiting_churn_tol_sec: Optional[int] = None
    waiting_churn_beta: Optional[float] = None
    onboard_churn_tol_sec: Optional[int] = None
    onboard_churn_beta: Optional[float] = None
    
    cvar_alpha: float = 0.95
    fairness_gamma: float = 1.0
    
    debug_mask: bool = False
    
    od_glob: str = "data/processed/od_mapped/*.parquet"
    graph_nodes_path: str = "data/processed/graph/layer2_nodes.parquet"
    graph_edges_path: str = "data/processed/graph/layer2_edges.parquet"
    graph_embeddings_path: str = "data/processed/graph/node2vec_embeddings.parquet"
    
    stop_to_sumo_edge_path: str = "data/processed/graph/stop_to_sumo_edge.json"
    
    travel_time_fallback_multiplier: float = 1.2
    # Hard-mask robustness: avoid action-space collapse when deadlines are already missed.
    hard_mask_skip_unrecoverable: bool = False
    hard_mask_slack_sec: float = 0.0
    
    def __post_init__(self) -> None:
        if self.waiting_churn_tol_sec is None:
            self.waiting_churn_tol_sec = int(self.churn_tol_sec)
        if self.waiting_churn_beta is None:
            self.waiting_churn_beta = float(self.churn_beta)
        if self.onboard_churn_tol_sec is None:
            self.onboard_churn_tol_sec = int(self.churn_tol_sec)
        if self.onboard_churn_beta is None:
            self.onboard_churn_beta = float(self.churn_beta)
        if self.hard_mask_slack_sec < 0:
            raise ValueError("hard_mask_slack_sec must be >= 0")


@dataclass
class SUMOVehicleState:
    """State of a vehicle in SUMO simulation."""
    
    vehicle_id: int
    sumo_vehicle_id: str
    current_stop: int
    target_stop: Optional[int] = None
    available_time: float = 0.0
    onboard: List[dict] = field(default_factory=list)
    visit_counts: Dict[int, int] = field(default_factory=dict)
    
    departure_time: Optional[float] = None
    arrival_time: Optional[float] = None
    is_moving: bool = False
    
    prior_travel_time: float = 0.0
    actual_travel_time: float = 0.0


class SUMOEnv:
    """SUMO-based environment for Stage 2 validation.
    
    This environment mirrors the EventDrivenEnv interface but uses SUMO/TraCI
    for realistic traffic dynamics. Key differences from Stage 1:
    - Travel times are measured from actual SUMO simulation (not static priors)
    - Traffic lights, congestion, and vehicle interactions affect travel
    - Enables Sim-to-Real validation and delta metrics
    
    The observation/action interface remains identical to Stage 1 for policy compatibility.
    """
    
    def __init__(self, config: SUMOEnvConfig, traci_connection: Any = None) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.traci = traci_connection
        
        self._load_graph()
        self._load_requests()
        self._load_stop_mapping()
        
        self.sim_to_real_deltas: List[Dict[str, float]] = []
        
    def _load_graph(self) -> None:
        """Load Layer-2 logical stop graph (same as Stage 1)."""
        nodes = pd.read_parquet(self.config.graph_nodes_path)
        edges = pd.read_parquet(self.config.graph_edges_path)
        self._validate_layer2_graph(nodes, edges)
        self._load_embeddings(nodes)
        
        edges = edges.copy()
        edges["source"] = edges["source"].astype(int)
        edges["target"] = edges["target"].astype(int)
        edges["travel_time_sec"] = edges["travel_time_sec"].astype(float)
        
        self.stop_ids = nodes["gnn_node_id"].astype(int).tolist()
        
        if "lon" in nodes.columns and "lat" in nodes.columns:
            self.stop_coords = {
                int(stop_id): (float(lon), float(lat))
                for stop_id, lon, lat in nodes[["gnn_node_id", "lon", "lat"]].itertuples(index=False, name=None)
            }
        else:
            self.stop_coords = {int(stop_id): (0.0, 0.0) for stop_id in self.stop_ids}
        
        stop_index = {int(stop_id): idx for idx, stop_id in enumerate(self.stop_ids)}
        src_idx = edges["source"].map(stop_index).astype(int).to_numpy()
        dst_idx = edges["target"].map(stop_index).astype(int).to_numpy()
        self.graph_edge_index = np.stack([src_idx, dst_idx], axis=0).astype(np.int64)
        
        graph_edge_features = np.zeros((len(edges), 4), dtype=np.float32)
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
        
        self.neighbors: Dict[int, List[Tuple[int, float]]] = {}
        for src, dst, travel in edges.itertuples(index=False, name=None):
            self.neighbors.setdefault(int(src), []).append((int(dst), float(travel)))
        
        self.prior_travel_times: Dict[Tuple[int, int], float] = {}
        for src, dst, travel in edges.itertuples(index=False, name=None):
            self.prior_travel_times[(int(src), int(dst))] = float(travel)
        
        adj: Dict[int, set[int]] = {int(stop_id): set() for stop_id in self.stop_ids}
        for src, dst, _travel in edges.itertuples(index=False, name=None):
            adj[int(src)].add(int(dst))
            adj[int(dst)].add(int(src))
        self.adj_undirected = {key: sorted(list(value)) for key, value in adj.items()}
        
        isolated = [stop_id for stop_id in self.stop_ids if stop_id not in self.neighbors]
        if isolated:
            LOG.warning("Layer 2 graph has %d stop(s) with no outgoing edges", len(isolated))
    
    def _validate_layer2_graph(self, nodes: pd.DataFrame, edges: pd.DataFrame) -> None:
        """Validate Layer-2 graph schema."""
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
    
    def _load_embeddings(self, nodes: pd.DataFrame) -> None:
        """Load Node2Vec embeddings for geo features."""
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
    
    def _load_requests(self) -> None:
        """Load OD requests (same as Stage 1)."""
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
        if self.config.max_requests:
            od = od.iloc[: self.config.max_requests].copy()
        
        t0 = od["tpep_pickup_datetime"].iloc[0]
        od["request_time_sec"] = (od["tpep_pickup_datetime"] - t0).dt.total_seconds()
        
        scheduled = od[["pickup_stop_id", "dropoff_stop_id", "request_time_sec"]].to_dict("records")
        for idx, req in enumerate(scheduled):
            req["request_id"] = int(idx)
            req["source"] = "scheduled"
            req["structural_unreachable"] = bool(od.iloc[idx].get("structural_unreachable", False))
        
        self.requests = []
        self.request_index: Dict[int, dict] = {}
        for req in scheduled:
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
    
    def _load_stop_mapping(self) -> None:
        """Load mapping from logical stops to SUMO edges/positions."""
        mapping_path = Path(self.config.stop_to_sumo_edge_path)
        if mapping_path.exists():
            import json
            with open(mapping_path, "r", encoding="utf-8") as f:
                self.stop_to_sumo = json.load(f)
            LOG.info("Loaded stop-to-SUMO mapping from %s", mapping_path)
        else:
            self.stop_to_sumo = {}
            LOG.warning("No stop-to-SUMO mapping found at %s, will use coordinate-based routing", mapping_path)
    
    def _shortest_time(self, src: int, dst: int) -> float:
        """Compute shortest path travel time using prior graph."""
        if src == dst:
            return 0.0
        try:
            return float(nx.shortest_path_length(self.graph, src, dst, weight="travel_time_sec"))
        except nx.NetworkXNoPath:
            return float("inf")
    
    def set_traci(self, traci_connection: Any) -> None:
        """Set TraCI connection for SUMO interaction."""
        self.traci = traci_connection
    
    def reset(self) -> Dict[str, float]:
        """Reset environment state for new episode."""
        self.current_time = 0.0
        
        if self.config.num_vehicles < 1:
            raise ValueError("num_vehicles must be >= 1")
        if self.config.vehicle_capacity < 1:
            raise ValueError("vehicle_capacity must be >= 1")
        
        self.vehicles: List[SUMOVehicleState] = [
            SUMOVehicleState(
                vehicle_id=idx,
                sumo_vehicle_id=f"minibus_{idx}",
                current_stop=int(self.rng.choice(self.stop_ids)),
            )
            for idx in range(self.config.num_vehicles)
        ]
        
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
        
        self.sim_to_real_deltas = []
        self.cumulative_prior_travel = 0.0
        self.cumulative_actual_travel = 0.0
        
        self.ready_vehicle_ids: List[int] = []
        self.active_vehicle_id: Optional[int] = None
        
        for req in self.requests:
            req["status"] = None
            req["pickup_time_sec"] = None
            req.pop("cancel_reason", None)
            req.pop("t_max_sec", None)
        
        for req in self.requests:
            if req.get("structural_unserviceable", False):
                self.structurally_unserviceable += 1
                req["status"] = "structurally_unserviceable"
        
        self._step_stats = self._init_step_stats()
        
        for vehicle in self.vehicles:
            vehicle.visit_counts = {int(vehicle.current_stop): 1}
            self.ready_vehicle_ids.append(vehicle.vehicle_id)
        
        if self.ready_vehicle_ids:
            self.active_vehicle_id = self.ready_vehicle_ids.pop(0)
        
        return self._observe()
    
    def _observe(self) -> Dict[str, float]:
        """Return current observation dict."""
        active_vehicle = self._get_active_vehicle()
        return {
            "time_sec": self.current_time,
            "active_vehicle_id": float(self.active_vehicle_id) if self.active_vehicle_id is not None else -1.0,
            "current_stop": float(active_vehicle.current_stop if active_vehicle else -1),
            "onboard": float(len(active_vehicle.onboard) if active_vehicle else 0),
            "waiting": float(sum(len(v) for v in self.waiting.values())),
            "ready_vehicles": float(len(self.ready_vehicle_ids)),
        }
    
    def _get_active_vehicle(self) -> Optional[SUMOVehicleState]:
        """Get currently active vehicle."""
        if self.active_vehicle_id is None:
            return None
        return self.vehicles[self.active_vehicle_id]
    
    @property
    def current_stop(self) -> int:
        vehicle = self._get_active_vehicle()
        if vehicle is None and self.vehicles:
            vehicle = self.vehicles[0]
        return int(vehicle.current_stop) if vehicle else -1
    
    @property
    def onboard(self) -> List[dict]:
        vehicle = self._get_active_vehicle()
        if vehicle is None and self.vehicles:
            vehicle = self.vehicles[0]
        return vehicle.onboard if vehicle else []
    
    def _churn_prob(self, wait_sec: float, beta: float, tol_sec: float) -> float:
        """Compute sigmoid churn probability."""
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
        """Compute CVaR (Conditional Value at Risk) at alpha quantile."""
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
        """Compute per-stop waiting risk metrics."""
        risks: Dict[int, Tuple[float, float, int]] = {}
        for stop_id, queue in self.waiting.items():
            if not queue:
                risks[int(stop_id)] = (0.0, 0.0, 0)
                continue
            waits = [self.current_time - req["request_time_sec"] for req in queue]
            probs = [self._waiting_churn_prob(wait_sec) for wait_sec in waits]
            risks[int(stop_id)] = (float(np.mean(probs)), self._cvar(probs), len(queue))
        return risks
    
    def _k_hop_nodes(self, center: int, k: int) -> List[int]:
        """Get k-hop neighborhood nodes."""
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
    
    def get_action_mask(self, debug: bool = False) -> Tuple[List[int], List[bool]]:
        """Compute action mask based on hard constraints (same logic as Stage 1)."""
        vehicle = self._get_active_vehicle()
        if vehicle is None:
            return [], []
        candidates = self.neighbors.get(vehicle.current_stop, [])
        if not candidates:
            return [], []
        
        actions = [dst for dst, _ in candidates]
        mask = []
        debug_entries = []
        hard_mask_skipped = []
        hard_mask_gate_by_id: Dict[int, HardMaskGate] = {}
        if vehicle.onboard:
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
                    slack_sec=float(self.config.hard_mask_slack_sec),
                    max_time_sec=DEFAULT_MAX_TIME_SEC,
                    skip_unrecoverable=bool(self.config.hard_mask_skip_unrecoverable),
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
            
            dropoffs_at_dst = sum(1 for pax in vehicle.onboard if pax["dropoff_stop_id"] == dst)
            projected_onboard = len(vehicle.onboard) - dropoffs_at_dst
            capacity_left = self.config.vehicle_capacity - projected_onboard
            waiting_at_dst = len(self.waiting.get(int(dst), []))
            
            if capacity_left <= 0 and waiting_at_dst > 0:
                feasible = False
                violations.append({
                    "type": "capacity",
                    "vehicle_id": int(vehicle.vehicle_id),
                    "projected_onboard": int(projected_onboard),
                    "capacity": int(self.config.vehicle_capacity),
                    "waiting_at_dst": int(waiting_at_dst),
                })
            
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
                if over_by > float(self.config.hard_mask_slack_sec):
                    feasible = False
                    violations.append({
                        "type": "hard_mask",
                        "request_id": pax.get("request_id"),
                        "dropoff_stop_id": pax["dropoff_stop_id"],
                        "eta_sec": float(eta_total),
                        "t_max_sec": float(pax["t_max_sec"]),
                        "over_by_sec": float(over_by),
                        "travel_time_sec": float(travel),
                        "remaining_sec": float(remaining),
                    })
                    break
            
            mask.append(feasible)
            if violations:
                debug_entries.append({
                    "action": int(dst),
                    "vehicle_id": int(vehicle.vehicle_id),
                    "violations": violations,
                })
        
        if hard_mask_skipped and (debug or self.config.debug_mask):
            debug_entries.insert(
                0,
                {
                    "type": "hard_mask_skip_unrecoverable",
                    "vehicle_id": int(vehicle.vehicle_id),
                    "slack_sec": float(self.config.hard_mask_slack_sec),
                    "skipped": hard_mask_skipped,
                },
            )
        
        if debug or self.config.debug_mask:
            self.last_mask_debug = debug_entries
        
        return actions, mask
    
    def get_feature_batch(self, k_hop: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Get feature batch for policy inference (same interface as Stage 1)."""
        vehicle = self._get_active_vehicle()
        stop_index = {stop_id: idx for idx, stop_id in enumerate(self.stop_ids)}
        
        if vehicle is None:
            node_features = np.zeros((len(self.stop_ids), 5), dtype=np.float32)
            risks = self._compute_waiting_risks()
            for stop_id, (risk_mean, risk_cvar, count) in risks.items():
                idx = stop_index[stop_id]
                node_features[idx, 0] = float(risk_mean)
                node_features[idx, 1] = float(risk_cvar)
                node_features[idx, 2] = float(count)
                node_features[idx, 3] = self.fairness_weight.get(int(stop_id), 1.0)
                node_features[idx, 4] = self.geo_embedding_scalar[int(stop_id)]
            return {
                "node_features": node_features,
                "edge_features": np.zeros((0, 4), dtype=np.float32),
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
        risks = self._compute_waiting_risks()
        for stop_id, (risk_mean, risk_cvar, count) in risks.items():
            idx = stop_index[stop_id]
            node_features[idx, 0] = float(risk_mean)
            node_features[idx, 1] = float(risk_cvar)
            node_features[idx, 2] = float(count)
            node_features[idx, 3] = self.fairness_weight.get(int(stop_id), 1.0)
            node_features[idx, 4] = self.geo_embedding_scalar[int(stop_id)]
        
        actions, mask = self.get_action_mask(debug=self.config.debug_mask)
        edge_features = np.zeros((len(actions), 4), dtype=np.float32)
        
        for i, action in enumerate(actions):
            travel_time = dict(self.neighbors.get(vehicle.current_stop, [])).get(action, 0.0)
            delta_eta_max = 0.0
            onboard_risks_before: List[float] = []
            onboard_risks_after: List[float] = []
            count_violation = 0.0
            
            for pax in vehicle.onboard:
                curr_eta = self._shortest_time(vehicle.current_stop, pax["dropoff_stop_id"])
                new_eta = travel_time + self._shortest_time(action, pax["dropoff_stop_id"])
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
            edge_features[i, 0] = float(delta_eta_max)
            edge_features[i, 1] = float(delta_cvar)
            edge_features[i, 2] = float(count_violation)
            edge_features[i, 3] = float(travel_time)
        
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
                edge_attr_list.append([0.0, 0.0, 0.0, float(travel_time)])
        
        if src_list:
            sub_edge_index = np.array([src_list, dst_list], dtype=np.int64)
            sub_edge_features = np.array(edge_attr_list, dtype=np.float32)
        else:
            sub_edge_index = np.zeros((2, 0), dtype=np.int64)
            sub_edge_features = np.zeros((0, 4), dtype=np.float32)
        
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
    
    def _init_step_stats(self) -> Dict[str, Any]:
        """Initialize per-step statistics."""
        return {
            "served": 0.0,
            "waiting_churned": 0.0,
            "waiting_timeouts": 0.0,
            "onboard_churned": 0.0,
            "waiting_churn_prob_sum": 0.0,
            "waiting_churn_prob_weighted_sum": 0.0,
            "waiting_churn_cvar": 0.0,
            "waiting_churn_probs": [],
            "onboard_delay_prob_sum": 0.0,
            "onboard_churn_prob_sum": 0.0,
            "onboard_churn_probs": [],
            "tacc_gain": 0.0,
            "boarded": 0.0,
            "denied_boarding": 0.0,
        }
    
    def get_prior_travel_time(self, src: int, dst: int) -> float:
        """Get prior (Stage 1) travel time between stops."""
        return self.prior_travel_times.get((int(src), int(dst)), float("inf"))
    
    def record_travel_delta(self, src: int, dst: int, prior_sec: float, actual_sec: float) -> None:
        """Record sim-to-real travel time delta."""
        delta = actual_sec - prior_sec
        delta_ratio = delta / prior_sec if prior_sec > 0 else 0.0
        
        self.sim_to_real_deltas.append({
            "src_stop": int(src),
            "dst_stop": int(dst),
            "prior_sec": float(prior_sec),
            "actual_sec": float(actual_sec),
            "delta_sec": float(delta),
            "delta_ratio": float(delta_ratio),
        })
        
        self.cumulative_prior_travel += prior_sec
        self.cumulative_actual_travel += actual_sec
    
    def get_sim_to_real_summary(self) -> Dict[str, float]:
        """Get summary statistics of sim-to-real deltas."""
        if not self.sim_to_real_deltas:
            return {
                "num_trips": 0,
                "mean_delta_sec": 0.0,
                "std_delta_sec": 0.0,
                "mean_delta_ratio": 0.0,
                "cumulative_prior_sec": 0.0,
                "cumulative_actual_sec": 0.0,
                "cumulative_delta_sec": 0.0,
            }
        
        deltas = [d["delta_sec"] for d in self.sim_to_real_deltas]
        ratios = [d["delta_ratio"] for d in self.sim_to_real_deltas]
        
        return {
            "num_trips": len(self.sim_to_real_deltas),
            "mean_delta_sec": float(np.mean(deltas)),
            "std_delta_sec": float(np.std(deltas)),
            "mean_delta_ratio": float(np.mean(ratios)),
            "cumulative_prior_sec": float(self.cumulative_prior_travel),
            "cumulative_actual_sec": float(self.cumulative_actual_travel),
            "cumulative_delta_sec": float(self.cumulative_actual_travel - self.cumulative_prior_travel),
        }
