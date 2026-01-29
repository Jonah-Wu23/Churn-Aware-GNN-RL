"""SUMO-based evaluator for Stage 2 validation.

This module implements a unified evaluator that runs trained policies in SUMO
and produces metrics compatible with the Stage 1 evaluator, plus sim-to-real
delta statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import json
import logging
import time

import numpy as np
import pandas as pd
import torch

from src.models.edge_q_gnn import EdgeQGNN
from src.utils.hashing import sha256_file
from src.utils.fairness import compute_service_volume_gini

from .sumo_env import SUMOEnv, SUMOEnvConfig
from .traci_adapter import (
    TraCIConfig,
    TraCIConnection,
    TraCISimulationRunner,
    StopRouteManager,
    SUMOVehicleController,
    TRACI_AVAILABLE,
)

LOG = logging.getLogger(__name__)


@dataclass
class SUMOEvalConfig:
    """Configuration for SUMO evaluation."""
    
    episodes: int = 5
    seed: int = 7
    policy: str = "edgeq"
    model_path: Optional[str] = None
    device: str = "cpu"
    max_steps: Optional[int] = None
    
    sumo_cfg_path: str = ""
    sumo_gui: bool = False
    sumo_warmup_steps: int = 100
    sumo_step_length: float = 1.0
    
    stop_to_sumo_edge_path: str = "data/processed/graph/stop_to_sumo_edge.json"





def _compute_wait_times(requests: List[dict]) -> List[float]:
    """Extract wait times from served requests."""
    waits = []
    for req in requests:
        pickup = req.get("pickup_time_sec")
        if pickup is None:
            continue
        wait = float(pickup) - float(req["request_time_sec"])
        waits.append(max(0.0, wait))
    return waits


def _compute_metrics(env: SUMOEnv, total_tacc: float) -> Dict[str, float]:
    """Compute metrics matching Stage 1 evaluator schema."""
    total_requests = float(len(env.requests))
    structural = float(env.structurally_unserviceable)
    waiting_churned = float(env.waiting_churned)
    waiting_timeouts = float(env.waiting_timeouts)
    onboard_churned = float(env.onboard_churned)
    served = float(env.served)
    
    non_structural = max(0.0, total_requests - structural)
    waiting_total = waiting_churned + waiting_timeouts
    algorithmic = waiting_total + onboard_churned
    
    service_rate = served / non_structural if non_structural > 0 else 0.0
    waiting_churn_rate = waiting_total / non_structural if non_structural > 0 else 0.0
    onboard_churn_rate = onboard_churned / non_structural if non_structural > 0 else 0.0
    algorithmic_churn_rate = algorithmic / non_structural if non_structural > 0 else 0.0
    structural_rate = structural / total_requests if total_requests > 0 else 0.0
    
    wait_times = _compute_wait_times(env.requests)
    wait_p95 = float(np.percentile(wait_times, 95)) if wait_times else 0.0
    
    # Use aligned vector (all Layer-2 stops) for reproducible cross-baseline Gini
    gini = compute_service_volume_gini(env.service_count_by_stop, env.stop_ids)
    
    return {
        "total_requests": total_requests,
        "served": served,
        "waiting_churned": waiting_churned,
        "waiting_timeouts": waiting_timeouts,
        "onboard_churned": onboard_churned,
        "structural_unserviceable": structural,
        "service_rate": float(service_rate),
        "waiting_churn_rate": float(waiting_churn_rate),
        "onboard_churn_rate": float(onboard_churn_rate),
        "algorithmic_churn_rate": float(algorithmic_churn_rate),
        "structural_unserviceable_rate": float(structural_rate),
        "tacc_total": float(total_tacc),
        "wait_time_p95_sec": float(wait_p95),
        "service_gini": float(gini),
    }


def _random_policy(features: Dict[str, np.ndarray], rng: np.random.Generator) -> Optional[int]:
    """Random policy for baseline comparison."""
    actions = features["actions"].astype(np.int64)
    mask = features["action_mask"].astype(bool)
    if len(actions) == 0:
        return None
    valid = np.where(mask)[0]
    if len(valid) == 0:
        return None
    idx = int(rng.choice(valid))
    return int(actions[idx])


def _greedy_policy(features: Dict[str, np.ndarray]) -> Optional[int]:
    """Greedy policy based on risk scores."""
    actions = features["actions"].astype(np.int64)
    mask = features["action_mask"].astype(bool)
    if len(actions) == 0:
        return None
    valid = np.where(mask)[0]
    if len(valid) == 0:
        return None
    
    node_features = features["node_features"]
    action_nodes = features["action_node_indices"].astype(np.int64)
    edge_features = features["edge_features"]
    
    scores = []
    for idx in valid:
        node_idx = int(action_nodes[idx])
        risk_mean = float(node_features[node_idx, 0])
        risk_cvar = float(node_features[node_idx, 1])
        count = float(node_features[node_idx, 2])
        fairness = float(node_features[node_idx, 3])
        edge = edge_features[idx]
        travel_time = float(edge[3])
        delta_eta = float(edge[0])
        delta_cvar = float(edge[1])
        score = (risk_mean + risk_cvar) * fairness + 0.1 * count - 0.001 * travel_time - 0.1 * delta_eta - 0.1 * delta_cvar
        scores.append((score, idx))
    
    best_idx = max(scores, key=lambda item: item[0])[1]
    return int(actions[best_idx])


def _edgeq_policy(
    features: Dict[str, np.ndarray],
    model: EdgeQGNN,
    device: torch.device,
) -> Optional[int]:
    """Edge-Q GNN policy using trained model."""
    actions = features["actions"].astype(np.int64)
    mask = features["action_mask"].astype(bool)
    if len(actions) == 0:
        return None
    valid = np.where(mask)[0]
    if len(valid) == 0:
        return None
    
    x = torch.tensor(features["node_features"], dtype=torch.float32, device=device)
    graph_edge_index = torch.tensor(features["graph_edge_index"], dtype=torch.long, device=device)
    graph_edge_features = torch.tensor(features["graph_edge_features"], dtype=torch.float32, device=device)
    dst = torch.tensor(features["action_node_indices"], dtype=torch.long, device=device)
    src = torch.full_like(dst, int(features["current_node_index"][0]), dtype=torch.long, device=device)
    action_edge_index = torch.stack([src, dst], dim=0)
    action_edge_attr = torch.tensor(features["edge_features"], dtype=torch.float32, device=device)
    
    data = {
        "node_features": x,
        "graph_edge_index": graph_edge_index,
        "graph_edge_features": graph_edge_features,
        "action_edge_index": action_edge_index,
        "edge_features": action_edge_attr,
    }
    
    with torch.no_grad():
        q = model(data).detach().cpu().numpy()
    
    q_masked = np.copy(q)
    q_masked[~mask] = -1e9
    idx = int(np.argmax(q_masked))
    return int(actions[idx])


def _build_sumo_env_config(env_cfg: Dict[str, Any], sumo_cfg: Dict[str, Any]) -> SUMOEnvConfig:
    """Build SUMOEnvConfig from YAML config dicts."""
    return SUMOEnvConfig(
        sumo_cfg_path=str(sumo_cfg.get("sumo_cfg_path", "")),
        sumo_binary=str(sumo_cfg.get("sumo_binary", "sumo")),
        sumo_gui=bool(sumo_cfg.get("sumo_gui", False)),
        sumo_step_length=float(sumo_cfg.get("sumo_step_length", 1.0)),
        sumo_port=int(sumo_cfg.get("sumo_port", 8813)),
        sumo_seed=int(sumo_cfg.get("sumo_seed", env_cfg.get("seed", 7))),
        sumo_warmup_steps=int(sumo_cfg.get("sumo_warmup_steps", 100)),
        
        mask_alpha=float(env_cfg.get("mask_alpha", 1.5)),
        walk_threshold_sec=int(env_cfg.get("walk_threshold_sec", 600)),
        max_requests=int(env_cfg.get("max_requests", 2000)),
        seed=int(env_cfg.get("seed", 7)),
        num_vehicles=int(env_cfg.get("num_vehicles", 1)),
        vehicle_capacity=int(env_cfg.get("vehicle_capacity", 6)),
        request_timeout_sec=int(env_cfg.get("request_timeout_sec", 600)),
        
        churn_tol_sec=int(env_cfg.get("churn_tol_sec", 300)),
        churn_beta=float(env_cfg.get("churn_beta", 0.02)),
        waiting_churn_tol_sec=env_cfg.get("waiting_churn_tol_sec"),
        waiting_churn_beta=env_cfg.get("waiting_churn_beta"),
        onboard_churn_tol_sec=env_cfg.get("onboard_churn_tol_sec"),
        onboard_churn_beta=env_cfg.get("onboard_churn_beta"),
        
        cvar_alpha=float(env_cfg.get("cvar_alpha", 0.95)),
        fairness_gamma=float(env_cfg.get("fairness_gamma", 1.0)),
        
        debug_mask=bool(env_cfg.get("debug_mask", False)),
        
        od_glob=env_cfg.get("od_glob", "data/processed/od_mapped/*.parquet"),
        graph_nodes_path=env_cfg.get("graph_nodes_path", "data/processed/graph/layer2_nodes.parquet"),
        graph_edges_path=env_cfg.get("graph_edges_path", "data/processed/graph/layer2_edges.parquet"),
        graph_embeddings_path=env_cfg.get(
            "graph_embeddings_path",
            "data/processed/graph/node2vec_embeddings.parquet",
        ),
        stop_to_sumo_edge_path=sumo_cfg.get(
            "stop_to_sumo_edge_path",
            "data/processed/graph/stop_to_sumo_edge.json",
        ),
        hard_mask_skip_unrecoverable=bool(env_cfg.get("hard_mask_skip_unrecoverable", False)),
        hard_mask_slack_sec=float(env_cfg.get("hard_mask_slack_sec", 0.0)),
    )


def _load_stop_to_edge_mapping(path: str) -> Dict[int, str]:
    """Load stop-to-SUMO-edge mapping from JSON file."""
    mapping_path = Path(path)
    if not mapping_path.exists():
        LOG.warning("Stop-to-edge mapping not found at %s, using empty mapping", path)
        return {}
    
    with open(mapping_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    
    return {int(k): str(v) for k, v in raw.items()}


def _build_hashes(env_cfg: Dict[str, Any]) -> tuple:
    """Build hash fingerprints for reproducibility."""
    graph_files = [
        env_cfg.get("graph_nodes_path", "data/processed/graph/layer2_nodes.parquet"),
        env_cfg.get("graph_edges_path", "data/processed/graph/layer2_edges.parquet"),
        env_cfg.get("graph_embeddings_path", "data/processed/graph/node2vec_embeddings.parquet"),
    ]
    graph_hashes = {}
    for path in graph_files:
        if Path(path).exists():
            graph_hashes[path] = sha256_file(path)
    
    od_glob = env_cfg.get("od_glob", "data/processed/od_mapped/*.parquet")
    od_paths = list(Path().glob(od_glob))
    od_hashes = {}
    for path in od_paths:
        od_hashes[str(path)] = sha256_file(str(path))
    
    return graph_hashes, od_hashes


class SUMOEpisodeRunner:
    """Runs a single SUMO evaluation episode without TraCI (simulation mode)."""
    
    def __init__(
        self,
        env: SUMOEnv,
        policy_fn: Callable[[Dict[str, np.ndarray]], Optional[int]],
        max_steps: int = 1000,
    ) -> None:
        self.env = env
        self.policy_fn = policy_fn
        self.max_steps = max_steps
    
    def run(self) -> Dict[str, Any]:
        """Run episode using prior travel times (no SUMO connection)."""
        self.env.reset()
        
        total_tacc = 0.0
        steps = 0
        
        for req in self.env.requests:
            if req["status"] is not None:
                continue
            if req.get("structural_unserviceable", False):
                continue
            req["status"] = "waiting"
            pickup_stop = req["pickup_stop_id"]
            self.env.waiting[pickup_stop].append(req)
        
        while steps < self.max_steps and not self.env.done:
            if not self.env.ready_vehicle_ids and self.env.active_vehicle_id is None:
                break
            
            if self.env.active_vehicle_id is None and self.env.ready_vehicle_ids:
                self.env.active_vehicle_id = self.env.ready_vehicle_ids.pop(0)
            
            features = self.env.get_feature_batch()
            action = self.policy_fn(features)
            
            if action is None:
                break
            
            vehicle = self.env._get_active_vehicle()
            if vehicle is None:
                break
            
            prior_travel = self.env.get_prior_travel_time(vehicle.current_stop, action)
            if not np.isfinite(prior_travel):
                prior_travel = 300.0
            
            self.env.current_time += prior_travel
            
            old_stop = vehicle.current_stop
            vehicle.current_stop = action
            vehicle.visit_counts[action] = vehicle.visit_counts.get(action, 0) + 1
            
            self.env.record_travel_delta(old_stop, action, prior_travel, prior_travel)
            
            stop_id = vehicle.current_stop
            dropped = [p for p in vehicle.onboard if p["dropoff_stop_id"] == stop_id]
            vehicle.onboard = [p for p in vehicle.onboard if p["dropoff_stop_id"] != stop_id]
            
            for pax in dropped:
                pax["status"] = "served"
                self.env.served += 1
                self.env.dropoff_count_by_stop[stop_id] += 1
                total_tacc += pax.get("direct_time_sec", 0.0)
            
            capacity_left = self.env.config.vehicle_capacity - len(vehicle.onboard)
            queue = self.env.waiting.get(stop_id, [])
            boarded = queue[:capacity_left]
            self.env.waiting[stop_id] = queue[capacity_left:]
            
            for req in boarded:
                req["pickup_time_sec"] = self.env.current_time
                req["t_max_sec"] = self.env.config.mask_alpha * req.get("direct_time_sec", float("inf"))
                req["status"] = "onboard"
                vehicle.onboard.append(req)
                wait_sec = self.env.current_time - req["request_time_sec"]
                self.env.acc_wait_time_by_stop[stop_id] += wait_sec
            
            if boarded:
                self.env.service_count_by_stop[stop_id] += len(boarded)
            
            self._apply_churn()
            
            self.env.active_vehicle_id = None
            if self.env.ready_vehicle_ids:
                self.env.active_vehicle_id = self.env.ready_vehicle_ids.pop(0)
            else:
                self.env.ready_vehicle_ids.append(vehicle.vehicle_id)
                self.env.active_vehicle_id = self.env.ready_vehicle_ids.pop(0)
            
            steps += 1
            
            if steps >= self.max_steps:
                self.env.done = True
        
        metrics = _compute_metrics(self.env, total_tacc)
        metrics["steps"] = float(steps)
        metrics["sim_to_real"] = self.env.get_sim_to_real_summary()
        
        return metrics
    
    def _apply_churn(self) -> None:
        """Apply churn to waiting and onboard passengers."""
        for stop_id, queue in self.env.waiting.items():
            remain = []
            for req in queue:
                wait_sec = self.env.current_time - req["request_time_sec"]
                
                if wait_sec > self.env.config.request_timeout_sec:
                    req["status"] = "churned_waiting"
                    req["cancel_reason"] = "timeout"
                    self.env.waiting_timeouts += 1
                    self.env.canceled_requests.append(req)
                    continue
                
                prob = self.env._waiting_churn_prob(wait_sec)
                if self.env.rng.random() < prob:
                    req["status"] = "churned_waiting"
                    req["cancel_reason"] = "probabilistic_churn"
                    self.env.waiting_churned += 1
                    self.env.canceled_requests.append(req)
                else:
                    remain.append(req)
            
            self.env.waiting[stop_id] = remain
        
        for vehicle in self.env.vehicles:
            remain = []
            for pax in vehicle.onboard:
                pickup_time = pax.get("pickup_time_sec", self.env.current_time)
                elapsed = self.env.current_time - pickup_time
                delay = max(0.0, elapsed - pax.get("direct_time_sec", elapsed))
                
                prob = self.env._onboard_churn_prob(delay)
                if self.env.rng.random() < prob:
                    pax["status"] = "churned_onboard"
                    pax["cancel_reason"] = "onboard_churn"
                    self.env.onboard_churned += 1
                    self.env.canceled_requests.append(pax)
                else:
                    remain.append(pax)
            
            vehicle.onboard = remain


def evaluate_sumo(
    config: Dict[str, Any],
    config_path: str | Path,
    run_dir: Optional[Path] = None,
    model_path_override: Optional[str] = None,
    use_traci: bool = False,
) -> Path:
    """Run SUMO evaluation and produce metrics compatible with Stage 1.
    
    Args:
        config: Full configuration dict from YAML
        config_path: Path to configuration file
        run_dir: Output directory for results
        model_path_override: Override model path from config
        use_traci: Whether to use actual SUMO/TraCI (requires SUMO installation)
    
    Returns:
        Path to output JSON file
    """
    env_cfg = config.get("env", {})
    eval_cfg = config.get("eval", {})
    sumo_cfg = config.get("sumo", {})
    model_cfg = config.get("model", {})
    
    eval_config = SUMOEvalConfig(
        episodes=int(eval_cfg.get("episodes", 5)),
        seed=int(eval_cfg.get("seed", env_cfg.get("seed", 7))),
        policy=str(eval_cfg.get("policy", "edgeq")),
        model_path=model_path_override or eval_cfg.get("model_path"),
        device=str(eval_cfg.get("device", "cpu")),
        max_steps=eval_cfg.get("max_steps") or int(env_cfg.get("max_horizon_steps", 1000)),
        sumo_cfg_path=str(sumo_cfg.get("sumo_cfg_path", "")),
        sumo_gui=bool(sumo_cfg.get("sumo_gui", False)),
        sumo_warmup_steps=int(sumo_cfg.get("sumo_warmup_steps", 100)),
        stop_to_sumo_edge_path=sumo_cfg.get(
            "stop_to_sumo_edge_path",
            "data/processed/graph/stop_to_sumo_edge.json",
        ),
    )
    
    if run_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = Path("reports") / "sumo_eval" / f"{eval_config.policy}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(eval_config.device)
    model: Optional[EdgeQGNN] = None
    
    if eval_config.policy == "edgeq":
        if not eval_config.model_path:
            raise ValueError("eval.model_path is required for edgeq policy in SUMO evaluation")
        
        model = EdgeQGNN(
            node_dim=int(model_cfg.get("node_dim", 5)),
            edge_dim=int(model_cfg.get("edge_dim", 4)),
            hidden_dim=int(model_cfg.get("hidden_dim", 32)),
            num_layers=int(model_cfg.get("num_layers", 2)),
            dropout=float(model_cfg.get("dropout", 0.0)),
            dueling=bool(model_cfg.get("dueling", False)),
        )
        model.load_state_dict(torch.load(eval_config.model_path, map_location=device))
        model.to(device)
        model.eval()
        LOG.info("Loaded Edge-Q model from %s", eval_config.model_path)
    
    sumo_env_config = _build_sumo_env_config(env_cfg, sumo_cfg)
    stop_to_edge = _load_stop_to_edge_mapping(eval_config.stop_to_sumo_edge_path)
    
    episode_rows: List[Dict[str, float]] = []
    sim_to_real_all: List[Dict[str, Any]] = []
    rng = np.random.default_rng(int(eval_config.seed))
    
    LOG.info("Starting SUMO evaluation: %d episodes, policy=%s, use_traci=%s",
             eval_config.episodes, eval_config.policy, use_traci)
    
    for ep in range(int(eval_config.episodes)):
        ep_seed = int(eval_config.seed) + ep
        sumo_env_config.seed = ep_seed
        sumo_env_config.sumo_seed = ep_seed
        
        env = SUMOEnv(sumo_env_config)
        
        def make_policy_fn():
            if eval_config.policy == "random":
                ep_rng = np.random.default_rng(ep_seed)
                return lambda f: _random_policy(f, ep_rng)
            elif eval_config.policy == "greedy":
                return _greedy_policy
            elif eval_config.policy == "edgeq":
                return lambda f: _edgeq_policy(f, model, device)
            else:
                raise ValueError(f"Unknown policy: {eval_config.policy}")
        
        policy_fn = make_policy_fn()
        
        if use_traci and TRACI_AVAILABLE and eval_config.sumo_cfg_path:
            traci_config = TraCIConfig(
                sumo_cfg_path=eval_config.sumo_cfg_path,
                sumo_gui=eval_config.sumo_gui,
                sumo_seed=ep_seed,
                sumo_warmup_steps=eval_config.sumo_warmup_steps,
            )
            
            runner = TraCISimulationRunner(
                traci_config=traci_config,
                sumo_env=env,
                policy_fn=policy_fn,
                stop_to_edge=stop_to_edge,
            )
            
            try:
                metrics = runner.run_episode(max_steps=eval_config.max_steps or 1000)
            except Exception as e:
                LOG.error("TraCI episode %d failed: %s", ep, e)
                continue
        else:
            if use_traci and not TRACI_AVAILABLE:
                LOG.warning("TraCI requested but not available, using simulation mode")
            
            runner = SUMOEpisodeRunner(
                env=env,
                policy_fn=policy_fn,
                max_steps=eval_config.max_steps or 1000,
            )
            metrics = runner.run()
        
        metrics["episode_index"] = float(ep)
        metrics["seed"] = float(ep_seed)
        
        sim_to_real = metrics.pop("sim_to_real", {})
        sim_to_real["episode_index"] = ep
        sim_to_real_all.append(sim_to_real)
        
        episode_rows.append(metrics)
        
        LOG.info("Episode %d: served=%.0f, service_rate=%.3f, steps=%.0f",
                 ep, metrics.get("served", 0), metrics.get("service_rate", 0), metrics.get("steps", 0))
    
    if not episode_rows:
        raise RuntimeError("No episodes completed successfully")
    
    df = pd.DataFrame(episode_rows)
    aggregate = {
        "episodes": float(len(df)),
        "policy": eval_config.policy,
        "use_traci": use_traci,
    }
    
    for col in df.columns:
        if col in {"episode_index", "seed"}:
            continue
        aggregate[f"{col}_mean"] = float(df[col].mean())
        aggregate[f"{col}_std"] = float(df[col].std(ddof=0))
    
    sim_to_real_agg = {
        "num_episodes": len(sim_to_real_all),
    }
    if sim_to_real_all:
        all_deltas = [s.get("mean_delta_sec", 0.0) for s in sim_to_real_all]
        all_ratios = [s.get("mean_delta_ratio", 0.0) for s in sim_to_real_all]
        sim_to_real_agg["mean_delta_sec_across_episodes"] = float(np.mean(all_deltas))
        sim_to_real_agg["mean_delta_ratio_across_episodes"] = float(np.mean(all_ratios))
    
    graph_hashes, od_hashes = _build_hashes(env_cfg)
    
    meta = {
        "config_path": str(config_path),
        "config_sha256": sha256_file(str(config_path)) if Path(config_path).exists() else "",
        "eval_config": {
            "episodes": eval_config.episodes,
            "seed": eval_config.seed,
            "policy": eval_config.policy,
            "model_path": eval_config.model_path,
            "device": eval_config.device,
            "max_steps": eval_config.max_steps,
            "sumo_cfg_path": eval_config.sumo_cfg_path,
            "use_traci": use_traci,
        },
        "env_config": env_cfg,
        "sumo_config": sumo_cfg,
        "model_config": model_cfg,
        "graph_hashes": graph_hashes,
        "od_hashes": od_hashes,
    }
    
    output = {
        "meta": meta,
        "aggregate": aggregate,
        "sim_to_real_summary": sim_to_real_agg,
        "episodes": episode_rows,
        "sim_to_real_per_episode": sim_to_real_all,
    }
    
    output_path = run_dir / "sumo_eval_results.json"
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    
    df.to_csv(run_dir / "sumo_eval_episodes.csv", index=False)
    
    sim_to_real_df = pd.DataFrame(sim_to_real_all)
    sim_to_real_df.to_csv(run_dir / "sim_to_real_deltas.csv", index=False)
    
    LOG.info("SUMO evaluation complete. Results saved to %s", output_path)
    
    return output_path
