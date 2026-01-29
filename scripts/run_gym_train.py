"""Diagnostic baseline CLI entrypoint for Gym training (not paper-aligned)."""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import load_config
from src.env.gym_env import EnvConfig, EventDrivenEnv
from src.models.edge_q_gnn import EdgeQGNN


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    logging.warning(
        "DIAGNOSTIC BASELINE ONLY: This script is not paper-aligned. It does NOT use DQNTrainer, "
        "curriculum learning, or the full Layer-2 graph message passing. It uses a star-edge-only "
        "edge_index from the current stop and is meant for quick smoke checks."
    )
    cfg = load_config(args.config)
    env_cfg = cfg.get("env", {})
    env = EventDrivenEnv(
        EnvConfig(
            max_horizon_steps=int(env_cfg.get("max_horizon_steps", 200)),
            mask_alpha=float(env_cfg.get("mask_alpha", 1.5)),
            walk_threshold_sec=int(env_cfg.get("walk_threshold_sec", 600)),
            max_requests=int(env_cfg.get("max_requests", 2000)),
            seed=int(env_cfg.get("seed", 7)),
            num_vehicles=int(env_cfg.get("num_vehicles", 1)),
            vehicle_capacity=int(env_cfg.get("vehicle_capacity", 6)),
            request_timeout_sec=int(env_cfg.get("request_timeout_sec", 600)),
            realtime_request_rate_per_sec=float(env_cfg.get("realtime_request_rate_per_sec", 0.0)),
            realtime_request_count=int(env_cfg.get("realtime_request_count", 0)),
            realtime_request_end_sec=float(env_cfg.get("realtime_request_end_sec", 0.0)),
            churn_tol_sec=int(env_cfg.get("churn_tol_sec", 300)),
            churn_beta=float(env_cfg.get("churn_beta", 0.02)),
            waiting_churn_tol_sec=env_cfg.get("waiting_churn_tol_sec"),
            waiting_churn_beta=env_cfg.get("waiting_churn_beta"),
            onboard_churn_tol_sec=env_cfg.get("onboard_churn_tol_sec"),
            onboard_churn_beta=env_cfg.get("onboard_churn_beta"),
            reward_service=float(env_cfg.get("reward_service", 1.0)),
            reward_waiting_churn_penalty=float(env_cfg.get("reward_waiting_churn_penalty", 1.0)),
            reward_onboard_churn_penalty=float(env_cfg.get("reward_onboard_churn_penalty", 1.0)),
            reward_travel_cost_per_sec=float(env_cfg.get("reward_travel_cost_per_sec", 0.0)),
            reward_tacc_weight=float(env_cfg.get("reward_tacc_weight", 1.0)),
            reward_onboard_delay_weight=float(env_cfg.get("reward_onboard_delay_weight", 0.1)),
            reward_cvar_penalty=float(env_cfg.get("reward_cvar_penalty", 1.0)),
            reward_fairness_weight=float(env_cfg.get("reward_fairness_weight", 1.0)),
            cvar_alpha=float(env_cfg.get("cvar_alpha", 0.95)),
            fairness_gamma=float(env_cfg.get("fairness_gamma", 1.0)),
            travel_time_multiplier=float(env_cfg.get("travel_time_multiplier", 1.0)),
            debug_mask=bool(env_cfg.get("debug_mask", False)),
            od_glob=env_cfg.get("od_glob", "data/processed/od_mapped/*.parquet"),
            graph_nodes_path=env_cfg.get("graph_nodes_path", "data/processed/graph/layer2_nodes.parquet"),
            graph_edges_path=env_cfg.get("graph_edges_path", "data/processed/graph/layer2_edges.parquet"),
            graph_embeddings_path=env_cfg.get(
                "graph_embeddings_path",
                "data/processed/graph/node2vec_embeddings.parquet",
            ),
        )
    )

    obs = env.reset()
    features = env.get_feature_batch()
    logging.info(
        "Feature shapes: nodes=%s edges=%s mask=%s",
        features["node_features"].shape,
        features["edge_features"].shape,
        features["action_mask"].shape,
    )

    model = EdgeQGNN(node_dim=5, edge_dim=4, hidden_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    total_reward = 0.0
    steps = 0
    while True:
        features = env.get_feature_batch()
        actions = features["actions"]
        action_node_indices = features["action_node_indices"]
        mask = features["action_mask"]
        if len(actions) == 0:
            break

        node_ids = features["node_ids"]
        node_index = {int(node_id): idx for idx, node_id in enumerate(node_ids.tolist())}
        current_stop_id = int(features["current_stop"][0])
        current_idx = int(node_index.get(current_stop_id, -1))
        if current_idx < 0:
            raise ValueError(f"Missing current_stop in node_ids: {current_stop_id}")

        edge_index = torch.tensor(
            [
                [current_idx] * len(actions),
                action_node_indices.tolist(),
            ],
            dtype=torch.long,
        )
        data = {
            "node_features": torch.tensor(features["node_features"], dtype=torch.float32),
            "edge_features": torch.tensor(features["edge_features"], dtype=torch.float32),
            "edge_index": edge_index,
        }
        q_values = model(data)

        valid_indices = [i for i, ok in enumerate(mask) if ok]
        if not valid_indices:
            break

        chosen_idx = int(valid_indices[int(torch.argmax(q_values[valid_indices]).item())])
        action = int(actions[chosen_idx])

        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1

        target = torch.tensor(float(reward), dtype=torch.float32)
        loss = (q_values[chosen_idx] - target) ** 2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if done:
            break

    logging.info(
        "Episode finished in %d steps, reward=%.2f, served=%d, waiting_churned=%d",
        steps,
        total_reward,
        int(info.get("served", 0)),
        int(info.get("waiting_churned", 0)),
    )
    logging.info("Structural unserviceable: %s", int(info.get("structural_unserviceable", 0)))
    service_by_stop = info.get("service_by_stop", {})
    waiting_by_stop = {int(stop_id): len(queue) for stop_id, queue in env.waiting.items()}
    nonzero_service = {int(k): int(v) for k, v in service_by_stop.items() if v}
    nonzero_waiting = {int(k): int(v) for k, v in waiting_by_stop.items() if v}
    logging.info("Service by stop (nonzero): %s", nonzero_service)
    logging.info("Waiting by stop (nonzero): %s", nonzero_waiting)


if __name__ == "__main__":
    main()
