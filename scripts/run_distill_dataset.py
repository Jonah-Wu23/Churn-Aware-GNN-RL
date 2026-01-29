"""Generate distillation dataset from a teacher EdgeQ model."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.env.gym_env import EventDrivenEnv
from src.train.runner import _build_env_config
from src.models.edge_q_gnn import EdgeQGNN
from src.utils.config import load_config
from src.utils.distill_features import build_action_vectors
from src.utils.feature_spec import get_edge_dim, validate_checkpoint_edge_dim
from src.utils.hashing import sha256_file


LOG = logging.getLogger(__name__)


def _load_teacher(
    model_path: Path,
    model_cfg: Dict[str, Any],
    env_cfg: Dict[str, Any],
    device: torch.device,
) -> EdgeQGNN:
    edge_dim = int(get_edge_dim(env_cfg))
    use_fleet_potential = bool(env_cfg.get("use_fleet_potential", False))
    model = EdgeQGNN(
        node_dim=int(model_cfg.get("node_dim", 5)),
        edge_dim=edge_dim,
        hidden_dim=int(model_cfg.get("hidden_dim", 32)),
        num_layers=int(model_cfg.get("num_layers", 2)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        dueling=bool(model_cfg.get("dueling", False)),
    )
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
    checkpoint_edge_dim = edge_dim
    if isinstance(state_dict, dict):
        q_head_key = "q_head.0.weight"
        if q_head_key in state_dict:
            q_head_in = state_dict[q_head_key].shape[1]
            hidden_dim = int(model_cfg.get("hidden_dim", 32))
            checkpoint_edge_dim = q_head_in - hidden_dim * 2
    validate_checkpoint_edge_dim(checkpoint_edge_dim, edge_dim, use_fleet_potential)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def _teacher_logits(
    model: EdgeQGNN,
    features: Dict[str, np.ndarray],
    device: torch.device,
) -> np.ndarray:
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
    return q


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--output", default="data/processed/distill/distill_dataset.npz")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--use-eval-env-overrides", action="store_true")
    parser.add_argument("--log-every", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    config_path = Path(args.config)
    cfg = load_config(config_path)
    env_cfg = dict(cfg.get("env", {}))
    if args.use_eval_env_overrides:
        eval_overrides = cfg.get("eval", {}).get("env_overrides", {})
        if isinstance(eval_overrides, dict):
            env_cfg.update(eval_overrides)
    if args.seed is not None:
        env_cfg["seed"] = int(args.seed)
    model_cfg = dict(cfg.get("model", {}))

    device = torch.device(args.device)
    teacher_path = Path(args.teacher_model)
    if not teacher_path.exists():
        raise FileNotFoundError(f"Teacher model not found: {teacher_path}")

    env = EventDrivenEnv(_build_env_config(env_cfg))
    teacher = _load_teacher(teacher_path, model_cfg, env_cfg, device)

    action_vectors_list: List[np.ndarray] = []
    action_masks_list: List[np.ndarray] = []
    teacher_logits_list: List[np.ndarray] = []
    target_idx_list: List[int] = []
    action_counts: List[int] = []
    feature_names: Optional[List[str]] = None

    total_steps = 0
    total_samples = 0
    for ep in range(int(args.episodes)):
        env.seed(int(env_cfg.get("seed", 7)) + ep)
        env.reset()
        done = False
        while not done:
            features = env.get_feature_batch()
            actions = features.get("actions", np.array([], dtype=np.int64)).astype(np.int64)
            mask = features.get("action_mask", np.array([], dtype=bool)).astype(bool)
            if len(actions) == 0 or not np.any(mask):
                done = True
                continue
            action_vectors, names = build_action_vectors(features, env)
            if feature_names is None:
                feature_names = list(names)
            teacher_logits = _teacher_logits(teacher, features, device)
            teacher_logits = teacher_logits.astype(np.float32)
            teacher_logits[~mask] = -1e9
            target_idx = int(np.argmax(teacher_logits))

            action_vectors_list.append(action_vectors)
            action_masks_list.append(mask.astype(bool))
            teacher_logits_list.append(teacher_logits)
            target_idx_list.append(target_idx)
            action_counts.append(int(len(actions)))
            total_samples += 1

            action = int(actions[target_idx])
            _obs, _reward, done, _info = env.step(action)
            total_steps += 1
            if args.max_steps is not None and total_steps >= int(args.max_steps):
                done = True
            if args.max_samples is not None and total_samples >= int(args.max_samples):
                done = True

            if args.log_every > 0 and total_steps % int(args.log_every) == 0:
                LOG.info("Collected steps=%d samples=%d", total_steps, total_samples)
        if args.max_samples is not None and total_samples >= int(args.max_samples):
            break

    if not action_vectors_list:
        raise RuntimeError("No samples collected; check environment and teacher model.")

    max_actions = int(max(action_counts))
    feature_dim = int(action_vectors_list[0].shape[1])
    sample_count = int(len(action_vectors_list))

    action_vectors_arr = np.zeros((sample_count, max_actions, feature_dim), dtype=np.float32)
    action_masks_arr = np.zeros((sample_count, max_actions), dtype=bool)
    teacher_logits_arr = np.full((sample_count, max_actions), -1e9, dtype=np.float32)
    target_idx_arr = np.zeros((sample_count,), dtype=np.int64)
    action_counts_arr = np.array(action_counts, dtype=np.int64)

    for idx, (vecs, mask, logits) in enumerate(
        zip(action_vectors_list, action_masks_list, teacher_logits_list)
    ):
        count = vecs.shape[0]
        action_vectors_arr[idx, :count, :] = vecs
        action_masks_arr[idx, :count] = mask
        teacher_logits_arr[idx, :count] = logits
        target_idx_arr[idx] = int(target_idx_list[idx])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        action_vectors=action_vectors_arr,
        action_mask=action_masks_arr,
        teacher_logits=teacher_logits_arr,
        target_idx=target_idx_arr,
        action_counts=action_counts_arr,
    )

    meta = {
        "config_path": str(config_path),
        "config_sha256": sha256_file(str(config_path)),
        "teacher_model_path": str(teacher_path),
        "teacher_model_sha256": sha256_file(str(teacher_path)),
        "use_fleet_potential": bool(env_cfg.get("use_fleet_potential", False)),
        "feature_dim": feature_dim,
        "max_actions": max_actions,
        "samples": sample_count,
        "feature_names": feature_names or [],
    }
    meta_path = output_path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="ascii")
    LOG.info("Saved dataset to %s (%d samples)", output_path, sample_count)


if __name__ == "__main__":
    main()
