"""消融实验训练脚本 (供 AutoDL 使用)

用法:
    python scripts/run_ablation_train.py --ablation no_edge --config configs/ablation/no_edge.yaml
    python scripts/run_ablation_train.py --ablation no_risk --config configs/ablation/no_risk.yaml
    python scripts/run_ablation_train.py --ablation no_curriculum --config configs/ablation/no_curriculum.yaml

消融类型:
    no_edge: 使用 NodeOnlyGNN (GAT)，移除边特征
    no_risk: 使用 RiskAblatedEnv，同时消融 Reward 和 Feature 层
    no_curriculum: 直接在 L3 阶段训练，跳过课程学习
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.env.gym_env import EnvConfig, EventDrivenEnv
from src.models.edge_q_gnn import EdgeQGNN
from src.models.node_only_gnn import NodeOnlyGNN
from src.ablation.ablation_env_wrapper import RiskAblatedEnv, ABLATION_TYPES
from src.train.dqn import DQNConfig, DQNTrainer, build_hashes
from src.train.runner import run_curriculum_training
from src.utils.config import load_config
from src.utils.feature_spec import get_edge_dim

LOG = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="消融实验训练脚本")
    parser.add_argument(
        "--ablation",
        required=True,
        choices=["no_edge", "no_risk", "no_curriculum"],
        help="消融类型",
    )
    parser.add_argument("--config", required=True, help="配置文件路径")
    parser.add_argument("--run-dir", default=None, help="保存目录")
    parser.add_argument("--device", default=None, help="计算设备 (覆盖配置)")
    return parser.parse_args()


def _build_env_config(env_cfg: Dict[str, Any]) -> EnvConfig:
    """从配置字典构建 EnvConfig"""
    return EnvConfig(
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
        reward_scale=float(env_cfg.get("reward_scale", 1.0)),
        reward_step_backlog_penalty=float(env_cfg.get("reward_step_backlog_penalty", 0.0)),
        reward_waiting_time_penalty_per_sec=float(env_cfg.get("reward_waiting_time_penalty_per_sec", 0.0)),
        demand_exhausted_min_time_sec=float(env_cfg.get("demand_exhausted_min_time_sec", 300.0)),
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
        travel_time_multiplier=float(env_cfg.get("travel_time_multiplier", 1.0)),
        time_split_mode=env_cfg.get("time_split_mode"),
        time_split_ratio=float(env_cfg.get("time_split_ratio", 0.3)),
    )


def _build_dqn_config(train_cfg: Dict[str, Any], env_cfg: Dict[str, Any]) -> DQNConfig:
    """从配置字典构建 DQNConfig"""
    return DQNConfig(
        seed=int(train_cfg.get("seed", env_cfg.get("seed", 7))),
        total_steps=int(train_cfg.get("total_steps", 200_000)),
        buffer_size=int(train_cfg.get("buffer_size", 10_000)),
        batch_size=int(train_cfg.get("batch_size", 64)),
        learning_starts=int(train_cfg.get("learning_starts", 2_000)),
        train_freq=int(train_cfg.get("train_freq", 1)),
        gradient_steps=int(train_cfg.get("gradient_steps", 1)),
        target_update_interval=int(train_cfg.get("target_update_interval", 2_000)),
        gamma=float(train_cfg.get("gamma", 0.99)),
        learning_rate=float(train_cfg.get("learning_rate", 1e-3)),
        max_grad_norm=float(train_cfg.get("max_grad_norm", 10.0)),
        double_dqn=bool(train_cfg.get("double_dqn", True)),
        epsilon_start=float(train_cfg.get("epsilon_start", 1.0)),
        epsilon_end=float(train_cfg.get("epsilon_end", 0.05)),
        epsilon_decay_steps=int(train_cfg.get("epsilon_decay_steps", 100_000)),
        log_every_steps=int(train_cfg.get("log_every_steps", 1_000)),
        checkpoint_every_steps=int(train_cfg.get("checkpoint_every_steps", 10_000)),
        device=str(train_cfg.get("device", "cpu")),
    )


def _build_model(
    model_cfg: Dict[str, Any],
    env_cfg: Dict[str, Any],
    ablation_type: str,
) -> torch.nn.Module:
    """根据消融类型构建模型"""
    model_type = model_cfg.get("type", "edge_q_gnn")
    
    if ablation_type == "no_edge" or model_type == "node_only_gnn":
        return NodeOnlyGNN(
            node_dim=int(model_cfg.get("node_dim", 5)),
            hidden_dim=int(model_cfg.get("hidden_dim", 32)),
            num_layers=int(model_cfg.get("num_layers", 2)),
            dropout=float(model_cfg.get("dropout", 0.0)),
            heads=int(model_cfg.get("heads", 4)),
            edge_dim=int(model_cfg.get("edge_dim", get_edge_dim(env_cfg))),
        )
    else:
        return EdgeQGNN(
            node_dim=int(model_cfg.get("node_dim", 5)),
            edge_dim=int(model_cfg.get("edge_dim", get_edge_dim(env_cfg))),
            hidden_dim=int(model_cfg.get("hidden_dim", 32)),
            num_layers=int(model_cfg.get("num_layers", 2)),
            dropout=float(model_cfg.get("dropout", 0.0)),
        )


def _build_env(env_config: EnvConfig, ablation_type: str) -> EventDrivenEnv:
    """根据消融类型构建环境"""
    if ablation_type == "no_risk":
        return RiskAblatedEnv(env_config)
    else:
        return EventDrivenEnv(env_config)


def _use_curriculum(cfg: Dict[str, Any], ablation_type: str) -> bool:
    """判断是否使用课程学习"""
    if ablation_type == "no_curriculum":
        # 检查是否只有 L3 阶段
        curriculum_cfg = cfg.get("curriculum", {})
        stages = curriculum_cfg.get("stages", [])
        if stages == ["L3"]:
            return False  # 只有 L3，不算课程学习
    
    curriculum_cfg = cfg.get("curriculum", {})
    stages = curriculum_cfg.get("stages")
    if isinstance(stages, list) and len(stages) > 1:
        return True
    return False


def _materialize_model_final(run_dir: Path) -> Optional[Path]:
    """Ensure model_final.pt exists in run_dir for downstream eval."""
    candidates = list(run_dir.rglob("edgeq_model_final.pt"))
    if not candidates:
        return None
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    target = run_dir / "model_final.pt"
    if target.resolve() != latest.resolve():
        shutil.copy2(latest, target)
    return target


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    
    ablation_type = args.ablation
    LOG.info(f"消融实验类型: {ablation_type}")
    LOG.info(f"配置文件: {args.config}")
    
    cfg = load_config(args.config)
    
    # 覆盖设备
    if args.device:
        cfg.setdefault("train", {})["device"] = args.device
    
    # 设置运行目录
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = Path("reports") / "runs" / f"ablation_{ablation_type}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    LOG.info(f"运行目录: {run_dir}")
    
    env_cfg = cfg.get("env", {})
    train_cfg = cfg.get("train", {})
    model_cfg = cfg.get("model", {})
    
    env_config = _build_env_config(env_cfg)
    dqn_config = _build_dqn_config(train_cfg, env_cfg)
    
    # 构建环境和模型
    env = _build_env(env_config, ablation_type)
    model = _build_model(model_cfg, env_cfg, ablation_type)
    model.to(torch.device(dqn_config.device))
    
    LOG.info(f"模型类型: {type(model).__name__}")
    LOG.info(f"环境类型: {type(env).__name__}")
    
    # 保存消融元数据
    ablation_meta = {
        "ablation_type": ablation_type,
        "description": ABLATION_TYPES.get(ablation_type, {}).get("description", ""),
        "model_type": type(model).__name__,
        "env_type": type(env).__name__,
        "config_path": str(args.config),
    }
    import json
    with open(run_dir / "ablation_meta.json", "w", encoding="utf-8") as f:
        json.dump(ablation_meta, f, indent=2, ensure_ascii=False)
    
    # 开始训练
    if _use_curriculum(cfg, ablation_type):
        LOG.info("使用课程学习")
        log_path = run_curriculum_training(args.config, run_dir=run_dir)
    else:
        LOG.info("直接训练（无课程学习）")
        graph_hashes, od_hashes = build_hashes(env_cfg)
        trainer = DQNTrainer(
            env=env,
            model=model,
            config=dqn_config,
            run_dir=run_dir,
            graph_hashes=graph_hashes,
            od_hashes=od_hashes,
            env_cfg=env_cfg,
        )
        log_path = trainer.train()
        trainer.close()

    model_final = _materialize_model_final(run_dir)
    if model_final is not None:
        LOG.info(f"模型已同步到: {model_final}")
    
    LOG.info(f"训练完成: {log_path}")


if __name__ == "__main__":
    main()
