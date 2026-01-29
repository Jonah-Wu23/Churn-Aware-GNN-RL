"""消融实验独立评估脚本

完全独立于现有评估脚本，但复用底层指标计算模块确保口径一致。

用法:
    python scripts/run_ablation_eval.py \\
        --ablation no_edge \\
        --model-path reports/runs/ablation_no_edge/model_final.pt \\
        --config configs/ablation/no_edge.yaml \\
        --output-dir reports/ablation_results

    # 批量评估所有消融变体
    python scripts/run_ablation_eval.py --batch \\
        --output-dir reports/ablation_results

输出:
    - ablation_results.json: 所有消融变体的指标汇总
    - ablation_comparison.md: 论文用表格格式
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.env.gym_env import EnvConfig
from src.models.edge_q_gnn import EdgeQGNN
from src.models.node_only_gnn import NodeOnlyGNN
from src.ablation.ablation_evaluator import (
    AblationEvaluator,
    AblationResult,
    generate_comparison_table,
    save_results,
)
from src.utils.config import load_config
from src.utils.feature_spec import get_edge_dim

LOG = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="消融实验独立评估脚本")
    parser.add_argument(
        "--ablation",
        choices=["no_edge", "no_risk", "no_curriculum", "full"],
        help="消融类型（单个评估时必需）",
    )
    parser.add_argument("--model-path", help="模型权重路径")
    parser.add_argument("--config", help="配置文件路径")
    parser.add_argument(
        "--output-dir",
        default="reports/ablation_results",
        help="输出目录",
    )
    parser.add_argument("--episodes", type=int, default=10, help="评估 episode 数")
    parser.add_argument("--device", default="cuda", help="计算设备")
    parser.add_argument("--seed", type=int, default=7, help="随机种子")
    parser.add_argument(
        "--batch",
        action="store_true",
        help="批量评估所有消融变体",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="使用随机权重测试流程（不需要预训练模型）",
    )
    return parser.parse_args()


def _build_env_config(env_cfg: Dict[str, Any]) -> EnvConfig:
    """从配置字典构建 EnvConfig"""
    return EnvConfig(
        max_horizon_steps=int(env_cfg.get("max_horizon_steps", 200)),
        max_sim_time_sec=env_cfg.get("max_sim_time_sec"),
        allow_stop_when_actions_exist=bool(env_cfg.get("allow_stop_when_actions_exist", False)),
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
        reward_service_transform=str(env_cfg.get("reward_service_transform", "none")),
        reward_service_transform_scale=float(env_cfg.get("reward_service_transform_scale", 1.0)),
        reward_tacc_transform=str(env_cfg.get("reward_tacc_transform", "none")),
        reward_tacc_transform_scale=float(env_cfg.get("reward_tacc_transform_scale", 1.0)),
        reward_onboard_delay_weight=float(env_cfg.get("reward_onboard_delay_weight", 0.1)),
        reward_cvar_penalty=float(env_cfg.get("reward_cvar_penalty", 1.0)),
        reward_fairness_weight=float(env_cfg.get("reward_fairness_weight", 1.0)),
        reward_congestion_penalty=float(env_cfg.get("reward_congestion_penalty", 0.0)),
        reward_scale=float(env_cfg.get("reward_scale", 1.0)),
        reward_step_backlog_penalty=float(env_cfg.get("reward_step_backlog_penalty", 0.0)),
        reward_waiting_time_penalty_per_sec=float(env_cfg.get("reward_waiting_time_penalty_per_sec", 0.0)),
        reward_potential_alpha=float(env_cfg.get("reward_potential_alpha", 0.0)),
        reward_potential_alpha_source=str(env_cfg.get("reward_potential_alpha_source", "env_default")),
        reward_potential_lost_weight=float(env_cfg.get("reward_potential_lost_weight", 0.0)),
        reward_potential_scale_with_reward_scale=bool(
            env_cfg.get("reward_potential_scale_with_reward_scale", True)
        ),
        demand_exhausted_min_time_sec=float(env_cfg.get("demand_exhausted_min_time_sec", 300.0)),
        cvar_alpha=float(env_cfg.get("cvar_alpha", 0.95)),
        fairness_gamma=float(env_cfg.get("fairness_gamma", 1.0)),
        debug_mask=bool(env_cfg.get("debug_mask", False)),
        debug_abort_on_alert=bool(env_cfg.get("debug_abort_on_alert", False)),
        debug_dump_dir=str(env_cfg.get("debug_dump_dir", "reports/debug/potential_alerts")),
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
        # FAEP configuration
        use_fleet_potential=bool(env_cfg.get("use_fleet_potential", False)),
        fleet_potential_mode=str(env_cfg.get("fleet_potential_mode", "next_stop")),
        fleet_potential_k=int(env_cfg.get("fleet_potential_k", 1)),
        fleet_potential_hybrid_center_weight=float(env_cfg.get("fleet_potential_hybrid_center_weight", 0.5)),
        fleet_potential_hybrid_neighbor_weight=float(env_cfg.get("fleet_potential_hybrid_neighbor_weight", 0.5)),
        fleet_potential_phi=str(env_cfg.get("fleet_potential_phi", "log1p_norm")),
        reward_terminal_backlog_penalty=float(env_cfg.get("reward_terminal_backlog_penalty", 0.0)),
        hard_mask_skip_unrecoverable=bool(env_cfg.get("hard_mask_skip_unrecoverable", False)),
        hard_mask_slack_sec=float(env_cfg.get("hard_mask_slack_sec", 0.0)),
    )


def _is_edgeq_state_dict(state_dict: Dict[str, torch.Tensor]) -> bool:
    """判断是否为 EdgeQGNN (TransformerConv) 的权重"""
    edgeq_markers = (
        "convs.0.lin_key.weight",
        "convs.0.lin_query.weight",
        "convs.0.lin_value.weight",
        "convs.0.lin_edge.weight",
        "value_head.0.weight",
    )
    return any(k in state_dict for k in edgeq_markers)


def _infer_edge_dim_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Optional[int]:
    """从权重推断 edge_dim（若存在 TransformerConv 的 lin_edge 权重）。"""
    key = "convs.0.lin_edge.weight"
    if key in state_dict:
        weight = state_dict[key]
        if hasattr(weight, "shape") and len(weight.shape) == 2:
            return int(weight.shape[1])
    return None


def _infer_dueling_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> bool:
    """从权重判断是否启用 dueling head。"""
    return "value_head.0.weight" in state_dict


def _load_model(
    ablation_type: str,
    model_cfg: Dict[str, Any],
    env_cfg: Dict[str, Any],
    model_path: Optional[Path],
    device: torch.device,
    dry_run: bool = False,
) -> torch.nn.Module:
    """加载模型（与训练结构/维度对齐）"""
    state_dict: Optional[Dict[str, torch.Tensor]] = None
    if not dry_run and model_path and model_path.exists():
        LOG.info(f"加载模型权重: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
        else:
            state_dict = checkpoint
    elif dry_run:
        LOG.info("Dry-run 模式: 使用随机权重")
    else:
        LOG.warning(f"模型权重文件不存在: {model_path}，使用随机权重")

    # 训练侧 EdgeQGNN 的 edge_dim 统一来自 get_edge_dim(env_cfg)，但如果权重给出更可信的维度，优先以权重为准
    edge_dim = int(get_edge_dim(env_cfg))
    if state_dict is not None:
        inferred_edge_dim = _infer_edge_dim_from_state_dict(state_dict)
        if inferred_edge_dim is not None and inferred_edge_dim != edge_dim:
            LOG.warning(
                "edge_dim mismatch: config=%s, checkpoint=%s; using checkpoint value",
                edge_dim,
                inferred_edge_dim,
            )
            edge_dim = inferred_edge_dim

    dueling = bool(model_cfg.get("dueling", False))
    if state_dict is not None:
        inferred_dueling = _infer_dueling_from_state_dict(state_dict)
        if inferred_dueling != dueling:
            LOG.warning(
                "dueling mismatch: config=%s, checkpoint=%s; using checkpoint value",
                dueling,
                inferred_dueling,
            )
            dueling = inferred_dueling

    use_edgeq = ablation_type != "no_edge"
    if ablation_type == "no_edge" and state_dict is not None:
        # 训练时若走 curriculum，会实际产出 EdgeQGNN 权重
        use_edgeq = _is_edgeq_state_dict(state_dict)

    if use_edgeq:
        model = EdgeQGNN(
            node_dim=int(model_cfg.get("node_dim", 5)),
            edge_dim=edge_dim,
            hidden_dim=int(model_cfg.get("hidden_dim", 256)),
            num_layers=int(model_cfg.get("num_layers", 3)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            dueling=dueling,
        )
    else:
        model = NodeOnlyGNN(
            node_dim=int(model_cfg.get("node_dim", 5)),
            hidden_dim=int(model_cfg.get("hidden_dim", 256)),
            num_layers=int(model_cfg.get("num_layers", 3)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            heads=int(model_cfg.get("heads", 4)),
            edge_dim=edge_dim,
        )

    if state_dict is not None and not dry_run:
        model.load_state_dict(state_dict)

    model.to(device)
    return model


def evaluate_single(
    ablation_type: str,
    config_path: Path,
    model_path: Optional[Path],
    device: torch.device,
    episodes: int,
    seed: int,
    dry_run: bool = False,
) -> AblationResult:
    """评估单个消融变体"""
    LOG.info(f"评估消融变体: {ablation_type}")
    
    cfg = load_config(str(config_path))
    env_cfg = cfg.get("env", {})
    model_cfg = cfg.get("model", {})

    # 先加载模型（可从 checkpoint 推断 edge_dim / dueling）
    model = _load_model(ablation_type, model_cfg, env_cfg, model_path, device, dry_run)

    # 若 checkpoint edge_dim 与 env_cfg 不一致，动态调整 env_cfg 的 FAEP 开关以匹配特征维度
    if not dry_run:
        state_dict = None
        if model_path and model_path.exists():
            checkpoint = torch.load(model_path, map_location=device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif isinstance(checkpoint, dict):
                state_dict = checkpoint
        if state_dict is not None:
            inferred_edge_dim = _infer_edge_dim_from_state_dict(state_dict)
            if inferred_edge_dim is not None:
                desired_faep = inferred_edge_dim == 5
                current_faep = bool(env_cfg.get("use_fleet_potential", False))
                if desired_faep != current_faep:
                    LOG.warning(
                        "Env FAEP mismatch: config use_fleet_potential=%s, checkpoint edge_dim=%s; overriding env config",
                        current_faep,
                        inferred_edge_dim,
                    )
                    env_cfg = dict(env_cfg)
                    env_cfg["use_fleet_potential"] = desired_faep

    env_config = _build_env_config(env_cfg)
    
    evaluator = AblationEvaluator(
        ablation_type=ablation_type,
        env_config=env_config,
        model=model,
        device=device,
        seed=seed,
    )
    
    result = evaluator.evaluate(episodes=episodes)
    result.model_path = str(model_path) if model_path else None
    
    LOG.info(f"评估完成: TACC={result.tacc:.1f}, Churn={result.churn_rate:.3f}, "
             f"WaitP95={result.wait_time_p95:.1f}s, Suburban={result.suburban_service_rate:.3f}")
    
    return result


def evaluate_batch(
    output_dir: Path,
    device: torch.device,
    episodes: int,
    seed: int,
    dry_run: bool = False,
) -> List[AblationResult]:
    """批量评估所有消融变体"""
    results: List[AblationResult] = []
    
    ablation_configs = {
        "full": ("configs/manhattan_curriculum_v13.yaml", None),
        "no_edge": ("configs/ablation/no_edge.yaml", None),
        "no_risk": ("configs/ablation/no_risk.yaml", None),
        "no_curriculum": ("configs/ablation/no_curriculum.yaml", None),
    }
    
    for ablation_type, (config_path, model_path) in ablation_configs.items():
        config_path = Path(config_path)
        if not config_path.exists():
            LOG.warning(f"配置文件不存在，跳过: {config_path}")
            continue
        
        # 尝试查找模型文件
        if model_path is None and not dry_run:
            possible_paths = [
                output_dir.parent / f"runs/ablation_{ablation_type}_latest/model_final.pt",
                output_dir.parent / f"runs/ablation_{ablation_type}_latest/edgeq_model_final.pt",
                output_dir.parent / f"runs/{ablation_type}/model_final.pt",
                output_dir.parent / f"runs/{ablation_type}/edgeq_model_final.pt",
            ]
            for p in possible_paths:
                if p.exists():
                    model_path = p
                    break
        
        try:
            result = evaluate_single(
                ablation_type=ablation_type,
                config_path=config_path,
                model_path=model_path,
                device=device,
                episodes=episodes,
                seed=seed,
                dry_run=dry_run,
            )
            results.append(result)
        except Exception as e:
            LOG.error(f"评估失败 ({ablation_type}): {e}")
            continue
    
    return results


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.batch:
        LOG.info("批量评估模式")
        results = evaluate_batch(
            output_dir=output_dir,
            device=device,
            episodes=args.episodes,
            seed=args.seed,
            dry_run=args.dry_run,
        )
    else:
        if not args.ablation or not args.config:
            LOG.error("单个评估模式需要 --ablation 和 --config 参数")
            sys.exit(1)
        
        model_path = Path(args.model_path) if args.model_path else None
        result = evaluate_single(
            ablation_type=args.ablation,
            config_path=Path(args.config),
            model_path=model_path,
            device=device,
            episodes=args.episodes,
            seed=args.seed,
            dry_run=args.dry_run,
        )
        results = [result]
    
    # 保存结果
    save_results(results, output_dir)
    
    # 打印对比表格
    print("\n" + "=" * 80)
    print("消融实验结果对比")
    print("=" * 80)
    print(generate_comparison_table(results))
    print("=" * 80)
    
    LOG.info(f"结果已保存至: {output_dir}")


if __name__ == "__main__":
    main()
