"""Smoke test for potential shaping invariants (CPU-only)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.build_info import get_build_id
from src.utils.config import load_config
from src.env.gym_env import EventDrivenEnv
from src.train.curriculum import generate_stage, load_nodes, load_od_frames
from src.train.runner import _build_env_config, _merge_env_cfg, _stage_specs_from_config
from src.utils.reward_hack_alerts import RewardHackAlertConfig, RewardHackDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", default="L1")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def _build_stage_env(config: Dict[str, Any], stage_name: str, seed: int) -> Dict[str, Any]:
    env_cfg = config.get("env", {})
    curriculum_cfg = config.get("curriculum", {})
    strict = bool(curriculum_cfg.get("strict_stage_params", False))
    stage_specs, stage_env_overrides_map = _stage_specs_from_config(curriculum_cfg, strict_stage_params=strict)
    target = None
    for spec in stage_specs:
        if spec.name == stage_name:
            target = spec
            break
    if target is None:
        raise ValueError(f"Stage {stage_name} not found in curriculum.")

    od = load_od_frames(env_cfg.get("od_glob", "data/processed/od_mapped/*.parquet"))
    nodes = load_nodes(env_cfg.get("graph_nodes_path", "data/processed/graph/layer2_nodes.parquet"))
    output_dir = Path("reports") / "debug" / "smoke_stage" / stage_name
    stage_output = generate_stage(od, nodes, target, output_dir=output_dir, seed=seed)

    stage_env_base = dict(env_cfg)
    stage_env_base["od_glob"] = str(stage_output.od_path)
    combined_stage_overrides = {
        **stage_output.env_overrides,
        **stage_env_overrides_map.get(stage_name, {}),
    }
    return _merge_env_cfg(stage_env_base, combined_stage_overrides, None)


def main() -> int:
    args = parse_args()
    build_id = get_build_id()
    print(f"BUILD_ID={build_id}")

    cfg = load_config(args.config)
    stage_env_cfg = _build_stage_env(cfg, args.stage, args.seed)
    env = EventDrivenEnv(_build_env_config(stage_env_cfg))
    rng = np.random.default_rng(int(args.seed))

    alert_cfg = RewardHackAlertConfig.from_dict(
        {
            "enabled": True,
            "debug_abort_on_alert": bool(stage_env_cfg.get("debug_abort_on_alert", True)),
            "debug_dump_dir": str(stage_env_cfg.get("debug_dump_dir", "reports/debug/potential_alerts")),
        }
    )
    detector = RewardHackDetector(alert_cfg)

    shaping_nonzero = 0
    env.reset()
    for step in range(int(args.steps)):
        features = env.get_feature_batch()
        actions = features["actions"].astype(np.int64)
        mask = features["action_mask"].astype(bool)
        valid = np.where(mask)[0]
        if len(actions) == 0 or len(valid) == 0:
            break
        action = int(actions[int(rng.choice(valid))])
        _, reward, done, info = env.step(action)

        reward_components = info.get("reward_components", {})
        reward_terms = dict(reward_components) if isinstance(reward_components, dict) else {}
        reward_terms.update(info.get("reward_components_raw", {}))
        reward_terms["reward_total"] = float(info.get("reward_total", reward))
        payload = {
            "reward_total": float(info.get("reward_total", reward)),
            "reward_terms": reward_terms,
            "reward_potential_alpha": float(info.get("reward_potential_alpha", 0.0)),
            "reward_potential_alpha_source": str(info.get("reward_potential_alpha_source", "unknown")),
            "reward_potential_lost_weight": float(info.get("reward_potential_lost_weight", 0.0)),
            "reward_potential_scale_with_reward_scale": bool(
                info.get("reward_potential_scale_with_reward_scale", False)
            ),
            "phi_before": float(info.get("phi_before", 0.0)),
            "phi_after": float(info.get("phi_after", 0.0)),
            "phi_delta": float(info.get("phi_delta", 0.0)),
            "phi_backlog_before": float(info.get("phi_backlog_before", 0.0)),
            "phi_backlog_after": float(info.get("phi_backlog_after", 0.0)),
            "lost_total_before": float(info.get("lost_total_before", 0.0)),
            "lost_total_after": float(info.get("lost_total_after", 0.0)),
            "waiting_churned_before": float(info.get("waiting_churned_before", 0.0)),
            "waiting_churned_after": float(info.get("waiting_churned_after", 0.0)),
            "onboard_churned_before": float(info.get("onboard_churned_before", 0.0)),
            "onboard_churned_after": float(info.get("onboard_churned_after", 0.0)),
            "structural_unserviceable_before": float(info.get("structural_unserviceable_before", 0.0)),
            "structural_unserviceable_after": float(info.get("structural_unserviceable_after", 0.0)),
            "waiting_remaining_before": float(info.get("waiting_remaining_before", 0.0)),
            "waiting_remaining_after": float(info.get("waiting_remaining_after", 0.0)),
            "onboard_remaining_before": float(info.get("onboard_remaining_before", 0.0)),
            "onboard_remaining_after": float(info.get("onboard_remaining_after", 0.0)),
            "step_served": float(info.get("step_served", 0.0)),
            "served_per_decision": float(info.get("step_served", 0.0)),
            "action_stop": int(action),
            "missing_keys": [],
        }
        try:
            detector.update(payload)
        except RuntimeError as exc:
            print(f"FAIL: {exc}")
            return 1

        if abs(float(info.get("reward_potential_shaping_raw", 0.0))) > 1e-9:
            shaping_nonzero += 1

        if done:
            break

    if shaping_nonzero <= 0:
        print("FAIL: shaping_nonzero_count=0")
        return 1
    print(f"PASS: shaping_nonzero_count={shaping_nonzero} steps={step + 1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
