"""Sensitivity evaluation (fixed policy, zero retraining).

Runs evaluation sweeps over key env/eval parameters while keeping the policy
checkpoint fixed. Outputs per-run eval_results.json plus consolidated
sensitivity_results.json and sensitivity_episodes.csv for plotting.
"""

from __future__ import annotations

import argparse
import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[1]

import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.evaluator import evaluate
from src.env.gym_env import EnvConfig
from src.utils.config import load_config
from src.utils.hashing import sha256_file

LOG = logging.getLogger(__name__)


PARAM_ENV_KEYS = {
    "mask_alpha": "mask_alpha",
    "fairness_gamma": "fairness_gamma",
    "cvar_alpha": "cvar_alpha",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run zero-retrain sensitivity evaluation.")
    parser.add_argument("--config", required=True, help="Base YAML config.")
    parser.add_argument("--model-path", required=True, help="Fixed policy checkpoint (.pt).")
    parser.add_argument("--policy", default="edgeq", help="Eval policy (default: edgeq).")
    parser.add_argument("--run-dir", default="reports/sensitivity", help="Output root directory.")
    parser.add_argument("--episodes", type=int, default=None, help="Override eval episodes.")
    parser.add_argument("--seed", type=int, default=None, help="Override eval seed.")
    parser.add_argument("--device", default=None, help="Override eval device.")
    parser.add_argument(
        "--param",
        choices=["mask_alpha", "fairness_gamma", "cvar_alpha", "k_hop", "all"],
        default="all",
        help="Run a single parameter sweep or all (default: all).",
    )
    parser.add_argument(
        "--values",
        nargs="*",
        default=None,
        help="Custom values for the selected param (space-separated).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip runs if eval_results.json already exists.",
    )
    return parser.parse_args()


def _unique_keep_order(values: Iterable[float]) -> List[float]:
    seen = set()
    out = []
    for v in values:
        key = float(v)
        if key in seen:
            continue
        seen.add(key)
        out.append(float(v))
    return out


def _format_value(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{value:.6g}".replace(".", "p")


def _default_env_value(env_cfg: Dict[str, Any], key: str, default: float) -> float:
    raw = env_cfg.get(key, default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _build_default_sweeps(env_cfg: Dict[str, Any]) -> Dict[str, List[float | int]]:
    base = EnvConfig()
    mask_alpha_base = _default_env_value(env_cfg, "mask_alpha", base.mask_alpha)
    fairness_gamma_base = _default_env_value(env_cfg, "fairness_gamma", base.fairness_gamma)
    cvar_alpha_base = _default_env_value(env_cfg, "cvar_alpha", base.cvar_alpha)

    mask_alpha_vals = _unique_keep_order(
        [mask_alpha_base * v for v in (0.7, 0.85, 1.0, 1.15, 1.3)]
    )
    fairness_gamma_vals = _unique_keep_order(
        [0.0, fairness_gamma_base * 0.5, fairness_gamma_base, fairness_gamma_base * 1.5, fairness_gamma_base * 2.0]
    )
    cvar_alpha_vals = _unique_keep_order([0.50, 0.75, cvar_alpha_base, 0.99, 0.995])
    for v in cvar_alpha_vals:
        if not (0.0 < float(v) <= 1.0):
            raise ValueError(f"cvar_alpha must be in (0, 1], got {v}")

    return {
        "mask_alpha": mask_alpha_vals,
        "fairness_gamma": fairness_gamma_vals,
        "cvar_alpha": cvar_alpha_vals,
        "k_hop": [1, 2, 3, 4, 5],
    }


def _apply_overrides(
    cfg: Dict[str, Any],
    param: str,
    value: float | int,
) -> Dict[str, Any]:
    cfg = deepcopy(cfg)
    eval_cfg = dict(cfg.get("eval", {}))
    env_overrides = dict(eval_cfg.get("env_overrides", {}))
    if param in PARAM_ENV_KEYS:
        env_overrides[PARAM_ENV_KEYS[param]] = float(value)
        eval_cfg["env_overrides"] = env_overrides
    elif param == "k_hop":
        eval_cfg["k_hop"] = int(value)
    else:
        raise ValueError(f"Unsupported param: {param}")
    cfg["eval"] = eval_cfg
    return cfg


def _collect_aggregate_row(
    *,
    param: str,
    value: float | int,
    run_dir: Path,
    eval_results: Dict[str, Any],
    model_sha: str,
    config_sha: str,
) -> Dict[str, Any]:
    aggregate = eval_results.get("aggregate", {})
    row = {
        "param": param,
        "value": float(value) if not isinstance(value, int) else int(value),
        "run_dir": str(run_dir),
        "episodes": int(aggregate.get("episodes", 0)),
        "policy": aggregate.get("policy"),
        "config_sha256": config_sha,
        "model_sha256": model_sha,
    }
    key_map = [
        "service_rate",
        "algorithmic_churn_rate",
        "wait_time_p95_sec",
        "service_gini",
        "tacc_total",
    ]
    for key in key_map:
        mean_key = f"{key}_mean"
        std_key = f"{key}_std"
        if mean_key in aggregate:
            row[mean_key] = aggregate.get(mean_key)
        if std_key in aggregate:
            row[std_key] = aggregate.get(std_key)
    return row


def _collect_episode_rows(
    param: str,
    value: float | int,
    eval_results: Dict[str, Any],
) -> List[Dict[str, Any]]:
    rows = []
    for ep in eval_results.get("episodes", []):
        row = {
            "param": param,
            "value": float(value) if not isinstance(value, int) else int(value),
            "episode_index": ep.get("episode_index"),
            "seed": ep.get("seed"),
            "service_rate": ep.get("service_rate"),
            "algorithmic_churn_rate": ep.get("algorithmic_churn_rate"),
            "wait_time_p95_sec": ep.get("wait_time_p95_sec"),
            "service_gini": ep.get("service_gini"),
            "tacc_total": ep.get("tacc_total"),
            "sim_time_sec": ep.get("sim_time_sec"),
            "end_reason": ep.get("end_reason"),
        }
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    import csv

    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    base_cfg = load_config(config_path)
    base_env_cfg = dict(base_cfg.get("env", {}))
    base_eval_cfg = dict(base_cfg.get("eval", {}))
    base_eval_cfg["policy"] = str(args.policy)
    base_eval_cfg["model_path"] = str(model_path)
    if args.episodes is not None:
        base_eval_cfg["episodes"] = int(args.episodes)
    if args.seed is not None:
        base_eval_cfg["seed"] = int(args.seed)
    if args.device is not None:
        base_eval_cfg["device"] = str(args.device)
    base_cfg["eval"] = base_eval_cfg

    sweeps = _build_default_sweeps(base_env_cfg)
    if args.param != "all":
        if args.values:
            if args.param == "k_hop":
                values = [int(v) for v in args.values]
            else:
                values = [float(v) for v in args.values]
            sweeps = {args.param: values}
        else:
            sweeps = {args.param: sweeps[args.param]}

    output_root = Path(args.run_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    config_sha = sha256_file(str(config_path))
    model_sha = sha256_file(str(model_path))

    aggregate_rows: List[Dict[str, Any]] = []
    episode_rows: List[Dict[str, Any]] = []

    for param, values in sweeps.items():
        for value in values:
            value_tag = _format_value(value)
            run_dir = output_root / param / value_tag
            eval_results_path = run_dir / "eval_results.json"
            if args.skip_existing and eval_results_path.exists():
                LOG.info("Skipping existing run: %s", run_dir)
                eval_results = json.loads(eval_results_path.read_text(encoding="utf-8"))
            else:
                cfg = _apply_overrides(base_cfg, param, value)
                run_dir.mkdir(parents=True, exist_ok=True)
                output_path = evaluate(cfg, config_path=str(config_path), run_dir=run_dir)
                eval_results = json.loads(Path(output_path).read_text(encoding="utf-8"))

            aggregate_rows.append(
                _collect_aggregate_row(
                    param=param,
                    value=value,
                    run_dir=run_dir,
                    eval_results=eval_results,
                    model_sha=model_sha,
                    config_sha=config_sha,
                )
            )
            episode_rows.extend(_collect_episode_rows(param, value, eval_results))

    summary = {
        "meta": {
            "config_path": str(config_path),
            "config_sha256": config_sha,
            "model_path": str(model_path),
            "model_sha256": model_sha,
            "policy": str(args.policy),
            "sweeps": sweeps,
        },
        "runs": aggregate_rows,
    }
    (output_root / "sensitivity_results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_csv(output_root / "sensitivity_episodes.csv", episode_rows)

    LOG.info("Sensitivity summary: %s", output_root / "sensitivity_results.json")
    LOG.info("Sensitivity episodes: %s", output_root / "sensitivity_episodes.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
