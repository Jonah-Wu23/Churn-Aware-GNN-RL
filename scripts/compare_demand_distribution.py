"""Compare curriculum stage demand distribution vs raw eval split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.train.curriculum import StageSpec, annotate_od, default_stages, generate_stage, load_nodes, load_od_frames
from src.utils.config import load_config


def _parse_stage_spec(cfg: Dict[str, Any], stage_name: str) -> StageSpec:
    curriculum_cfg = cfg.get("curriculum", {})
    stage_params = curriculum_cfg.get("stage_params", {}) if isinstance(curriculum_cfg, dict) else {}
    defaults = {spec.name: spec for spec in default_stages()}
    base = defaults.get(stage_name, StageSpec(name=stage_name, description="Custom stage"))
    overrides = stage_params.get(stage_name, {}) if isinstance(stage_params, dict) else {}
    stagespec_fields = {
        "name",
        "description",
        "density_multiplier",
        "sample_fraction",
        "time_scale",
        "center_quantile",
        "edge_quantile",
        "short_trip_quantile",
        "long_trip_quantile",
        "center_ratio",
        "churn_tol_override_sec",
        "travel_time_multiplier",
    }
    od_params = {k: overrides[k] for k in overrides if k in stagespec_fields}
    params = {**base.__dict__, **od_params, "name": stage_name}
    return StageSpec(**params)


def _apply_time_split(od: pd.DataFrame, mode: str | None, ratio: float) -> pd.DataFrame:
    if mode is None:
        return od
    mode = str(mode).lower().strip()
    if mode not in {"train", "eval"}:
        return od
    n = len(od)
    if n == 0:
        return od
    split = int(max(0, min(n, round(n * float(ratio)))))
    if mode == "train":
        return od.iloc[:split].copy()
    return od.iloc[split:].copy()


def _time_density_stats(od: pd.DataFrame) -> Tuple[Dict[str, float], pd.DataFrame]:
    if od.empty:
        summary = {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0, "std": 0.0}
        return summary, pd.DataFrame(columns=["hour", "count"])
    hours = od["tpep_pickup_datetime"].dt.floor("H")
    counts = hours.value_counts().sort_index()
    series = counts.astype(float)
    summary = {
        "mean": float(series.mean()),
        "p50": float(series.quantile(0.5)),
        "p95": float(series.quantile(0.95)),
        "max": float(series.max()),
        "std": float(series.std(ddof=0)),
    }
    df = counts.rename_axis("hour").reset_index(name="count")
    return summary, df


def _compute_stats(
    od: pd.DataFrame,
    nodes: pd.DataFrame,
    quantiles: Dict[str, float],
) -> Dict[str, float]:
    annotated = annotate_od(od, nodes)
    total = float(len(annotated))
    if total == 0:
        return {
            "total": 0.0,
            "center_ratio": 0.0,
            "edge_ratio": 0.0,
            "short_ratio": 0.0,
            "long_ratio": 0.0,
        }
    center_ratio = float((annotated["pickup_dist_center_m"] <= quantiles["center_max"]).mean())
    edge_ratio = float((annotated["pickup_dist_center_m"] >= quantiles["edge_min"]).mean())
    short_ratio = float((annotated["trip_dist_m"] <= quantiles["short_max"]).mean())
    long_ratio = float((annotated["trip_dist_m"] >= quantiles["long_min"]).mean())
    return {
        "total": total,
        "center_ratio": center_ratio,
        "edge_ratio": edge_ratio,
        "short_ratio": short_ratio,
        "long_ratio": long_ratio,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Config YAML path")
    parser.add_argument("--stage", default="L3", help="Curriculum stage to compare")
    parser.add_argument("--output-dir", default="reports/audit/demand_compare", help="Output directory")
    args = parser.parse_args()

    cfg = load_config(args.config)
    env_cfg = cfg.get("env", {})
    stage = _parse_stage_spec(cfg, args.stage)

    base_od = load_od_frames(env_cfg.get("od_glob", "data/processed/od_mapped/*.parquet"))
    nodes = load_nodes(env_cfg.get("graph_nodes_path", "data/processed/graph/layer2_nodes.parquet"))

    base_annotated = annotate_od(base_od, nodes)
    quantiles = {
        "center_max": float(base_annotated["pickup_dist_center_m"].quantile(stage.center_quantile)),
        "edge_min": float(base_annotated["pickup_dist_center_m"].quantile(stage.edge_quantile)),
        "short_max": float(base_annotated["trip_dist_m"].quantile(stage.short_trip_quantile)),
        "long_min": float(base_annotated["trip_dist_m"].quantile(stage.long_trip_quantile)),
    }

    eval_od = _apply_time_split(
        base_od.sort_values("tpep_pickup_datetime").reset_index(drop=True),
        env_cfg.get("time_split_mode"),
        float(env_cfg.get("time_split_ratio", 0.3)),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_output = generate_stage(base_od, nodes, stage, output_dir, seed=int(env_cfg.get("seed", 7)))
    stage_od = pd.read_parquet(stage_output.od_path)

    eval_stats = _compute_stats(eval_od, nodes, quantiles)
    stage_stats = _compute_stats(stage_od, nodes, quantiles)

    eval_time_stats, eval_time_df = _time_density_stats(eval_od)
    stage_time_stats, stage_time_df = _time_density_stats(stage_od)

    eval_time_df["dataset"] = "eval_raw"
    stage_time_df["dataset"] = f"curriculum_{stage.name}"
    time_df = pd.concat([eval_time_df, stage_time_df], ignore_index=True)

    output = {
        "config_path": str(args.config),
        "stage": stage.__dict__,
        "quantiles": quantiles,
        "eval_raw": {
            "stats": eval_stats,
            "time_density": eval_time_stats,
        },
        f"curriculum_{stage.name}": {
            "stats": stage_stats,
            "time_density": stage_time_stats,
            "stage_meta": stage_output.meta,
            "stage_od_path": str(stage_output.od_path),
        },
        "differences": {
            "center_ratio_delta": float(stage_stats["center_ratio"] - eval_stats["center_ratio"]),
            "edge_ratio_delta": float(stage_stats["edge_ratio"] - eval_stats["edge_ratio"]),
            "short_ratio_delta": float(stage_stats["short_ratio"] - eval_stats["short_ratio"]),
            "long_ratio_delta": float(stage_stats["long_ratio"] - eval_stats["long_ratio"]),
        },
    }

    (output_dir / "demand_distribution_compare.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="ascii"
    )
    time_df.to_csv(output_dir / "demand_density_by_hour.csv", index=False)

    print(f"Wrote: {output_dir / 'demand_distribution_compare.json'}")
    print(f"Wrote: {output_dir / 'demand_density_by_hour.csv'}")


if __name__ == "__main__":
    main()
