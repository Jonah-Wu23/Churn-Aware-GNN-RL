"""Structured curriculum scenario generation for OD demand."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import json

import numpy as np
import pandas as pd

from src.env.gym_env import _haversine_meters


@dataclass(frozen=True)
class StageSpec:
    name: str
    description: str
    density_multiplier: float = 1.0
    sample_fraction: float = 1.0
    time_scale: float = 1.0
    center_quantile: float = 0.3
    edge_quantile: float = 0.7
    short_trip_quantile: float = 0.3
    long_trip_quantile: float = 0.7
    center_ratio: float = 0.8
    churn_tol_override_sec: Optional[int] = None
    travel_time_multiplier: float = 1.0


@dataclass(frozen=True)
class StageOutput:
    stage: StageSpec
    od_path: Path
    env_overrides: Dict[str, float | int]
    meta: Dict[str, object]


def load_od_frames(od_glob: str | Path) -> pd.DataFrame:
    if Path(od_glob).is_absolute():
        paths = [Path(od_glob)]
    else:
        paths = list(Path().glob(str(od_glob)))
    if not paths:
        raise FileNotFoundError(f"No OD files match {od_glob}")
    frames = [pd.read_parquet(path) for path in paths]
    od = pd.concat(frames, ignore_index=True)
    required = {"pickup_stop_id", "dropoff_stop_id", "tpep_pickup_datetime"}
    if not required.issubset(set(od.columns)):
        raise ValueError("OD data must include pickup_stop_id/dropoff_stop_id/tpep_pickup_datetime")
    od = od.sort_values("tpep_pickup_datetime").reset_index(drop=True)
    return od


def load_nodes(nodes_path: str | Path) -> pd.DataFrame:
    nodes = pd.read_parquet(nodes_path)
    required = {"gnn_node_id", "lon", "lat"}
    if not required.issubset(set(nodes.columns)):
        raise ValueError("Layer 2 nodes must include gnn_node_id/lon/lat for curriculum sampling")
    nodes = nodes[list(required)].copy()
    nodes["gnn_node_id"] = nodes["gnn_node_id"].astype(int)
    return nodes


def annotate_od(od: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    nodes = nodes.set_index("gnn_node_id")
    for col in ("pickup_stop_id", "dropoff_stop_id"):
        if od[col].isna().any():
            raise ValueError(f"OD data includes null {col} values")
    pickup = od["pickup_stop_id"].astype(int)
    dropoff = od["dropoff_stop_id"].astype(int)
    pickup_nodes = nodes.loc[pickup]
    dropoff_nodes = nodes.loc[dropoff]

    center_lon = float(nodes["lon"].mean())
    center_lat = float(nodes["lat"].mean())
    pickup_dist_center = _haversine_meters(
        pickup_nodes["lon"].to_numpy(),
        pickup_nodes["lat"].to_numpy(),
        np.full(len(od), center_lon),
        np.full(len(od), center_lat),
    )
    trip_dist = _haversine_meters(
        pickup_nodes["lon"].to_numpy(),
        pickup_nodes["lat"].to_numpy(),
        dropoff_nodes["lon"].to_numpy(),
        dropoff_nodes["lat"].to_numpy(),
    )
    annotated = od.copy()
    annotated["pickup_dist_center_m"] = pickup_dist_center
    annotated["trip_dist_m"] = trip_dist
    return annotated


def _scale_timestamps(od: pd.DataFrame, time_scale: float) -> pd.DataFrame:
    if time_scale <= 0:
        raise ValueError("time_scale must be positive")
    if time_scale == 1.0:
        return od
    t0 = od["tpep_pickup_datetime"].iloc[0]
    deltas = (od["tpep_pickup_datetime"] - t0).dt.total_seconds()
    scaled = od.copy()
    scaled["tpep_pickup_datetime"] = t0 + pd.to_timedelta(deltas * float(time_scale), unit="s")
    return scaled


def _weighted_sample(df: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    if n <= 0:
        return df.iloc[0:0].copy()
    replace = n > len(df)
    return df.sample(n=n, replace=replace, random_state=int(rng.integers(0, 2**31 - 1)))


def build_stage_od(
    base_od: pd.DataFrame,
    stage: StageSpec,
    rng: np.random.Generator,
    quantiles: Dict[str, float],
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    if stage.name == "L0":
        n = int(max(1, len(base_od) * float(stage.sample_fraction)))
        sampled = _weighted_sample(base_od, n, rng)
        sampled = _scale_timestamps(sampled, stage.time_scale)
        meta = {"selection": "global_sparse", "target_count": n}
        return sampled, meta

    if stage.name in {"L1"}:
        center_max = quantiles["center_max"]
        short_max = quantiles["short_max"]
        subset = base_od[
            (base_od["pickup_dist_center_m"] <= center_max)
            & (base_od["trip_dist_m"] <= short_max)
            & (~base_od.get("structural_unreachable", pd.Series(False, index=base_od.index)).astype(bool))
        ]
        target = int(max(1, len(base_od) * float(stage.density_multiplier)))
        sampled = _weighted_sample(subset, target, rng)
        sampled = _scale_timestamps(sampled, stage.time_scale)
        meta = {
            "selection": "center_short",
            "target_count": target,
            "center_max_m": float(center_max),
            "short_max_m": float(short_max),
        }
        return sampled, meta

    if stage.name in {"L2", "bait"}:
        center_max = quantiles["center_max"]
        short_max = quantiles["short_max"]
        edge_min = quantiles["edge_min"]
        long_min = quantiles["long_min"]
        reachable_mask = ~base_od.get("structural_unreachable", pd.Series(False, index=base_od.index)).astype(bool)
        center_subset = base_od[
            (base_od["pickup_dist_center_m"] <= center_max)
            & (base_od["trip_dist_m"] <= short_max)
            & reachable_mask
        ]
        edge_subset = base_od[
            (base_od["pickup_dist_center_m"] >= edge_min)
            & (base_od["trip_dist_m"] >= long_min)
            & reachable_mask
        ]
        target = int(max(1, len(base_od) * float(stage.density_multiplier)))
        center_target = int(round(target * float(stage.center_ratio)))
        edge_target = int(max(0, target - center_target))
        center_sample = _weighted_sample(center_subset, center_target, rng)
        edge_sample = _weighted_sample(edge_subset, edge_target, rng)
        sampled = pd.concat([center_sample, edge_sample], ignore_index=True)
        sampled = sampled.sort_values("tpep_pickup_datetime").reset_index(drop=True)
        sampled = _scale_timestamps(sampled, stage.time_scale)
        meta = {
            "selection": "center_short_plus_edge_long",
            "target_count": target,
            "center_target": center_target,
            "edge_target": edge_target,
            "center_max_m": float(center_max),
            "short_max_m": float(short_max),
            "edge_min_m": float(edge_min),
            "long_min_m": float(long_min),
        }
        return sampled, meta

    if stage.name in {"L3", "surge", "L4"}:
        target = int(max(1, len(base_od) * float(stage.density_multiplier)))
        sampled = _weighted_sample(base_od, target, rng)
        sampled = sampled.sort_values("tpep_pickup_datetime").reset_index(drop=True)
        sampled = _scale_timestamps(sampled, stage.time_scale)
        meta = {"selection": "realistic_or_surge", "target_count": int(len(sampled))}
        return sampled, meta

    raise ValueError(f"Unknown stage name: {stage.name}")


def default_stages() -> List[StageSpec]:
    return [
        StageSpec(
            name="L0",
            description="Sparse global demand, infinite tolerance to learn connectivity.",
            sample_fraction=0.1,
            time_scale=2.0,
            churn_tol_override_sec=10**9,
        ),
        StageSpec(
            name="L1",
            description="Center high-density short trips to induce bait behavior.",
            density_multiplier=1.2,
            time_scale=0.6,
        ),
        StageSpec(
            name="L2",
            description="Center bait plus edge sparse long trips (moral dilemma).",
            density_multiplier=1.2,
            time_scale=0.7,
            center_ratio=0.8,
        ),
        StageSpec(
            name="L3",
            description="Realistic Poisson-like demand with true tolerance.",
            density_multiplier=1.0,
            time_scale=1.0,
        ),
        StageSpec(
            name="L4",
            description="Stress phase with surge and incident-like slowdown.",
            density_multiplier=1.5,
            time_scale=0.6,
            travel_time_multiplier=1.2,
        ),
    ]


def stress_stages() -> List[StageSpec]:
    return [
        StageSpec(
            name="bait",
            description="Center short-trip bait with scarce edge long trips.",
            density_multiplier=1.1,
            time_scale=0.7,
            center_ratio=0.9,
        ),
        StageSpec(
            name="surge",
            description="Overload surge with compressed arrivals and travel slowdown.",
            density_multiplier=1.5,
            time_scale=0.5,
            travel_time_multiplier=1.2,
        ),
    ]


def generate_stage(
    base_od: pd.DataFrame,
    nodes: pd.DataFrame,
    stage: StageSpec,
    output_dir: Path,
    seed: int,
) -> StageOutput:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(seed))
    annotated = annotate_od(base_od, nodes)
    quantiles = {
        "center_max": float(annotated["pickup_dist_center_m"].quantile(stage.center_quantile)),
        "edge_min": float(annotated["pickup_dist_center_m"].quantile(stage.edge_quantile)),
        "short_max": float(annotated["trip_dist_m"].quantile(stage.short_trip_quantile)),
        "long_min": float(annotated["trip_dist_m"].quantile(stage.long_trip_quantile)),
    }
    stage_od, meta = build_stage_od(annotated, stage, rng, quantiles)
    keep_cols = ["pickup_stop_id", "dropoff_stop_id", "tpep_pickup_datetime"]
    if "structural_unreachable" in stage_od.columns:
        keep_cols.append("structural_unreachable")
    stage_od = stage_od[keep_cols].copy()

    od_path = output_dir / f"od_{stage.name}.parquet"
    stage_od.to_parquet(od_path, index=False)

    env_overrides: Dict[str, float | int] = {
        # NOTE: Do NOT override max_requests here. Let the config file value apply.
        # The curriculum generates a stage-specific OD file, and max_requests
        # in the config controls how many records are loaded per episode.
    }
    if stage.churn_tol_override_sec is not None:
        env_overrides["churn_tol_sec"] = int(stage.churn_tol_override_sec)
        env_overrides["waiting_churn_tol_sec"] = int(stage.churn_tol_override_sec)
        env_overrides["onboard_churn_tol_sec"] = int(stage.churn_tol_override_sec)
    if stage.travel_time_multiplier != 1.0:
        env_overrides["travel_time_multiplier"] = float(stage.travel_time_multiplier)

    meta_payload = {
        "stage": stage.name,
        "description": stage.description,
        "seed": int(seed),
        "counts": {
            "base": int(len(base_od)),
            "stage": int(len(stage_od)),
        },
        "quantiles": quantiles,
        "params": {
            "density_multiplier": float(stage.density_multiplier),
            "sample_fraction": float(stage.sample_fraction),
            "time_scale": float(stage.time_scale),
            "center_ratio": float(stage.center_ratio),
            "churn_tol_override_sec": stage.churn_tol_override_sec,
            "travel_time_multiplier": float(stage.travel_time_multiplier),
        },
        "meta": meta,
    }
    audit_path = output_dir / f"audit_{stage.name}.json"
    audit_path.write_text(json.dumps(meta_payload, ensure_ascii=False, indent=2), encoding="ascii")

    return StageOutput(stage=stage, od_path=od_path, env_overrides=env_overrides, meta=meta_payload)
