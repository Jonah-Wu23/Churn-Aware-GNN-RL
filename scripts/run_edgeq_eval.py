"""CLI entrypoint for EdgeQ evaluation (no baselines)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.evaluator import evaluate
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", default=None)
    return parser.parse_args()


def _resolve_from_run_dir(run_dir: Path, config: Dict[str, Any]) -> Path:
    run_meta = run_dir / "run_meta.json"
    if run_meta.exists():
        payload = json.loads(run_meta.read_text(encoding="utf-8"))
        model_path = payload.get("model_path_final") or payload.get("model_path_latest")
        if model_path:
            candidate = Path(model_path)
            if candidate.exists():
                return candidate

    direct_final = run_dir / "edgeq_model_final.pt"
    if direct_final.exists():
        return direct_final
    direct_latest = run_dir / "edgeq_model_latest.pt"
    if direct_latest.exists():
        return direct_latest

    stage_dirs = [path for path in run_dir.iterdir() if path.is_dir() and path.name.startswith("stage_")]
    if stage_dirs:
        curriculum = config.get("curriculum", {})
        stages = curriculum.get("stages", [])
        if isinstance(stages, list):
            for stage in reversed(stages):
                stage_dir = run_dir / f"stage_{stage}"
                candidate = stage_dir / "edgeq_model_final.pt"
                if candidate.exists():
                    return candidate
                candidate = stage_dir / "edgeq_model_latest.pt"
                if candidate.exists():
                    return candidate
        for stage_dir in sorted(stage_dirs, key=lambda p: p.name, reverse=True):
            candidate = stage_dir / "edgeq_model_final.pt"
            if candidate.exists():
                return candidate
            candidate = stage_dir / "edgeq_model_latest.pt"
            if candidate.exists():
                return candidate

    candidates = list(run_dir.rglob("edgeq_model_final.pt"))
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)
    candidates = list(run_dir.rglob("edgeq_model_latest.pt"))
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)

    raise FileNotFoundError(f"No EdgeQ model found under {run_dir}")


def _find_latest_run_dir(base_dir: Path) -> Optional[Path]:
    if not base_dir.exists():
        return None
    candidates = list(base_dir.rglob("edgeq_model_final.pt"))
    if candidates:
        newest = max(candidates, key=lambda p: p.stat().st_mtime)
        return newest.parent
    candidates = list(base_dir.rglob("edgeq_model_latest.pt"))
    if candidates:
        newest = max(candidates, key=lambda p: p.stat().st_mtime)
        return newest.parent
    return None


def _infer_root_dir(model_path: Path) -> Path:
    parent = model_path.parent
    if parent.name.startswith("stage_") and parent.parent.exists():
        return parent.parent
    return parent


def _resolve_model_path(args_run_dir: Optional[str], config: Dict[str, Any]) -> Tuple[Path, Optional[Path]]:
    if args_run_dir:
        run_dir = Path(args_run_dir)
        model_path = _resolve_from_run_dir(run_dir, config)
        return model_path, _infer_root_dir(model_path)

    latest_dir = _find_latest_run_dir(Path("reports") / "runs")
    if latest_dir is None:
        raise FileNotFoundError("No training run found under reports/runs")
    model_path = _resolve_from_run_dir(latest_dir, config)
    return model_path, _infer_root_dir(model_path)


def _build_eval_output_dir(run_root: Optional[Path]) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base = Path("reports") / "eval"
    if run_root is None:
        return base / f"edgeq_{timestamp}"
    return base / f"edgeq_{run_root.name}_{timestamp}"


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    cfg = load_config(args.config)
    model_path, run_root = _resolve_model_path(args.run_dir, cfg)
    logging.info("Using EdgeQ model: %s", model_path)

    eval_cfg = dict(cfg.get("eval", {}))
    eval_cfg["policy"] = "edgeq"
    eval_cfg["model_path"] = str(model_path)
    cfg["eval"] = eval_cfg

    output_dir = _build_eval_output_dir(run_root)
    output_path = evaluate(cfg, config_path=args.config, run_dir=output_dir)
    logging.info("Eval results written to %s", output_path)


if __name__ == "__main__":
    main()
