#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export PYTHONPATH="$PWD"

CONFIG="configs/manhattan_curriculum_v13.yaml"

EDGEQ_MODEL="reports/runs/L3/stage_L3/edgeq_model_best.pt"
MAPPO_MODEL="reports/mappo_train/run_20260125_205928/models/actor.pt"
CPO_MODEL="reports/cpo_train/run_20260125_210038/best_model.pkl"
MOHITO_MODEL="reports/mohito_train/run_20260124_214941/mohito_actor_best.pth"
WU2024_MODEL="reports/wu2024_train/run_20260124_180842/wu2024_model_best.pt"

check_exists() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Missing model file: $path" >&2
    exit 1
  fi
}

check_exists "$EDGEQ_MODEL"
check_exists "$MAPPO_MODEL"
check_exists "$CPO_MODEL"
check_exists "$MOHITO_MODEL"
check_exists "$WU2024_MODEL"

echo "Evaluating edgeq..."
python scripts/run_eval.py --config "$CONFIG" \
  --policy edgeq \
  --model-path "$EDGEQ_MODEL" \
  --run-dir "reports/eval/edgeq"

echo "Evaluating hcride..."
python scripts/run_eval.py --config "$CONFIG" \
  --policy hcride \
  --run-dir "reports/eval/hcride"

echo "Evaluating mappo..."
python scripts/run_eval.py --config "$CONFIG" \
  --policy mappo \
  --model-path "$MAPPO_MODEL" \
  --run-dir "reports/eval/mappo"

echo "Evaluating cpo..."
python scripts/run_eval.py --config "$CONFIG" \
  --policy cpo \
  --model-path "$CPO_MODEL" \
  --run-dir "reports/eval/cpo"

echo "Evaluating mohito..."
python scripts/run_eval.py --config "$CONFIG" \
  --policy mohito \
  --model-path "$MOHITO_MODEL" \
  --run-dir "reports/eval/mohito"

echo "Evaluating wu2024..."
python scripts/run_eval.py --config "$CONFIG" \
  --policy wu2024 \
  --model-path "$WU2024_MODEL" \
  --run-dir "reports/eval/wu2024"

echo "All baselines evaluated."
