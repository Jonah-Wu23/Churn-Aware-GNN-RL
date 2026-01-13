# Repository Guidelines

# 严禁简单的实现，严禁最小化实现，严禁不完整实现，严禁漏洞漏西
# 不能重复造轮子，一旦你有什么可能在github上存在的库啊，算法之类，请告诉我，我会git clone

## Project Structure and Module Organization
- `src/` contains the core Python modules, organized by responsibility: `data/`, `graph/`, `env/`, `models/`, `train/`, `sim_sumo/`, `eval/`, and `utils/`.
- `docs/` and `specs/` hold the engineering narrative and formal specifications.
- `configs/` stores YAML configuration files (example: `configs/manhattan.yaml`).
- `data/` is split into `raw/`, `interim/`, `processed/`, and `external/` with `data/manifest.md` tracking provenance.
- `reports/audit/` stores required audit JSONs for mapping and graph construction.
- `tests/` contains pytest-based checks (schema, connectivity, masks, rewards).

## Build, Test, and Development Commands
- Build logical graph: `python scripts/build_graph.py --config configs/manhattan.yaml` (creates Layer 2 graph and stop map).
- Train (skeleton): `python scripts/run_gym_train.py --config configs/manhattan.yaml` (entrypoint for Gym training).
- SUMO validation (skeleton): `python scripts/run_sumo_eval.py --config configs/manhattan.yaml` (entrypoint for TraCI validation).
- Run tests: `pytest` (expects placeholder tests to be implemented).

## Coding Style and Naming Conventions
- Python, 4-space indentation, PEP 8 style.
- Use `snake_case` for functions/variables, `PascalCase` for classes, and module names in `snake_case`.
- Keep config keys lower_snake_case and path values relative to repo root.
- Prefer `logging` or `loguru` for structured logging; no print-based logging in production modules.

## Testing Guidelines
- Framework: pytest.
- Test files should follow `tests/test_*.py` naming.
- Minimum coverage focus: schema validation, graph connectivity, mask compliance, reward logic.

### Recommended Test Commands
Core repo tests are isolated from baseline dependencies via `pytest.ini` configuration.

```bash
# Run all core tests (recommended - uses pytest.ini defaults)
pytest -q tests

# Run with verbose output for debugging
pytest -v tests

# Run a specific test file
pytest tests/test_churn_game.py

# Run tests matching a pattern
pytest -k "churn" tests
```

**Important**: The `baselines/` directory contains third-party reference implementations with their own dependencies. These are excluded from test discovery by default. To run baseline-specific tests, set up each baseline's environment separately per its own README.

## Commit and Pull Request Guidelines
- No Git history is present in this repository; no established commit message convention is detectable.
- Recommended format: Conventional Commits (e.g., `feat: add od mapping audit report`).
- PRs should include a concise description, linked issue (if any), and validation notes (commands run, metrics or audit outputs).

## Reproducibility and Audit Expectations
- Any data mapping or graph build must emit `reports/audit/*.json` with mismatch and barrier metrics.
- Record random seeds, config used, and dataset hashes for every experiment.
- Report structural unreachability separately and exclude it from algorithmic churn metrics.

## Repository Status Snapshot (Auto-Updated)
- Event-driven Gym env, hard mask budgets, CVaR risk, fairness weights: implemented.
- Graph build emits audit JSON + SVG, fixes zero travel times, supports zero-in/out pruning.
- Node2Vec embeddings via local `node2vec/` clone: implemented.
- DQNTrainer exists but CLI runner remains baseline/diagnostic; evaluator/curriculum/baselines pending.
