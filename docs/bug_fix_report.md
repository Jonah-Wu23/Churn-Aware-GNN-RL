# Bug Fix Report

## Repair Summary
- Bugs found: 5
- Bugs fixed: 5
- Tests run: None (not requested)

## Detailed Fixes

### Bug #1: NodeOnlyGNN 缺失 edge_dim 导致训练崩溃
- Fix: `NodeOnlyGNN` 增加 `edge_dim` 属性并支持传参；`run_ablation_train.py` 以 `get_edge_dim(env_cfg)` 传入
- Files: `src/models/node_only_gnn.py`, `scripts/run_ablation_train.py`

### Bug #2: 配置读取使用 ASCII 导致中文字段报错
- Fix: `load_config` 改为 `utf-8` 编码读取
- File: `src/utils/config.py`

### Bug #3: 训练输出模型名与评估/文档不一致
- Fix: 训练结束后自动复制最新 `edgeq_model_final.pt` 为 `model_final.pt`
- File: `scripts/run_ablation_train.py`

### Bug #4: 评估脚本找不到模型路径
- Fix: 批量评估时增加对 `edgeq_model_final.pt` 的探测路径
- File: `scripts/run_ablation_eval.py`

### Bug #5: 批量评估使用旧配置 v10
- Fix: 改为 `configs/manhattan_curriculum_v13.yaml`
- File: `scripts/run_ablation_eval.py`

## Documentation Sync
- 修正消融评估示例路径为带 `{timestamp}` 的训练目录
- File: `如何运行.md`

## Tests
- Not run (user did not request)
