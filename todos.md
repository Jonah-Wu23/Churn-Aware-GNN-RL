# Project TODOs

## 协作规则
- 中文讨论，英文代码/文档。

---

# ✅ 已完成模块 (A-K)

| 模块 | 状态 | 备注 |
|------|------|------|
| A. Event-Driven Gym | ✅ 完成 | 多车、容量、事件驱动 |
| B. Churn Model | ✅ 完成 | 等待+车上sigmoid流失 |
| C. Risk Aggregation | ✅ 完成 | CVaR + Fairness Weight |
| D. Graph Abstraction | ✅ 完成 | Layer-2 stop graph |
| E. State/Feature Spec | ✅ 完成 | 5-dim node, 4-dim edge |
| F. Hard Mask | ✅ 完成 | Budget commitment + mask debug |
| G. Edge-Q GNN | ✅ 完成 | ECC实现，GAT可选 (见下方) |
| H. Reward + TACC | ✅ 完成 | North-star metric |
| I. Curriculum Learning | ✅ 完成 | L0-L3 + Bait/Surge |
| J. Evaluation + Baselines | ✅ 完成 | HCRide/MAPPO/CPO/MOHITO/Wu2024 |
| K. Doc vs Impl Alignment | ✅ 完成 | EdgeQ train/eval scripts |

---

# 🔲 待完成事项

## 0. EdgeQ 训练协议紧急修复 (最高优先级)

> **背景**：L3训练已完成但未收敛（rho≈0.28，远低于trigger_rho=0.8）。L1/L2被"超时推进"而非"达标换关"，phase3出现明显回撤。

### 0.1 修复关卡转换机制 ✅ (2026-01-22)
- [x] **禁止超时换关**：`require_rho_transition=True`，支持 fail_fast/forced 策略
- [x] 增加 `stage_max_steps`、`stage_extension_steps`、`max_stage_extensions` 配置
- [x] 添加日志警告：当接近max_steps但rho未达标时发出提示

### 0.2 修复epsilon重置问题 ✅ (2026-01-22)
- [x] **排查L3 epsilon回跳原因**：`global_step` 跨phase保持连续
- [x] 确保跨phase时epsilon继承：`epsilon_by_stage` 配置 + `model._global_step` 传递
- [x] 验证fix：L3 phase2→phase3时epsilon平滑衰减

### 0.3 修复phase3奖励切换导致的回撤 ✅ (2026-01-22)
- [x] **实现渐进过渡**：`reward_ramp.py` 模块实现线性插值
- [x] 公式：`alpha = phase_step / reward_ramp_steps`，`weight = (1-alpha)*W2 + alpha*W3`
- [x] phase间奖励权重线性插值过渡

### 0.4 改进模型选择策略 ✅ (2026-01-22)
- [x] **停止用训练极值选模型**：改用 `evaluation_checkpointer.py`
- [x] 实现**固定seed评估选模**：
  - `eval_seeds: [42, 123, 456, 789, 1000]`，epsilon=0
  - 用评估rho均值选best checkpoint
- [x] 选模准则：mean_rho 为主，service_rate 为次
- [x] 保留 `best_rho`, `best_service_rate`, `last` checkpoint

### 0.5 验收标准 ⏳ (待实际训练验证)
- [ ] L1/L2不再超时推进，必须rho≥trigger才换关
- [ ] L3 phase切换时epsilon连续、奖励渐进
- [ ] 使用固定seed评估选出的best模型，rho≥0.5
- [ ] 训练曲线显示phase3不再出现回撤

---

## 1. MOHITO/Wu2024 域内训练 (高优先级)

训练器已实现，需要在Autodl上运行完整训练：

```bash
# Wu2024 (200k steps, ~2-4h on GPU)
nohup python scripts/run_wu2024_train.py --config configs/manhattan.yaml --device cuda > wu2024_train.log 2>&1 &

# MOHITO (需要torch_geometric)
nohup python scripts/run_mohito_train.py --config configs/manhattan.yaml --device cuda > mohito_train.log 2>&1 &
```

**验收标准：**
- [ ] Wu2024 训练完成，服务率≥30%
- [ ] MOHITO 训练完成，服务率≥30%
- [ ] 保存训练曲线、checkpoint、config hash

---

## 2. 训练后评估

完成训练后，需要用统一evaluator重新评估：

```bash
python scripts/run_eval.py --config configs/manhattan.yaml \
    --policy wu2024 --model-path reports/wu2024_train/run_xxx/wu2024_model_final.pt
```

**验收标准：**
- [ ] 训练后MOHITO/Wu2024服务率≥30%，Gini有效
- [ ] 所有baselines使用相同test seeds评估
- [ ] 输出相同格式的eval_results.json

---

## 3. 文档更新 (论文提交前)

- [ ] 更新README实验设置：训练预算、seeds、超参数
- [ ] 明确声明：\"All baselines trained in-domain to convergence\"
- [ ] (可选) 附录添加zero-shot结果作为参考

---

## 4. 可选项

- [ ] **GAT变体**：G部分提到的可选架构
- [ ] **Teacher-Student蒸馏**：部署优化

---

# 📁 关键文件索引

| 功能 | 文件 |
|------|------|
| Wu2024训练器 | `src/train/wu2024_trainer.py` |
| MOHITO训练器 | `src/train/mohito_trainer.py` |
| Wu2024训练脚本 | `scripts/run_wu2024_train.py` |
| MOHITO训练脚本 | `scripts/run_mohito_train.py` |
| 统一评估器 | `src/eval/evaluator.py` |
| 训练配置 | `configs/manhattan.yaml` (mohito_train/wu2024_train) |
| 训练状态文档 | `docs/baseline_training_status.md` |

---

# 📋 Autodl 快速参考

```bash
# 上传
rsync -avz --exclude='__pycache__' . root@xxx.autodl.com:/root/minibus/

# 安装
pip install -r requirements.txt
pip install torch-geometric torch-scatter torch-sparse  # MOHITO only

# 训练
nohup python scripts/run_wu2024_train.py --config configs/manhattan.yaml --device cuda > wu2024.log 2>&1 &

# 监控
tail -f wu2024.log

# 下载
rsync -avz root@xxx.autodl.com:/root/minibus/reports/ ./reports/
```
