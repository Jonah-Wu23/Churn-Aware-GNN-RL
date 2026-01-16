# Structural Unserviceable 问题诊断报告

**日期**: 2026-01-14  
**问题**: 训练日志显示 `structural_unserviceable: 1715`，与可达性审计报告 99.88% 可达率矛盾

---

## 问题现象

训练 L1 阶段日志：
```json
{"type": "episode", "step": 2000, "served": 29, "waiting_churned": 297, "structural_unserviceable": 1715}
```

用户疑惑：为什么有 1715 个请求被标记为"结构性不可服务"？

---

## 诊断过程

### 1. 验证数据一致性

| 数据项 | 哈希值 | 状态 |
|--------|--------|------|
| layer2_nodes.parquet | `8b72d28d05...` | ✅ 一致 |
| layer2_edges.parquet | `889cc26ac9...` | ✅ 一致 |
| od_L1.parquet | `0a55429025...` | ✅ 一致 |

### 2. 使用不同 max_requests 验证

| max_requests | 加载请求数 | structural 数量 |
|--------------|-----------|-----------------|
| 2,000 | 2,000 | **0** |
| 3,248,834 | 3,248,834 | **1,715** |

---

## 根本原因

### 问题不是 Bug，而是统计口径

1. **Curriculum L1 阶段**生成了 **324 万**条采样订单
2. 其中有 **1715 条**跨强连通分量（无有向路径）
3. 训练时 `max_requests` 被设置为整个订单池大小（324 万）
4. `structurally_unserviceable` 统计的是**整个订单池**中的不可达请求

### 关键数据

- 图有 **4 个强连通分量**：343, 193, 22, 1 个节点
- 不可达比例：1715 / 3,248,834 = **0.053%**（非常小）
- 每个 episode 实际处理：约 **330 个**请求

---

## 结论

| 方面 | 结论 |
|------|------|
| 训练效果 | ✅ 不受影响 |
| 模型学习 | ✅ 正常 |
| 日志显示 | ⚠️ 1715 是全局统计，不是每轮失败数 |

### 日志解读

```
structural_unserviceable: 1715  ← 整个 324 万订单中有 1715 个死单
served: 29                      ← 本轮成功服务 29 个乘客
waiting_churned: 297            ← 本轮流失 297 个乘客
```

---

## 可选优化

如需减少内存占用和启动时间，可在 `configs/manhattan.yaml` 添加：

```yaml
env:
  max_requests: 10000  # 限制加载的订单数
```

这样 `structural_unserviceable` 会降至约 0-5 个。

---

## 附录：验证命令

```bash
# 验证不同 max_requests 的 structural 数量
python -c "
from src.env.gym_env import EnvConfig, EventDrivenEnv
config = EnvConfig(max_requests=3248834, od_glob='reports/runs/.../od_L1.parquet', ...)
env = EventDrivenEnv(config)
print(f'structural: {env.structurally_unserviceable}')  # 输出: 1715
"
```
