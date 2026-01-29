"""深入分析 structural 值在不同 episode 中的变化模式 - 保存到文件"""
import json
from pathlib import Path

log_path = Path("reports/runs/curriculum_20260114_165103/stage_L1/train_log.jsonl")
output_path = Path("reports/episode_analysis.txt")

lines = []
def log(msg):
    lines.append(msg)
    print(msg)

log("=== 深入分析 structural_unserviceable 的模式 ===\n")

episodes = []
with open(log_path, "r") as f:
    for line in f:
        data = json.loads(line)
        if data.get("type") == "episode":
            episodes.append({
                "step": data["step"],
                "episode_steps": data.get("episode_steps", 0),
                "served": data.get("served", 0),
                "waiting_churned": data.get("waiting_churned", 0),
                "onboard_churned": data.get("onboard_churned", 0),
                "structural": data.get("structural_unserviceable", 0),
            })

log("所有 Episode 详情:")
log("-" * 100)
log(f"{'Step':>8} {'EpSteps':>10} {'Served':>8} {'WaitChurn':>10} {'OnChurn':>8} {'Struct':>8} {'Total':>8}")
log("-" * 100)

for ep in episodes:
    total = ep['served'] + ep['waiting_churned'] + ep['onboard_churned'] + ep['structural']
    log(f"{ep['step']:>8} {ep['episode_steps']:>10} {ep['served']:>8} {ep['waiting_churned']:>10} {ep['onboard_churned']:>8} {ep['structural']:>8} {total:>8}")

log("\n=== 模式分析 ===")

# 分析 structural=1715 和 structural=0 的 episode 特征
struct_1715 = [ep for ep in episodes if ep['structural'] == 1715]
struct_0 = [ep for ep in episodes if ep['structural'] == 0]

log(f"\nstructural=1715 的 episode 数量: {len(struct_1715)}")
if struct_1715:
    avg_steps = sum(ep['episode_steps'] for ep in struct_1715) / len(struct_1715)
    log(f"  平均 episode_steps: {avg_steps:.0f}")
    log(f"  episode_steps 值: {set(ep['episode_steps'] for ep in struct_1715)}")

log(f"\nstructural=0 的 episode 数量: {len(struct_0)}")
if struct_0:
    avg_steps = sum(ep['episode_steps'] for ep in struct_0) / len(struct_0)
    log(f"  平均 episode_steps: {avg_steps:.0f}")
    log(f"  episode_steps 值:")
    for ep in struct_0:
        log(f"    step={ep['step']}, episode_steps={ep['episode_steps']}")

log("\n=== 关键发现 ===")
if struct_1715 and all(ep['episode_steps'] == 2000 for ep in struct_1715):
    log("[!] structural=1715 的所有 episode 的 episode_steps 都是 2000（max_horizon_steps）")
    log("    这说明这些 episode 是因为达到了 max_horizon_steps 而结束的")
    
if struct_0 and all(ep['episode_steps'] < 2000 for ep in struct_0):
    log("[!] structural=0 的所有 episode 的 episode_steps 都小于 2000")
    log("    这说明这些 episode 是因为其他原因（如 event_queue 为空）提前结束的")
    log("    提前结束时，尚未遍历全部请求，所以 structural 计数为 0")

# 计算比例
log("\n=== 数值分析 ===")
log(f"每个完整 episode（2000 steps）的 total requests: ~{2000 + 1715} = 3715")
log(f"其中 structural: 1715 ({1715/3715*100:.1f}%)")
log(f"其中 effective (served + churned): ~330 ({330/3715*100:.1f}%)")

log("\n这个比例非常异常！1715/3715 = 46% 的请求是不可服务的")
log("但我们之前的诊断脚本确认前 2000 条 OD 全部有效（0 个 structural）")
log("\n可能的原因:")
log("1. 训练使用的是不同位置的 OD 数据")
log("2. 图数据在加载时被修改（prune_zero_in/out）")
log("3. OD 数据中的节点 ID 与图中的节点 ID 有偏移")

# 保存到文件
with open(output_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
log(f"\n结果已保存到: {output_path}")
