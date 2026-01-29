"""解析训练日志中的元数据配置"""
import json
from pathlib import Path

log_path = Path("reports/runs/curriculum_20260114_165103/stage_L1/train_log.jsonl")
with open(log_path, "r") as f:
    line = f.readline()
    data = json.loads(line)
    
# 保存完整的元数据到文件
with open("reports/train_meta_debug.json", "w") as f:
    json.dump(data, f, indent=2)

print("=== 训练元数据 ===")
print(json.dumps(data, indent=2))
