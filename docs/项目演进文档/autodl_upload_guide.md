# AutoDL 上传文件清单

## 必须上传的文件/目录

### 1. 源代码 (最重要！)
```
src/           ← 整个目录，包含最新的 bug 修复
configs/       ← 配置文件
scripts/       ← 运行脚本
```

### 2. 数据文件
```
data/processed/graph/
  ├── layer2_nodes.parquet    (28 KB)
  ├── layer2_edges.parquet    (114 KB)
  └── node2vec_embeddings.parquet (159 KB)

data/processed/od_mapped/
  ├── yellow_tripdata_2025-09.parquet (77 MB)
  └── yellow_tripdata_2025-10.parquet (83 MB)
```

### 3. 依赖文件
```
requirements.txt
```

## 快速打包命令

在项目根目录运行以下 PowerShell 命令：

```powershell
# 创建一个干净的目录结构用于上传
$dest = "autodl_upload"
New-Item -ItemType Directory -Path $dest -Force

# 复制源代码
Copy-Item -Path "src" -Destination $dest -Recurse
Copy-Item -Path "configs" -Destination $dest -Recurse
Copy-Item -Path "scripts" -Destination $dest -Recurse

# 复制数据（保留目录结构）
New-Item -ItemType Directory -Path "$dest/data/processed/graph" -Force
New-Item -ItemType Directory -Path "$dest/data/processed/od_mapped" -Force
Copy-Item -Path "data/processed/graph/*" -Destination "$dest/data/processed/graph/"
Copy-Item -Path "data/processed/od_mapped/*" -Destination "$dest/data/processed/od_mapped/"

# 复制依赖文件
Copy-Item -Path "requirements.txt" -Destination $dest

# 打包成 zip
Compress-Archive -Path $dest -DestinationPath "autodl_upload.zip" -Force

Write-Host "打包完成: autodl_upload.zip"
```

## 验证清单

上传后在 AutoDL 上运行以下命令验证：

```bash
# 验证代码版本
git log --oneline -1  # 或者检查关键文件是否存在

# 验证数据完整性
python -c "import pandas as pd; print('nodes:', len(pd.read_parquet('data/processed/graph/layer2_nodes.parquet')))"
python -c "import pandas as pd; print('edges:', len(pd.read_parquet('data/processed/graph/layer2_edges.parquet')))"

# 验证 structural 计算正常
python scripts/reproduce_structural.py
# 应该显示 structural_unserviceable: 0
```

## 注意事项

1. **确保上传最新代码**: `src/env/gym_env.py` 是关键文件
2. **不需要上传的**:
   - `.git/` 目录
   - `reports/` 目录（训练会自动生成）
   - `cache/` 目录
   - 原始 CSV 文件（已转换为 parquet）
3. **建议使用 Git**: 如果可能，直接在 AutoDL 上 `git clone` 或 `git pull` 获取最新代码
