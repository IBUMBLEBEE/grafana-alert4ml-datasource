# dataset

本目录包含测试数据集和工具脚本。

## 前置准备

### 1. 克隆 NAB 数据集

```bash
git clone https://github.com/numenta/NAB.git dataset/NAB
```

执行后目录结构如下：

```
dataset/
  NAB/
    data/
      realAWSCloudwatch/
      realKnownCause/
      ...
    labels/
      combined_windows.json
      combined_labels.json
```

### 2. 安装 Python 依赖

```bash
cd dataset
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

---

## main.py 用法

脚本路径：`dataset/main.py`

### 导入 NAB 数据到 Elasticsearch

```bash
# 导入全部类别（默认 ES 地址 http://192.168.3.58:9200）
python dataset/main.py

# 指定 ES 地址
python dataset/main.py --es-url http://localhost:9200

# 指定 ES API Key（可选）
python dataset/main.py --es-url http://localhost:9200 --api-key <your-key>

# 只导入指定类别
python dataset/main.py --categories realAWSCloudwatch,realKnownCause

# 预览模式（不写入 ES，只打印统计）
python dataset/main.py --dry-run

# 时间平移：把数据集最晚日期对齐到今天（保留时间部分不变）
python dataset/main.py --shift-time
```

**ES 索引命名规则：** `nab-ts-data-{category小写}`

例：`realAWSCloudwatch` → `nab-ts-data-realaWScloudwatch`

**索引字段：**

| 字段 | 类型 | 说明 |
|---|---|---|
| `@timestamp` | date | ISO8601 UTC |
| `value` | double | 指标值 |
| `metric` | keyword | 文件名（如 `ec2_cpu_utilization_24ae8d`） |
| `category` | keyword | 类别目录名（如 `realAWSCloudwatch`） |

---

### 删除已导入的索引

```bash
# 删除全部类别索引
python dataset/main.py --delete

# 删除指定类别索引
python dataset/main.py --delete --categories realAWSCloudwatch
```

---

### 导出带异常标注的 CSV（供 Go/Rust 单元测试使用）

```bash
# 导出全部类别，默认输出到 dataset/testdata/
python dataset/main.py --export

# 导出指定类别和文件
python dataset/main.py --export --categories realAWSCloudwatch --files ec2_cpu_utilization_24ae8d

# 指定异常标注模式（默认 window：窗口内所有点标为异常）
python dataset/main.py --export --label-mode window   # 推荐，覆盖更完整
python dataset/main.py --export --label-mode point    # 只标注精确异常点

# 指定时间戳格式（默认 ms）
python dataset/main.py --export --timestamp-unit ms   # Unix 毫秒（默认）
python dataset/main.py --export --timestamp-unit s    # Unix 秒
python dataset/main.py --export --timestamp-unit iso  # ISO8601 UTC

# 限制每个文件最多输出行数（用于生成小型测试 fixture）
python dataset/main.py --export --limit 500
```

**导出 CSV 字段：**

| 字段 | 类型 | 说明 |
|---|---|---|
| `timestamp_ms` | int64 | Unix 毫秒时间戳（`--timestamp-unit ms`） |
| `timestamp_s` | int64 | Unix 秒时间戳（`--timestamp-unit s`） |
| `timestamp` | str | ISO8601 UTC（`--timestamp-unit iso`） |
| `value` | float | 指标值 |
| `is_anomaly` | int | 0 = 正常，1 = 异常 |

---

### 导出 history/current 分割文件（用于 Dynamics/Forecast 测试）

```bash
# 以第一个异常区间为中心，分割成 history 和 current 两份
python dataset/main.py --export --split

# 自定义 current 窗口大小（行数，默认 200）
python dataset/main.py --export --split --split-current-rows 300

# 选择第二个异常区间（0=第一个，默认）
python dataset/main.py --export --split --window-index 1
```

输出文件示例：

```
dataset/testdata/
  realAWSCloudwatch/
    ec2_cpu_utilization_24ae8d_history.csv
    ec2_cpu_utilization_24ae8d_current.csv
```

---

## testdata 目录约定

`dataset/testdata/` 下存放仓库固定测试数据，供 Go/Rust 单元测试直接引用。

- 不要在测试代码里内联长时序或临时生成独立 CSV
- 不要把测试数据散落在仓库其他目录
- 文件名格式：`{metric}_{suffix}.csv`（suffix 为 `history` 或 `current` 或空）
