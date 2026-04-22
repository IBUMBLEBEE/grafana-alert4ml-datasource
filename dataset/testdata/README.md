# testdata — 单元测试数据集说明

本目录下的 CSV 文件由 `dataset/main.py --export` 从 [NAB（Numenta Anomaly Benchmark）](https://github.com/numenta/NAB) 数据集生成，用于 Go / Rust 后端单元测试。

---

## 文件命名规则

```
{period}_{label}[_{role}]_{short_id}.csv
```

| 段 | 取值 | 含义 |
|----|------|------|
| `period` | `p24h` / `p7d` / `p24h7d` / `pnone` | 自相关法检测到的主周期 |
| `label`  | `anom` / `clean` | 是否含异常标注（is_anomaly=1 的行） |
| `role`   | `hist` / `curr`（可省略） | split 模式：history 段 / current 段；省略表示完整序列 |
| `short_id` | 原始文件缩短名 | 区分同类文件 |

---

## CSV 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `timestamp_ms` | int64 | Unix 毫秒时间戳（real* 类别使用） |
| `timestamp_s`  | int64 | Unix 秒时间戳（artificial* 类别使用） |
| `value`        | float64 | 指标值 |
| `is_anomaly`   | int (0/1) | 0 = 正常，1 = 异常 |

> `timestamp_ms` 直接对应 `rsod_core::ad::RawPoint.timestamp_ms: i64`。

---

## 标注模式

| 模式 | 参数 | 含义 |
|------|------|------|
| **window**（默认） | `--label-mode window` | `combined_windows.json` 中的异常时间区间内所有点标为 1 |
| **point** | `--label-mode point` | 仅 `combined_labels.json` 中的精确时间戳点标为 1 |

---

## 周期性一览

按周期强度（自相关系数 ACF）选择数据集，覆盖 24h、7d 两种典型周期：

| 文件前缀（目录/） | 间隔 | ACF 24h | ACF 7d | 周期类型 | 有异常 |
|---|---|---|---|---|---|
| `artificialNoAnomaly/p24h_clean_art_daily_no_noise` | 5 min | **1.0** | **1.0** | 24h（纯正弦基线） | 否 |
| `artificialNoAnomaly/p24h_clean_art_daily_small_noise` | 5 min | **0.99** | **0.99** | 24h（带噪声基线） | 否 |
| `artificialWithAnomaly/p24h_anom_art_daily_*` | 5 min | 0.86–0.93 | 0.87–0.94 | 24h | 是 |
| `realAWSCloudwatch/p24h_{}_rds_cc0c53` | 5 min | **0.71** | –0.30 | **24h** | 是 |
| `realAWSCloudwatch/p24h_{}_rds_e47b3b` | 5 min | **0.68** | –0.38 | **24h** | 是 |
| `realAWSCloudwatch/p24h_{}_ec2_53ea38` | 5 min | **0.62** | 0.56 | **24h** | 是 |
| `realAdExchange/p24h7d_{}_exchange2_cpm` | 60 min | **0.82** | **0.69** | **24h + 7d** | 是 |
| `realAdExchange/p24h7d_{}_exchange2_cpc` | 60 min | **0.82** | **0.60** | **24h + 7d** | 是 |
| `realKnownCause/p24h7d_{}_asg_cpu` | 5 min | **0.77** | **0.64** | **24h + 7d** | 是 |
| `realKnownCause/p7d_{}_nyc_taxi` | 30 min | **0.80** | **0.92** | **7d（最强）** | 是 |

---

## 目录结构

```
testdata/
├── artificialNoAnomaly/               # 纯周期基线（无异常，timestamp_s）
│   ├── p24h_clean_art_daily_no_noise.csv          # 完美正弦，4032 行，14 天
│   └── p24h_clean_art_daily_small_noise.csv       # 带小噪声正弦，4032 行，14 天
├── artificialWithAnomaly/             # 人工合成异常序列（point 模式，timestamp_s）
│   ├── p24h_anom_art_daily_flatmiddle.csv
│   ├── p24h_anom_art_daily_jumpsdown.csv
│   ├── p24h_anom_art_daily_jumpsup.csv
│   ├── p24h_anom_art_daily_nojump.csv
│   ├── p24h_anom_art_increase_spike_density.csv
│   └── p24h_anom_art_load_balancer_spikes.csv
├── realAWSCloudwatch/                 # 真实 AWS CloudWatch（window 模式，timestamp_ms，5min）
│   ├── pnone_anom_ec2_24ae8d.csv              # 完整序列，24h 周期弱（ACF=0.12）
│   ├── pnone_clean_hist_ec2_24ae8d.csv        # ^ split history
│   ├── pnone_anom_curr_ec2_24ae8d.csv         # ^ split current
│   ├── p24h_clean_hist_ec2_53ea38.csv         # 24h 周期中（ACF=0.62）split history
│   ├── p24h_anom_curr_ec2_53ea38.csv          # ^ split current
│   ├── p24h_clean_hist_rds_cc0c53.csv         # 24h 周期强（ACF=0.71）split history
│   ├── p24h_anom_curr_rds_cc0c53.csv          # ^ split current
│   ├── p24h_clean_hist_rds_e47b3b.csv         # 24h 周期强（ACF=0.68）split history
│   └── p24h_anom_curr_rds_e47b3b.csv          # ^ split current
├── realAdExchange/                    # 广告交换平台（window 模式，timestamp_ms，1h 间隔）
│   ├── p24h7d_clean_hist_exchange2_cpc.csv    # 24h+7d 双周期 split history
│   ├── p24h7d_anom_curr_exchange2_cpc.csv     # ^ split current
│   ├── p24h7d_clean_hist_exchange2_cpm.csv    # 24h+7d 双周期 split history
│   └── p24h7d_anom_curr_exchange2_cpm.csv     # ^ split current
└── realKnownCause/                    # 真实已知原因异常（window 模式，timestamp_ms）
    ├── p24h7d_clean_hist_asg_cpu.csv          # 24h+7d 双周期，50 天基线
    ├── p24h7d_anom_curr_asg_cpu.csv           # ^ split current（ASG 错误配置）
    ├── p7d_clean_hist_nyc_taxi.csv            # 7d 周期最强（ACF=0.92），118 天基线
    └── p7d_anom_curr_nyc_taxi.csv             # ^ split current（万圣节高峰）
```

---

## 各数据集详情

### artificialNoAnomaly（纯周期基线）

采样间隔：5 分钟，时间范围：2014-04-01 ~ 2014-04-14，共 4032 行，ACF 24h = 1.0。  
标注模式：`window`（`timestamp_s`），is_anomaly 全为 0。

| 文件 | 值域 | 用途 |
|------|------|------|
| `p24h_clean_art_daily_no_noise.csv` | [10, 90] | 完美 24h 正弦，无噪声，建立周期检测基线 |
| `p24h_clean_art_daily_small_noise.csv` | [8, 92] | 24h 正弦 + 小高斯噪声，测试抗噪能力 |

**适用测试场景**：验证检测器在纯周期信号上不产生误报。

---

### artificialWithAnomaly（人工合成）

采样间隔：5 分钟，时间范围：2014-04-01 ~ 2014-04-14，共 4032 行，ACF 24h ≈ 0.86–0.93。  
标注模式：`point`（`timestamp_s`）。

| 文件 | 异常模式 | 精确异常时间戳 | 异常窗口 | 值域 |
|------|----------|----------------|----------|------|
| `p24h_anom_art_daily_flatmiddle.csv` | 中段值归零（flatline） | 2014-04-11 00:00:00 | 2014-04-10 07:15 ~ 2014-04-11 16:45 | [-22, 88] |
| `p24h_anom_art_daily_jumpsdown.csv`  | 值突然跳降 | 2014-04-11 09:00:00 | 2014-04-10 16:15 ~ 2014-04-12 01:45 | [18, 88] |
| `p24h_anom_art_daily_jumpsup.csv`    | 值突然跳升 | 2014-04-11 09:00:00 | 2014-04-10 16:15 ~ 2014-04-12 01:45 | [18, 165] |
| `p24h_anom_art_daily_nojump.csv`     | 基线漂移无跳变 | 2014-04-11 09:00:00 | 2014-04-10 16:15 ~ 2014-04-12 01:45 | [18, 88] |
| `p24h_anom_art_increase_spike_density.csv` | 毛刺密度增加 | 2014-04-07 23:10:00 | 2014-04-07 06:25 ~ 2014-04-08 15:55 | [0, 20] |
| `p24h_anom_art_load_balancer_spikes.csv` | 负载均衡器周期性尖峰 | 2014-04-11 04:35:00 | 2014-04-10 11:50 ~ 2014-04-11 21:20 | [0, 3.23] |

**适用测试场景**：异常类型识别、边界检测（flatmiddle / jump）、稀疏异常检测。

---

### realAWSCloudwatch（真实 AWS EC2/RDS CPU）

指标：EC2/RDS 实例 CPU 使用率（%），采样间隔：5 分钟。  
标注模式：`window`（`timestamp_ms`）。

#### `pnone_*_ec2_24ae8d` — ACF 24h=0.12（周期弱，作为对照）

| 文件 | 行数 | 时间范围 | 异常行数 |
|------|------|----------|----------|
| `pnone_anom_ec2_24ae8d.csv`       | 4032 | 2014-02-14 ~ 2014-02-28 | 402 |
| `pnone_clean_hist_ec2_24ae8d.csv` | 3297 | 2014-02-14 ~ 2014-02-26 | 0   |
| `pnone_anom_curr_ec2_24ae8d.csv`  | 731  | 2014-02-26 ~ 2014-02-28 | 402 |

异常窗口：2014-02-26 13:45 ~ 2014-02-27 06:25，2014-02-27 08:55 ~ 2014-02-28 01:35

#### `p24h_*_ec2_53ea38` — ACF 24h=0.62（**24h 周期中等**）

| 文件 | 行数 | 时间范围 | 异常行数 |
|------|------|----------|----------|
| `p24h_clean_hist_ec2_53ea38.csv` | 1108 | 2014-02-14 ~ 2014-02-18 | 0   |
| `p24h_anom_curr_ec2_53ea38.csv`  | 777  | 2014-02-18 ~ 2014-02-21 | 201 |

异常窗口（取第 0 个）：2014-02-19 10:50 ~ 2014-02-20 03:30

#### `p24h_*_rds_cc0c53` — ACF 24h=0.71（**24h 周期强**）

| 文件 | 行数 | 时间范围 | 异常行数 |
|------|------|----------|----------|
| `p24h_clean_hist_rds_cc0c53.csv` | 2692 | 2014-02-14 ~ 2014-02-23 | 0   |
| `p24h_anom_curr_rds_cc0c53.csv`  | 1276 | 2014-02-23 ~ 2014-02-28 | 402 |

异常窗口（取第 0 个）：2014-02-24 22:50 ~ 2014-02-25 15:35

#### `p24h_*_rds_e47b3b` — ACF 24h=0.68（**24h 周期强**）

| 文件 | 行数 | 时间范围 | 异常行数 |
|------|------|----------|----------|
| `p24h_clean_hist_rds_e47b3b.csv` | 558  | 2014-04-10 ~ 2014-04-11 | 0   |
| `p24h_anom_curr_rds_e47b3b.csv`  | 777  | 2014-04-11 ~ 2014-04-14 | 201 |

异常窗口（取第 0 个）：2014-04-12 22:32 ~ 2014-04-13 15:12

---

### realAdExchange（广告交易平台，1h 间隔）

采样间隔：60 分钟，标注模式：`window`（`timestamp_ms`）。  
**24h + 7d 双周期**（ACF 24h ≈ 0.82，ACF 7d ≈ 0.60–0.69）。

| 文件 | 行数 | 时间范围 | 异常行数 |
|------|------|----------|----------|
| `p24h7d_clean_hist_exchange2_cpc.csv` | 76  | 2011-07-01 ~ 2011-07-04 | 0   |
| `p24h7d_anom_curr_exchange2_cpc.csv`  | 499 | 2011-07-04 ~ 2011-07-24 | 163 |
| `p24h7d_clean_hist_exchange2_cpm.csv` | 397 | 2011-07-01 ~ 2011-07-17 | 0   |
| `p24h7d_anom_curr_exchange2_cpm.csv`  | 786 | 2011-07-17 ~ 2011-08-19 | 162 |

`cpc` 异常窗口：2011-07-11 04:00 ~ 2011-07-17 22:00  
`cpm` 异常窗口 1（取第 0 个）：2011-07-24 14:00 ~ 2011-07-27 22:00  

**适用测试场景**：测试检测器在 1h 粒度、双周期（业务流量）数据上的表现。

---

### realKnownCause（真实已知原因异常）

标注模式：`window`（`timestamp_ms`）。

#### `p24h7d_*_asg_cpu` — **24h + 7d 双周期**，ACF 24h=0.77，ACF 7d=0.64

采样间隔：5 分钟，62 天，ASG 错误配置导致的 CPU 异常。

| 文件 | 行数 | 时间范围 | 异常行数 |
|------|------|----------|----------|
| `p24h7d_clean_hist_asg_cpu.csv` | 14535 | 2014-05-14 ~ 2014-07-03 | 0    |
| `p24h7d_anom_curr_asg_cpu.csv`  | 3515  | 2014-07-03 ~ 2014-07-15 | 1499 |

异常窗口：2014-07-10 12:29 ~ 2014-07-15 17:19  
**适用测试场景**：检测器处理有充足历史数据（50 天基线）的双周期长序列；7d 工作日/周末模式验证。

#### `p7d_*_nyc_taxi` — **7d 周期最强**，ACF 7d=0.92，ACF 24h=0.80

采样间隔：30 分钟，纽约出租车载客量。取第 0 个异常窗口（万圣节高峰）。

| 文件 | 行数 | 时间范围 | 异常行数 |
|------|------|----------|----------|
| `p7d_clean_hist_nyc_taxi.csv` | 5639 | 2014-07-01 ~ 2014-10-26 | 0   |
| `p7d_anom_curr_nyc_taxi.csv`  | 607  | 2014-10-26 ~ 2014-11-08 | 207 |

异常窗口（万圣节）：2014-10-30 15:30 ~ 2014-11-03 22:30  
**适用测试场景**：7d 周期强信号 + 节假日异常；验证模型对周期性流量异常的识别能力。

---

## 使用示例

### Rust（rsod-ad 单元测试）

```rust
use rsod_utils::read_csv_to_vec;
use rsod_core::ad::{MinimalAdRequest, RawPoint};

fn load_raw_points(path: &str) -> Vec<RawPoint> {
    // CSV 格式: timestamp_ms,value,is_anomaly
    let data: Vec<[f64; 3]> = read_csv_to_vec(path);
    data.iter().map(|r| RawPoint {
        timestamp_ms: r[0] as i64,
        value: r[1],
    }).collect()
}

#[test]
fn test_detect_periodic_24h() {
    // p24h — rds_cc0c53: ACF 24h=0.71，强 24h 周期
    let history = load_raw_points("../../dataset/testdata/realAWSCloudwatch/p24h_clean_hist_rds_cc0c53.csv");
    let current = load_raw_points("../../dataset/testdata/realAWSCloudwatch/p24h_anom_curr_rds_cc0c53.csv");
    let req = MinimalAdRequest { history, current, ..Default::default() };
    let resp = rsod_ad::detect::detect_online(req).unwrap();
    assert!(resp.points.iter().any(|p| p.is_anomaly));
}

#[test]
fn test_detect_periodic_7d() {
    // p7d — nyc_taxi: ACF 7d=0.92，7d 周期最强
    let history = load_raw_points("../../dataset/testdata/realKnownCause/p7d_clean_hist_nyc_taxi.csv");
    let current = load_raw_points("../../dataset/testdata/realKnownCause/p7d_anom_curr_nyc_taxi.csv");
    let req = MinimalAdRequest { history, current, ..Default::default() };
    let resp = rsod_ad::detect::detect_online(req).unwrap();
    assert!(resp.points.iter().any(|p| p.is_anomaly));
}

#[test]
fn test_no_false_alarm_on_clean_periodic() {
    // p24h clean — 完美 24h 正弦，期望无误报（is_anomaly 全为 0）
    let data = load_raw_points("../../dataset/testdata/artificialNoAnomaly/p24h_clean_art_daily_no_noise.csv");
    // ... 构造 history + current，期望 resp.points 全部 is_anomaly == false
}
```

### Go（plugin 单元测试）

```go
// 读取 CSV 构造 data.Frame，然后调用 rsod.MiniAdDetect(currentFrame, historyFrame, opts)
// is_anomaly 列用于断言输出帧中的异常点位置
```

---

## 重新生成数据

export 命令按 NAB 原始文件名生成 CSV，生成后需手动运行 mv 重命名。

```bash
cd dataset

# 无异常基线（24h 周期）
python main.py --export \
  --categories artificialNoAnomaly \
  --files art_daily_no_noise,art_daily_small_noise \
  --timestamp-unit s
# 重命名: 加 p24h_clean_ 前缀

# 人工合成异常序列（24h 周期）
python main.py --export \
  --categories artificialWithAnomaly \
  --label-mode point --timestamp-unit s
# 重命名: 加 p24h_anom_ 前缀

# 24h 周期真实 AWS 数据（split）
python main.py --export \
  --categories realAWSCloudwatch \
  --files ec2_cpu_utilization_53ea38,rds_cpu_utilization_cc0c53,rds_cpu_utilization_e47b3b \
  --split --split-current-rows 576 --window-index 0 --timestamp-unit ms
# 重命名: ec2_cpu_utilization_53ea38_history → p24h_clean_hist_ec2_53ea38，依此类推

# ec2_cpu_utilization_24ae8d（完整 + split）
python main.py --export \
  --categories realAWSCloudwatch \
  --files ec2_cpu_utilization_24ae8d \
  --timestamp-unit ms
python main.py --export \
  --categories realAWSCloudwatch \
  --files ec2_cpu_utilization_24ae8d \
  --split --split-current-rows 300 --timestamp-unit ms

# 24h+7d 双周期（广告交换）
python main.py --export \
  --categories realAdExchange \
  --files exchange-2_cpm_results,exchange-2_cpc_results \
  --split --split-current-rows 336 --window-index 0 --timestamp-unit ms

# 24h+7d 双周期（ASG CPU）
python main.py --export \
  --categories realKnownCause \
  --files cpu_utilization_asg_misconfiguration \
  --split --split-current-rows 4032 --timestamp-unit ms

# 7d 周期最强（NYC Taxi）
python main.py --export \
  --categories realKnownCause \
  --files nyc_taxi \
  --split --split-current-rows 400 --window-index 0 --timestamp-unit ms
```

更多参数详见 `python main.py --export --help`。
