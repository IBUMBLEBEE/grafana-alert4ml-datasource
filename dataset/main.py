#!/home/ibumblebee/code/go/src/grafana-alert4ml-datasource/dataset/.venv/bin/python
"""
NAB (Numenta Anomaly Benchmark) 数据集导入 Elasticsearch 脚本

数据来源: 本地目录 dataset/NAB/data（已 git clone https://github.com/numenta/NAB.git）
CSV 格式: timestamp (YYYY-MM-DD HH:MM:SS), value

索引命名规则: nab-ts-data-{category} (小写)
  例: nab-ts-data-realaWSCloudwatch -> nab-ts-data-realaWSCloudwatch

用法:
    # 导入全部类别（默认）
    python scripts/import_nab_to_es.py

    # 指定类别
    python scripts/import_nab_to_es.py --categories realAWSCloudwatch,realKnownCause

    # 仅预览，不写入 ES
    python scripts/import_nab_to_es.py --dry-run

索引字段:
    @timestamp  date     ISO8601 UTC 时间
    value       double   指标值
    metric      keyword  文件名 (如 ec2_cpu_utilization_24ae8d)
    category    keyword  类别目录名 (如 realAWSCloudwatch)
"""

import argparse
import csv
import json
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.request import urlopen, Request

# ── 默认配置 ──────────────────────────────────────────────────────────────────

# 脚本所在目录的父目录为项目根目录
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
DEFAULT_DATA_DIR = str(_PROJECT_ROOT / "dataset" / "NAB" / "data")
DEFAULT_ES_URL   = "http://192.168.3.58:9200"
INDEX_PREFIX     = "nab-ts-data"   # 最终索引: nab-ts-data-{category小写}

# ── ES 索引 Mapping ───────────────────────────────────────────────────────────

INDEX_MAPPING = {
    "mappings": {
        "properties": {
            "@timestamp": {"type": "date"},
            "value":      {"type": "double"},
            "metric":     {"type": "keyword"},
            "category":   {"type": "keyword"},
        }
    },
    "settings": {
        "number_of_shards":   1,
        "number_of_replicas": 0,
    },
}

# ── CSV 解析 ──────────────────────────────────────────────────────────────────

def parse_csv_file(csv_path: Path) -> list[dict]:
    """读取本地 NAB CSV，返回文档列表（不含 metric/category 字段）"""
    records = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts_str  = row.get("timestamp", "").strip()
            val_str = row.get("value", "").strip()
            if not ts_str or not val_str:
                continue
            try:
                dt    = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                value = float(val_str)
            except ValueError:
                continue
            records.append({
                "@timestamp": dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "value":      value,
            })
    return records

# ── ES 工具函数 ───────────────────────────────────────────────────────────────

def _es_req(method: str, path: str, body, es_url: str, api_key: str):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"ApiKey {api_key}"
    data = json.dumps(body).encode() if body is not None else None
    req  = Request(f"{es_url}{path}", data=data, headers=headers, method=method)
    try:
        with urlopen(req, timeout=60) as resp:
            return json.loads(resp.read())
    except Exception as exc:
        detail = ""
        if hasattr(exc, "read"):
            detail = "\n" + exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"ES {method} {path} -> {exc}{detail}") from exc


def create_index_if_absent(index: str, es_url: str, api_key: str):
    try:
        _es_req("PUT", f"/{index}", INDEX_MAPPING, es_url, api_key)
        print(f"  [INDEX] '{index}' created")
    except RuntimeError as exc:
        if "resource_already_exists_exception" in str(exc):
            print(f"  [INDEX] '{index}' already exists, skip")
        else:
            raise


def bulk_index(index: str, docs: list[dict], es_url: str, api_key: str) -> int:
    """分批 Bulk 写入，返回成功写入数"""
    BATCH = 500
    total = 0
    for i in range(0, len(docs), BATCH):
        batch = docs[i: i + BATCH]
        lines = []
        for doc in batch:
            lines.append(json.dumps({"index": {"_index": index}}))
            lines.append(json.dumps(doc))
        body = ("\n".join(lines) + "\n").encode("utf-8")

        headers = {"Content-Type": "application/x-ndjson"}
        if api_key:
            headers["Authorization"] = f"ApiKey {api_key}"
        req = Request(f"{es_url}/_bulk", data=body, headers=headers, method="POST")
        try:
            with urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read())
        except Exception as exc:
            print(f"    [ERROR] bulk failed: {exc}")
            continue

        if result.get("errors"):
            errs = [
                item["index"]["error"]
                for item in result["items"]
                if "error" in item.get("index", {})
            ]
            print(f"    [WARN] {len(errs)} doc errors, first: {errs[0]}")

        total += sum(
            1 for item in result["items"]
            if item.get("index", {}).get("result") in ("created", "updated")
        )
    return total


def find_max_date(cats: dict[str, list]) -> date | None:
    """扫描所有 CSV 文件的最后一行，找到全局最晚日期（NAB 文件按时间顺序排列）"""
    max_dt: datetime | None = None
    for csv_files in cats.values():
        for csv_path in csv_files:
            with csv_path.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            if not rows:
                continue
            ts_str = rows[-1].get("timestamp", "").strip()
            if not ts_str:
                continue
            try:
                dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                if max_dt is None or dt > max_dt:
                    max_dt = dt
            except ValueError:
                pass
    return max_dt.date() if max_dt else None


def apply_time_shift(docs: list[dict], offset_days: int) -> None:
    """原地修改 docs 中的 @timestamp：日期偏移 offset_days 天，时间部分不变"""
    delta = timedelta(days=offset_days)
    for doc in docs:
        dt = datetime.strptime(doc["@timestamp"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        doc["@timestamp"] = (dt + delta).strftime("%Y-%m-%dT%H:%M:%SZ")


def delete_indices(
    cats: dict[str, list],
    es_url: str,
    api_key: str,
    selected_cats: list[str],
):
    """删除导入时创建的 ES 索引，selected_cats 为空则删除全部类别索引"""
    targets = selected_cats if selected_cats else list(cats.keys())
    print(f"ES:      {es_url}")
    print(f"Deleting {len(targets)} indices ...\n")
    for category in targets:
        idx = index_name(category)
        try:
            _es_req("DELETE", f"/{idx}", None, es_url, api_key)
            print(f"  [DELETED] {idx}")
        except RuntimeError as exc:
            if "index_not_found_exception" in str(exc):
                print(f"  [NOT FOUND] {idx}, skip")
            else:
                print(f"  [ERROR] {idx}: {exc}")
    print("\nDone.")

# ── CSV 导出（带异常标注）────────────────────────────────────────────────────

def load_anomaly_windows(nab_root: Path) -> dict[str, list[tuple[datetime, datetime]]]:
    """加载 combined_windows.json，返回 {relative_key: [(start, end), ...]}"""
    windows_file = nab_root / "labels" / "combined_windows.json"
    if not windows_file.exists():
        return {}
    with windows_file.open(encoding="utf-8") as f:
        raw: dict[str, list] = json.load(f)

    result: dict[str, list[tuple[datetime, datetime]]] = {}
    for key, windows in raw.items():
        parsed = []
        for w in windows:
            try:
                # window格式: ["2014-04-10 07:15:00.000000", "2014-04-11 16:45:00.000000"]
                start = datetime.strptime(w[0].split(".")[0], "%Y-%m-%d %H:%M:%S")
                end   = datetime.strptime(w[1].split(".")[0], "%Y-%m-%d %H:%M:%S")
                parsed.append((start, end))
            except (ValueError, IndexError):
                pass
        result[key] = parsed
    return result


def load_anomaly_labels(nab_root: Path) -> dict[str, set[datetime]]:
    """加载 combined_labels.json，返回 {relative_key: {datetime, ...}}"""
    labels_file = nab_root / "labels" / "combined_labels.json"
    if not labels_file.exists():
        return {}
    with labels_file.open(encoding="utf-8") as f:
        raw: dict[str, list] = json.load(f)

    result: dict[str, set[datetime]] = {}
    for key, timestamps in raw.items():
        pts: set[datetime] = set()
        for ts in timestamps:
            try:
                pts.add(datetime.strptime(ts, "%Y-%m-%d %H:%M:%S"))
            except ValueError:
                pass
        result[key] = pts
    return result


def is_in_window(dt: datetime, windows: list[tuple[datetime, datetime]]) -> bool:
    return any(start <= dt <= end for start, end in windows)


def export_labeled_csv(
    data_dir: Path,
    nab_root: Path,
    output_dir: Path,
    selected_cats: list[str],
    selected_metrics: list[str],
    label_mode: str,       # "window" | "point"
    timestamp_unit: str,   # "ms" | "s" | "iso"
    limit: int,            # 0 = unlimited
    split: bool,           # True: 分成 history / current 两份
    split_current_rows: int,  # split 模式下 current 窗口大小（行数）
    window_index: int = 0, # split 时使用第几个异常 cluster（0=第一个）
):
    """将 NAB 数据集导出为带 is_anomaly 标注的 CSV 文件，供 Go/Rust 单元测试使用。

    输出 CSV 字段:
        timestamp_ms  int64   Unix 毫秒时间戳（--timestamp-unit ms，默认）
                   或
        timestamp_s   int64   Unix 秒时间戳（--timestamp-unit s）
                   或
        timestamp     str     ISO8601 UTC（--timestamp-unit iso）
        value         float   指标值
        is_anomaly    int     0 = 正常, 1 = 异常
    """
    all_cats = discover_categories(data_dir)
    if not all_cats:
        print(f"[ERROR] No category directories found in: {data_dir}", file=sys.stderr)
        sys.exit(1)

    # 过滤类别
    if selected_cats:
        cats = {k: all_cats[k] for k in selected_cats if k in all_cats}
    else:
        cats = all_cats

    # 加载标注
    windows_map = load_anomaly_windows(nab_root)
    labels_map  = load_anomaly_labels(nab_root)

    output_dir.mkdir(parents=True, exist_ok=True)

    ts_col_name = {"ms": "timestamp_ms", "s": "timestamp_s", "iso": "timestamp"}.get(timestamp_unit, "timestamp_ms")

    total_files = 0
    for category, csv_files in cats.items():
        for csv_path in csv_files:
            metric = csv_path.stem
            if selected_metrics and metric not in selected_metrics:
                continue

            # 相对键，与 NAB 标注 JSON 中的键对应
            rel_key = f"{category}/{csv_path.name}"

            windows = windows_map.get(rel_key, [])
            labels  = labels_map.get(rel_key, set())

            # 读取并标注
            records: list[tuple[datetime, float, int]] = []
            with csv_path.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    ts_str  = row.get("timestamp", "").strip()
                    val_str = row.get("value", "").strip()
                    if not ts_str or not val_str:
                        continue
                    try:
                        dt    = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                        value = float(val_str)
                    except ValueError:
                        continue

                    if label_mode == "window":
                        anomaly = 1 if is_in_window(dt, windows) else 0
                    else:  # point
                        anomaly = 1 if dt in labels else 0

                    records.append((dt, value, anomaly))

            if not records:
                continue

            if limit > 0:
                records = records[:limit]

            def ts_val(dt: datetime) -> str:
                dt_utc = dt.replace(tzinfo=timezone.utc)
                if timestamp_unit == "ms":
                    return str(int(dt_utc.timestamp() * 1000))
                elif timestamp_unit == "s":
                    return str(int(dt_utc.timestamp()))
                else:
                    return dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

            if split:
                # 将异常索引按间距 > split_current_rows 聚类，按 window_index 选取
                anomaly_indices = [i for i, (_, _, a) in enumerate(records) if a == 1]
                if anomaly_indices:
                    clusters: list[list[int]] = []
                    cur_cluster: list[int] = [anomaly_indices[0]]
                    for idx in anomaly_indices[1:]:
                        if idx - cur_cluster[-1] > split_current_rows:
                            clusters.append(cur_cluster)
                            cur_cluster = [idx]
                        else:
                            cur_cluster.append(idx)
                    clusters.append(cur_cluster)
                    wi = min(window_index, len(clusters) - 1)
                    cluster = clusters[wi]
                    first_anomaly = cluster[0]
                    last_anomaly  = cluster[-1]
                    # current 窗口：以选定异常区间为中心
                    half = split_current_rows // 2
                    cur_start = max(0, first_anomaly - half)
                    cur_end   = min(len(records), last_anomaly + half + 1)
                else:
                    # 无异常：后 split_current_rows 行为 current
                    cur_start = max(0, len(records) - split_current_rows)
                    cur_end   = len(records)

                history_records = records[:cur_start]
                current_records = records[cur_start:cur_end]

                for suffix, subset in [("history", history_records), ("current", current_records)]:
                    if not subset:
                        continue
                    out_path = output_dir / category / f"{metric}_{suffix}.csv"
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    with out_path.open("w", newline="", encoding="utf-8") as out_f:
                        w = csv.writer(out_f)
                        w.writerow([ts_col_name, "value", "is_anomaly"])
                        for dt, val, anomaly in subset:
                            w.writerow([ts_val(dt), val, anomaly])
                    print(f"  -> {out_path.relative_to(output_dir.parent)}  ({len(subset)} rows, {sum(a for _, _, a in subset)} anomalies)")
            else:
                out_path = output_dir / category / f"{metric}.csv"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with out_path.open("w", newline="", encoding="utf-8") as out_f:
                    w = csv.writer(out_f)
                    w.writerow([ts_col_name, "value", "is_anomaly"])
                    for dt, val, anomaly in records:
                        w.writerow([ts_val(dt), val, anomaly])
                anomaly_count = sum(a for _, _, a in records)
                print(f"  -> {out_path.relative_to(output_dir.parent)}  ({len(records)} rows, {anomaly_count} anomalies)")
            total_files += 1

    print(f"\nDone. Exported {total_files} file(s) to {output_dir}")

# ── 主逻辑 ────────────────────────────────────────────────────────────────────

def discover_categories(data_dir: Path) -> dict[str, list[Path]]:
    """扫描本地目录，返回 {category: [csv_path, ...]}"""
    result = {}
    for cat_dir in sorted(data_dir.iterdir()):
        if not cat_dir.is_dir():
            continue
        csvs = sorted(cat_dir.glob("*.csv"))
        if csvs:
            result[cat_dir.name] = csvs
    return result


def index_name(category: str) -> str:
    """nab-ts-data-{category小写}，ES 索引名不允许大写"""
    return f"{INDEX_PREFIX}-{category.lower()}"


def run_import(
    data_dir: Path,
    es_url: str,
    api_key: str,
    selected_cats: list[str],
    dry_run: bool,
    shift_time: bool = False,
):
    all_cats = discover_categories(data_dir)
    if not all_cats:
        print(f"[ERROR] No category directories found in: {data_dir}", file=sys.stderr)
        sys.exit(1)

    # 过滤类别
    if selected_cats:
        unknown = set(selected_cats) - set(all_cats)
        if unknown:
            print(f"[ERROR] Unknown categories: {', '.join(sorted(unknown))}", file=sys.stderr)
            print(f"Available: {', '.join(sorted(all_cats))}", file=sys.stderr)
            sys.exit(1)
        cats = {k: all_cats[k] for k in selected_cats}
    else:
        cats = all_cats

    total_files  = sum(len(v) for v in cats.values())
    total_docs   = 0

    # 计算时间偏移量（全局最晚日期 → 今天，整体向前平移）
    offset_days = 0
    if shift_time:
        max_date = find_max_date(cats)
        if max_date:
            offset_days = (date.today() - max_date).days
        print(f"Time shift: {offset_days:+d} days  "
              f"({max_date} -> {date.today()})" if offset_days else "Time shift: disabled (no valid dates found)")

    print(f"Data dir:   {data_dir}")
    print(f"ES:         {es_url}")
    print(f"Index rule: {INDEX_PREFIX}-{{category}}")
    print(f"Categories: {', '.join(cats)}")
    print(f"Files:      {total_files}")
    print(f"Dry-run:    {dry_run}")
    print()

    for category, csv_files in cats.items():
        idx = index_name(category)
        print(f"[{category}]  ->  {idx}  ({len(csv_files)} files)")

        if not dry_run:
            create_index_if_absent(idx, es_url, api_key)

        for csv_path in csv_files:
            metric = csv_path.stem
            print(f"  {metric}.csv", end=" ... ", flush=True)

            docs = parse_csv_file(csv_path)
            for d in docs:
                d["metric"]   = metric
                d["category"] = category

            if shift_time and offset_days:
                apply_time_shift(docs, offset_days)

            print(f"{len(docs)} records", end="")

            if dry_run:
                print("  [dry-run]")
                total_docs += len(docs)
                continue

            ok = bulk_index(idx, docs, es_url, api_key)
            total_docs += ok
            print(f"  -> {ok} indexed")

        print()

    print(f"Done. Total documents indexed: {total_docs}")

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Import local NAB dataset into Elasticsearch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python scripts/import_nab_to_es.py
  python scripts/import_nab_to_es.py --categories realAWSCloudwatch,realKnownCause
  python scripts/import_nab_to_es.py --dry-run

  # 导出带异常标注的 CSV（用于 Go/Rust 单元测试）
  python main.py --export
  python main.py --export --categories realAWSCloudwatch --files ec2_cpu_utilization_24ae8d
  python main.py --export --label-mode window --timestamp-unit ms --output-dir dataset/testdata
  python main.py --export --split --split-current-rows 200 --limit 2000
""",
    )
    p.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help=f"NAB data directory (default: {DEFAULT_DATA_DIR})",
    )
    p.add_argument(
        "--es-url",
        default=DEFAULT_ES_URL,
        help=f"Elasticsearch endpoint (default: {DEFAULT_ES_URL})",
    )
    p.add_argument(
        "--api-key",
        default="",
        help="ES API key for authentication (optional)",
    )
    p.add_argument(
        "--categories",
        default="",
        help="Comma-separated category names to import/export; omit for all",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse files and print stats without writing to ES",
    )
    p.add_argument(
        "--shift-time",
        action="store_true",
        help="Shift all dates so the earliest date maps to today; time-of-day is preserved",
    )
    p.add_argument(
        "--delete",
        action="store_true",
        help="Delete the target indices from ES and exit (does not import data)",
    )

    # ── 导出模式 ──────────────────────────────────────────────────────────────
    p.add_argument(
        "--export",
        action="store_true",
        help="Export NAB data as labeled CSV files for Go/Rust unit tests",
    )
    p.add_argument(
        "--output-dir",
        default=str(_PROJECT_ROOT / "dataset" / "testdata"),
        help="Output directory for exported CSV files (default: dataset/testdata)",
    )
    p.add_argument(
        "--files",
        default="",
        help="Comma-separated metric file stems to export (e.g. ec2_cpu_utilization_24ae8d); omit for all",
    )
    p.add_argument(
        "--label-mode",
        choices=["window", "point"],
        default="window",
        help="window: mark all points inside anomaly windows (default); point: mark only the labeled timestamps",
    )
    p.add_argument(
        "--timestamp-unit",
        choices=["ms", "s", "iso"],
        default="ms",
        help="Timestamp format in exported CSV: ms (unix ms, default), s (unix seconds), iso (ISO8601 UTC)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max rows per exported file (0 = unlimited); useful for small unit-test fixtures",
    )
    p.add_argument(
        "--split",
        action="store_true",
        help="Split each file into {metric}_history.csv and {metric}_current.csv centred on the anomaly window",
    )
    p.add_argument(
        "--split-current-rows",
        type=int,
        default=200,
        help="Number of rows in the 'current' window when using --split (default: 200)",
    )
    p.add_argument(
        "--window-index",
        type=int,
        default=0,
        help="Which anomaly cluster to center the split around (0=first, default: 0)",
    )
    return p.parse_args()


def main():
    args    = parse_args()
    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        print(f"[ERROR] Data directory not found: {data_dir}", file=sys.stderr)
        print("Please run: git clone https://github.com/numenta/NAB.git dataset/NAB", file=sys.stderr)
        sys.exit(1)

    selected = [c.strip() for c in args.categories.split(",") if c.strip()] if args.categories else []
    es_url   = args.es_url.rstrip("/")

    if args.delete:
        all_cats = discover_categories(data_dir)
        if not all_cats:
            print(f"[ERROR] No categories found in: {data_dir}", file=sys.stderr)
            sys.exit(1)
        delete_indices(all_cats, es_url, args.api_key, selected)
        return

    if args.export:
        selected_metrics = [m.strip() for m in args.files.split(",") if m.strip()] if args.files else []
        nab_root = data_dir.parent  # dataset/NAB/
        export_labeled_csv(
            data_dir          = data_dir,
            nab_root          = nab_root,
            output_dir        = Path(args.output_dir),
            selected_cats     = selected,
            selected_metrics  = selected_metrics,
            label_mode        = args.label_mode,
            timestamp_unit    = args.timestamp_unit,
            limit             = args.limit,
            split             = args.split,
            split_current_rows = args.split_current_rows,
            window_index      = args.window_index,
        )
        return

    run_import(
        data_dir      = data_dir,
        es_url        = es_url,
        api_key       = args.api_key,
        selected_cats = selected,
        dry_run       = args.dry_run,
        shift_time    = args.shift_time,
    )


if __name__ == "__main__":
    main()
    