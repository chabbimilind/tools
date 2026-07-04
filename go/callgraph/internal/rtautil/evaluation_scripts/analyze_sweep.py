#!/usr/bin/env python3
"""
Parse the per-target .log files produced by run_oss_rta_sweep.sh and emit
speedup / memory-overhead / scaling / parallel-efficiency tables.

Usage:
    ./analyze_sweep.py <output_dir>

Each <target>.log inside <output_dir> is expected to contain Infof-style
lines of the form:
    INFO rta_duration_secs_<flavor>_<N>_workers=<float>
    INFO rta_peak_heap_delta_mb_<flavor>_<N>_workers=<float>
    INFO rta_callgraph_nodes_<flavor>_<N>_workers=<float>
    INFO rta_callgraph_edges_<flavor>_<N>_workers=<float>
    INFO Finished CG construction-only evaluation
plus a few other metric lines we don't compute over.

We anchor on FLAVOR_BASELINE for speedup ratios and on
PRTA_KUMO_NONBLOCKING (the parallel "production" flavor) for scaling and
efficiency.
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

FLAVOR_BASELINE = "srta"
PRTA_FLAVOR = "prta_kumo_nonblocking"
COMPLETION_MARKER = "Finished CG construction-only evaluation"
JSON_METRICS_MARKER = "oss_rta_metrics"

# Worker counts the harness reports for the parallel flavor.
PRTA_WORKER_GRID = [1, 2, 4, 8, 16, 32, 64]

# Parses a metric key like "rta_duration_secs_srta_kumo_nonblocking_8_workers"
# into (metric_name, flavor, workers). The metric_name is the longest match
# from the known prefix set; flavor is the chunk between metric and `_<N>_workers`.
METRIC_KEY_RX = re.compile(
    r"^(rta_(?:duration_secs|cpu_time_ms|peak_heap_delta_mb|alloc_bytes_total_mb|"
    r"alloc_objects_total|callgraph_nodes|callgraph_edges|reachable_funcs|"
    r"implements_calls|concrete_types|interface_types|implements_success|"
    r"implements_fail|checks_from_interfaces|checks_from_implementations))_"
    r"([a-z_]+?)_(\d+)_workers$",
)

# Per-metric Infof line, kept as a fallback for old logs without the JSON dump.
METRIC_LINE_RX = re.compile(
    r"\b(rta_(?:duration_secs|peak_heap_delta_mb|callgraph_nodes|callgraph_edges|reachable_funcs|"
    r"cpu_time_ms|alloc_bytes_total_mb|alloc_objects_total|implements_calls|concrete_types|"
    r"interface_types|implements_success|implements_fail|checks_from_interfaces|"
    r"checks_from_implementations))_"
    r"([a-z_]+?)_(\d+)_workers=([0-9eE+.\-]+)",
)


def _split_metric_key(key: str) -> tuple[str, str, int] | None:
    m = METRIC_KEY_RX.match(key)
    if not m:
        return None
    return m.group(1), m.group(2), int(m.group(3))


def parse_log(path: Path) -> tuple[dict[tuple[str, str, int], float], bool]:
    """Return ({(metric, flavor, workers): value}, completed)."""
    metrics: dict[tuple[str, str, int], float] = {}
    completed = False
    json_line: str | None = None
    with path.open() as f:
        for line in f:
            if COMPLETION_MARKER in line:
                completed = True
            if JSON_METRICS_MARKER in line:
                json_line = line
            else:
                m = METRIC_LINE_RX.search(line)
                if m:
                    metric, flavor, workers, raw = m.group(1), m.group(2), int(m.group(3)), m.group(4)
                    try:
                        metrics[(metric, flavor, workers)] = float(raw)
                    except ValueError:
                        pass

    # JSON dump is the canonical source — overrides any per-line values.
    if json_line is not None:
        # Extract the {"metrics": {...}} payload from the tail of the line.
        # The line shape is: TIMESTAMP INFO  oss_rta_metrics<TAB>{"metrics":{...}}
        idx = json_line.find('{"metrics"')
        if idx >= 0:
            try:
                obj = json.loads(json_line[idx:].strip())
                for k, v in obj.get("metrics", {}).items():
                    parsed = _split_metric_key(k)
                    if parsed is None:
                        continue
                    try:
                        metrics[parsed] = float(v)
                    except (TypeError, ValueError):
                        pass
            except json.JSONDecodeError:
                pass
    return metrics, completed


def collect(output_dir: Path) -> dict[str, dict[tuple[str, str, int], float]]:
    targets: dict[str, dict[tuple[str, str, int], float]] = {}
    skipped: list[str] = []
    for log in sorted(output_dir.glob("*.log")):
        name = log.stem
        metrics, done = parse_log(log)
        if not done:
            skipped.append(f"{name} (no completion marker)")
            continue
        if not metrics:
            skipped.append(f"{name} (no metric lines)")
            continue
        targets[name] = metrics
    if skipped:
        print(f"# Skipped {len(skipped)} log(s):", file=sys.stderr)
        for s in skipped:
            print(f"#   {s}", file=sys.stderr)
    return targets


def fmt_float(v: float | None, digits: int = 2) -> str:
    return "-" if v is None else f"{v:.{digits}f}"


def get(metrics: dict[tuple[str, str, int], float], metric: str, flavor: str, workers: int) -> float | None:
    return metrics.get((metric, flavor, workers))


def report_codebase_size(targets):
    print("\n## Codebase size (from prta_kumo_nonblocking @ 64 workers callgraph)")
    print()
    rows = []
    for name, m in targets.items():
        nodes = get(m, "rta_callgraph_nodes", PRTA_FLAVOR, 64) or get(m, "rta_callgraph_nodes", PRTA_FLAVOR, 1)
        edges = get(m, "rta_callgraph_edges", PRTA_FLAVOR, 64) or get(m, "rta_callgraph_edges", PRTA_FLAVOR, 1)
        reach = get(m, "rta_reachable_funcs", PRTA_FLAVOR, 64) or get(m, "rta_reachable_funcs", PRTA_FLAVOR, 1)
        if nodes is None:
            continue
        rows.append((name, int(nodes), int(edges or 0), int(reach or 0)))
    rows.sort(key=lambda r: r[1])
    print(f"{'target':30}  {'cg_nodes':>10}  {'cg_edges':>10}  {'reachable':>10}")
    print(f"{'-'*30}  {'-'*10}  {'-'*10}  {'-'*10}")
    for name, n, e, r in rows:
        print(f"{name:30}  {n:>10,}  {e:>10,}  {r:>10,}")


def report_speedup_over_srta(targets):
    print("\n## Speedup over srta (sequential baseline)")
    print(f"# baseline: rta_duration_secs_{FLAVOR_BASELINE}_1_workers")
    print("# speedup_X = baseline_duration / X_duration  (higher = faster than srta)")
    print()
    flavors = ["srta_kumo", "prta_kumo_nonblocking@1", "prta_kumo_nonblocking@8", "prta_kumo_nonblocking@32", "prta_kumo_nonblocking@64"]
    header = f"{'target':30}  {'srta(s)':>9}  " + "  ".join(f"{f:>22}" for f in flavors)
    print(header)
    print("-" * len(header))
    speedups: dict[str, list[float]] = defaultdict(list)
    for name, m in sorted(targets.items()):
        base = get(m, "rta_duration_secs", FLAVOR_BASELINE, 1)
        if base is None or base == 0:
            continue
        cells = [f"{base:>9.3f}"]
        for f in flavors:
            if "@" in f:
                flavor, w = f.split("@")
                w = int(w)
            else:
                flavor, w = f, 1
            d = get(m, "rta_duration_secs", flavor, w)
            if d is None or d == 0:
                cells.append(f"{'-':>22}")
            else:
                s = base / d
                cells.append(f"{s:>20.2f}x")
                speedups[f].append(s)
        print(f"{name:30}  " + "  ".join(cells))
    print("\n# Median speedup across targets:")
    for f in flavors:
        vals = speedups[f]
        if vals:
            print(
                f"#   {f:35} median {statistics.median(vals):>6.2f}x   "
                f"p10 {sorted(vals)[max(0, len(vals)//10)]:>5.2f}x   "
                f"p90 {sorted(vals)[min(len(vals)-1, len(vals)*9//10)]:>5.2f}x",
            )


def report_memory_overhead(targets):
    print("\n## Memory overhead vs srta peak heap delta")
    print("# metric: rta_peak_heap_delta_mb_<flavor>_<workers>")
    print("# overhead_X = X_peak_mb / srta_peak_mb")
    print()
    flavors = ["srta_kumo", "prta_kumo_nonblocking@1", "prta_kumo_nonblocking@8", "prta_kumo_nonblocking@32", "prta_kumo_nonblocking@64"]
    header = f"{'target':30}  {'srta(MB)':>10}  " + "  ".join(f"{f:>22}" for f in flavors)
    print(header)
    print("-" * len(header))
    overheads: dict[str, list[float]] = defaultdict(list)
    for name, m in sorted(targets.items()):
        base = get(m, "rta_peak_heap_delta_mb", FLAVOR_BASELINE, 1)
        if base is None or base == 0:
            continue
        cells = [f"{base:>10.1f}"]
        for f in flavors:
            if "@" in f:
                flavor, w = f.split("@")
                w = int(w)
            else:
                flavor, w = f, 1
            p = get(m, "rta_peak_heap_delta_mb", flavor, w)
            if p is None:
                cells.append(f"{'-':>22}")
            else:
                ratio = p / base if base > 0 else float("inf")
                cells.append(f"{p:>10.1f} ({ratio:>4.2f}x)")
                overheads[f].append(ratio)
        print(f"{name:30}  " + "  ".join(cells))
    print("\n# Median heap-delta ratio (parallel/srta) across targets:")
    for f in flavors:
        vals = overheads[f]
        if vals:
            print(f"#   {f:35} median {statistics.median(vals):>5.2f}x")


def report_scaling(targets):
    print("\n## Scaling — prta_kumo_nonblocking duration as N grows")
    print(f"# all values in seconds; columns are worker counts {PRTA_WORKER_GRID}")
    print()
    header = f"{'target':30}  " + "  ".join(f"{w:>8}" for w in PRTA_WORKER_GRID)
    print(header)
    print("-" * len(header))
    for name, m in sorted(targets.items()):
        cells = []
        for w in PRTA_WORKER_GRID:
            d = get(m, "rta_duration_secs", PRTA_FLAVOR, w)
            cells.append(f"{'-':>8}" if d is None else f"{d:>8.3f}")
        print(f"{name:30}  " + "  ".join(cells))


def report_parallel_efficiency(targets):
    print("\n## Parallel efficiency — (T_1 / T_N) / N for prta_kumo_nonblocking")
    print("# 1.0 = ideal linear scaling; lower = diminishing returns")
    print()
    header = f"{'target':30}  " + "  ".join(f"{w:>6}" for w in PRTA_WORKER_GRID)
    print(header)
    print("-" * len(header))
    eff_by_w: dict[int, list[float]] = defaultdict(list)
    for name, m in sorted(targets.items()):
        t1 = get(m, "rta_duration_secs", PRTA_FLAVOR, 1)
        if t1 is None or t1 == 0:
            continue
        cells = []
        for w in PRTA_WORKER_GRID:
            tn = get(m, "rta_duration_secs", PRTA_FLAVOR, w)
            if tn is None or tn == 0:
                cells.append(f"{'-':>6}")
            else:
                eff = (t1 / tn) / w
                eff_by_w[w].append(eff)
                cells.append(f"{eff:>6.2f}")
        print(f"{name:30}  " + "  ".join(cells))
    print("\n# Median parallel efficiency across targets per worker count:")
    for w in PRTA_WORKER_GRID:
        vals = eff_by_w[w]
        if vals:
            print(f"#   N={w:<3}  median {statistics.median(vals):>5.2f}   min {min(vals):>5.2f}   max {max(vals):>5.2f}")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    output_dir = Path(argv[1])
    if not output_dir.is_dir():
        print(f"error: not a directory: {output_dir}", file=sys.stderr)
        return 1
    targets = collect(output_dir)
    if not targets:
        print("error: no completed targets found", file=sys.stderr)
        return 1
    print(f"# sweep output: {output_dir}")
    print(f"# completed targets: {len(targets)}")
    report_codebase_size(targets)
    report_speedup_over_srta(targets)
    report_memory_overhead(targets)
    report_scaling(targets)
    report_parallel_efficiency(targets)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
