#!/usr/bin/env python3
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

ALGO_COLUMNS = {
    "cea": "cea",
    "milp": "milp",
    "psa": "psa",
    "routing-pca": "ortools",
    "routing-pci": "ortools_pci",
    "cp_sat": "ortools_cpsat",
}

FEASIBLE_SOLVERS = set(ALGO_COLUMNS.keys())

COLUMN_ORDER = ["cea", "milp", "psa", "ortools", "ortools_pci", "ortools_cpsat"]

TARGET_LABELS = ["distance", "empty", "price"]


def cell_n_orders(cell):
    parts = cell.split("_")
    return int(parts[1]) if len(parts) >= 2 else None


def load_records(results_json_path):
    data = json.loads(Path(results_json_path).read_text())
    out = []
    for rec in data.get("records", []):
        algo = rec["algo"]
        if algo not in ALGO_COLUMNS:
            continue
        if rec.get("status") in ("Skipped", "Failed"):
            continue
        value = rec.get("value")
        if value is None or value <= 0:
            continue
        n = cell_n_orders(rec["cell"])
        if n is None:
            continue
        out.append({
            "algo": algo,
            "orders": n,
            "target": rec["objective"],
            "problem_key": f"{rec['cell']}/{rec['instance_id']}",
            "obj": float(value),
            "exec_s": rec["runtime_ms"] / 1000.0,
        })
    return out


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def aggregate_raw(records, target):
    series = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r["target"] != target:
            continue
        series[r["algo"]][r["orders"]].append(r["obj"])
    return {a: {o: mean(v) for o, v in d.items()} for a, d in series.items()}


def aggregate_gap(records, target):
    best = {}
    for r in records:
        if r["target"] != target:
            continue
        key = r["problem_key"]
        if key not in best or r["obj"] < best[key]:
            best[key] = r["obj"]
    series = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r["target"] != target:
            continue
        b = best.get(r["problem_key"])
        if not b or b <= 0:
            continue
        series[r["algo"]][r["orders"]].append(100.0 * (r["obj"] - b) / b)
    return {a: {o: mean(v) for o, v in d.items()} for a, d in series.items()}


def aggregate_time(records, target):
    series = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r["target"] != target:
            continue
        series[r["algo"]][r["orders"]].append(r["exec_s"])
    return {a: {o: mean(v) for o, v in d.items()} for a, d in series.items()}


COLUMN_TO_ALGO = {col: algo for algo, col in ALGO_COLUMNS.items()}


def write_dat(series, all_orders, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["size  " + "  ".join(COLUMN_ORDER)]
    for n in all_orders:
        row = [str(n)]
        for col in COLUMN_ORDER:
            algo = COLUMN_TO_ALGO.get(col)
            v = series.get(algo, {}).get(n) if algo else None
            if v is None or (isinstance(v, float) and math.isnan(v)):
                row.append("nan")
            else:
                row.append(f"{v:.6g}")
        lines.append("  ".join(row))
    output_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_json", help="Path to results/<run-id>/results.json")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for .dat files (default: <run-id>/dat/ next to results.json)",
    )
    args = parser.parse_args()

    results_path = Path(args.results_json).resolve()
    if not results_path.is_file():
        raise SystemExit(f"not found: {results_path}")
    output_dir = Path(args.output_dir) if args.output_dir else results_path.parent / "dat"

    records = load_records(results_path)
    if not records:
        raise SystemExit(f"no usable records in {results_path}")
    all_orders = sorted({r["orders"] for r in records})
    print(
        f"loaded {len(records)} records from {results_path}; orders={all_orders}"
    )

    for target in TARGET_LABELS:
        write_dat(aggregate_raw(records, target), all_orders, output_dir / f"r05-raw-{target}.dat")
        write_dat(aggregate_gap(records, target), all_orders, output_dir / f"r05-gap-{target}.dat")
        write_dat(aggregate_time(records, target), all_orders, output_dir / f"r05-time-{target}.dat")


if __name__ == "__main__":
    main()
