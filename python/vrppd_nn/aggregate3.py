"""Aggregate Stage 3 results into tables, figures and paired tests.

  python -m vrppd_nn.aggregate3 --root results/stage3

Works on whatever batches are finished: outputs of a batch without batches/<batch>/DONE
(missing or still running) stay out of every table, figure and CSV, and summary.md
starts with a coverage table. tables/ and figures/ are cleared first, so nothing from an
earlier aggregation survives. Neural runs must use the tau of selection.json.
Layout read under --root:
  batches/<batch>/manifest.json, DONE, STARTED, probe.json   (see BATCHES)
  lns/<group>/<operator>/<cell>/<instance>_s<seed>.json      LNS run files; groups:
      iters<N>, time<N>s, ksweep-k<frac>, objective_<price|empty>,
      select-tau<tau> (validation instances, only used by write_selection)
  reference/cea_ref_<cell>.json, reference/objective_<obj>/cea_ref_<cell>.json
  models/{b0,v1}/{grid,trainsize,reg,seeds}/<name>/, models/{b0,v1}/best.json, models/v2/<run>/
  selection/selection.json, label_stats/stats_*.md
Writes into --root: runs.csv, summary.csv, summary.md, stats.csv,
training_summary.csv, tables/*.tex (table rows only) and figures/*.png.

Gaps are relative to the CEA reference of the instance (same objective) and to
the best value known for it (CEA and every LNS run on that instance and objective),
as in Stage 2. Objective values of one solution differ by about 1e-4 km between
macOS and Linux, so every consistency check uses a relative tolerance.

Pipeline helpers used by scripts/stage3/run_all.sh:
  python -m vrppd_nn.aggregate3 pick-best <model dir>...
      prints the directory whose metrics.json has the highest selection_value
  python -m vrppd_nn.aggregate3 select --root R --val-dir D --model NAME=WEIGHTS ...
      chooses tau per model, the best V2 run and best_model from lns/select-tau*/
      and writes selection/selection.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import sys

import matplotlib
import numpy as np
import pandas as pd

from vrppd_nn.aggregate import CELL_ORDER, reference_row, run_row, time_to_target
from vrppd_nn.fileio import write_atomic, write_json_atomic
from vrppd_nn.stats import holm, paired_wilcoxon

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Queue order of scripts/stage3/run_all.sh (a test checks that `run_all.sh --list` agrees).
BATCHES = ["data", "b0", "v1", "ref-10_10", "ref-10_20", "ref-20_50", "v2-a", "v2-b", "v2-c", "v2-d", "v2-e",
           "select", "iters-10_10", "iters-10_20", "iters-20_50", "time-10_10", "time-10_20", "time-20_50",
           "ksweep", "large-30_100", "large-50_200", "large-100_500", "obj-price", "obj-empty"]
OPERATORS = ["random", "worst", "related", "adaptive", "b0", "v1", "v2", "adaptive_n"]
NEURAL = ["b0", "v1", "v2", "adaptive_n"]
BASELINES = ["random", "adaptive"]
# Categorical palette in fixed slot order (the Stage 2 operators keep their colours).
COLORS = {"random": "#2a78d6", "worst": "#eb6834", "related": "#1baf7a", "adaptive": "#eda100",
          "b0": "#e87ba4", "v1": "#008300", "v2": "#6250d6", "adaptive_n": "#e34948"}
SLOTS = list(COLORS.values())
SMALL_CELLS = ["10_10", "10_20", "20_50"]
LARGE_CELLS = ["30_100", "50_200", "100_500"]
KIND_ORDER = ["iters", "time", "ksweep", "objective"]
REL_TOL = 1e-6
GRID_POINTS = 200
TEST_INSTANCES = "problem-gen/problems"
RUN_COLUMNS = ["budget", "operator", "cell", "instance", "seed", "initial", "best", "iterations", "elapsed_ms",
               "accepted", "improved_current", "new_best_count", "best_iteration", "best_elapsed_ms", "k",
               "destroy_ms", "repair_ms", "valid", "kind", "split", "objective", "k_frac", "tau", "host", "threads",
               "cpu_list", "budget_iters", "budget_secs", "iters_to_1pct", "secs_to_1pct"]
REF_COLUMNS = ["cell", "instance", "cea_best", "cea_mean", "cea_runs", "cea_mean_runtime_s", "objective", "host",
               "threads", "cpu_list", "file"]


# ---------------------------------------------------------------- loading

def parse_group(group: str) -> dict:
    """What a run group directory under lns/ holds (see the module docstring)."""
    info = {"kind": None, "split": "test", "objective": "distance", "k_frac": np.nan}
    if re.fullmatch(r"iters\d+", group):
        info["kind"] = "iters"
    elif re.fullmatch(r"time\d+(\.\d+)?s", group):
        info["kind"] = "time"
    elif m := re.fullmatch(r"ksweep-k(\d+(\.\d+)?)", group):
        info.update(kind="ksweep", k_frac=float(m.group(1)))
    elif m := re.fullmatch(r"objective_(price|empty)", group):
        info.update(kind="objective", objective=m.group(1))
    elif re.fullmatch(r"select-tau\d+(\.\d+)?", group):
        info.update(kind="select", split="val")
    else:
        raise ValueError(f"unknown run group lns/{group}/")
    return info


def load_references3(root: str) -> pd.DataFrame:
    """CEA references keyed by (cell, instance, objective), with the CPU setup they ran on."""
    rows = []
    paths = sorted(glob.glob(os.path.join(root, "reference", "cea_ref_*.json"))
                   + glob.glob(os.path.join(root, "reference", "objective_*", "cea_ref_*.json")))
    for path in paths:
        parent = os.path.basename(os.path.dirname(path))
        expected = parent[len("objective_"):] if parent.startswith("objective_") else "distance"
        with open(path) as f:
            records = json.load(f)
        for r in records:
            objective = r.get("objective", "distance")
            if objective != expected:
                raise ValueError(f"{path}: record {r['cell']}/{r['instance']} has objective {objective!r}, "
                                 f"files there must hold {expected!r}")
            rows.append({**reference_row(r), "objective": objective, "host": r.get("host", ""),
                         "threads": r.get("threads", 0), "cpu_list": r.get("cpu_list", ""), "file": path})
    return pd.DataFrame(rows, columns=REF_COLUMNS)


def grid_axis(budget_iters, budget_secs) -> tuple[str, np.ndarray]:
    """x key and grid of a run's curve: elapsed time for wall-time runs, else log-spaced iterations."""
    if pd.notna(budget_secs) and budget_secs > 0:
        return "elapsed_ms", np.linspace(0.0, 1000.0 * budget_secs, GRID_POINTS)
    return "iteration", np.geomspace(1, max(budget_iters if pd.notna(budget_iters) else 1, 2), GRID_POINTS)


def grid_curve(curve: list[dict], budget_iters, budget_secs) -> np.ndarray:
    """Best-so-far value of a run at the points of grid_axis (step interpolation)."""
    xkey, grid = grid_axis(budget_iters, budget_secs)
    x = np.array([row[xkey] for row in curve], dtype=float)
    y = np.array([row["best"] for row in curve], dtype=float)
    idx = np.clip(np.searchsorted(x, grid, side="right") - 1, 0, len(x) - 1)
    return y[idx]


def load_runs3(root: str, refs: pd.DataFrame | None = None, group_glob: str = "*") -> tuple[pd.DataFrame, dict]:
    """Every LNS run file under lns/<group_glob>/ in one pass (curves are not kept whole:
    only iteration and wall-time runs keep a gridded best-so-far curve, keyed by
    (group, operator, cell, instance, seed)). Raises ValueError when a file's objective
    is not the one of its group, or its best value differs from its own objective total."""
    if refs is None:
        refs = load_references3(root)
    targets = {(r.cell, r.instance, r.objective): r.cea_best for r in refs.itertuples()}
    rows, curves = [], {}
    for path in sorted(glob.glob(os.path.join(root, "lns", group_glob, "*", "*", "*.json"))):
        group, operator, cell = path.split(os.sep)[-4:-1]
        info = parse_group(group)
        with open(path) as f:
            d = json.load(f)
        objective = d.get("objective", "distance")
        if objective != info["objective"]:
            raise ValueError(f"{path}: objective is {objective!r} but runs under lns/{group}/ must be "
                             f"{info['objective']!r}")
        total = d.get(f"best_{objective}")
        if total is not None and np.isfinite(total) and not np.isclose(d["best"], total, rtol=REL_TOL, atol=0.0):
            raise ValueError(f"{path}: best {d['best']} differs from best_{objective} {total}")
        row = run_row(d, group, operator, cell)
        row.update(kind=info["kind"], split=info["split"], objective=objective, k_frac=info["k_frac"],
                   tau=d.get("tau", np.nan), host=d.get("host", ""), threads=d.get("threads", 0),
                   cpu_list=d.get("cpu_list", ""), budget_iters=d.get("budget_iters"),
                   budget_secs=d.get("budget_secs"))
        target = targets.get((cell, row["instance"], objective)) if info["split"] == "test" else None
        row["iters_to_1pct"], row["secs_to_1pct"] = (time_to_target(d["curve"], 1.01 * target)
                                                     if target is not None else (np.nan, np.nan))
        if info["kind"] in ("iters", "time"):
            key = (group, operator, cell, row["instance"], row["seed"])
            curves[key] = grid_curve(d["curve"], d.get("budget_iters"), d.get("budget_secs"))
        rows.append(row)
    return pd.DataFrame(rows, columns=RUN_COLUMNS), curves


def add_gaps(runs: pd.DataFrame, refs: pd.DataFrame) -> pd.DataFrame:
    """Test runs with the CEA reference, best known value and both gaps (%)."""
    keys = ["cell", "instance", "objective"]
    df = runs[runs.split == "test"].merge(refs[keys + ["cea_best"]], on=keys, how="left")
    best_known = df.groupby(keys)["best"].min().rename("lns_best_known").reset_index()
    df = df.merge(best_known, on=keys, how="left")
    df["best_known"] = np.fmin(df["lns_best_known"], df["cea_best"].fillna(np.inf))
    df["gap_cea_pct"] = 100 * (df["best"] - df["cea_best"]) / df["cea_best"]
    df["gap_best_pct"] = 100 * (df["best"] - df["best_known"]) / df["best_known"]
    df["iters_per_s"] = 1000 * df["iterations"] / df["elapsed_ms"].clip(lower=1e-6)
    df["accept_rate"] = df["accepted"] / df["iterations"].clip(lower=1)
    df["destroy_share"] = df["destroy_ms"] / (df["destroy_ms"] + df["repair_ms"]).clip(lower=1e-9)
    df["reached_1pct"] = np.where(df["cea_best"].notna(), df["iters_to_1pct"].notna().astype(float), np.nan)
    return df


def run_batch(kind: str, cell: str, objective: str) -> str:
    """The batch of scripts/stage3/run_all.sh that writes runs of this kind, class and objective."""
    if kind in ("select", "ksweep"):
        return kind
    if kind == "objective":
        return f"obj-{objective}"
    if cell in LARGE_CELLS:
        return f"large-{cell}"
    return f"{kind}-{cell}"  # iters-<cell>, time-<cell>


def reference_batch(cell: str, objective: str) -> str:
    """The batch that writes the CEA reference of this class and objective."""
    if objective != "distance":
        return f"obj-{objective}"
    return f"large-{cell}" if cell in LARGE_CELLS else f"ref-{cell}"


# ---------------------------------------------------------------- checks

def check_timed_runs(runs: pd.DataFrame, refs: pd.DataFrame | None = None) -> None:
    """Raise ValueError (naming the class) if the wall-time files of one class, its
    time-budget runs and CEA references, come from more than one (host, threads, CPU list)."""
    cols = ["cell", "objective", "host", "threads", "cpu_list"]
    parts = [runs.loc[runs.kind == "time", cols]]
    if refs is not None and len(refs):
        parts.append(refs[cols])
    timed = pd.concat(parts, ignore_index=True)
    problems = []
    for (cell, objective), sub in timed.groupby(["cell", "objective"]):
        setups = sub.groupby(["host", "threads", "cpu_list"]).size()
        if len(setups) > 1:
            desc = "; ".join(f"host={h!r} threads={t} cpus={c!r} ({n} files)" for (h, t, c), n in setups.items())
            problems.append(f"class {cell} ({objective}): wall-time files come from more than one CPU setup: {desc}")
    if problems:
        raise ValueError("\n".join(problems))


def check_neural_taus(runs: pd.DataFrame, selection: dict) -> None:
    """Raise ValueError naming the run directory when a neural run of the main, k-sweep, large
    or objective studies used another tau than selection.json gives its operator."""
    expected = {"b0": selection["models"].get("b0", {}).get("tau"), "v1": selection["models"].get("v1", {}).get("tau"),
                "v2": (selection.get("v2") or {}).get("tau"), "adaptive_n": selection["best_model"]["tau"]}
    expected = {op: tau for op, tau in expected.items() if tau is not None}
    neural = runs[runs.kind.isin(["iters", "time", "ksweep", "objective"]) & runs.operator.isin(list(expected))]
    bad = []
    for (group, op, cell), g in neural.groupby(["budget", "operator", "cell"]):
        taus = g["tau"].astype(float)
        wrong = ~np.isclose(taus, expected[op], rtol=0.0, atol=1e-12)
        if wrong.any():
            bad.append(f"lns/{group}/{op}/{cell}: tau {sorted(set(taus[wrong]))}, selection.json gives {expected[op]}")
    if bad:
        raise ValueError("neural runs do not use the selected tau (kept from an older selection?):\n" + "\n".join(bad))


def _strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _strings(v)


def check_selection(selection) -> None:
    """Raise ValueError if a selection (dict, or path to selection.json) lists any test instance."""
    if not isinstance(selection, dict):
        with open(selection) as f:
            selection = json.load(f)
    leaks = sorted({s for s in _strings(selection) if TEST_INSTANCES in s})
    if leaks:
        raise ValueError(f"selection lists {len(leaks)} test instance path(s) ({TEST_INSTANCES}), e.g. {leaks[0]}")


# ---------------------------------------------------------------- pipeline helpers

def pick_best(model_dirs: list[str]) -> str:
    """The model directory whose metrics.json has the highest selection_value (validation).
    All candidates must have been ranked by the same metric (select_by)."""
    best, best_value, metrics = None, -np.inf, set()
    for d in model_dirs:
        path = os.path.join(d, "metrics.json")
        if not os.path.exists(path):
            continue
        m = _read_json(path)
        metrics.add(m.get("select_by"))
        value = m["selection_value"]
        if value is not None and np.isfinite(value) and value > best_value:
            best, best_value = d, value
    if len(metrics) > 1:
        raise SystemExit(f"models ranked by different metrics {sorted(map(str, metrics))}: not comparable")
    if best is None:
        raise SystemExit(f"no finished model among {len(model_dirs)} directories")
    return best


def _instance_file(val_dir: str, cell: str, instance: str) -> str:
    for name in (f"{instance}.json", f"{instance}_*.json"):
        found = sorted(glob.glob(os.path.join(glob.escape(os.path.join(val_dir, cell)), name)))
        if found:
            return found[0]
    raise ValueError(f"no instance file for {cell}/{instance} under {val_dir}")


def write_selection(root: str, val_dir: str, weights: dict[str, str]) -> dict:
    """Choose tau per model (lowest mean objective over the validation search runs in
    lns/select-tau*/, all cells, instances and repeats pooled), the best V2 run and
    best_model among b0, v1 and that V2 run; write selection/selection.json."""
    runs, _ = load_runs3(root, refs=pd.DataFrame(columns=REF_COLUMNS), group_glob="select-tau*")
    if runs.empty:
        raise ValueError(f"no selection runs under {root}/lns/select-tau*/")
    missing = sorted(set(weights) - set(runs.operator))
    if missing:
        raise ValueError(f"no selection runs for {missing}")
    runs = runs[runs.operator.isin(list(weights))]
    covered = {key: frozenset(zip(g.cell, g.instance, g.seed)) for key, g in runs.groupby(["operator", "tau"])}
    taus = {op: frozenset(g.tau) for op, g in runs.groupby("operator")}
    if len(set(covered.values())) != 1 or len(set(taus.values())) != 1:
        raise ValueError("selection runs are incomplete: models or taus differ in their (instance, seed) sets")
    files = sorted({_instance_file(val_dir, c, i) for c, i in zip(runs.cell, runs.instance)})
    means = runs.groupby(["operator", "tau"])["best"].mean()
    models = {}
    for name in weights:
        by_tau = means.loc[name]
        tau = float(by_tau.idxmin())
        models[name] = {"weights": weights[name], "tau": tau, "mean_objective": float(by_tau.min()),
                        "mean_objective_by_tau": {str(float(t)): float(v) for t, v in by_tau.items()},
                        "runs": int((runs.operator == name).sum() / len(by_tau)), "instance_files": files}
    v2_names = sorted(n for n in weights if n.startswith("v2"))
    v2 = None
    if v2_names:
        best_v2 = min(v2_names, key=lambda n: models[n]["mean_objective"])
        v2 = {"name": best_v2, **models[best_v2]}
    candidates = {op: m for op, m in (("b0", models.get("b0")), ("v1", models.get("v1")), ("v2", v2)) if m}
    best_op = min(candidates, key=lambda op: candidates[op]["mean_objective"])
    best_model = {"name": v2["name"] if best_op == "v2" else best_op, "operator": best_op,
                  **{k: v for k, v in candidates[best_op].items() if k != "name"}}
    selection = {
        "rule": "lowest mean objective over the validation search runs (cells, instances and repeats pooled)",
        "iterations": int(runs["budget_iters"].max()),
        "repeats": int(runs.groupby(["operator", "tau", "cell", "instance"]).size().max()),
        "taus": sorted(float(t) for t in runs["tau"].unique()),
        "validation_dir": val_dir, "instance_files": files,
        "models": models, "v2": v2, "best_model": best_model,
    }
    check_selection(selection)
    os.makedirs(os.path.join(root, "selection"), exist_ok=True)
    write_json_atomic(os.path.join(root, "selection", "selection.json"), selection, indent=2)
    return selection


# ---------------------------------------------------------------- coverage and training

def batch_coverage(root: str) -> pd.DataFrame:
    rows = []
    for b in BATCHES:
        bdir = os.path.join(root, "batches", b)
        manifest = {}
        if os.path.exists(os.path.join(bdir, "manifest.json")):
            with open(os.path.join(bdir, "manifest.json")) as f:
                manifest = json.load(f)
        started = None  # STARTED holds "<host> <start time>"
        if os.path.exists(os.path.join(bdir, "STARTED")):
            with open(os.path.join(bdir, "STARTED")) as f:
                started = f.read().split() or ["?"]
        status = "done" if os.path.exists(os.path.join(bdir, "DONE")) else "started" if started else "missing"
        host = manifest.get("host") or (started[0] if started else "")
        rows.append({"batch": b, "status": status, "host": host, "date": manifest.get("end", "")})
    return pd.DataFrame(rows)


def _read_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def training_summary(root: str, done: set[str] | None = None) -> pd.DataFrame:
    """One row per trained model of a finished batch (all batches when done is None); for the
    seed runs of the best B0 and V1 configuration also their mean and standard deviation.
    train_jobs is the number of training processes the batch ran at once (from its manifest):
    with more than one, train_seconds includes the contention."""
    rows = []
    for family in ("b0", "v1"):
        if done is not None and family not in done:
            continue
        manifest = os.path.join(root, "batches", family, "manifest.json")
        jobs = (_read_json(manifest).get("train_jobs") or {}).get(family, np.nan) if os.path.exists(manifest) else np.nan
        fdir = os.path.join(root, "models", family)
        best = _read_json(os.path.join(fdir, "best.json")) if os.path.exists(os.path.join(fdir, "best.json")) else {}
        fam_rows = {}
        for group in ("grid", "reg", "trainsize", "seeds"):
            for mdir in sorted(glob.glob(os.path.join(fdir, group, "*"))):
                if not os.path.exists(os.path.join(mdir, "metrics.json")):
                    continue
                m, c = _read_json(os.path.join(mdir, "metrics.json")), _read_json(os.path.join(mdir, "config.json"))
                rel = f"{group}/{os.path.basename(mdir)}"
                fam_rows[rel] = {
                    "family": family, "group": group, "name": os.path.basename(mdir), "is_best": rel == best.get("best"),
                    "params": m["n_params"], "spearman": m["spearman"], "p_at_k": m["precision_at_k"],
                    "p_at_k_best_try": m.get("p_at_k_best_try", np.nan),
                    "p_at_k_random": m.get("precision_at_k_random"), "select_by": m.get("select_by"),
                    "selection_value": m.get("selection_value"),
                    "best_epoch": m["best_epoch"], "epochs_run": m["epochs_run"], "train_seconds": m["train_seconds"],
                    "loss": c.get("loss"), "target": c.get("target"), "width": c.get("width"), "depth": c.get("depth"),
                    "d": c.get("d"), "n_att": c.get("n_att"), "dropout": c.get("dropout"),
                    "weight_decay": c.get("weight_decay"), "train_instances": c.get("train_instances"),
                    "seed": c.get("seed"), "train_jobs": jobs,
                    # absent from models trained before these were recorded
                    "threads": m.get("threads", np.nan), "peak_rss_mb": m.get("peak_rss_mb", np.nan)}
        rows.extend(fam_rows.values())
        seed_rows = [fam_rows[s] for s in best.get("seeds", []) if s in fam_rows]
        if seed_rows:
            sdf = pd.DataFrame(seed_rows)
            num = ["params", "spearman", "p_at_k", "p_at_k_best_try", "selection_value", "best_epoch", "epochs_run",
                   "train_seconds", "peak_rss_mb"]
            name = os.path.basename(best["best"])
            values_num = sdf[num].astype(float)
            for stat, values in (("seeds mean", values_num.mean()), ("seeds sd", values_num.std())):
                rows.append({"family": family, "group": stat, "name": f"{name} ({len(sdf)} seeds)", **values.to_dict()})
    for vdir in sorted(glob.glob(os.path.join(root, "models", "v2", "*"))):
        wpath = os.path.join(vdir, "weights.json")
        if not os.path.exists(wpath) or (done is not None and f"v2-{os.path.basename(vdir)}" not in done):
            continue
        w = _read_json(wpath)
        tc = w.get("train_config", {})
        finish = _read_json(os.path.join(vdir, "DONE")) if os.path.exists(os.path.join(vdir, "DONE")) else {}
        rows.append({"family": "v2", "group": "reinforce", "name": f"v2-{os.path.basename(vdir)}",
                     "params": int(sum(np.prod(p["shape"]) for p in w["params"].values())),
                     "best_epoch": tc.get("episode"), "epochs_run": finish.get("episode"),
                     "train_seconds": 3600 * finish["hours"] if "hours" in finish else np.nan,
                     "val_reward": tc.get("val_reward"), "init": tc.get("init"), "samples": tc.get("samples"),
                     "entropy_beta": tc.get("entropy_beta"), "dropout": tc.get("dropout"), "seed": tc.get("seed"),
                     "d": w["config"]["d"], "n_att": w["config"]["n_att"], "train_jobs": 1})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- tables

def _order(values, order) -> list:
    return [v for v in order if v in set(values)] + sorted(set(values) - set(order))


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Per (kind, budget, objective, k_frac, cell, operator): means of both gaps with the
    standard deviation over instances (of per-instance means) and over repeats (mean of
    per-instance standard deviations), objective values, speed and time to 1 % of CEA."""
    keys = ["kind", "budget", "objective", "k_frac", "cell", "operator"]
    per_inst = df.groupby(keys + ["instance"], dropna=False).agg(
        gc=("gap_cea_pct", "mean"), gb=("gap_best_pct", "mean"),
        rc=("gap_cea_pct", "std"), rb=("gap_best_pct", "std"))
    spread = per_inst.groupby(keys, dropna=False).agg(
        sd_inst_gap_cea_pct=("gc", "std"), sd_rep_gap_cea_pct=("rc", "mean"),
        sd_inst_gap_best_pct=("gb", "std"), sd_rep_gap_best_pct=("rb", "mean"))
    main = df.groupby(keys, dropna=False).agg(
        runs=("best", "size"), instances=("instance", "nunique"), mean_best=("best", "mean"), sd_best=("best", "std"),
        mean_gap_cea_pct=("gap_cea_pct", "mean"), mean_gap_best_pct=("gap_best_pct", "mean"),
        mean_iters=("iterations", "mean"), mean_elapsed_s=("elapsed_ms", lambda x: x.mean() / 1000),
        iters_per_s=("iters_per_s", "mean"), reached_1pct=("reached_1pct", "mean"),
        mean_iters_to_1pct=("iters_to_1pct", "mean"), mean_secs_to_1pct=("secs_to_1pct", "mean"),
        accept_rate=("accept_rate", "mean"), destroy_share=("destroy_share", "mean"), valid=("valid", "all"))
    s = main.join(spread).reset_index()
    rank = {"kind": KIND_ORDER, "cell": CELL_ORDER, "operator": OPERATORS}
    sort_cols = []
    for col in ("kind", "budget", "objective", "k_frac", "cell", "operator"):
        if col in rank:
            s[f"_{col}"] = s[col].map({v: i for i, v in enumerate(_order(s[col], rank[col]))})
            sort_cols.append(f"_{col}")
        else:
            sort_cols.append(col)
    return s.sort_values(sort_cols).drop(columns=[c for c in sort_cols if c.startswith("_")]).reset_index(drop=True)


def _gap_metric(sub: pd.DataFrame) -> str:
    return "gap_cea_pct" if len(sub) and sub["cea_best"].notna().all() else "gap_best_pct"


def paired_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Every neural operator against random and adaptive, per class and pooled over the
    classes of a study (main = small test classes, large, or one objective), on
    per-instance means over repeats of the gap (to CEA when every run has a reference,
    else to the best known value). Holm's correction within each (study, budget)."""
    rows = []
    for (kind, budget), sb in df[df.kind.isin(["iters", "time", "objective"])].groupby(["kind", "budget"]):
        studies = ([(budget, sb)] if kind == "objective" else
                   [("main", sb[sb.cell.isin(SMALL_CELLS)]), ("large", sb[~sb.cell.isin(SMALL_CELLS)])])
        for study, sub in studies:
            if sub.empty:
                continue
            metric = _gap_metric(sub)
            per_inst = sub.groupby(["cell", "instance", "operator"])[metric].mean().unstack("operator")
            cells = _order(sub.cell, CELL_ORDER)
            family = []  # per-instance means: pairs are matched by (cell, instance), never by row order
            for op in _order([o for o in per_inst.columns if o in NEURAL], OPERATORS):
                for base in [b for b in BASELINES if b in per_inst.columns]:
                    for cell in cells + ["pooled"]:
                        pi = per_inst if cell == "pooled" else per_inst.loc[[cell]]
                        r = paired_wilcoxon(pi[op].to_numpy(), pi[base].to_numpy())
                        if r["n"]:
                            family.append({"study": study, "kind": kind, "budget": budget, "cell": cell,
                                           "operator": op, "baseline": base, "metric": metric, **r})
            for row, p_adj in zip(family, holm([r["p"] for r in family])):
                row["p_holm"] = p_adj
            rows.extend(family)
    return pd.DataFrame(rows, columns=["study", "kind", "budget", "cell", "operator", "baseline", "metric", "n",
                                       "n_nonzero", "median_diff", "p", "p_holm"])


def _tex(text) -> str:
    return str(text).replace("_", r"\_").replace("%", r"\%")


def _pm(m, s) -> str:
    if pd.isna(m):
        return "--"
    return f"{m:+.2f} ± {s:.2f}" if pd.notna(s) else f"{m:+.2f}"


def _num(v, spec: str) -> str:
    return "--" if v is None or pd.isna(v) else format(v, spec)


def write_tex(path: str, rows: list[list]) -> None:
    """Rows of a LaTeX table body (cells joined by &, one row per line)."""
    write_atomic(path, "".join(" & ".join(_tex(c).replace("±", r"$\pm$") for c in r) + r" \\" + "\n" for r in rows))


def md_table(header: list[str], rows: list[list]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out + [""]


def gap_rows(s: pd.DataFrame, lead: list[str]) -> list[list]:
    return [[*(getattr(r, c) for c in lead), r.operator, r.runs, _pm(r.mean_gap_cea_pct, r.sd_inst_gap_cea_pct),
             _num(r.sd_rep_gap_cea_pct, ".2f"), _pm(r.mean_gap_best_pct, r.sd_inst_gap_best_pct),
             _num(r.mean_best, ".1f"), _num(r.iters_per_s, ".0f")]
            for r in s.itertuples()]


GAP_HEADER = ["operator", "runs", "gap to CEA % (± sd over instances)", "sd over repeats", "gap to best known %",
              "objective (mean)", "it/s"]


# ---------------------------------------------------------------- figures

def _style(ax, title, xlabel, ylabel):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)


def _line(op):
    return {"color": COLORS.get(op, "#888"), "lw": 2, "ls": "-" if op in NEURAL else "--", "label": op}


def fig_search_curves(df: pd.DataFrame, curves: dict, kind: str, path: str) -> bool:
    sub = df[(df.kind == kind) & df.cell.isin(SMALL_CELLS)]
    cells = _order(sub.cell, SMALL_CELLS)
    if not cells:
        return False
    fig, axes = plt.subplots(1, len(cells), figsize=(5 * len(cells), 3.8), squeeze=False)
    for ax, cell in zip(axes[0], cells):
        sc = sub[sub.cell == cell]
        metric = _gap_metric(sc)
        r0 = sc.iloc[0]
        _, grid = grid_axis(r0.budget_iters, r0.budget_secs)
        for op in _order(sc.operator, OPERATORS):
            ys = []
            for r in sc[sc.operator == op].itertuples():
                ref = r.cea_best if metric == "gap_cea_pct" else r.best_known
                ys.append(100 * (curves[(r.budget, r.operator, r.cell, r.instance, r.seed)] - ref) / ref)
            ax.plot(grid / 1000 if kind == "time" else grid, np.mean(ys, axis=0), **_line(op))
        ax.axhline(0, color="#52514e", lw=1, ls=":")
        if kind == "iters":
            ax.set_xscale("log")
        ax.set_ylim(top=min(ax.get_ylim()[1], 40))  # the start solution's gap would flatten the rest
        _style(ax, f"{cell} ({r0.budget})", "elapsed time (s)" if kind == "time" else "iteration",
               "gap to CEA reference (%)" if metric == "gap_cea_pct" else "gap to best known (%)")
    axes[0][0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def fig_by_x(s: pd.DataFrame, panels: list[tuple[str, pd.DataFrame]], x: str, xlabel: str, path: str,
             categorical: bool = False) -> bool:
    """One panel per (title, subset of summary rows): mean gap against x, one line per operator."""
    panels = [(t, p) for t, p in panels if not p.empty]
    if not panels:
        return False
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 3.8), squeeze=False)
    for ax, (title, p) in zip(axes[0], panels):
        metric = "mean_gap_cea_pct" if p["mean_gap_cea_pct"].notna().all() else "mean_gap_best_pct"
        xs = _order(p[x], CELL_ORDER) if categorical else sorted(p[x].unique())
        for op in _order(p.operator, OPERATORS):
            q = p[p.operator == op].set_index(x).reindex(xs)
            pos = np.arange(len(xs)) if categorical else xs
            ax.plot(pos, q[metric].to_numpy(dtype=float), marker="o", ms=6, **_line(op))
        if categorical:
            ax.set_xticks(np.arange(len(xs)), xs)
        ax.axhline(0, color="#52514e", lw=1, ls=":")
        _style(ax, title, xlabel, "gap to CEA reference (%)" if metric == "mean_gap_cea_pct" else "gap to best known (%)")
    axes[0][0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def fig_loss_curves(fdir: str, groups: list[str], panel_of, label_of, path: str) -> bool:
    """Train (dotted) and validation (solid) loss of the models in fdir/<group>/*, one
    panel per panel_of(group, config), one colour per label_of(config)."""
    items = []
    for group in groups:
        for mdir in sorted(glob.glob(os.path.join(fdir, group, "*"))):
            cp = os.path.join(mdir, "loss_curve.csv")
            if os.path.exists(cp) and os.path.exists(os.path.join(mdir, "config.json")):
                c = _read_json(os.path.join(mdir, "config.json"))
                items.append((panel_of(group, c), label_of(c), pd.read_csv(cp)))
    if not items:
        return False
    panels = _order([p for p, _, _ in items], [])
    labels = _order([lab for _, lab, _ in items], [])
    color = {lab: SLOTS[i % len(SLOTS)] for i, lab in enumerate(labels)}
    ncols = min(len(panels), 2)
    nrows = -(-len(panels) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 3.8 * nrows), squeeze=False)
    for ax, panel in zip(axes.flat, panels):
        for p, lab, lc in items:
            if p == panel:
                ax.plot(lc.epoch, lc.train_loss, color=color[lab], lw=1.5, ls=":")
                ax.plot(lc.epoch, lc.val_loss, color=color[lab], lw=2, label=lab)
        _style(ax, panel, "epoch", "loss (dotted = train, solid = validation)")
        ax.legend(frameon=False, fontsize=7)
    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def fig_v2(root: str, path: str, done: set[str]) -> bool:
    runs = [d for d in sorted(glob.glob(os.path.join(root, "models", "v2", "*")))
            if os.path.exists(os.path.join(d, "train_log.csv")) and f"v2-{os.path.basename(d)}" in done]
    if not runs:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for i, d in enumerate(runs):
        name, col = f"v2-{os.path.basename(d)}", SLOTS[i % len(SLOTS)]
        tl = pd.read_csv(os.path.join(d, "train_log.csv"))
        axes[0].plot(tl.episode, tl.mean_reward.rolling(10, min_periods=1).mean(), color=col, lw=2, label=name)
        axes[0].plot(tl.episode, tl.random_reward.rolling(10, min_periods=1).mean(), color=col, lw=1, ls=":")
        vp = os.path.join(d, "val_log.csv")
        if os.path.exists(vp):
            vl = pd.read_csv(vp)
            axes[1].plot(vl.episode, vl.val_reward, color=col, lw=2, marker="o", ms=4, label=name)
    _style(axes[0], "training reward (10-episode mean; dotted = random sets)", "episode", "reward (% improvement)")
    _style(axes[1], "validation reward", "episode", "reward (% improvement)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def fig_trainsize(root: str, path: str) -> bool:
    bp = os.path.join(root, "models", "b0", "best.json")
    if not os.path.exists(bp):
        return False
    points = []
    for n, rel in _read_json(bp).get("trainsize", {}).items():
        mp = os.path.join(root, "models", "b0", rel, "metrics.json")
        if os.path.exists(mp):
            m = _read_json(mp)
            points.append((int(n), m["precision_at_k"], m.get("p_at_k_best_try", np.nan), m["spearman"]))
    if not points:
        return False
    points.sort()
    n, pk, pb, sp = (np.array(v) for v in zip(*points))
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    for ax, y, lab in ((axes[0], pk, "p@k (own target)"), (axes[1], pb, "p@k best try"), (axes[2], sp, "Spearman")):
        ax.plot(n, y, color=SLOTS[0], lw=2, marker="o", ms=6)
        ax.set_xscale("log")
        ax.minorticks_off()
        ax.set_xticks(n, [str(v) for v in n])
        _style(ax, f"validation {lab}", "training instances", lab)
    fig.suptitle("B0, best configuration, by training-set size")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


# ---------------------------------------------------------------- main

def aggregate(root: str) -> None:
    figdir, tabdir = os.path.join(root, "figures"), os.path.join(root, "tables")
    for d in (figdir, tabdir):  # nothing from an earlier aggregation may survive
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
    for name in ("runs.csv", "summary.csv", "stats.csv", "training_summary.csv"):
        if os.path.exists(os.path.join(root, name)):
            os.remove(os.path.join(root, name))
    md = ["# Stage 3 summary", "", "## Coverage", ""]
    cov = batch_coverage(root)
    md += md_table(["batch", "status", "host", "finished"], cov.values.tolist())
    md += ["Only batches marked done enter the tables, figures and CSV files below.", ""]
    done = set(cov.loc[cov.status == "done", "batch"])
    notes = []

    refs = load_references3(root)
    refs = refs[pd.Series([reference_batch(c, o) in done for c, o in zip(refs.cell, refs.objective)],
                          index=refs.index, dtype=bool)]
    runs, curves = load_runs3(root, refs)
    runs = runs[pd.Series([run_batch(k, c, o) in done for k, c, o in zip(runs.kind, runs.cell, runs.objective)],
                          index=runs.index, dtype=bool)]
    sel_path = os.path.join(root, "selection", "selection.json")
    if os.path.exists(sel_path):
        check_selection(sel_path)
        check_neural_taus(runs, _read_json(sel_path))
    elif runs.operator.isin(NEURAL).any():
        notes.append("No selection/selection.json: the tau of the neural runs was not checked.")
    df = add_gaps(runs, refs)
    val_runs = runs[runs.split == "val"]
    pd.concat([df, val_runs], ignore_index=True).to_csv(os.path.join(root, "runs.csv"), index=False)

    # wall-time classes: one CPU setup per class, or no wall-time table for it
    refused = set()
    for cell in _order(df.loc[df.kind == "time", "cell"], CELL_ORDER):
        try:
            check_timed_runs(df[df.cell == cell], refs[(refs.cell == cell) & (refs.objective == "distance")])
        except ValueError as e:
            refused.add(cell)
            msg = f"Refused the wall-time table of class {cell}: {e}"
            print(msg, file=sys.stderr)
            notes.append(msg)
    for (cell, objective), _ in refs[refs.objective != "distance"].groupby(["cell", "objective"]):
        try:
            check_timed_runs(df.iloc[0:0], refs[(refs.cell == cell) & (refs.objective == objective)])
        except ValueError as e:
            notes.append(f"Warning: {e}")
    df = df[~((df.kind == "time") & df.cell.isin(refused))]
    probe = os.path.join(root, "batches", "large-100_500", "probe.json")
    if os.path.exists(probe) and "large-100_500" in done:
        p = _read_json(probe)
        notes.append(f"Class 100_500: a probe of the {p['operator']} operator ran at {p['iters_per_s']:.1f} it/s, so one "
                     f"{p['full_iters']}-iteration run takes about {p['projected_s']:.0f} s (limit {p['limit_s']} s); "
                     f"the iteration budget for this class is {p['iterations']}"
                     + (" (cut)." if p["iterations"] < p["full_iters"] else "."))
    if notes:
        md += ["## Notes", ""] + [f"- {n}" for n in notes] + [""]
    md += ["Gap to CEA % is relative to the CEA reference of the instance (best of its runs, same objective); gap to "
           "best known % is relative to the best value known for the instance (CEA and all LNS runs). Means over "
           "instances and repeats; sd over instances is of per-instance means, sd over repeats the mean "
           "per-instance sd.", ""]

    if not refs.empty:
        rr = refs.groupby(["objective", "cell"]).agg(instances=("instance", "nunique"), runs=("cea_runs", "max"),
                                                     mean_best=("cea_best", "mean"), runtime=("cea_mean_runtime_s", "mean"),
                                                     host=("host", lambda h: ",".join(sorted(set(h))))).reset_index()
        rr["_cell"] = rr["cell"].map({c: i for i, c in enumerate(_order(rr["cell"], CELL_ORDER))})
        rows = [[r.objective, r.cell, r.instances, r.runs, f"{r.mean_best:.1f}", f"{r.runtime:.1f}", r.host]
                for r in rr.sort_values(["objective", "_cell"]).itertuples()]
        md += ["## CEA references", ""] + md_table(["objective", "cell", "instances", "runs", "mean best",
                                                     "mean runtime (s)", "host"], rows)
        write_tex(os.path.join(tabdir, "references.tex"), rows)

    s = summarise(df) if not df.empty else pd.DataFrame()
    if not s.empty:
        s.to_csv(os.path.join(root, "summary.csv"), index=False)
        sections = [("main_iters", "Main comparison, iteration budget", (s.kind == "iters") & s.cell.isin(SMALL_CELLS), ["cell"]),
                    ("main_time", "Main comparison, wall time", (s.kind == "time") & s.cell.isin(SMALL_CELLS), ["cell"]),
                    ("ksweep", "Destroy size sweep", s.kind == "ksweep", ["cell", "k_frac"]),
                    ("large", "Large classes", s.kind.isin(["iters", "time"]) & ~s.cell.isin(SMALL_CELLS), ["cell", "budget"]),
                    ("objectives", "Price and empty-distance objectives", s.kind == "objective", ["objective", "cell"])]
        for name, title, mask, lead in sections:
            sub = s[mask]
            if sub.empty:
                continue
            rows = gap_rows(sub, lead)
            md += [f"## {title}", ""] + md_table(lead + GAP_HEADER, rows)
            write_tex(os.path.join(tabdir, f"{name}.tex"), rows)

        st = paired_tests(df)
        st.to_csv(os.path.join(root, "stats.csv"), index=False)
        if not st.empty:
            rows = [[r.study, r.budget, r.cell, r.operator, r.baseline, r.n, r.n_nonzero, f"{r.median_diff:+.3f}",
                     f"{r.p:.2g}", f"{r.p_holm:.2g}"] for r in st.itertuples()]
            md += ["## Paired tests", "",
                   "Wilcoxon signed-rank test over instances on per-instance mean gaps (column metric in stats.csv); "
                   "median difference = operator minus baseline in percentage points; Holm's correction within each "
                   "study and budget.", ""]
            md += md_table(["study", "budget", "cell", "operator", "baseline", "n", "n non-zero", "median diff", "p",
                            "p (Holm)"], rows)
            write_tex(os.path.join(tabdir, "stats.tex"), rows)

        fig_search_curves(df, curves, "iters", os.path.join(figdir, "search_curves_iters.png"))
        fig_search_curves(df, curves, "time", os.path.join(figdir, "search_curves_time.png"))
        ks = s[s.kind == "ksweep"]
        fig_by_x(ks, [(c, ks[ks.cell == c]) for c in _order(ks.cell, CELL_ORDER)], "k_frac",
                 "destroy size (fraction of orders)", os.path.join(figdir, "k_sweep.png"))
        lg = s[s.kind.isin(["iters", "time"]) & ~s.cell.isin(SMALL_CELLS)]
        fig_by_x(lg, [("iteration budget", lg[lg.kind == "iters"]), ("wall-time budget", lg[lg.kind == "time"])],
                 "cell", "class", os.path.join(figdir, "large_classes.png"), categorical=True)
        ob = s[s.kind == "objective"]
        fig_by_x(ob, [(o, ob[ob.objective == o]) for o in _order(ob.objective, ["price", "empty"])], "cell", "class",
                 os.path.join(figdir, "objectives.png"), categorical=True)

    if "select" in done and os.path.exists(sel_path):
        sel = _read_json(sel_path)
        rows = [[name, f"{m['tau']:g}", f"{m['mean_objective']:.2f}",
                 " ".join(f"{t}:{v:.1f}" for t, v in m["mean_objective_by_tau"].items())]
                for name, m in sel["models"].items()]
        md += ["## Selection (validation instances only)", "",
               f"Best V2 run: {sel['v2']['name'] if sel['v2'] else '--'}; best model: {sel['best_model']['name']} "
               f"(tau {sel['best_model']['tau']:g}); {len(sel['instance_files'])} validation instances, "
               f"{sel['iterations']} iterations, {sel['repeats']} repeats.", ""]
        md += md_table(["model", "tau", "mean objective", "mean objective by tau"], rows)
        write_tex(os.path.join(tabdir, "selection.tex"), rows)

    tr = training_summary(root, done)
    if not tr.empty:
        tr.to_csv(os.path.join(root, "training_summary.csv"), index=False)
        md += ["## Training", ""]
        for family in ("b0", "v1"):
            t = tr[tr.family == family]
            if t.empty:
                continue
            seeds = t.group.str.startswith("seeds ")
            rows = [[r.group, r.name, _num(r.params, ".0f"), _num(r.spearman, ".3f"), _num(r.p_at_k, ".3f"),
                     _num(r.p_at_k_best_try, ".3f"), _num(r.best_epoch, ".1f" if s else ".0f"),
                     _num(r.train_seconds, ".0f")]
                    for r, s in zip(t.itertuples(), seeds)]
            md += [f"### {family.upper()}", "",
                   "Validation metrics at the kept epoch: Spearman and p@k against the model's own training target "
                   "(mean or top), p@k best try = share of the predicted top k inside the state's best try (the "
                   "same for every target; models are ranked by it).", ""]
            md += md_table(["group", "model", "parameters", "Spearman", "p@k (own target)", "p@k best try",
                            "best epoch", "train s"], rows)
            write_tex(os.path.join(tabdir, f"training_{family}.tex"), rows)
        t = tr[tr.family == "v2"]
        if not t.empty:
            rows = [[r.name, "V1" if isinstance(r.init, str) and r.init else "scratch", r.samples, r.entropy_beta,
                     _num(r.params, ".0f"), _num(r.val_reward, ".4f"), r.best_epoch, r.epochs_run,
                     _num(r.train_seconds / 3600, ".2f")]
                    for r in t.itertuples()]
            md += ["### V2", ""] + md_table(["run", "start", "S", "beta", "parameters", "best val reward",
                                             "best episode", "episodes", "hours"], rows)
            write_tex(os.path.join(tabdir, "training_v2.tex"), rows)
    fig_loss_curves(os.path.join(root, "models", "b0"), ["grid"] if "b0" in done else [],
                    lambda g, c: f"{c['loss']} loss, target {c.get('target', 'mean')}",
                    lambda c: f"w{c['width']} d{c['depth']}", os.path.join(figdir, "loss_curves_b0.png"))
    fig_loss_curves(os.path.join(root, "models", "v1"), ["grid", "reg"] if "v1" in done else [],
                    lambda g, c: "model size" if g == "grid" else "regularisation of the best size",
                    lambda c: f"d{c['d']} n_att{c['n_att']} dropout {c['dropout']:g} wd {c['weight_decay']:g}",
                    os.path.join(figdir, "loss_curves_v1.png"))
    fig_v2(root, os.path.join(figdir, "v2_training.png"), done)
    if "b0" in done:
        fig_trainsize(root, os.path.join(figdir, "trainsize.png"))

    for sp in sorted(glob.glob(os.path.join(root, "label_stats", "stats_*.md"))) if "data" in done else []:
        with open(sp) as f:
            md += [f"## Label quality ({os.path.basename(sp)})", "", f.read(), ""]
    write_atomic(os.path.join(root, "summary.md"), "\n".join(md) + "\n")
    print(f"aggregate3: {len(df)} test runs, {len(val_runs)} selection runs, {len(refs)} references -> {root}")


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "pick-best":
        print(pick_best(argv[1:]))
        return
    if argv and argv[0] == "select":
        ap = argparse.ArgumentParser(prog="aggregate3 select")
        ap.add_argument("--root", required=True)
        ap.add_argument("--val-dir", required=True)
        ap.add_argument("--model", action="append", required=True, help="NAME=WEIGHTS (weights path relative to --root)")
        args = ap.parse_args(argv[1:])
        sel = write_selection(args.root, args.val_dir, dict(m.split("=", 1) for m in args.model))
        print(json.dumps({k: sel[k] for k in ("v2", "best_model")}, default=str)[:400])
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/stage3")
    aggregate(ap.parse_args(argv).root)


if __name__ == "__main__":
    main()
