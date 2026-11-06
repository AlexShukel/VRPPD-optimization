"""Aggregate Stage 2 results into summary.csv, markdown tables and PNG figures.

Reads:
  <root>/lns/<budget>/<operator>/<cell>/*.json   (LNS run files, curve inside)
  <root>/reference/cea_ref_<cell>.json           (CEA reference values)
  <root>/models/grid/*/metrics.json, loss_curve.csv ; <root>/models/best/*/...
  <root>/labels/stats_*.md
Writes:
  <root>/summary.csv, <root>/summary.md, <root>/figures/*.png
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OPERATOR_ORDER = ["random", "worst", "related", "adaptive", "neural", "neural_tau"]
COLORS = {"random": "#2a78d6", "worst": "#eb6834", "related": "#1baf7a", "adaptive": "#eda100",
          "neural": "#e87ba4", "neural_tau": "#6250d6", "cea": "#52514e"}
CELL_ORDER = ["10_10", "10_20", "20_50", "30_100", "50_200", "100_500"]


def load_runs(root: str) -> pd.DataFrame:
    rows = []
    curves = {}
    for path in glob.glob(os.path.join(root, "lns", "*", "*", "*", "*.json")):
        parts = path.split(os.sep)
        budget, operator, cell = parts[-4], parts[-3], parts[-2]
        with open(path) as f:
            d = json.load(f)
        key = (budget, operator, cell, d["instance"], d["seed"])
        curves[key] = d["curve"]
        rows.append({
            "budget": budget, "operator": operator, "cell": cell, "instance": d["instance"], "seed": d["seed"],
            "initial": d["initial"], "best": d["best"], "iterations": d["iterations"], "elapsed_ms": d["elapsed_ms"],
            "accepted": d["accepted"], "improved_current": d["improved_current"], "new_best_count": d["new_best_count"],
            "best_iteration": d["best_iteration"], "best_elapsed_ms": d["best_elapsed_ms"], "k": d["k"],
            "destroy_ms": d["destroy_ms"], "repair_ms": d["repair_ms"], "valid": d["valid"],
        })
    df = pd.DataFrame(rows)
    return df, curves


def load_reference(root: str) -> pd.DataFrame:
    rows = []
    for path in glob.glob(os.path.join(root, "reference", "cea_ref_*.json")):
        with open(path) as f:
            for r in json.load(f):
                rows.append({"cell": r["cell"], "instance": r["instance"], "cea_best": r["best"],
                             "cea_mean": float(np.mean(r["runs"])), "cea_runs": len(r["runs"]),
                             "cea_mean_runtime_s": float(np.mean(r["runtimes_ms"])) / 1000.0})
    return pd.DataFrame(rows)


def time_to_target(curve: list[dict], target: float) -> tuple[float, float]:
    for row in curve:
        if row["best"] <= target + 1e-9:
            return row["iteration"], row["elapsed_ms"] / 1000.0
    return np.nan, np.nan


def fmt(m, s):
    return f"{m:.1f} ± {s:.1f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/stage2")
    args = ap.parse_args()
    root = args.root
    figdir = os.path.join(root, "figures")
    os.makedirs(figdir, exist_ok=True)

    df, curves = load_runs(root)
    if df.empty:
        print("no LNS runs found")
        return
    ref = load_reference(root)
    df = df.merge(ref, on=["cell", "instance"], how="left")
    # best known per (cell, instance) over everything (CEA + all LNS runs)
    best_known = df.groupby(["cell", "instance"])["best"].min().rename("lns_best_known").reset_index()
    df = df.merge(best_known, on=["cell", "instance"], how="left")
    df["best_known"] = np.fmin(df["lns_best_known"], df["cea_best"].fillna(np.inf))
    df["gap_cea_pct"] = 100 * (df["best"] - df["cea_best"]) / df["cea_best"]
    df["gap_best_pct"] = 100 * (df["best"] - df["best_known"]) / df["best_known"]
    df["iters_per_s"] = 1000 * df["iterations"] / df["elapsed_ms"].clip(lower=1e-6)
    tt = [time_to_target(curves[(r.budget, r.operator, r.cell, r.instance, r.seed)], 1.01 * r.cea_best)
          if pd.notna(r.cea_best) else (np.nan, np.nan) for r in df.itertuples()]
    df["iters_to_1pct"] = [t[0] for t in tt]
    df["secs_to_1pct"] = [t[1] for t in tt]
    df["reached_1pct"] = df["iters_to_1pct"].notna()
    df.to_csv(os.path.join(root, "runs.csv"), index=False)

    g = df.groupby(["budget", "operator", "cell"])
    summary = g.agg(
        runs=("best", "size"), instances=("instance", "nunique"),
        mean_best=("best", "mean"), sd_best=("best", "std"),
        mean_gap_cea_pct=("gap_cea_pct", "mean"), sd_gap_cea_pct=("gap_cea_pct", "std"),
        mean_gap_best_pct=("gap_best_pct", "mean"), sd_gap_best_pct=("gap_best_pct", "std"),
        mean_iters=("iterations", "mean"), mean_elapsed_s=("elapsed_ms", lambda x: x.mean() / 1000),
        iters_per_s=("iters_per_s", "mean"),
        reached_1pct=("reached_1pct", "mean"),
        mean_iters_to_1pct=("iters_to_1pct", "mean"), mean_secs_to_1pct=("secs_to_1pct", "mean"),
        mean_accept_rate=("accepted", "mean"), destroy_share=("destroy_ms", "mean"), repair_share=("repair_ms", "mean"),
        valid=("valid", "all"),
    ).reset_index()
    summary["mean_accept_rate"] = summary["mean_accept_rate"] / summary["mean_iters"]
    tot = (summary["destroy_share"] + summary["repair_share"]).clip(lower=1e-9)
    summary["destroy_share"] = summary["destroy_share"] / tot
    summary["repair_share"] = summary["repair_share"] / tot
    summary["operator"] = pd.Categorical(summary["operator"], [o for o in OPERATOR_ORDER if o in set(summary["operator"])] + sorted(set(summary["operator"]) - set(OPERATOR_ORDER)))
    summary["cell"] = pd.Categorical(summary["cell"], [c for c in CELL_ORDER if c in set(summary["cell"])] + sorted(set(summary["cell"]) - set(CELL_ORDER)))
    summary = summary.sort_values(["budget", "cell", "operator"])
    summary.to_csv(os.path.join(root, "summary.csv"), index=False)

    # Wilcoxon-free simple win counts: per (budget, cell, instance) mean over seeds, which operator is best
    md = ["# Stage 2 summary", ""]
    md.append("Gap % is relative to the CEA reference (best of the CEA runs, 60 s budget); gap-best is relative to the best value known for the instance over CEA and all LNS runs. Mean ± sd over instances × repeats.")
    md.append("")
    if not ref.empty:
        md.append("## CEA reference (distance objective)")
        md.append("")
        md.append("| cell | instances | runs per instance | mean best | mean CEA runtime (s) |")
        md.append("|---|---|---|---|---|")
        for cell, sub in ref.groupby("cell"):
            md.append(f"| {cell} | {len(sub)} | {int(sub.cea_runs.iloc[0])} | {sub.cea_best.mean():.1f} | {sub.cea_mean_runtime_s.mean():.1f} |")
        md.append("")
    for budget, sb in summary.groupby("budget", sort=False):
        md.append(f"## Budget: {budget}")
        md.append("")
        md.append("| cell | operator | runs | objective (mean ± sd) | gap vs CEA % (mean ± sd) | gap vs best-known % | iterations (mean) | it/s | time to 1 % of CEA: reached / iters / s | accept rate | destroy share of time |")
        md.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for r in sb.itertuples():
            t1 = f"{100 * r.reached_1pct:.0f}% / {r.mean_iters_to_1pct:.0f} / {r.mean_secs_to_1pct:.2f}" if r.reached_1pct > 0 else "0% / – / –"
            md.append(f"| {r.cell} | {r.operator} | {r.runs} | {fmt(r.mean_best, r.sd_best)} | {r.mean_gap_cea_pct:+.2f} ± {r.sd_gap_cea_pct:.2f} | {r.mean_gap_best_pct:+.2f} ± {r.sd_gap_best_pct:.2f} | {r.mean_iters:.0f} | {r.iters_per_s:.0f} | {t1} | {r.mean_accept_rate:.2f} | {r.destroy_share:.2f} |")
        md.append("")
        # per-instance wins
        per_inst = df[df.budget == budget].groupby(["cell", "instance", "operator"])["best"].mean().reset_index()
        wins = []
        for (cell, inst), sub in per_inst.groupby(["cell", "instance"]):
            m = sub["best"].min()
            for op in sub[sub["best"] <= m + 1e-6]["operator"]:
                wins.append({"cell": cell, "operator": op})
        if wins:
            w = pd.DataFrame(wins).groupby(["cell", "operator"]).size().unstack(fill_value=0)
            md.append("Number of instances (out of 20) on which the operator's mean over repeats is the best (ties count for all):")
            md.append("")
            md.append("| cell | " + " | ".join(str(c) for c in w.columns) + " |")
            md.append("|---|" + "---|" * len(w.columns))
            for cell, row in w.iterrows():
                md.append(f"| {cell} | " + " | ".join(str(int(v)) for v in row.values) + " |")
            md.append("")

    # ---------- figures: search curves ----------
    for budget in sorted(df.budget.unique()):
        cells = [c for c in CELL_ORDER if c in set(df[df.budget == budget].cell)]
        if not cells:
            continue
        for xkey, xlabel, fname in (("iteration", "iteration", "iters"), ("elapsed_ms", "elapsed time (s)", "time")):
            fig, axes = plt.subplots(1, len(cells), figsize=(5 * len(cells), 3.8), squeeze=False)
            for ax, cell in zip(axes[0], cells):
                sub = df[(df.budget == budget) & (df.cell == cell)]
                ops = [o for o in OPERATOR_ORDER if o in set(sub.operator)]
                xmax = sub["iterations"].max() if xkey == "iteration" else sub["elapsed_ms"].max()
                grid = np.geomspace(1, max(xmax, 2), 200) if xkey == "iteration" else np.linspace(0, xmax, 200)
                for op in ops:
                    ys = []
                    for r in sub[sub.operator == op].itertuples():
                        c = curves[(r.budget, r.operator, r.cell, r.instance, r.seed)]
                        x = np.array([row[xkey] for row in c], dtype=float)
                        y = np.array([row["best"] for row in c], dtype=float)
                        ref_v = r.cea_best if pd.notna(r.cea_best) else r.best_known
                        idx = np.searchsorted(x, grid, side="right") - 1
                        idx = np.clip(idx, 0, len(x) - 1)
                        ys.append(100 * (y[idx] - ref_v) / ref_v)
                    ys = np.array(ys)
                    xs = grid / 1000.0 if xkey == "elapsed_ms" else grid
                    ax.plot(xs, ys.mean(axis=0), color=COLORS.get(op, "#888"), lw=2, label=op)
                ax.axhline(0, color="#52514e", lw=1, ls="--")
                if xkey == "iteration":
                    ax.set_xscale("log")
                ax.set_title(f"{cell} ({budget})")
                ax.set_xlabel(xlabel)
                ax.set_ylabel("gap to CEA reference (%)")
                ax.grid(alpha=0.25)
                ax.set_ylim(min(-5, ax.get_ylim()[0]), 40)
            axes[0][0].legend(frameon=False)
            fig.tight_layout()
            fig.savefig(os.path.join(figdir, f"search_curves_{budget}_{fname}.png"), dpi=130)
            plt.close(fig)

    # ---------- training results ----------
    grid_rows = []
    for mdir in sorted(glob.glob(os.path.join(root, "models", "*", "*"))):
        mp = os.path.join(mdir, "metrics.json")
        if not os.path.exists(mp):
            continue
        with open(mp) as f:
            m = json.load(f)
        with open(os.path.join(mdir, "config.json")) as f:
            c = json.load(f)
        grid_rows.append({"group": os.path.basename(os.path.dirname(mdir)), "name": os.path.basename(mdir),
                          "width": c["width"], "depth": c["depth"], "loss": c["loss"], "seed": c["seed"],
                          "best_epoch": m["best_epoch"], "epochs_run": m["epochs_run"], "val_loss": m["best_val_loss"],
                          "val_mse_z": m["mse"], "val_spearman": m["spearman"], "val_p_at_k": m["precision_at_k"],
                          "p_at_k_random": m["precision_at_k_random"], "best_set_recall": m["best_set_recall"],
                          "train_seconds": m["train_seconds"], "n_params": m["n_params"],
                          "n_train_states": m["n_train_states"], "n_val_states": m["n_val_states"]})
    if grid_rows:
        gdf = pd.DataFrame(grid_rows)
        gdf.to_csv(os.path.join(root, "training_summary.csv"), index=False)
        md.append("## B0 training")
        md.append("")
        md.append("Validation metrics at the best (early-stopped) epoch. MSE is on per-state z-scores (predictions also z-scored per state). Spearman = mean per-state rank correlation; p@k = overlap of predicted and true top-k (random expectation in the last column).")
        md.append("")
        md.append("| group | width | depth | loss | seed | epochs (best/run) | val loss | val MSE(z) | Spearman | p@k | p@k random | best-set recall | params | train s |")
        md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in gdf.sort_values(["group", "loss", "width", "depth", "seed"]).itertuples():
            md.append(f"| {r.group} | {r.width} | {r.depth} | {r.loss} | {r.seed} | {r.best_epoch}/{r.epochs_run} | {r.val_loss:.4f} | {r.val_mse_z:.3f} | {r.val_spearman:.3f} | {r.val_p_at_k:.3f} | {r.p_at_k_random:.3f} | {r.best_set_recall:.3f} | {r.n_params} | {r.train_seconds:.0f} |")
        md.append("")
        # loss curves figure (grid)
        gdirs = sorted(glob.glob(os.path.join(root, "models", "grid", "*")))
        if gdirs:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            for i, loss in enumerate(["mse", "rank"]):
                ax = axes[i]
                for mdir in gdirs:
                    cp = os.path.join(mdir, "loss_curve.csv")
                    if not os.path.exists(cp):
                        continue
                    with open(os.path.join(mdir, "config.json")) as f:
                        c = json.load(f)
                    if c["loss"] != loss:
                        continue
                    lc = pd.read_csv(cp)
                    lab = f"w{c['width']} d{c['depth']}"
                    col = {(64, 2): "#2a78d6", (64, 3): "#eb6834", (128, 2): "#1baf7a", (128, 3): "#eda100"}.get((c["width"], c["depth"]), "#888")
                    ax.plot(lc.epoch, lc.train_loss, color=col, lw=1.5, ls=":", label=f"{lab} train")
                    ax.plot(lc.epoch, lc.val_loss, color=col, lw=2, label=f"{lab} val")
                ax.set_title(f"{loss} loss (dotted = train, solid = validation)")
                ax.set_xlabel("epoch")
                ax.set_ylabel("loss")
                ax.grid(alpha=0.25)
                ax.legend(frameon=False, fontsize=7, ncol=2)
            fig.tight_layout()
            fig.savefig(os.path.join(figdir, "loss_curves_grid.png"), dpi=130)
            plt.close(fig)
        # predicted vs true + spearman histogram for best model (first seed)
        best_dirs = sorted(glob.glob(os.path.join(root, "models", "best", "*")))
        if best_dirs:
            vp = os.path.join(best_dirs[0], "val_predictions.jsonl")
            if os.path.exists(vp):
                zs, ps, rhos = [], [], []
                from scipy.stats import spearmanr
                with open(vp) as f:
                    for line in f:
                        d = json.loads(line)
                        m = np.array(d["mask"], dtype=bool)
                        z = np.array(d["z"])[m]
                        p = np.array(d["pred"])[m]
                        if len(z) >= 3:
                            pz = (p - p.mean()) / (p.std() if p.std() > 1e-12 else 1)
                            zs.append(z)
                            ps.append(pz)
                            rhos.append(spearmanr(p, z).correlation)
                zs = np.concatenate(zs)
                ps = np.concatenate(ps)
                fig, axes = plt.subplots(1, 2, figsize=(11, 4))
                axes[0].scatter(zs, ps, s=4, alpha=0.15, color="#2a78d6")
                axes[0].set_xlabel("true score (per-state z)")
                axes[0].set_ylabel("predicted score (per-state z)")
                axes[0].set_title(f"validation: {len(rhos)} states, {len(zs)} orders")
                axes[0].grid(alpha=0.25)
                axes[1].hist(np.nan_to_num(rhos), bins=40, color="#eb6834")
                axes[1].axvline(np.nanmean(rhos), color="#0b0b0b", lw=1.5, ls="--")
                axes[1].set_xlabel("Spearman per validation state")
                axes[1].set_ylabel("states")
                axes[1].set_title(f"mean Spearman = {np.nanmean(rhos):.3f}")
                axes[1].grid(alpha=0.25)
                fig.tight_layout()
                fig.savefig(os.path.join(figdir, "pred_vs_true_best_model.png"), dpi=130)
                plt.close(fig)

    # label stats
    for sp in sorted(glob.glob(os.path.join(root, "labels", "stats_*.md"))):
        md.append(f"## Label quality ({os.path.basename(sp)})")
        md.append("")
        with open(sp) as f:
            md.append(f.read())
        md.append("")

    with open(os.path.join(root, "summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
