"""Label-quality statistics for a labels JSONL:
 - spread of try improvements per state (std, IQR, best vs median)
 - how often the best try beats the median try by > X % of the objective
 - correlation (Pearson, Spearman) of each of the 8 features with the z-scored scores
 - Spearman / precision@k of the single feature `route_saving` (= the "worst" rule)

Usage: python -m vrppd_nn.label_stats --labels train.jsonl --out stats.json [--md stats.md]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.stats import pearsonr, spearmanr

from vrppd_nn.data import FEATURE_NAMES, load_states
from vrppd_nn.metrics import evaluate_predictions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--md", default=None)
    ap.add_argument("--thresholds", default="0.5,1,2,5", help="percent of objective")
    args = ap.parse_args()
    states = load_states(args.labels)
    thresholds = [float(t) for t in args.thresholds.split(",")]

    rel_std, best_minus_median_pct, frac_positive, best_pct, median_pct = [], [], [], [], []
    for s in states:
        t = s.try_improvements
        obj = max(s.objective, 1e-9)
        rel_std.append(100 * t.std() / obj)
        best_minus_median_pct.append(100 * (t.max() - np.median(t)) / obj)
        best_pct.append(100 * t.max() / obj)
        median_pct.append(100 * np.median(t) / obj)
        frac_positive.append(float((t > 1e-9).mean()))
    bmm = np.array(best_minus_median_pct)

    # feature correlations with z-scores (all masked rows pooled)
    X = np.concatenate([s.features[s.mask] for s in states])
    Z = np.concatenate([s.z[s.mask] for s in states])
    corr = {}
    for j, name in enumerate(FEATURE_NAMES):
        x = X[:, j]
        if x.std() < 1e-12:
            corr[name] = {"pearson": 0.0, "spearman": 0.0}
        else:
            corr[name] = {"pearson": float(pearsonr(x, Z)[0]), "spearman": float(spearmanr(x, Z).correlation)}
    # single-feature baselines (ranking by one feature, descending)
    baselines = {}
    for j, name in enumerate(FEATURE_NAMES):
        met = evaluate_predictions(states, [s.features[:, j] for s in states])
        baselines[name] = {"spearman": met["spearman"], "precision_at_k": met["precision_at_k"]}
    rand = evaluate_predictions(states, [np.random.default_rng(0).normal(size=s.features.shape[0]) for s in states])

    per_cell = {}
    for cell in sorted({s.cell for s in states}):
        sub = [i for i, s in enumerate(states) if s.cell == cell]
        per_cell[cell] = {
            "n_states": len(sub),
            "mean_rel_std_pct": float(np.mean([rel_std[i] for i in sub])),
            "mean_best_minus_median_pct": float(np.mean([bmm[i] for i in sub])),
            "mean_best_pct": float(np.mean([best_pct[i] for i in sub])),
            "mean_median_pct": float(np.mean([median_pct[i] for i in sub])),
            "mean_frac_positive_tries": float(np.mean([frac_positive[i] for i in sub])),
        }
    out = {
        "n_states": len(states),
        "n_rows": int(len(Z)),
        "M": int(states[0].try_improvements.shape[0]) if states else 0,
        "mean_rel_std_pct": float(np.mean(rel_std)),
        "median_rel_std_pct": float(np.median(rel_std)),
        "mean_best_minus_median_pct": float(bmm.mean()),
        "mean_best_try_pct": float(np.mean(best_pct)),
        "mean_median_try_pct": float(np.mean(median_pct)),
        "mean_frac_positive_tries": float(np.mean(frac_positive)),
        "frac_states_best_beats_median_by": {f"{t}%": float((bmm > t).mean()) for t in thresholds},
        "feature_correlation_with_z": corr,
        "single_feature_ranking": baselines,
        "random_ranking": {"spearman": rand["spearman"], "precision_at_k": rand["precision_at_k"]},
        "per_cell": per_cell,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    if args.md:
        lines = [f"Label statistics for `{args.labels}`: {out['n_states']} states, {out['n_rows']} order rows, M = {out['M']}.", ""]
        lines.append("| cell | states | try-improvement std (% of obj) | best − median try (% of obj) | best try (% of obj) | median try (% of obj) | share of tries with gain > 0 |")
        lines.append("|---|---|---|---|---|---|---|")
        for c, v in per_cell.items():
            lines.append(f"| {c} | {v['n_states']} | {v['mean_rel_std_pct']:.3f} | {v['mean_best_minus_median_pct']:.3f} | {v['mean_best_pct']:.3f} | {v['mean_median_pct']:.3f} | {v['mean_frac_positive_tries']:.3f} |")
        lines.append("")
        lines.append("Share of states where the best try beats the median try by more than X % of the objective: " +
                     ", ".join(f"{k}: {v:.3f}" for k, v in out["frac_states_best_beats_median_by"].items()))
        lines.append("")
        lines.append("| feature | Pearson r with z-score | Spearman ρ with z-score | Spearman (rank by this feature alone, per state) | precision@k (feature alone) |")
        lines.append("|---|---|---|---|---|")
        for name in FEATURE_NAMES:
            lines.append(f"| {name} | {corr[name]['pearson']:.3f} | {corr[name]['spearman']:.3f} | {baselines[name]['spearman']:.3f} | {baselines[name]['precision_at_k']:.3f} |")
        lines.append(f"| (random order) | – | – | {rand['spearman']:.3f} | {rand['precision_at_k']:.3f} |")
        with open(args.md, "w") as f:
            f.write("\n".join(lines) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k not in ("feature_correlation_with_z", "single_feature_ranking", "per_cell")}))


if __name__ == "__main__":
    main()
