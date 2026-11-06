"""Validation metrics for per-order scores: MSE on z-scores (predictions z-scored
per state), mean per-state Spearman, precision@k (overlap of predicted and true
top-k), and recall of the best try's set."""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from vrppd_nn.data import State, zscore


def evaluate_predictions(states: list[State], preds: list[np.ndarray]) -> dict:
    mses, rhos, pks, best_hits, rand_pks = [], [], [], [], []
    for s, p in zip(states, preds):
        m = s.mask
        if m.sum() < 3:
            continue
        pz, _ = zscore(np.where(m, p, np.nan))
        mses.append(float(np.mean((pz[m] - s.z[m]) ** 2)))
        rho = spearmanr(p[m], s.z[m]).correlation
        rhos.append(0.0 if np.isnan(rho) else float(rho))
        k = min(s.k, int(m.sum()))
        idx = np.where(m)[0]
        top_pred = set(idx[np.argsort(-p[m])[:k]])
        top_true = set(idx[np.argsort(-s.z[m])[:k]])
        pks.append(len(top_pred & top_true) / k)
        rand_pks.append(k / int(m.sum()))
        best = set(np.where(s.in_best)[0])
        if best:
            best_hits.append(len(top_pred & best) / len(best))
    return {
        "mse": float(np.mean(mses)) if mses else float("nan"),
        "spearman": float(np.mean(rhos)) if rhos else float("nan"),
        "precision_at_k": float(np.mean(pks)) if pks else float("nan"),
        "precision_at_k_random": float(np.mean(rand_pks)) if rand_pks else float("nan"),
        "best_set_recall": float(np.mean(best_hits)) if best_hits else float("nan"),
        "n_states": len(rhos),
        "spearman_per_state": rhos,
    }
