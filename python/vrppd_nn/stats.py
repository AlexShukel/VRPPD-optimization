"""Paired tests for the Stage 3 comparisons: Wilcoxon signed-rank and Holm's correction."""
from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon


def paired_wilcoxon(a: np.ndarray, b: np.ndarray) -> dict:
    """Two-sided Wilcoxon signed-rank test of a - b over the pairs where both values
    exist. Zero differences are dropped (scipy's default); p = 1.0 when every
    difference is zero. Returns n (pairs used), median_diff (median of a - b) and p."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired samples differ in shape: {a.shape} vs {b.shape}")
    keep = ~(np.isnan(a) | np.isnan(b))
    d = a[keep] - b[keep]
    if d.size == 0:
        return {"n": 0, "median_diff": float("nan"), "p": float("nan")}
    p = 1.0 if np.all(d == 0) else float(wilcoxon(d).pvalue)
    return {"n": int(d.size), "median_diff": float(np.median(d)), "p": p}


def holm(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the input order, capped at 1."""
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    order = np.argsort(p, kind="stable")
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adjusted[i] = running
    return adjusted.tolist()
