"""Load labelled search states (JSONL written by `label`) into padded tensors.

One state = one instance + one solution; rows = orders. Targets are the raw
scores z-scored within the state (score - mean) / std, so that only the order
of scores within a state matters. Orders never sampled (score null) are masked.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass

import numpy as np
import torch

N_FEATURES = 8
FEATURE_NAMES = [
    "route_saving", "start_to_pickup", "pickup_to_delivery", "load_factor",
    "price_km", "orders_on_route", "nearest_other", "relative_position",
]


@dataclass
class State:
    instance: str
    cell: str
    iteration: int
    k: int
    features: np.ndarray  # (n, 8) raw
    scores: np.ndarray  # (n,) raw, nan where never sampled
    z: np.ndarray  # (n,) per-state z-score, 0 where masked
    mask: np.ndarray  # (n,) bool
    in_best: np.ndarray  # (n,) bool
    try_improvements: np.ndarray  # (M,)
    objective: float
    split: str = ""
    # per vehicle: (order index, "pickup" | "delivery"); the label file's order ids
    # mapped to indices through order_ids. None for lines without routes.
    routes: list[list[tuple[int, str]]] | None = None
    order_ids: np.ndarray | None = None  # (n,) order id of each row


def zscore(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-state z-scores of the finite entries; exactly 0 when their spread is below 1e-12."""
    mask = np.isfinite(scores)
    z = np.zeros_like(scores)
    if mask.sum() >= 2:
        s = scores[mask]
        std = s.std()
        if std > 1e-12:
            z[mask] = (s - s.mean()) / std
    return z, mask


def route_indices(routes: list[list[list]], order_ids: list[int]) -> list[list[tuple[int, str]]]:
    """Label routes hold (order id, kind); map the ids to row indices (position in order_ids)."""
    index = {oid: i for i, oid in enumerate(order_ids)}
    try:
        return [[(index[oid], sys.intern(kind)) for oid, kind in r] for r in routes]
    except KeyError as e:
        raise ValueError(f"route stop has order id {e.args[0]} that is not in order_ids") from None


TARGET_FIELDS = {"mean": "scores", "top": "top_share"}


def load_states(path: str, limit: int | None = None, target: str = "mean") -> list[State]:
    """`target="mean"` z-scores `scores` (mean improvement); `target="top"` z-scores
    `top_share` (absent from Stage 2 files). `State.scores` always holds the mean scores."""
    if target not in TARGET_FIELDS:
        raise ValueError(f"unknown target {target!r}; expected one of {sorted(TARGET_FIELDS)}")
    field = TARGET_FIELDS[target]
    out: list[State] = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            scores = np.array([np.nan if s is None else s for s in d["scores"]], dtype=float)
            if field == "scores":
                tgt = scores
            else:
                if field not in d:
                    raise ValueError(f"{path}: label line has no '{field}' field, required for target={target!r} "
                                     "(file predates this field; relabel it)")
                tgt = np.array([np.nan if s is None else s for s in d[field]], dtype=float)
            z, mask = zscore(tgt)
            if "in_best" not in d:
                raise ValueError(f"{path}: label line has no 'in_best' field (the orders of the state's best try)")
            routes = order_ids = None
            if "routes" in d and "order_ids" in d:
                routes = route_indices(d["routes"], d["order_ids"])
                order_ids = np.asarray(d["order_ids"], dtype=np.int64)
            out.append(State(
                instance=d["instance"], cell=d["cell"], iteration=int(d["iteration"]), k=int(d["k"]),
                features=np.asarray(d["features"], dtype=float), scores=scores, z=z, mask=mask,
                in_best=np.asarray(d["in_best"], dtype=bool),
                try_improvements=np.asarray(d["try_improvements"], dtype=float),
                objective=float(d["objective"]),
                split=d.get("split", ""), routes=routes, order_ids=order_ids,
            ))
            if limit and len(out) >= limit:
                break
    return out


def subsample_instances(states: list[State], n_instances: int, seed: int = 0) -> list[State]:
    """Keep every state of `n_instances` instances. The instance order is one seeded
    permutation of the sorted names, so smaller selections are prefixes (nested)."""
    names = sorted({s.instance for s in states})
    order = np.random.default_rng(seed).permutation(len(names))
    keep = {names[i] for i in order[:n_instances]}
    return [s for s in states if s.instance in keep]


def feature_stats(states: list[State]) -> tuple[np.ndarray, np.ndarray]:
    x = np.concatenate([s.features for s in states], axis=0)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std > 1e-12, std, 1.0)
    return mean, std


class StateBatcher:
    """Yield padded mini-batches of states: x (B, n_max, 8), y (B, n_max), mask (B, n_max), k (B,), n (B,)."""

    def __init__(self, states: list[State], mean: np.ndarray, std: np.ndarray, batch_states: int,
                 shuffle: bool, seed: int = 0):
        self.states = states
        self.mean = mean
        self.std = std
        self.batch_states = batch_states
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return (len(self.states) + self.batch_states - 1) // self.batch_states

    def __iter__(self):
        idx = np.arange(len(self.states))
        if self.shuffle:
            self.rng.shuffle(idx)
        for b in range(0, len(idx), self.batch_states):
            batch = [self.states[i] for i in idx[b:b + self.batch_states]]
            yield self.collate(batch)

    def collate(self, batch: list[State]):
        n_max = max(s.features.shape[0] for s in batch)
        B = len(batch)
        x = np.zeros((B, n_max, N_FEATURES), dtype=np.float32)
        y = np.zeros((B, n_max), dtype=np.float32)
        m = np.zeros((B, n_max), dtype=bool)
        k = np.zeros((B,), dtype=np.int64)
        for i, s in enumerate(batch):
            n = s.features.shape[0]
            x[i, :n] = (s.features - self.mean) / self.std
            y[i, :n] = s.z
            m[i, :n] = s.mask
            k[i] = s.k
        n = np.array([s.features.shape[0] for s in batch], dtype=np.int64)
        return (torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(m), torch.from_numpy(k), torch.from_numpy(n))
