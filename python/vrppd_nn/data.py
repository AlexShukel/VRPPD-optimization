"""Load labelled search states (JSONL written by `label`) into padded tensors.

One state = one instance + one solution; rows = orders. Targets are the raw
scores z-scored within the state (score - mean) / std, so that only the order
of scores within a state matters. Orders never sampled (score null) are masked.
"""
from __future__ import annotations

import json
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


def zscore(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(scores)
    z = np.zeros_like(scores)
    if mask.sum() >= 2:
        s = scores[mask]
        std = s.std()
        z[mask] = (s - s.mean()) / (std if std > 1e-12 else 1.0)
    return z, mask


def load_states(path: str, limit: int | None = None) -> list[State]:
    out: list[State] = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            scores = np.array([np.nan if s is None else s for s in d["scores"]], dtype=float)
            z, mask = zscore(scores)
            out.append(State(
                instance=d["instance"], cell=d["cell"], iteration=int(d["iteration"]), k=int(d["k"]),
                features=np.asarray(d["features"], dtype=float), scores=scores, z=z, mask=mask,
                in_best=np.asarray(d["in_best"], dtype=bool),
                try_improvements=np.asarray(d["try_improvements"], dtype=float),
                objective=float(d["objective"]),
            ))
            if limit and len(out) >= limit:
                break
    return out


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
