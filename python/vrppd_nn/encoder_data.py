"""V1 inputs: one token per vehicle start, pickup and delivery, the links of the
current routes, and padded mini-batches of labelled states.

Layout for V vehicles and N orders, T = V + 2N tokens: the start of vehicle v is
token v, the pickup of order o is token V + o, its delivery is token V + N + o.
Token features (F_TOK = 7): type one-hot (start, pickup, delivery), latitude,
longitude, demand (1 / loadFactor, 0 for a start), price per km (the vehicle's for
a start, 0 otherwise). Columns 3..6 are standardised with training-set statistics
(in float64, then cast to float32); padding tokens are all zeros.

Instance JSON fields read: vehicles[].startLocation.{latitude,longitude},
vehicles[].priceKm, orders[].pickupLocation.{latitude,longitude},
orders[].deliveryLocation.{latitude,longitude}, orders[].loadFactor.
"""
from __future__ import annotations

import glob
import json
import os
from collections import Counter
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch

from vrppd_nn.data import State

F_TOK = 7
STD_COLS = slice(3, 7)  # latitude, longitude, demand, price per km
KIND_OFFSET = {"pickup": 0, "delivery": 1}  # token = V + offset * N + order


@dataclass
class TokenState:
    tok: np.ndarray  # float32 (T, 7), raw
    prev: np.ndarray  # int64 (T,), previous token on the route, -1 for a start token
    next: np.ndarray  # int64 (T,), next token on the route, -1 after the last stop
    route: np.ndarray  # int64 (T,), vehicle index of every token
    n_vehicles: int
    n_orders: int


def instance_tokens(instance: dict) -> np.ndarray:
    """Raw token features (T, 7) of an instance; they do not depend on the routes."""
    vehicles, orders = instance["vehicles"], instance["orders"]
    V, N = len(vehicles), len(orders)
    tok = np.zeros((V + 2 * N, F_TOK), dtype=np.float32)
    for v, veh in enumerate(vehicles):
        loc = veh["startLocation"]
        tok[v] = (1, 0, 0, loc["latitude"], loc["longitude"], 0.0, veh["priceKm"])
    for o, order in enumerate(orders):
        demand = 1.0 / order["loadFactor"]
        p, q = order["pickupLocation"], order["deliveryLocation"]
        tok[V + o] = (0, 1, 0, p["latitude"], p["longitude"], demand, 0.0)
        tok[V + N + o] = (0, 0, 1, q["latitude"], q["longitude"], demand, 0.0)
    return tok


def build_tokens(instance: dict, routes: list[list[tuple[int, str]]]) -> TokenState:
    """Tokens plus route links. `routes[v]` lists vehicle v's stops as (order index,
    "pickup" | "delivery"); every order must appear once as pickup and once as delivery."""
    V, N = len(instance["vehicles"]), len(instance["orders"])
    if len(routes) != V:
        raise ValueError(f"{len(routes)} routes for {V} vehicles")
    T = V + 2 * N
    prev = np.full(T, -1, dtype=np.int64)
    nxt = np.full(T, -1, dtype=np.int64)
    route = np.full(T, -1, dtype=np.int64)
    route[:V] = np.arange(V)
    for v, stops in enumerate(routes):
        last = v
        for o, kind in stops:
            if not 0 <= o < N or kind not in KIND_OFFSET:
                raise ValueError(f"bad stop ({o}, {kind!r}) on route {v}")
            t = V + KIND_OFFSET[kind] * N + o
            if route[t] >= 0:
                raise ValueError(f"order {o} {kind} appears twice")
            route[t], prev[t], nxt[last] = v, last, t
            last = t
    missing = np.flatnonzero(route < 0)
    if len(missing):
        t = int(missing[0]) - V
        raise ValueError(f"order {t % N} {'pickup' if t < N else 'delivery'} is on no route")
    return TokenState(instance_tokens(instance), prev, nxt, route, V, N)


def load_instance(instances_root: str, split: str, cell: str, instance: str) -> dict:
    """Instance file of a label line. The Stage 2 pipeline lays instances out as
    <root>/<split>/<cell>/<id>.json (thesis naming <id>_<timestamp>.json also works)
    and `label` names them "<split>_<cell>_<id>"."""
    prefix = f"{split}_{cell}_"
    inst_id = instance[len(prefix):] if instance.startswith(prefix) else instance
    cell_dir = os.path.join(instances_root, split, cell)
    path = os.path.join(cell_dir, f"{inst_id}.json")
    if not os.path.exists(path):
        matches = glob.glob(os.path.join(glob.escape(cell_dir), f"{glob.escape(inst_id)}_*.json"))
        if len(matches) != 1:
            raise FileNotFoundError(f"no unique instance file for {instance!r} in {cell_dir}")
        path = matches[0]
    with open(path) as f:
        return json.load(f)


def _instance_key(s: State) -> tuple[str, str, str]:
    return s.split, s.cell, s.instance


def state_tokens(s: State, instances_root: str, cache: dict) -> TokenState:
    key = _instance_key(s)
    if key not in cache:
        cache[key] = load_instance(instances_root, *key)
    inst = cache[key]
    if s.routes is None:
        raise ValueError(f"{s.instance}: label line has no routes/order_ids, which V1 needs")
    ids = [o["id"] for o in inst["orders"]]
    if s.order_ids.tolist() != ids:
        raise ValueError(f"{s.instance}: order_ids of the label line differ from the instance file's orders")
    return build_tokens(inst, s.routes)


def stats_of_tokens(arrays: list[np.ndarray], counts: list[int] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard deviation of columns 3..6 over all rows of the raw token
    arrays, array i counted counts[i] times; a zero standard deviation becomes 1."""
    x = np.concatenate([a[:, STD_COLS] for a in arrays]).astype(np.float64)
    w = np.repeat(np.ones(len(arrays)) if counts is None else np.asarray(counts, dtype=float),
                  [len(a) for a in arrays])
    mean = np.average(x, axis=0, weights=w)
    std = np.sqrt(np.average((x - mean) ** 2, axis=0, weights=w))
    return mean, np.where(std > 1e-12, std, 1.0)


def token_stats(states: list[State], instances_root: str) -> tuple[np.ndarray, np.ndarray]:
    """tok_mean / tok_std over all tokens of the given (training) states."""
    counts = Counter(_instance_key(s) for s in states)
    arrays = [instance_tokens(load_instance(instances_root, *key)) for key in counts]
    return stats_of_tokens(arrays, list(counts.values()))


class TokenBatch(NamedTuple):
    """Padded batch; the first eight fields are the V1Encoder.forward arguments."""
    tok: torch.Tensor  # float32 (B, T, 7), standardised, zero rows for padding
    pad: torch.Tensor  # bool (B, T), True for padding tokens
    prev: torch.Tensor  # int64 (B, T), -1 for none and for padding
    next: torch.Tensor  # int64 (B, T)
    route: torch.Tensor  # int64 (B, T), -1 for padding
    n_vehicles: torch.Tensor  # int64 (B,)
    n_orders: torch.Tensor  # int64 (B,)
    order_pad: torch.Tensor  # bool (B, N), True for padding orders
    y: torch.Tensor  # float32 (B, N), per-state z-scored targets
    mask: torch.Tensor  # bool (B, N), True where the target is known

    def inputs(self) -> tuple[torch.Tensor, ...]:
        return tuple(self[:8])

    def to(self, device: torch.device) -> "TokenBatch":
        return TokenBatch(*(t.to(device) for t in self))


def collate_tokens(items: list[TokenState], tok_mean: np.ndarray, tok_std: np.ndarray,
                   z: list[np.ndarray] | None = None, mask: list[np.ndarray] | None = None) -> TokenBatch:
    """Pad and standardise token states; state b keeps positions 0..V_b+2N_b-1."""
    B = len(items)
    T = max(len(s.tok) for s in items)
    N = max(s.n_orders for s in items)
    tok = np.zeros((B, T, F_TOK), dtype=np.float32)
    pad = np.ones((B, T), dtype=bool)
    links = np.full((3, B, T), -1, dtype=np.int64)  # prev, next, route
    order_pad = np.ones((B, N), dtype=bool)
    y = np.zeros((B, N), dtype=np.float32)
    m = np.zeros((B, N), dtype=bool)
    for i, s in enumerate(items):
        t, n = len(s.tok), s.n_orders
        x = s.tok.astype(np.float64)
        x[:, STD_COLS] = (x[:, STD_COLS] - tok_mean) / tok_std
        tok[i, :t] = x
        pad[i, :t] = False
        links[:, i, :t] = (s.prev, s.next, s.route)
        order_pad[i, :n] = False
        if z is not None:
            y[i, :n] = z[i]
            m[i, :n] = mask[i]
    nv = np.array([s.n_vehicles for s in items], dtype=np.int64)
    no = np.array([s.n_orders for s in items], dtype=np.int64)
    return TokenBatch(*(torch.from_numpy(a) for a in (tok, pad, links[0], links[1], links[2], nv, no, order_pad, y, m)))


class TokenBatcher:
    """Mini-batches of labelled states as V1 inputs plus the B0 targets (z, mask)."""

    def __init__(self, states: list[State], instances_root: str, tok_mean: np.ndarray, tok_std: np.ndarray,
                 batch_states: int, shuffle: bool, seed: int = 0):
        cache: dict = {}
        self.states = states
        self.tokens = [state_tokens(s, instances_root, cache) for s in states]
        self.tok_mean = np.asarray(tok_mean, dtype=np.float64)
        self.tok_std = np.asarray(tok_std, dtype=np.float64)
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
            sel = idx[b:b + self.batch_states]
            yield collate_tokens([self.tokens[i] for i in sel], self.tok_mean, self.tok_std,
                                 [self.states[i].z for i in sel], [self.states[i].mask for i in sel])
