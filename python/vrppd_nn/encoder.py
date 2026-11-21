"""V1: transformer-style encoder over stop tokens and the current routes, one score
per order. Every layer is an explicitly named Linear / LayerNorm (no fused
attention modules) because the Rust forward pass (crates/vrppd-lns/src/encoder.rs)
re-implements it from the exported parameters. Linear weights are (out, in), y = x W^T + b.

Layers, in order (d = embedding size, LayerNorm eps 1e-5, post-norm):
  embed      h = embed(tok)                                    Linear(7, d)
  att0..     h = ln1(h + o(MHA(q h, k h, v h)))                q, k, v, o: Linear(d, d)
             h = ln2(h + ff2(ReLU(ff1(h))))                    ff1: Linear(d, ff_mult d), ff2 back to d
             MHA: head j uses features j*d/heads..(j+1)*d/heads of q, k, v; scores
             q.k / sqrt(d/heads), softmax over non-padding keys; heads concatenated in order.
  nb         h = ln(h + ff(ReLU(w3([h ; w1 h_prev + w2 h_next]))))
             w1, w2: Linear(d, d) without bias (a missing neighbour gives a zero vector),
             w3: Linear(2d, d), ff: Linear(d, d)
  rs         h = ln(h + w4 m_route(t)), m_r = mean of h over the tokens of route r
             (start token included), w4: Linear(d, d)
  att{n_att}.. n_att more attention blocks
  head       score_o = l2(ReLU(l1([h_pickup(o) ; h_delivery(o)])))   l1: Linear(2d, d), l2: Linear(d, 1)
Dropout acts only after the ReLU inside the attention blocks' feed-forward part.
No positional encoding.
"""
from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import torch
from torch import nn

from vrppd_nn.encoder_data import F_TOK
from vrppd_nn.fileio import write_json_atomic

LN_EPS = 1e-5
SCHEMA_KEYS = ("model", "config", "tok_mean", "tok_std", "params")


def gather_tokens(h: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """h (B, T, d), idx (B, M) token positions or -1 -> (B, M, d), zero rows where idx is -1."""
    g = torch.gather(h, 1, idx.clamp(min=0).unsqueeze(-1).expand(-1, -1, h.shape[-1]))
    return torch.where((idx >= 0).unsqueeze(-1), g, torch.zeros_like(g))


class AttentionBlock(nn.Module):
    def __init__(self, d: int, heads: int, ff_mult: int, dropout: float):
        super().__init__()
        if d % heads:
            raise ValueError(f"d={d} is not divisible by heads={heads}")
        self.heads = heads
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)
        self.ln1 = nn.LayerNorm(d, eps=LN_EPS)
        self.ff1 = nn.Linear(d, ff_mult * d)
        self.ff2 = nn.Linear(ff_mult * d, d)
        self.ln2 = nn.LayerNorm(d, eps=LN_EPS)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        B, T, d = h.shape
        dh = d // self.heads

        def split(x: torch.Tensor) -> torch.Tensor:  # (B, T, d) -> (B, heads, T, dh)
            return x.view(B, T, self.heads, dh).transpose(1, 2)

        q, k, v = split(self.q(h)), split(self.k(h)), split(self.v(h))
        scores = q @ k.transpose(-1, -2) / math.sqrt(dh)  # (B, heads, query, key)
        scores = scores.masked_fill(pad[:, None, None, :], float("-inf"))
        att = (torch.softmax(scores, dim=-1) @ v).transpose(1, 2).reshape(B, T, d)
        h = self.ln1(h + self.o(att))
        return self.ln2(h + self.ff2(self.drop(torch.relu(self.ff1(h)))))


class NeighbourLayer(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.w1 = nn.Linear(d, d, bias=False)
        self.w2 = nn.Linear(d, d, bias=False)
        self.w3 = nn.Linear(2 * d, d)
        self.ff = nn.Linear(d, d)
        self.ln = nn.LayerNorm(d, eps=LN_EPS)

    def forward(self, h: torch.Tensor, prev: torch.Tensor, next: torch.Tensor) -> torch.Tensor:
        around = self.w1(gather_tokens(h, prev)) + self.w2(gather_tokens(h, next))
        return self.ln(h + self.ff(torch.relu(self.w3(torch.cat([h, around], dim=-1)))))


class RouteSummaryLayer(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.w4 = nn.Linear(d, d)
        self.ln = nn.LayerNorm(d, eps=LN_EPS)

    def forward(self, h: torch.Tensor, route: torch.Tensor, n_routes: int) -> torch.Tensor:
        member = (route.unsqueeze(-1) == torch.arange(n_routes, device=h.device)).to(h.dtype)  # (B, T, R)
        total = member.transpose(1, 2) @ h  # (B, R, d)
        mean = total / member.sum(dim=1).clamp(min=1).unsqueeze(-1)
        return self.ln(h + self.w4(member @ mean))  # padding tokens (route -1) get a zero mean


class ScoreHead(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.l1 = nn.Linear(2 * d, d)
        self.l2 = nn.Linear(d, 1)

    def forward(self, h: torch.Tensor, n_vehicles: torch.Tensor, n_orders: torch.Tensor,
                order_pad: torch.Tensor) -> torch.Tensor:
        o = torch.arange(order_pad.shape[1], device=h.device)
        pickup = (n_vehicles.unsqueeze(1) + o).masked_fill(order_pad, 0)
        delivery = (n_vehicles.unsqueeze(1) + n_orders.unsqueeze(1) + o).masked_fill(order_pad, 0)
        x = torch.cat([gather_tokens(h, pickup), gather_tokens(h, delivery)], dim=-1)
        return self.l2(torch.relu(self.l1(x))).squeeze(-1).masked_fill(order_pad, 0.0)


class V1Encoder(nn.Module):
    def __init__(self, d: int = 64, heads: int = 4, n_att: int = 2, ff_mult: int = 2, dropout: float = 0.1):
        super().__init__()
        self.d, self.heads, self.n_att, self.ff_mult = d, heads, n_att, ff_mult
        self.embed = nn.Linear(F_TOK, d)
        for i in range(2 * n_att):
            setattr(self, f"att{i}", AttentionBlock(d, heads, ff_mult, dropout))
        self.nb = NeighbourLayer(d)
        self.rs = RouteSummaryLayer(d)
        self.head = ScoreHead(d)

    def config(self) -> dict[str, int]:
        return {"d": self.d, "heads": self.heads, "n_att": self.n_att, "ff_mult": self.ff_mult}

    def forward(self, tok: torch.Tensor, pad: torch.Tensor, prev: torch.Tensor, next: torch.Tensor,
                route: torch.Tensor, n_vehicles: torch.Tensor, n_orders: torch.Tensor,
                order_pad: torch.Tensor) -> torch.Tensor:
        """Scores (B, N); 0 at padding orders. Inputs as yielded by encoder_data.TokenBatch."""
        h = self.embed(tok)
        for i in range(self.n_att):
            h = getattr(self, f"att{i}")(h, pad)
        h = self.nb(h, prev, next)
        h = self.rs(h, route, int(n_vehicles.max()))
        for i in range(self.n_att, 2 * self.n_att):
            h = getattr(self, f"att{i}")(h, pad)
        return self.head(h, n_vehicles, n_orders, order_pad)


def model_to_json(model: V1Encoder, tok_mean: np.ndarray, tok_std: np.ndarray,
                  extra: dict[str, Any] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "model": "v1",
        "config": model.config(),
        "tok_mean": [float(v) for v in tok_mean],
        "tok_std": [float(v) for v in tok_std],
        "params": {name: {"shape": list(t.shape), "data": t.detach().cpu().reshape(-1).tolist()}
                   for name, t in model.state_dict().items()},
    }
    if extra:
        clash = sorted(set(extra) & set(SCHEMA_KEYS))
        if clash:
            raise ValueError(f"extra keys {clash} would overwrite the weights schema")
        out.update(extra)
    return out


def export_json(model: V1Encoder, tok_mean: np.ndarray, tok_std: np.ndarray, path: str,
                extra: dict[str, Any] | None = None) -> None:
    write_json_atomic(path, model_to_json(model, tok_mean, tok_std, extra))


def load_json(path: str) -> tuple[V1Encoder, np.ndarray, np.ndarray]:
    """Model (in eval mode) and token statistics from an exported weights file."""
    with open(path) as f:
        w = json.load(f)
    if w.get("model") != "v1":
        raise ValueError(f"{path}: not a V1 weights file (model={w.get('model')!r})")
    cfg = w["config"]
    model = V1Encoder(d=cfg["d"], heads=cfg["heads"], n_att=cfg["n_att"], ff_mult=cfg["ff_mult"])
    params = {name: torch.tensor(p["data"], dtype=torch.float32).reshape(p["shape"]) for name, p in w["params"].items()}
    model.load_state_dict(params)  # strict: a missing, unexpected or misshapen parameter raises
    return model.eval(), np.asarray(w["tok_mean"], dtype=float), np.asarray(w["tok_std"], dtype=float)
