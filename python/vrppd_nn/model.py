"""B0: per-order MLP on 8 hand-made features, plus JSON export/import compatible
with the Rust forward pass in crates/vrppd-lns/src/mlp.rs."""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import torch
from torch import nn


class B0(nn.Module):
    def __init__(self, n_features: int = 8, width: int = 64, depth: int = 2, dropout: float = 0.1):
        """`depth` hidden layers of `width` units, ReLU, dropout; linear output of size 1."""
        super().__init__()
        layers: list[nn.Module] = []
        d_in = n_features
        for _ in range(depth):
            layers.append(nn.Linear(d_in, width))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d_in = width
        layers.append(nn.Linear(d_in, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def linear_layers(self) -> list[nn.Linear]:
        return [m for m in self.net if isinstance(m, nn.Linear)]


def mlp_to_json(layers: list[tuple[np.ndarray, np.ndarray]], mean: np.ndarray, std: np.ndarray,
                extra: dict[str, Any] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "n_features": int(len(mean)),
        "feature_mean": [float(v) for v in mean],
        "feature_std": [float(v) for v in std],
        "activation": "relu",
        "layers": [{"weight": w.astype(float).tolist(), "bias": b.astype(float).tolist()} for w, b in layers],
    }
    if extra:
        out.update(extra)
    return out


def model_to_json(model: B0, mean: np.ndarray, std: np.ndarray, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    layers = [(l.weight.detach().cpu().numpy(), l.bias.detach().cpu().numpy()) for l in model.linear_layers()]
    return mlp_to_json(layers, mean, std, extra)


def save_json(model: B0, mean: np.ndarray, std: np.ndarray, path: str, extra: dict[str, Any] | None = None) -> None:
    with open(path, "w") as f:
        json.dump(model_to_json(model, mean, std, extra), f)


def forward_numpy(weights: dict[str, Any], raw: np.ndarray) -> np.ndarray:
    """Reference forward pass on raw (unstandardised) features, shape (n, n_features)."""
    mean = np.asarray(weights["feature_mean"], dtype=float)
    std = np.asarray(weights["feature_std"], dtype=float)
    std = np.where(std > 0, std, 1.0)
    x = (np.asarray(raw, dtype=float) - mean) / std
    layers = weights["layers"]
    for i, layer in enumerate(layers):
        w = np.asarray(layer["weight"], dtype=float)
        b = np.asarray(layer["bias"], dtype=float)
        x = x @ w.T + b
        if i + 1 < len(layers):
            x = np.maximum(x, 0.0)
    return x[:, 0]
