"""Write a small random MLP + reference inputs/outputs (numpy forward pass) so the
Rust forward pass can be checked against Python bit-for-bit (within 1e-6)."""
import json
import sys

import numpy as np

from vrppd_nn.model import mlp_to_json, forward_numpy


def main(out_dir: str) -> None:
    rng = np.random.default_rng(1234)
    n_features = 8
    widths = [n_features, 16, 8, 1]
    layers = []
    for i in range(len(widths) - 1):
        w = rng.normal(0, 0.5, size=(widths[i + 1], widths[i]))
        b = rng.normal(0, 0.1, size=(widths[i + 1],))
        layers.append((w, b))
    mean = rng.normal(0, 1, size=n_features)
    std = rng.uniform(0.5, 2.0, size=n_features)
    weights = mlp_to_json(layers, mean, std)
    inputs = rng.normal(0, 3, size=(12, n_features))
    outputs = forward_numpy(weights, inputs)
    with open(f"{out_dir}/mlp_fixture_weights.json", "w") as f:
        json.dump(weights, f)
    with open(f"{out_dir}/mlp_fixture_io.json", "w") as f:
        json.dump({"inputs": inputs.tolist(), "outputs": outputs.tolist()}, f)
    print(f"wrote fixture with {len(inputs)} rows to {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1])
