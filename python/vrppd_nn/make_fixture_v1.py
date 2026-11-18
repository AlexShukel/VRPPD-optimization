"""Write the V1 parity fixture for the Rust forward pass (crates/vrppd-lns/src/encoder.rs):
a randomly initialised V1Encoder(d=16, heads=2, n_att=1) in eval mode, one small
hand-built instance, three route sets and the PyTorch scores of each.

The file is itself a V1 weights file (model, config, tok_mean, tok_std, params);
the instance, routes and scores sit under the extra key "fixture", so the Rust
test needs no other file. Seeds are fixed, so a rerun writes the same file.

Usage: cd python && ../.venv/bin/python -m vrppd_nn.make_fixture_v1 [out.json]
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
from torch import nn

from vrppd_nn.encoder import V1Encoder, model_to_json
from vrppd_nn.encoder_data import build_tokens, collate_tokens, instance_tokens, stats_of_tokens
from vrppd_nn.fileio import write_json_atomic

TORCH_SEED = 20261118
INSTANCE_SEED = 7
N_VEHICLES, N_ORDERS = 3, 7
DEFAULT_OUT = os.path.join(os.path.dirname(__file__), "..", "..", "crates", "vrppd-lns", "tests", "fixtures",
                           "v1_fixture.json")

P, D = "pickup", "delivery"
# Order indices. Every route set serves all seven orders, pickup before delivery.
ROUTE_SETS = {
    "normal": [
        [(0, P), (1, P), (0, D), (2, P), (1, D), (2, D)],
        [(3, P), (3, D), (4, P), (4, D)],
        [(5, P), (6, P), (6, D), (5, D)],
    ],
    "unused_vehicle": [
        [(0, P), (1, P), (1, D), (2, P), (0, D), (2, D), (3, P), (3, D)],
        [],
        [(4, P), (5, P), (4, D), (6, P), (5, D), (6, D)],
    ],
    "single_order_route": [
        [(0, P), (0, D)],
        [(1, P), (2, P), (2, D), (3, P), (1, D), (3, D)],
        [(4, P), (5, P), (6, P), (5, D), (4, D), (6, D)],
    ],
}


def make_instance(rng: np.random.Generator) -> dict:
    """Instance in the problem-gen JSON format (vrppd_core::Problem)."""
    count = iter(range(1000))

    def loc() -> dict:
        return {"hash": f"fx{next(count)}", "latitude": round(float(rng.uniform(45, 56)), 6),
                "longitude": round(float(rng.uniform(2, 16)), 6)}

    return {
        "vehicles": [{"id": 100 + v, "priceKm": round(float(rng.uniform(1, 3)), 4), "startLocation": loc()}
                     for v in range(N_VEHICLES)],
        "orders": [{"id": 500 + 3 * o, "pickupLocation": loc(), "deliveryLocation": loc(),
                    "loadFactor": int(rng.integers(1, 9))} for o in range(N_ORDERS)],
    }


def make_model() -> V1Encoder:
    torch.manual_seed(TORCH_SEED)
    model = V1Encoder(d=16, heads=2, n_att=1)
    # LayerNorm starts at weight 1, bias 0; move them so the fixture also checks
    # that the Rust pass applies them.
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.LayerNorm):
                m.weight.uniform_(0.5, 1.5)
                m.bias.normal_(0.0, 0.1)
    return model.eval()


def main(out: str = DEFAULT_OUT) -> None:
    inst = make_instance(np.random.default_rng(INSTANCE_SEED))
    tok_mean, tok_std = stats_of_tokens([instance_tokens(inst)])
    model = make_model()
    states = []
    for name, routes in ROUTE_SETS.items():
        batch = collate_tokens([build_tokens(inst, routes)], tok_mean, tok_std)
        with torch.no_grad():
            scores = model(*batch.inputs())[0]
        states.append({"name": name, "routes": [[[o, kind] for o, kind in r] for r in routes],
                       "scores": [float(s) for s in scores]})
    # The route-dependent layers must change the scores, or the fixture would not test them.
    for a, b in zip(states, states[1:]):
        assert np.abs(np.subtract(a["scores"], b["scores"])).max() > 1e-3, (a["name"], b["name"])
    fixture = {"torch_seed": TORCH_SEED, "instance_seed": INSTANCE_SEED, "torch_version": torch.__version__,
               "instance": inst, "states": states}
    write_json_atomic(out, model_to_json(model, tok_mean, tok_std, extra={"fixture": fixture}))
    for s in states:
        print(f"{s['name']:>18}: " + " ".join(f"{v:+.4f}" for v in s["scores"]))
    print(f"wrote {os.path.normpath(out)}")


if __name__ == "__main__":
    main(*sys.argv[1:])
