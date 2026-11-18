import json
import os

import numpy as np
import pytest
import torch

from vrppd_nn import train_v1
from vrppd_nn.data import load_states
from vrppd_nn.encoder import V1Encoder, export_json, load_json
from vrppd_nn.encoder_data import build_tokens, collate_tokens, load_instance, stats_of_tokens

P, D = "pickup", "delivery"

# Vehicle 0 serves orders 0, 1, 2, 4, 5 (interleaved), vehicle 1 is unused,
# vehicle 2 serves order 3 only. Order indices, not ids.
ROUTES = [
    [(0, P), (1, P), (0, D), (2, P), (4, P), (1, D), (2, D), (5, P), (4, D), (5, D)],
    [],
    [(3, P), (3, D)],
]


def make_instance(rng, n_vehicles=3, n_orders=6, id0=1000):
    def loc():
        return {"hash": "h", "latitude": float(rng.uniform(44, 55)), "longitude": float(rng.uniform(-5, 15))}
    return {
        "vehicles": [{"id": 10 + v, "startLocation": loc(), "priceKm": float(rng.uniform(1, 3))}
                     for v in range(n_vehicles)],
        "orders": [{"id": id0 + 7 * o, "pickupLocation": loc(), "deliveryLocation": loc(),
                    "loadFactor": float(rng.integers(1, 10))} for o in range(n_orders)],
    }


def scores_of(model, items, mean, std):
    batch = collate_tokens(items, mean, std)
    with torch.no_grad():
        return model(*batch.inputs()).numpy()


@pytest.fixture
def setup():
    inst = make_instance(np.random.default_rng(0))
    mean, std = stats_of_tokens([build_tokens(inst, ROUTES).tok])
    torch.manual_seed(0)
    model = V1Encoder().eval()
    return inst, model, mean, std


def test_token_layout():
    inst = make_instance(np.random.default_rng(0))
    ts = build_tokens(inst, ROUTES)
    assert (ts.n_vehicles, ts.n_orders) == (3, 6)
    assert ts.tok.shape == (15, 7) and ts.tok.dtype == np.float32
    # starts 0..2, pickup of order o at 3 + o, delivery at 9 + o
    np.testing.assert_array_equal(ts.tok[:, :3].argmax(axis=1), [0] * 3 + [1] * 6 + [2] * 6)
    np.testing.assert_array_equal(ts.tok[:, :3].sum(axis=1), np.ones(15))
    veh, order = inst["vehicles"][1], inst["orders"][4]
    loc = veh["startLocation"]
    np.testing.assert_allclose(ts.tok[1, 3:], [loc["latitude"], loc["longitude"], 0.0, veh["priceKm"]], rtol=1e-6)
    p, q, demand = order["pickupLocation"], order["deliveryLocation"], 1.0 / order["loadFactor"]
    np.testing.assert_allclose(ts.tok[3 + 4, 3:], [p["latitude"], p["longitude"], demand, 0.0], rtol=1e-6)
    np.testing.assert_allclose(ts.tok[9 + 4, 3:], [q["latitude"], q["longitude"], demand, 0.0], rtol=1e-6)
    #             s0  s1  s2  P0  P1  P2  P3  P4  P5  D0  D1  D2  D3  D4  D5
    exp_prev = [-1, -1, -1, 0, 3, 9, 2, 5, 11, 4, 7, 10, 6, 8, 13]
    exp_next = [3, -1, 6, 4, 9, 7, 12, 10, 13, 5, 11, 8, -1, 14, -1]
    exp_route = [0, 1, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0]
    np.testing.assert_array_equal(ts.prev, exp_prev)
    np.testing.assert_array_equal(ts.next, exp_next)
    np.testing.assert_array_equal(ts.route, exp_route)
    assert ts.prev.dtype == ts.next.dtype == ts.route.dtype == np.int64


def test_build_tokens_rejects_incomplete_routes():
    inst = make_instance(np.random.default_rng(0))
    with pytest.raises(ValueError, match="order 3"):
        build_tokens(inst, [ROUTES[0], [], []])
    with pytest.raises(ValueError, match="twice"):
        build_tokens(inst, [ROUTES[0], [(3, P), (3, D)], [(3, P), (3, D)]])


def test_order_permutation_equivariance(setup):
    inst, model, mean, std = setup
    perm = np.array([3, 0, 5, 1, 4, 2])  # new order i is old order perm[i]
    inv = np.argsort(perm)
    inst2 = dict(inst, orders=[inst["orders"][i] for i in perm])
    routes2 = [[(int(inv[o]), kind) for o, kind in r] for r in ROUTES]
    s1 = scores_of(model, [build_tokens(inst, ROUTES)], mean, std)[0]
    s2 = scores_of(model, [build_tokens(inst2, routes2)], mean, std)[0]
    assert s1.std() > 1e-4
    np.testing.assert_allclose(s2, s1[perm], atol=1e-5)


def test_padding_invariance(setup):
    inst, model, mean, std = setup
    big = make_instance(np.random.default_rng(1), n_vehicles=4, n_orders=9, id0=5000)
    big_routes = [[s for o in range(9) if o % 4 == v for s in ((o, P), (o, D))] for v in range(4)]
    small_tok, big_tok = build_tokens(inst, ROUTES), build_tokens(big, big_routes)
    alone = scores_of(model, [small_tok], mean, std)[0]
    after = scores_of(model, [big_tok, small_tok], mean, std)
    before = scores_of(model, [small_tok, big_tok], mean, std)
    assert after.shape == (2, 9)
    np.testing.assert_allclose(after[1, :6], alone, atol=1e-5)
    np.testing.assert_allclose(before[0, :6], alone, atol=1e-5)
    np.testing.assert_array_equal(after[1, 6:], 0.0)
    np.testing.assert_allclose(after[0], scores_of(model, [big_tok], mean, std)[0], atol=1e-5)


def test_scores_depend_on_routes(setup):
    inst, model, mean, std = setup
    moved = [ROUTES[0], [(3, P), (3, D)], []]  # order 3 moves from vehicle 2 to vehicle 1
    s1 = scores_of(model, [build_tokens(inst, ROUTES)], mean, std)[0]
    s2 = scores_of(model, [build_tokens(inst, moved)], mean, std)[0]
    assert np.abs(s1 - s2).max() > 1e-4


def test_parameter_names_follow_contract():
    model = V1Encoder(d=16, heads=2, n_att=1, ff_mult=2)
    shapes = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    expected = {"embed.weight": (16, 7), "embed.bias": (16,), "nb.w1.weight": (16, 16), "nb.w2.weight": (16, 16),
                "nb.w3.weight": (16, 32), "nb.w3.bias": (16,), "nb.ff.weight": (16, 16), "nb.ff.bias": (16,),
                "nb.ln.weight": (16,), "nb.ln.bias": (16,), "rs.w4.weight": (16, 16), "rs.w4.bias": (16,),
                "rs.ln.weight": (16,), "rs.ln.bias": (16,), "head.l1.weight": (16, 32), "head.l1.bias": (16,),
                "head.l2.weight": (1, 16), "head.l2.bias": (1,)}
    for blk in ("att0", "att1"):
        for lin in "qkvo":
            expected[f"{blk}.{lin}.weight"] = (16, 16)
            expected[f"{blk}.{lin}.bias"] = (16,)
        expected.update({f"{blk}.ff1.weight": (32, 16), f"{blk}.ff1.bias": (32,), f"{blk}.ff2.weight": (16, 32),
                         f"{blk}.ff2.bias": (16,), f"{blk}.ln1.weight": (16,), f"{blk}.ln1.bias": (16,),
                         f"{blk}.ln2.weight": (16,), f"{blk}.ln2.bias": (16,)})
    assert shapes == expected


def test_export_roundtrip(setup, tmp_path):
    inst, model, mean, std = setup
    path = str(tmp_path / "weights.json")
    export_json(model, mean, std, path, extra={"training": "test"})
    w = json.load(open(path))
    assert w["model"] == "v1" and w["training"] == "test"
    assert w["config"] == {"d": 64, "heads": 4, "n_att": 2, "ff_mult": 2}
    assert len(w["tok_mean"]) == len(w["tok_std"]) == 4
    assert set(w["params"]) == set(model.state_dict())
    assert w["params"]["att0.ff1.weight"]["shape"] == [128, 64]
    assert len(w["params"]["att0.ff1.weight"]["data"]) == 128 * 64
    model2, mean2, std2 = load_json(path)
    np.testing.assert_array_equal(mean2, mean)
    np.testing.assert_array_equal(std2, std)
    tokens = [build_tokens(inst, ROUTES)]
    np.testing.assert_allclose(scores_of(model2, tokens, mean2, std2), scores_of(model, tokens, mean, std), atol=1e-6)
    with pytest.raises(ValueError, match="config"):
        export_json(model, mean, std, path, extra={"config": {}})


def test_load_instance_follows_stage2_layout(tmp_path):
    inst = make_instance(np.random.default_rng(0))
    cell_dir = tmp_path / "val" / "10_20"
    cell_dir.mkdir(parents=True)
    (cell_dir / "7.json").write_text(json.dumps(inst))
    (cell_dir / "8_20261001.json").write_text(json.dumps(inst))  # thesis naming <i>_<timestamp>.json
    assert load_instance(str(tmp_path), "val", "10_20", "val_10_20_7") == inst
    assert load_instance(str(tmp_path), "val", "10_20", "val_10_20_8") == inst
    with pytest.raises(FileNotFoundError):
        load_instance(str(tmp_path), "val", "10_20", "val_10_20_9")


def write_dataset(tmp_path, seed=0):
    """Instances under <root>/<split>/c/<i>.json and label lines that store order ids in `routes`."""
    rng = np.random.default_rng(seed)
    root = tmp_path / "instances"
    paths = {}
    for split, n_inst in (("train", 6), ("val", 3)):
        cell_dir = root / split / "c"
        cell_dir.mkdir(parents=True)
        lines = []
        for i in range(n_inst):
            n = 5 + i % 3
            inst = make_instance(rng, n_vehicles=3, n_orders=n, id0=100 * i + 1)
            (cell_dir / f"{i}.json").write_text(json.dumps(inst))
            ids = [o["id"] for o in inst["orders"]]
            for it in range(2):
                veh = rng.integers(0, 3, size=n)
                routes = [[s for o in np.flatnonzero(veh == v) for s in ([ids[o], P], [ids[o], D])] for v in range(3)]
                scores = [None if o == 0 else float(rng.normal()) for o in range(n)]
                lines.append({"instance": f"{split}_c_{i}", "cell": "c", "split": split, "iteration": it, "k": 2,
                              "objective": 1.0, "n_orders": n, "n_vehicles": 3, "routes": routes, "order_ids": ids,
                              "features": rng.normal(size=(n, 8)).tolist(), "scores": scores,
                              "top_share": [None if s is None else float(rng.uniform()) for s in scores],
                              "in_best": [o < 2 for o in range(n)], "try_improvements": [0.1, 0.2]})
        paths[split] = str(tmp_path / f"{split}.jsonl")
        with open(paths[split], "w") as f:
            f.writelines(json.dumps(line) + "\n" for line in lines)
    return str(root), paths["train"], paths["val"]


def test_label_routes_are_mapped_to_order_indices(tmp_path):
    _root, train, _val = write_dataset(tmp_path)
    s = load_states(train)[0]
    raw = json.loads(open(train).readline())
    index = {oid: i for i, oid in enumerate(raw["order_ids"])}
    assert s.split == "train"
    assert s.routes == [[(index[oid], kind) for oid, kind in r] for r in raw["routes"]]


def test_train_v1_two_epochs(tmp_path):
    root, train, val = write_dataset(tmp_path)
    out = tmp_path / "out"
    common = ["--train", train, "--val", val, "--instances-root", root, "--device", "cpu",
              "--d", "16", "--heads", "2", "--n-att", "1"]
    train_v1.main([*common, "--out", str(out), "--epochs", "2"])
    w = json.load(open(out / "weights.json"))
    assert w["model"] == "v1" and w["config"]["d"] == 16
    met = json.load(open(out / "metrics.json"))
    assert len(met["train_loss_curve"]) == 2 and met["n_train_states"] == 12 and met["n_val_states"] == 6
    assert met["best_epoch"] == int(np.argmin(met["val_loss_curve"])) + 1
    assert sorted(os.listdir(out)) == ["config.json", "loss_curve.csv", "metrics.json", "val_predictions.jsonl",
                                       "weights.json"]
    model, mean, _std = load_json(str(out / "weights.json"))
    assert model.config() == {"d": 16, "heads": 2, "n_att": 1, "ff_mult": 2}
    # --init continues from the exported weights and keeps their token statistics
    out2 = tmp_path / "out2"
    train_v1.main([*common, "--out", str(out2), "--epochs", "1", "--init", str(out / "weights.json")])
    np.testing.assert_array_equal(np.asarray(json.load(open(out2 / "weights.json"))["tok_mean"]), mean)
    with pytest.raises(ValueError, match="--d"):
        train_v1.main([*common, "--d", "32", "--out", str(tmp_path / "out3"), "--epochs", "1",
                       "--init", str(out / "weights.json")])
