import json
import os

import numpy as np

from vrppd_nn import fileio, train
from vrppd_nn.data import State, zscore
from vrppd_nn.metrics import evaluate_predictions
from vrppd_nn.model import forward_numpy


def run(tmp_path, write_labels, *extra, epochs=2):
    tr = write_labels("train.jsonl", 20, seed=1)
    va = write_labels("val.jsonl", 6, seed=2)
    out = tmp_path / "out"
    train.main(["--train", tr, "--val", va, "--out", str(out), "--epochs", str(epochs),
                "--device", "cpu", *extra])
    return out


def test_train_writes_loss_curves(tmp_path, write_labels):
    out = run(tmp_path, write_labels)
    met = json.load(open(out / "metrics.json"))
    assert len(met["train_loss_curve"]) == 2 and len(met["val_loss_curve"]) == 2
    assert met["n_parameters"] > 0 and met["train_seconds"] >= 0
    w = json.load(open(out / "weights.json"))
    assert forward_numpy(w, np.zeros((4, 8))).shape == (4,)


def test_train_options_top_width_and_instances(tmp_path, write_labels):
    out = run(tmp_path, write_labels, "--target", "top", "--width", "256", "--select-by", "spearman",
              "--train-instances", "8")
    met = json.load(open(out / "metrics.json"))
    assert met["n_train_states"] == 8 and met["target"] == "top"
    w = json.load(open(out / "weights.json"))
    assert len(w["layers"][0]["weight"]) == 256


def test_select_by_is_recorded_and_epoch_stays_on_val_loss(tmp_path, write_labels):
    out = run(tmp_path, write_labels, "--select-by", "p_at_k", epochs=4)
    met = json.load(open(out / "metrics.json"))
    assert met["select_by"] == "p_at_k"
    assert met["selection_value"] == met["precision_at_k"]
    curve = met["val_loss_curve"]
    assert met["best_epoch"] == int(np.argmin(curve)) + 1
    assert met["best_val_loss"] == min(curve)


def test_every_output_is_written_as_tmp_then_renamed(tmp_path, write_labels, monkeypatch):
    renames = []
    real_replace = os.replace

    def record(src, dst):
        renames.append((str(src), str(dst)))
        real_replace(src, dst)

    monkeypatch.setattr(fileio.os, "replace", record)
    out = run(tmp_path, write_labels)
    files = sorted(os.listdir(out))
    assert files == ["config.json", "loss_curve.csv", "metrics.json", "val_predictions.jsonl", "weights.json"]
    assert sorted(renames) == sorted((str(out / f) + ".tmp", str(out / f)) for f in files)
    assert renames[-1][1].endswith("metrics.json")  # written last: its presence marks a finished run


def state_with_best_try(best: list[int], n: int = 8, k: int = 3) -> State:
    scores = np.linspace(-1.0, 1.0, n)
    z, mask = zscore(scores)
    in_best = np.zeros(n, dtype=bool)
    in_best[best] = True
    return State(instance="i", cell="c", iteration=0, k=k, features=np.zeros((n, 8)), scores=scores, z=z,
                 mask=mask, in_best=in_best, try_improvements=np.zeros(2), objective=1.0)


def test_p_at_k_best_try_is_share_of_top_k_in_best_try():
    pred = np.array([9.0, 8.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # model's top 3: orders 0, 1, 2
    same, disjoint, one = (state_with_best_try(b) for b in ([0, 1, 2], [5, 6, 7], [2, 6, 7]))
    assert evaluate_predictions([same], [pred])["p_at_k_best_try"] == 1.0
    assert evaluate_predictions([disjoint], [pred])["p_at_k_best_try"] == 0.0
    met = evaluate_predictions([same, disjoint, one], [pred] * 3)
    assert met["p_at_k_best_try"] == (1.0 + 0.0 + 1 / 3) / 3


def test_select_by_best_try_records_its_value(tmp_path, write_labels):
    out = run(tmp_path, write_labels, "--select-by", "best_try")
    with open(out / "metrics.json") as f:
        met = json.load(f)
    assert met["select_by"] == "best_try"
    assert met["selection_value"] == met["p_at_k_best_try"]
    assert 0.0 <= met["p_at_k_best_try"] <= 1.0
