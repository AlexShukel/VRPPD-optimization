import json
import os

import numpy as np

from vrppd_nn import fileio, train
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
