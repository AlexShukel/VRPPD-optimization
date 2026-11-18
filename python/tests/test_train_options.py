import json

import numpy as np

from vrppd_nn import train
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
