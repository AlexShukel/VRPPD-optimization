"""Train the B0 MLP on labelled states.

Usage:
  python -m vrppd_nn.train --train labels/train.jsonl --val labels/val.jsonl \
      --out results/stage2/models/w64_d2_mse_s0 --width 64 --depth 2 --loss mse --seed 0

Writes to --out: weights.json (Rust-compatible), loss_curve.csv (epoch, train_loss,
val_loss, val_spearman, val_p_at_k), val_predictions.jsonl, config.json and, last,
metrics.json (final validation metrics on the best-epoch weights). Every file is
written as *.tmp and renamed. The arguments, training loop and outputs are shared
with train_v1 (V1 encoder).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import resource
import sys
import time
from typing import Callable

import numpy as np
import torch
from torch import nn

from vrppd_nn.data import State, StateBatcher, feature_stats, load_states, subsample_instances
from vrppd_nn.fileio import write_atomic, write_json_atomic
from vrppd_nn.metrics import evaluate_predictions
from vrppd_nn.model import B0, save_json

torch.set_num_threads(4)  # default for importers; train and train_v1 take --threads


def peak_rss_mb() -> float:
    """Peak resident memory of this process in MB (ru_maxrss: bytes on macOS, kilobytes on Linux)."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def masked_mse(pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = (pred - y) ** 2
    return (diff * mask).sum() / mask.sum().clamp(min=1)


def listnet_loss(pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Listwise (ListNet, top-1) loss per state: cross-entropy between softmax of the
    true z-scores and softmax of the predicted scores, averaged over states."""
    neg = torch.finfo(pred.dtype).min
    p_true = torch.softmax(y.masked_fill(~mask, neg), dim=1)
    log_p = torch.log_softmax(pred.masked_fill(~mask, neg), dim=1)
    log_p = log_p.masked_fill(~mask, 0.0)
    return -(p_true * log_p).sum(dim=1).mean()


LOSSES = {"mse": masked_mse, "rank": listnet_loss}
# --select-by choice -> key of evaluate_predictions stored as selection_value
SELECT_BY = {"spearman": "spearman", "p_at_k": "precision_at_k", "best_try": "p_at_k_best_try"}


def model_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


# forward(model, batch, device) -> (pred (B, n_max), y, mask, n orders per state)
Forward = Callable[[nn.Module, tuple, torch.device], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]


def b0_forward(model: B0, batch: tuple, dev: torch.device):
    x, y, m, _k, n = batch
    return model(x.to(dev)), y.to(dev), m.to(dev), n


def predict_states(model: nn.Module, batcher, forward: Forward = b0_forward) -> list[np.ndarray]:
    model.eval()
    dev = model_device(model)
    out: list[np.ndarray] = []
    with torch.no_grad():
        for batch in batcher:
            pred, _y, _m, n = forward(model, batch, dev)
            p = pred.cpu().numpy()
            for i in range(p.shape[0]):
                out.append(p[i, :int(n[i])])
    return out


def run_epoch(model, batcher, loss_fn, opt=None, forward: Forward = b0_forward) -> float:
    total, count = 0.0, 0
    dev = model_device(model)
    for batch in batcher:
        if opt is not None:
            model.train()
            opt.zero_grad()
        pred, y, m, _n = forward(model, batch, dev)
        loss = loss_fn(pred, y, m)
        if opt is not None:
            loss.backward()
            opt.step()
        total += float(loss.detach()) * pred.shape[0]
        count += pred.shape[0]
    return total / max(count, 1)


def add_common_args(ap: argparse.ArgumentParser) -> None:
    """Arguments shared by train (B0) and train_v1 (V1)."""
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", choices=["mean", "top"], default="mean", help="label field to learn")
    ap.add_argument("--select-by", choices=sorted(SELECT_BY), default="p_at_k",
                    help="validation metric recorded as selection_value, used to rank configurations (epochs stay on "
                         "val loss); best_try = precision at k against the best try, the same for every target")
    ap.add_argument("--train-instances", type=int, default=None, help="train on all states of N instances (nested)")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--loss", choices=sorted(LOSSES), default="mse")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-states", type=int, default=8, help="states per mini-batch (~256 order rows)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="use only the first N states (smoke tests)")
    ap.add_argument("--threads", type=int, default=4, help="torch CPU threads of this process")


def load_split_states(args: argparse.Namespace) -> tuple[list[State], list[State]]:
    train_states = load_states(args.train, args.limit, args.target)
    if args.train_instances is not None:
        train_states = subsample_instances(train_states, args.train_instances, args.seed)
    val_states = load_states(args.val, args.limit, args.target)
    return train_states, val_states


def fit(model: nn.Module, train_b, val_b, val_states: list[State], args: argparse.Namespace,
        forward: Forward = b0_forward) -> tuple[dict, list[np.ndarray], list[tuple]]:
    """Adam; early stopping on validation loss; the best-loss epoch is restored.
    Returns (validation metrics, validation predictions, loss-curve rows)."""
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = LOSSES[args.loss]
    metric_key = SELECT_BY[args.select_by]
    best_val, best_epoch, best_state = float("inf"), -1, None
    rows = []
    t_train = time.time()
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_b, loss_fn, opt, forward)
        model.eval()
        with torch.no_grad():
            va = run_epoch(model, val_b, loss_fn, None, forward)
        preds = predict_states(model, val_b, forward)
        met = evaluate_predictions(val_states, preds)
        rows.append((epoch, tr, va, met["spearman"], met["precision_at_k"]))
        print(f"epoch {epoch:3d} train {tr:.4f} val {va:.4f} spearman {met['spearman']:.4f} p@k {met['precision_at_k']:.4f}", flush=True)
        if best_state is None or va < best_val - 1e-6:
            best_val, best_epoch = va, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= args.patience:
            print(f"early stop at epoch {epoch} (best {best_epoch})")
            break
    train_s = time.time() - t_train

    model.load_state_dict(best_state)
    preds = predict_states(model, val_b, forward)
    met = evaluate_predictions(val_states, preds)
    n_params = sum(p.numel() for p in model.parameters())
    met.update({"best_epoch": best_epoch, "best_val_loss": best_val, "epochs_run": len(rows),
                "train_seconds": train_s, "device": str(model_device(model)),
                "n_params": n_params, "n_parameters": n_params,
                "select_by": args.select_by, "selection_value": met[metric_key], "target": args.target,
                "train_loss_curve": [r[1] for r in rows], "val_loss_curve": [r[2] for r in rows]})
    return met, preds, rows


def write_outputs(out: str, args: argparse.Namespace, met: dict, rows: list[tuple], val_states: list[State],
                  preds: list[np.ndarray], save_weights: Callable[[str], None]) -> None:
    """Every file goes through *.tmp + rename; metrics.json comes last, so its presence
    means the run finished."""
    curve = "epoch,train_loss,val_loss,val_spearman,val_p_at_k\n" + "".join(
        ",".join(f"{v}" for v in r) + "\n" for r in rows)
    write_atomic(os.path.join(out, "loss_curve.csv"), curve)
    write_json_atomic(os.path.join(out, "config.json"), vars(args), indent=2)
    save_weights(os.path.join(out, "weights.json"))
    # per-state predictions on validation for scatter plots
    write_atomic(os.path.join(out, "val_predictions.jsonl"), "".join(
        json.dumps({"instance": s.instance, "cell": s.cell, "iteration": s.iteration, "z": s.z.tolist(),
                    "mask": s.mask.tolist(), "pred": p.tolist(), "k": s.k}) + "\n"
        for s, p in zip(val_states, preds)))
    met.update({"threads": torch.get_num_threads(), "peak_rss_mb": peak_rss_mb()})
    write_json_atomic(os.path.join(out, "metrics.json"), met, indent=2)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--depth", type=int, default=2)
    args = ap.parse_args(argv)

    torch.set_num_threads(args.threads)
    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    train_states, val_states = load_split_states(args)
    device = resolve_device(args.device)
    mean, std = feature_stats(train_states)
    load_s = time.time() - t0

    train_b = StateBatcher(train_states, mean, std, args.batch_states, shuffle=True, seed=args.seed)
    val_b = StateBatcher(val_states, mean, std, args.batch_states, shuffle=False)
    model = B0(width=args.width, depth=args.depth, dropout=args.dropout).to(device)
    met, preds, rows = fit(model, train_b, val_b, val_states, args)
    met.update({"load_seconds": load_s, "n_train_states": len(train_states), "n_val_states": len(val_states)})
    model.cpu()
    extra = {"config": {"width": args.width, "depth": args.depth, "loss": args.loss, "seed": args.seed,
                        "best_epoch": met["best_epoch"], "target": args.target}}
    write_outputs(args.out, args, met, rows, val_states, preds,
                  lambda path: save_json(model, mean, std, path, extra=extra))
    print(json.dumps(met))


if __name__ == "__main__":
    main()
