"""Train the B0 MLP on labelled states.

Usage:
  python -m vrppd_nn.train --train labels/train.jsonl --val labels/val.jsonl \
      --out results/stage2/models/w64_d2_mse_s0 --width 64 --depth 2 --loss mse --seed 0

Writes to --out: weights.json (Rust-compatible), loss_curve.csv (epoch, train_loss,
val_loss, val_spearman, val_p_at_k), metrics.json (final validation metrics on the
best-epoch weights) and config.json.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np
import torch
from torch import nn

from vrppd_nn.data import StateBatcher, feature_stats, load_states
from vrppd_nn.metrics import evaluate_predictions
from vrppd_nn.model import B0, save_json

torch.set_num_threads(4)


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


def predict_states(model: B0, batcher: StateBatcher) -> list[np.ndarray]:
    model.eval()
    out: list[np.ndarray] = []
    with torch.no_grad():
        for x, _y, _m, _k, n in batcher:
            p = model(x).numpy()
            for i in range(x.shape[0]):
                out.append(p[i, :int(n[i])])
    return out


def run_epoch(model, batcher, loss_fn, opt=None) -> float:
    total, count = 0.0, 0
    for x, y, m, _k, _n in batcher:
        if opt is not None:
            model.train()
            opt.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y, m)
        if opt is not None:
            loss.backward()
            opt.step()
        total += float(loss) * x.shape[0]
        count += x.shape[0]
    return total / max(count, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--loss", choices=["mse", "rank"], default="mse")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-states", type=int, default=8, help="states per mini-batch (~256 order rows)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="use only the first N states (smoke tests)")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    train_states = load_states(args.train, args.limit)
    val_states = load_states(args.val, args.limit)
    mean, std = feature_stats(train_states)
    load_s = time.time() - t0

    train_b = StateBatcher(train_states, mean, std, args.batch_states, shuffle=True, seed=args.seed)
    val_b = StateBatcher(val_states, mean, std, args.batch_states, shuffle=False)
    model = B0(width=args.width, depth=args.depth, dropout=args.dropout)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = masked_mse if args.loss == "mse" else listnet_loss

    best_val, best_epoch, best_state = float("inf"), -1, None
    rows = []
    t_train = time.time()
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_b, loss_fn, opt)
        model.eval()
        with torch.no_grad():
            va = run_epoch(model, val_b, loss_fn, None)
        preds = predict_states(model, val_b)
        met = evaluate_predictions(val_states, preds)
        rows.append((epoch, tr, va, met["spearman"], met["precision_at_k"]))
        print(f"epoch {epoch:3d} train {tr:.4f} val {va:.4f} spearman {met['spearman']:.4f} p@k {met['precision_at_k']:.4f}", flush=True)
        if va < best_val - 1e-6:
            best_val, best_epoch = va, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= args.patience:
            print(f"early stop at epoch {epoch} (best {best_epoch})")
            break
    train_s = time.time() - t_train

    model.load_state_dict(best_state)
    preds = predict_states(model, val_b)
    met = evaluate_predictions(val_states, preds)
    met.update({"best_epoch": best_epoch, "best_val_loss": best_val, "epochs_run": len(rows),
                "train_seconds": train_s, "load_seconds": load_s,
                "n_train_states": len(train_states), "n_val_states": len(val_states),
                "n_params": sum(p.numel() for p in model.parameters())})
    with open(os.path.join(args.out, "loss_curve.csv"), "w") as f:
        f.write("epoch,train_loss,val_loss,val_spearman,val_p_at_k\n")
        for r in rows:
            f.write(",".join(f"{v}" for v in r) + "\n")
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(met, f, indent=2)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    save_json(model, mean, std, os.path.join(args.out, "weights.json"),
              extra={"config": {"width": args.width, "depth": args.depth, "loss": args.loss, "seed": args.seed,
                                "best_epoch": best_epoch}})
    # per-state predictions on validation for scatter plots
    with open(os.path.join(args.out, "val_predictions.jsonl"), "w") as f:
        for s, p in zip(val_states, preds):
            f.write(json.dumps({"instance": s.instance, "cell": s.cell, "iteration": s.iteration,
                                "z": s.z.tolist(), "mask": s.mask.tolist(), "pred": p.tolist(), "k": s.k}) + "\n")
    print(json.dumps(met))


if __name__ == "__main__":
    main()
