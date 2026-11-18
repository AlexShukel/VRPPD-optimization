"""Train the V1 encoder on labelled states (same targets, optimiser, early stopping
and outputs as B0 in train.py).

Usage:
  python -m vrppd_nn.train_v1 --train labels/train.jsonl --val labels/val.jsonl \
      --instances-root results/stage2/instances --out results/stage3/models/v1_d64_a2_mse_s0 \
      --d 64 --heads 4 --n-att 2 --loss mse --seed 0

--instances-root holds <split>/<cell>/<id>.json, the layout of the Stage 2 pipeline.
--init starts from an exported V1 weights file: its architecture must match
--d/--heads/--n-att, and its token statistics are kept.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

from vrppd_nn.encoder import V1Encoder, export_json, load_json
from vrppd_nn.encoder_data import TokenBatch, TokenBatcher, token_stats
from vrppd_nn.train import add_common_args, fit, load_split_states, resolve_device, set_seed, write_outputs


def v1_forward(model: V1Encoder, batch: TokenBatch, dev: torch.device):
    b = batch.to(dev)
    return model(*b.inputs()), b.y, b.mask, batch.n_orders


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--d", type=int, default=64, help="embedding size")
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--n-att", type=int, default=2, help="attention blocks before and after the route layers")
    ap.add_argument("--instances-root", required=True, help="directory with <split>/<cell>/<id>.json")
    ap.add_argument("--init", default=None, help="start from these V1 weights (weights.json)")
    args = ap.parse_args(argv)

    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    train_states, val_states = load_split_states(args)
    device = resolve_device(args.device)
    model = V1Encoder(d=args.d, heads=args.heads, n_att=args.n_att, dropout=args.dropout)
    if args.init:
        init, tok_mean, tok_std = load_json(args.init)
        if init.config() != model.config():
            raise ValueError(f"--init {args.init} has config {init.config()}; pass matching --d/--heads/--n-att")
        model.load_state_dict(init.state_dict())
    else:
        tok_mean, tok_std = token_stats(train_states, args.instances_root)
    train_b = TokenBatcher(train_states, args.instances_root, tok_mean, tok_std, args.batch_states,
                           shuffle=True, seed=args.seed)
    val_b = TokenBatcher(val_states, args.instances_root, tok_mean, tok_std, args.batch_states, shuffle=False)
    load_s = time.time() - t0

    model.to(device)
    met, preds, rows = fit(model, train_b, val_b, val_states, args, forward=v1_forward)
    met.update({"load_seconds": load_s, "n_train_states": len(train_states), "n_val_states": len(val_states)})
    model.cpu()
    extra = {"training": "supervised",
             "train_config": {"loss": args.loss, "seed": args.seed, "best_epoch": met["best_epoch"],
                              "target": args.target, "dropout": args.dropout, "init": args.init}}
    write_outputs(args.out, args, met, rows, val_states, preds,
                  lambda path: export_json(model, tok_mean, tok_std, path, extra=extra))
    print(json.dumps(met))


if __name__ == "__main__":
    main()
