"""V2: train the V1 encoder against the search with REINFORCE.

Usage:
  python -m vrppd_nn.reinforce --train-instances-dir results/stage3/instances/train \
      --val-instances-dir results/stage3/instances/val --out results/stage3/models/v2_a \
      --init results/stage3/models/v1/weights.json --samples 4 --entropy-beta 0.01 \
      --batch-envs 32 --horizon 200 --lr 1e-4 --max-hours 3 --seed 0 --d 64 --n-att 2 --resume

The policy of an environment is softmax(scores) over its orders; a removal set is
k orders drawn from it without replacement. An episode takes --batch-envs instance
files (every *.json under --train-instances-dir, at any depth), drawn without
replacement with a generator seeded by (--seed, episode), builds a new
vrppd_py.BatchEnv on them, resets it and runs --horizon search iterations. At each
iteration: one forward pass on the current routes; --samples sets per environment
from the policy and as many uniformly random sets, all scored by BatchEnv.evaluate
(relative improvement in %, the reward); one Adam step on the REINFORCE loss with
the gradient norm clipped to 1; then BatchEnv.step with each environment's first
policy set. With one sample per state the baseline is a moving average of the batch
mean reward (decay 0.95), carried across iterations, episodes and resumes.

Validation runs every 10 episodes and after the last one: one episode on the first
32 files (sorted by path) under --val-instances-dir, environment and sampling seed
fixed, model in eval mode; its score is the mean reward of the sets the search steps
with. Token statistics come from --init, or from the training instances when
training from scratch.

Writes to --out, every file as *.tmp then renamed:
  train_log.csv  per episode: episode, hours, mean_reward, random_reward, entropy, loss
  val_log.csv    per validation: episode, hours, val_reward
  weights.json   the best validation checkpoint as V1 weights ("training": "reinforce")
  checkpoint.pt  model, optimiser, episode, elapsed hours, best validation reward,
                 baseline and generator states as of the last validation
  DONE           when the run ends normally (--max-hours or --max-episodes reached)
--resume continues from checkpoint.pt and drops log rows after its episode;
--max-hours counts the hours stored in the checkpoint plus this process's.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from typing import Any, Callable

import numpy as np
import torch
from torch import nn

from vrppd_nn.encoder import V1Encoder, export_json, load_json
from vrppd_nn.encoder_data import build_tokens, collate_tokens, instance_tokens, stats_of_tokens
from vrppd_nn.fileio import save_atomic, write_atomic
from vrppd_nn.train import resolve_device

VAL_EVERY = 10  # episodes between validations
N_VAL = 32  # validation instances
VAL_SEED = 1  # environment and sampling seed of every validation episode, the same in every run
EMA_DECAY = 0.95  # moving-average baseline when there is one sample per state
CLIP_NORM = 1.0
TRAIN_LOG_HEADER = "episode,hours,mean_reward,random_reward,entropy,loss"
VAL_LOG_HEADER = "episode,hours,val_reward"

# policy() -> (scores (B, N), order_pad (B, N)) of the environments' current states
Policy = Callable[[], tuple[torch.Tensor, torch.Tensor]]


def sequence_logprob(scores: torch.Tensor, order_pad: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
    """Log-probability (B, S) of drawing the orders seq[b, s] (Long (B, S, K)) in that
    order without replacement from softmax(scores[b]) over the non-padding orders: the
    sum over picks of the log-softmax over the orders still available. Entries -1 (after
    the last pick of a set) add nothing."""
    B, S, K = seq.shape
    N = scores.shape[1]
    real = (~order_pad).unsqueeze(1)  # (B, 1, N)
    logits = scores.unsqueeze(1).expand(B, S, N)
    avail = real.expand(B, S, N)
    total = scores.new_zeros(B, S)
    for t in range(K):
        idx = seq[:, :, t]
        live = idx >= 0
        # A finished set keeps every order available, so its softmax stays finite and
        # contributes no NaN to the gradient.
        allowed = torch.where(live.unsqueeze(-1), avail, real)
        logp = torch.log_softmax(logits.masked_fill(~allowed, float("-inf")), dim=-1)
        pick = idx.clamp(min=0).unsqueeze(-1)
        total = total + torch.where(live, logp.gather(-1, pick).squeeze(-1), 0.0)
        avail = avail & ~torch.zeros_like(avail).scatter(-1, pick, live.unsqueeze(-1))
    return total


def draw_sets(scores: torch.Tensor, order_pad: torch.Tensor, k: list[int], n_samples: int,
              generator: torch.Generator | None = None) -> torch.Tensor:
    """S sets per environment of k[b] orders drawn in order without replacement from
    softmax(scores[b]) over its non-padding orders, as the k[b] largest of score plus
    Gumbel noise (Gumbel top-k). Long (B, S, max k) on the scores' device, -1 after the
    k[b]-th pick. The noise is drawn on the CPU from `generator` (the global generator
    if None), so a draw does not depend on the device."""
    B, N = scores.shape
    n = (~order_pad).sum(dim=1).tolist()
    if len(k) != B:
        raise ValueError(f"{len(k)} destroy sizes for {B} environments")
    for b, (kb, nb) in enumerate(zip(k, n)):
        if not 0 <= kb <= nb:
            raise ValueError(f"environment {b}: cannot draw {kb} of {nb} orders")
    kmax = max(k, default=0)
    u = torch.rand((B, n_samples, N), generator=generator, dtype=torch.float64)
    gumbel = -torch.log(-torch.log(u.clamp(min=torch.finfo(torch.float64).tiny)))
    keys = scores.detach().cpu().double().unsqueeze(1) + gumbel
    keys = keys.masked_fill(order_pad.cpu().unsqueeze(1), float("-inf"))
    idx = keys.topk(kmax, dim=-1).indices
    after = torch.arange(kmax) >= torch.tensor(k).view(B, 1, 1)
    return idx.masked_fill(after, -1).to(scores.device)


def sample_sets(scores: torch.Tensor, order_pad: torch.Tensor, k: list[int], n_samples: int,
                generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Sets (Long (B, S, max k), -1 after the k[b]-th pick) and the log-probability of
    each drawn sequence (B, S), differentiable in `scores`."""
    sets = draw_sets(scores, order_pad, k, n_samples, generator)
    return sets, sequence_logprob(scores, order_pad, sets)


def first_pick_entropy(scores: torch.Tensor, order_pad: torch.Tensor) -> torch.Tensor:
    """Entropy (B,) of softmax(scores) over the non-padding orders."""
    logp = torch.log_softmax(scores.masked_fill(order_pad, float("-inf")), dim=-1).masked_fill(order_pad, 0.0)
    return -(logp.exp() * logp).sum(dim=-1)


def reinforce_loss(logp: torch.Tensor, reward: torch.Tensor, entropy: torch.Tensor, beta: float,
                   ema_baseline: float | None) -> torch.Tensor:
    """REINFORCE loss with a baseline and an entropy bonus. The baseline is the mean
    reward of the S samples of the same state when S > 1, otherwise `ema_baseline`.
    Advantages are divided by their (population) standard deviation over the batch,
    or by 1 when it is below 1e-8."""
    if logp.shape[1] > 1:
        baseline = reward.mean(dim=1, keepdim=True)
    elif ema_baseline is None:
        raise ValueError("one sample per state needs ema_baseline")
    else:
        baseline = ema_baseline
    adv = reward - baseline
    std = adv.std(correction=0)
    adv = adv / torch.where(std < 1e-8, torch.ones_like(std), std)
    return -(adv.detach() * logp).mean() - beta * entropy.mean()


def removal_lists(sets: torch.Tensor, k: list[int]) -> list[list[list[int]]]:
    """(B, S, max k) sets -> per environment, S lists of k[b] order indices."""
    return [[row[:kb] for row in env_sets] for env_sets, kb in zip(sets.tolist(), k)]


def random_sets(n_orders: list[int], k: list[int], n_samples: int, rng: np.random.Generator) -> list[list[list[int]]]:
    return [[rng.choice(n, size=kb, replace=False).tolist() for _ in range(n_samples)] for n, kb in zip(n_orders, k)]


def train_iteration(policy: Policy, env, opt: torch.optim.Optimizer, n_samples: int, beta: float,
                    ema: float | None, generator: torch.Generator,
                    rng: np.random.Generator) -> tuple[dict[str, float], float | None]:
    """One search iteration: score the current states, draw `n_samples` policy sets and
    as many random sets per environment, evaluate them, take one optimiser step, then
    step the environments with the first policy set. `env` needs k(), evaluate() and
    step() as in vrppd_py.BatchEnv. `ema` is the moving-average baseline used when
    n_samples is 1 (None before the first iteration: it starts at the first batch
    mean). Returns the iteration's means (reward, random_reward, entropy, loss) and
    the updated `ema`."""
    scores, order_pad = policy()
    k = env.k()
    sets, logp = sample_sets(scores, order_pad, k, n_samples, generator)
    chosen = removal_lists(sets, k)
    rand = random_sets((~order_pad).sum(dim=1).tolist(), k, n_samples, rng)
    out = env.evaluate([c + r for c, r in zip(chosen, rand)])
    reward = torch.tensor([row[:n_samples] for row in out], dtype=scores.dtype, device=scores.device)
    batch_mean = float(reward.mean())
    if n_samples == 1 and ema is None:
        ema = batch_mean
    entropy = first_pick_entropy(scores, order_pad)
    loss = reinforce_loss(logp, reward, entropy, beta, ema if n_samples == 1 else None)
    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_([p for group in opt.param_groups for p in group["params"]], CLIP_NORM)
    opt.step()
    if n_samples == 1:
        ema = EMA_DECAY * ema + (1 - EMA_DECAY) * batch_mean
    env.step([c[0] for c in chosen])
    stats = {"reward": batch_mean, "random_reward": float(np.mean([row[n_samples:] for row in out])),
             "entropy": float(entropy.detach().mean()), "loss": float(loss.detach())}
    return stats, ema


def encoder_policy(model: V1Encoder, instances: list[dict], env, tok_mean: np.ndarray, tok_std: np.ndarray,
                   device: torch.device) -> Policy:
    """V1 scores of the environments' current routes (order indices, as build_tokens takes them)."""
    def policy() -> tuple[torch.Tensor, torch.Tensor]:
        items = [build_tokens(inst, routes) for inst, routes in zip(instances, env.routes())]
        batch = collate_tokens(items, tok_mean, tok_std).to(device)
        return model(*batch.inputs()), batch.order_pad
    return policy


def validate(model: V1Encoder, instances: list[dict], paths: list[str], make_env, tok_mean: np.ndarray,
             tok_std: np.ndarray, device: torch.device, horizon: int) -> float:
    """Mean reward of the sets the search steps with, over one fixed-seed episode."""
    model.eval()
    env = make_env(paths, VAL_SEED, horizon=horizon, threads=0)
    env.reset()
    policy = encoder_policy(model, instances, env, tok_mean, tok_std, device)
    generator = torch.Generator().manual_seed(VAL_SEED)
    rewards: list[float] = []
    with torch.no_grad():
        for _ in range(horizon):
            scores, order_pad = policy()
            k = env.k()
            chosen = [sets[0] for sets in removal_lists(draw_sets(scores, order_pad, k, 1, generator), k)]
            rewards += [row[0] for row in env.evaluate([[c] for c in chosen])]
            env.step(chosen)
    model.train()
    return float(np.mean(rewards))


def list_instances(directory: str) -> list[str]:
    """Every *.json under `directory` (at any depth), sorted by path."""
    paths = sorted(glob.glob(os.path.join(glob.escape(directory), "**", "*.json"), recursive=True))
    if not paths:
        raise SystemExit(f"no instance files (*.json) under {directory}")
    return paths


def read_instance(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def read_log(path: str, header: str, last_episode: int) -> list[str]:
    """Rows of an earlier log up to `last_episode`; later rows were not covered by the checkpoint."""
    if not os.path.exists(path):
        return []
    with open(path) as f:
        lines = f.read().splitlines()
    if lines and lines[0] != header:
        raise SystemExit(f"{path}: unexpected header {lines[0]!r}")
    return [line for line in lines[1:] if line and int(line.split(",")[0]) <= last_episode]


def write_log(path: str, header: str, rows: list[str]) -> None:
    write_atomic(path, header + "\n" + "".join(row + "\n" for row in rows))


def rng_states(generator: torch.Generator, rng: np.random.Generator) -> dict[str, Any]:
    return {"torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "sampling": generator.get_state(), "numpy": rng.bit_generator.state}


def restore_rng_states(states: dict[str, Any], generator: torch.Generator, rng: np.random.Generator) -> None:
    torch.set_rng_state(states["torch"])
    if states["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states["cuda"])
    generator.set_state(states["sampling"])
    rng.bit_generator.state = states["numpy"]


def main(argv: list[str] | None = None, make_env=None) -> None:
    """`make_env(paths, seed, horizon=, threads=)` builds the environments; vrppd_py.BatchEnv by default."""
    ap = argparse.ArgumentParser(description="Train the V1 encoder against the search with REINFORCE (V2).")
    ap.add_argument("--train-instances-dir", required=True)
    ap.add_argument("--val-instances-dir", required=True, help="the first 32 files (sorted by path) are used")
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", default=None, help="start from these V1 weights (weights.json) and their token statistics")
    ap.add_argument("--samples", type=int, default=4, help="policy samples per state")
    ap.add_argument("--entropy-beta", type=float, default=0.01)
    ap.add_argument("--batch-envs", type=int, default=32, help="environments (instances) per episode")
    ap.add_argument("--horizon", type=int, default=200, help="search iterations per episode")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-hours", type=float, default=3.0, help="checked between episodes, summed over resumes")
    ap.add_argument("--max-episodes", type=int, default=None, help="stop after this episode (smoke runs, tests)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d", type=int, default=64, help="embedding size")
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--n-att", type=int, default=2, help="attention blocks before and after the route layers")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--resume", action="store_true", help="continue from OUT/checkpoint.pt if it exists")
    args = ap.parse_args(argv)
    if args.samples < 1 or args.horizon < 1:
        raise SystemExit("--samples and --horizon must be at least 1")
    if make_env is None:
        import vrppd_py  # imported here so that the pure functions work without the binding
        make_env = vrppd_py.BatchEnv

    t_start = time.time()
    out = args.out
    ck_path, weights_path = os.path.join(out, "checkpoint.pt"), os.path.join(out, "weights.json")
    log_path, val_log_path = os.path.join(out, "train_log.csv"), os.path.join(out, "val_log.csv")
    done_path = os.path.join(out, "DONE")
    if not args.resume and (os.path.exists(ck_path) or os.path.exists(log_path)):
        raise SystemExit(f"{out} holds an earlier run: pass --resume to continue it, or another --out")
    os.makedirs(out, exist_ok=True)
    if os.path.exists(done_path):
        os.remove(done_path)
    train_paths = list_instances(args.train_instances_dir)
    val_paths = list_instances(args.val_instances_dir)[:N_VAL]
    val_instances = [read_instance(p) for p in val_paths]
    device = resolve_device(args.device)

    torch.manual_seed(args.seed)
    model = V1Encoder(d=args.d, heads=args.heads, n_att=args.n_att, dropout=args.dropout)
    ck = torch.load(ck_path, map_location="cpu") if args.resume and os.path.exists(ck_path) else None
    if ck is not None:
        model.load_state_dict(ck["model"])
        tok_mean, tok_std = np.asarray(ck["tok_mean"]), np.asarray(ck["tok_std"])
    elif args.init:
        init, tok_mean, tok_std = load_json(args.init)
        if init.config() != model.config():
            raise SystemExit(f"--init {args.init} has config {init.config()}; pass matching --d/--heads/--n-att")
        model.load_state_dict(init.state_dict())
    else:
        tok_mean, tok_std = stats_of_tokens([instance_tokens(read_instance(p)) for p in train_paths])
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    generator = torch.Generator().manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    state: dict[str, Any] = {"episode": 0, "hours": 0.0, "val_episode": -1, "best_val": None, "best_episode": None,
                             "ema": None}
    if ck is not None:
        opt.load_state_dict(ck["optimizer"])
        restore_rng_states(ck["rng"], generator, rng)
        state = {key: ck[key] for key in state}
        print(f"resuming after episode {state['episode']} ({state['hours']:.3f} h)", flush=True)
    hours_before = state["hours"]

    def hours() -> float:
        return hours_before + (time.time() - t_start) / 3600

    log_rows = read_log(log_path, TRAIN_LOG_HEADER, state["episode"])
    val_rows = read_log(val_log_path, VAL_LOG_HEADER, state["episode"])
    write_log(log_path, TRAIN_LOG_HEADER, log_rows)
    write_log(val_log_path, VAL_LOG_HEADER, val_rows)

    def validate_and_checkpoint() -> None:
        episode = state["episode"]
        val = validate(model, val_instances, val_paths, make_env, tok_mean, tok_std, device, args.horizon)
        val_rows.append(f"{episode},{hours()},{val}")
        write_log(val_log_path, VAL_LOG_HEADER, val_rows)
        print(f"validation after episode {episode}: reward {val:.4f}", flush=True)
        if state["best_val"] is None or val > state["best_val"]:
            state["best_val"], state["best_episode"] = val, episode
            extra = {"training": "reinforce",
                     "train_config": {"init": args.init, "samples": args.samples, "entropy_beta": args.entropy_beta,
                                      "batch_envs": args.batch_envs, "horizon": args.horizon, "lr": args.lr,
                                      "dropout": args.dropout, "seed": args.seed, "episode": episode,
                                      "val_reward": val}}
            model.eval()
            export_json(model, tok_mean, tok_std, weights_path, extra=extra)
            model.train()
        state["val_episode"], state["hours"] = episode, hours()
        ck_out = {"model": model.state_dict(), "optimizer": opt.state_dict(), **state,
                  "tok_mean": [float(v) for v in tok_mean], "tok_std": [float(v) for v in tok_std],
                  "rng": rng_states(generator, rng), "args": vars(args)}
        save_atomic(ck_path, lambda tmp: torch.save(ck_out, tmp))

    while (args.max_episodes is None or state["episode"] < args.max_episodes) and hours() < args.max_hours:
        episode = state["episode"] + 1
        pick_rng = np.random.default_rng([args.seed, episode])
        pick = pick_rng.choice(len(train_paths), size=min(args.batch_envs, len(train_paths)), replace=False)
        paths = [train_paths[i] for i in pick]
        env = make_env(paths, int(pick_rng.integers(2**63)), horizon=args.horizon, threads=0)
        env.reset()
        policy = encoder_policy(model, [read_instance(p) for p in paths], env, tok_mean, tok_std, device)
        rows = []
        for _ in range(args.horizon):
            stats, state["ema"] = train_iteration(policy, env, opt, args.samples, args.entropy_beta, state["ema"],
                                                  generator, rng)
            rows.append(stats)
        mean = {key: float(np.mean([r[key] for r in rows])) for key in rows[0]}
        state["episode"] = episode
        log_rows.append(f"{episode},{hours()},{mean['reward']},{mean['random_reward']},{mean['entropy']},{mean['loss']}")
        write_log(log_path, TRAIN_LOG_HEADER, log_rows)
        print(f"episode {episode} {hours():.3f} h reward {mean['reward']:.4f} random {mean['random_reward']:.4f} "
              f"entropy {mean['entropy']:.3f} loss {mean['loss']:.4f}", flush=True)
        if episode % VAL_EVERY == 0:
            validate_and_checkpoint()
    if state["val_episode"] != state["episode"]:
        validate_and_checkpoint()
    write_atomic(done_path, json.dumps({"episode": state["episode"], "hours": hours(),
                                        "best_episode": state["best_episode"], "best_val": state["best_val"]}) + "\n")


if __name__ == "__main__":
    main()
