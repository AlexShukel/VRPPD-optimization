import json
import math
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from vrppd_nn import reinforce
from vrppd_nn.encoder import V1Encoder, export_json, load_json
from vrppd_nn.reinforce import (first_pick_entropy, reinforce_loss, sample_sets, sequence_logprob,
                                train_iteration)

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "small-instances"


def ordered_pairs(n, must_contain=None):
    return [(i, j) for i in range(n) for j in range(n)
            if i != j and (must_contain is None or must_contain in (i, j))]


def test_sequence_logprob_sums_to_one():
    torch.manual_seed(0)
    pairs = torch.tensor(ordered_pairs(4)).unsqueeze(0)  # all 12 ordered pairs of N = 4, k = 2
    scores = torch.randn(1, 4)
    p = sequence_logprob(scores, torch.zeros(1, 4, dtype=torch.bool), pairs).exp()
    assert p.shape == (1, 12)
    assert float(p.sum()) == pytest.approx(1.0, abs=1e-5)
    # The same 4 orders plus 2 padding orders with large scores, which must take no probability.
    padded = torch.cat([scores, torch.full((1, 2), 10.0)], dim=1)
    pad = torch.tensor([[False] * 4 + [True] * 2])
    assert float(sequence_logprob(padded, pad, pairs).exp().sum()) == pytest.approx(1.0, abs=1e-5)


def test_samples_are_distinct_and_unpadded():
    torch.manual_seed(1)
    n, k = [8, 5, 3], [3, 2, 3]
    scores = torch.randn(3, 8)
    pad = torch.arange(8)[None, :] >= torch.tensor(n)[:, None]
    scores = scores.masked_fill(pad, 50.0)  # padding orders look attractive
    gen = torch.Generator().manual_seed(0)
    sets, logp = sample_sets(scores, pad, k, 40, generator=gen)
    assert sets.shape == (3, 40, 3) and sets.dtype == torch.long and logp.shape == (3, 40)
    for b in range(3):
        for s in range(40):
            row = sets[b, s].tolist()
            picked = row[:k[b]]
            assert len(set(picked)) == k[b]
            assert all(0 <= o < n[b] for o in picked)
            assert row[k[b]:] == [-1] * (3 - k[b])
    assert torch.isfinite(logp).all() and (logp <= 0).all()
    torch.testing.assert_close(logp, sequence_logprob(scores, pad, sets))
    with pytest.raises(ValueError):
        sample_sets(scores, pad, [3, 6, 3], 2)  # environment 1 has only 5 orders


def test_draws_follow_sequence_probabilities():
    torch.manual_seed(2)
    scores = torch.randn(1, 4) * 1.5
    pad = torch.zeros(1, 4, dtype=torch.bool)
    sets, _ = sample_sets(scores, pad, [2], 20000, generator=torch.Generator().manual_seed(3))
    pairs = ordered_pairs(4)
    counts = dict.fromkeys(pairs, 0)
    for row in sets[0].tolist():
        counts[tuple(row)] += 1
    expected = sequence_logprob(scores, pad, torch.tensor(pairs).unsqueeze(0)).exp()[0]
    observed = torch.tensor([counts[p] / 20000 for p in pairs])
    torch.testing.assert_close(observed, expected, atol=0.01, rtol=0)


def test_first_pick_entropy_of_uniform_scores():
    pad = torch.tensor([[False] * 6, [False] * 3 + [True] * 3])
    scores = torch.zeros(2, 6).masked_fill(pad, 7.0)
    torch.testing.assert_close(first_pick_entropy(scores, pad), torch.tensor([math.log(6), math.log(3)]))


def test_loss_finite_when_all_rewards_equal():
    scores = torch.randn(3, 5, requires_grad=True)
    pad = torch.zeros(3, 5, dtype=torch.bool)
    for n_samples, ema in ((4, None), (1, 2.5)):
        entropy = first_pick_entropy(scores, pad)
        _sets, logp = sample_sets(scores, pad, [2, 2, 2], n_samples)
        reward = torch.full((3, n_samples), 2.5)
        loss = reinforce_loss(logp, reward, entropy, 0.01, ema)
        assert torch.isfinite(loss)
        torch.testing.assert_close(loss, -0.01 * entropy.mean())
        (grad,) = torch.autograd.grad(loss, scores)
        assert torch.isfinite(grad).all()
    with pytest.raises(ValueError):
        reinforce_loss(logp, reward, entropy, 0.01, None)  # one sample needs the moving-average baseline


class ToyEnv:
    """Six orders; a removal set earns 1 when it contains order 0, else 0."""

    def __init__(self, n_envs):
        self.n_envs = n_envs

    def k(self):
        return [2] * self.n_envs

    def evaluate(self, removals):
        return [[1.0 if 0 in s else 0.0 for s in sets] for sets in removals]

    def step(self, removals):
        return [True] * len(removals)


def test_policy_learns_toy_reward():
    torch.manual_seed(0)
    theta = torch.zeros(6, requires_grad=True)
    pad = torch.zeros(4, 6, dtype=torch.bool)
    opt = torch.optim.Adam([theta], lr=0.05)
    env, gen, rng = ToyEnv(4), torch.Generator().manual_seed(0), np.random.default_rng(0)
    ema = None
    for _ in range(300):
        stats, ema = train_iteration(lambda: (theta.expand(4, -1), pad), env, opt, 4, 0.01, ema, gen, rng)
    with_zero = torch.tensor(ordered_pairs(6, must_contain=0)).unsqueeze(0)
    p_zero = float(sequence_logprob(theta.detach()[None], pad[:1], with_zero).exp().sum())
    assert p_zero > 0.9
    assert stats["reward"] > stats["random_reward"]


def make_instance(rng, n_vehicles, n_orders):
    def loc():
        return {"hash": "h", "latitude": float(rng.uniform(44, 55)), "longitude": float(rng.uniform(-5, 15))}
    return {
        "vehicles": [{"id": v, "startLocation": loc(), "priceKm": float(rng.uniform(1, 3))} for v in range(n_vehicles)],
        "orders": [{"id": 100 + o, "pickupLocation": loc(), "deliveryLocation": loc(),
                    "loadFactor": float(rng.integers(1, 10))} for o in range(n_orders)],
    }


class StubEnv:
    """BatchEnv stand-in: every order sits on vehicle 0 (all pickups, then all
    deliveries) and a set earns the number of orders 0 and 1 in it."""

    def __init__(self, paths, seed, horizon=200, threads=0):
        self.sizes = []
        for p in paths:
            with open(p) as f:
                d = json.load(f)
            self.sizes.append((len(d["vehicles"]), len(d["orders"])))

    def reset(self):
        pass

    def k(self):
        return [min(2, n) for _v, n in self.sizes]

    def routes(self):
        return [[[(o, "pickup") for o in range(n)] + [(o, "delivery") for o in range(n)]] + [[] for _ in range(v - 1)]
                for v, n in self.sizes]

    def evaluate(self, removals):
        return [[float(sum(o < 2 for o in s)) for s in sets] for sets in removals]

    def step(self, removals):
        return [True] * len(removals)


@pytest.fixture
def stub_dirs(tmp_path):
    rng = np.random.default_rng(0)
    dirs = []
    for split, sizes in (("train", [(2, 4), (3, 5), (2, 6), (3, 3), (2, 5)]), ("val", [(2, 4), (3, 6)])):
        d = tmp_path / split / "cell"
        d.mkdir(parents=True)
        for i, (v, n) in enumerate(sizes):
            (d / f"{i}.json").write_text(json.dumps(make_instance(rng, v, n)))
        dirs.append(str(tmp_path / split))
    return dirs


def stub_args(dirs, out, *extra):
    return ["--train-instances-dir", dirs[0], "--val-instances-dir", dirs[1], "--out", str(out),
            "--batch-envs", "3", "--horizon", "3", "--samples", "2", "--d", "16", "--heads", "2", "--n-att", "1",
            "--lr", "1e-3", "--device", "cpu", *extra]


def read_log(out):
    lines = (out / "train_log.csv").read_text().splitlines()
    assert lines[0] == "episode,hours,mean_reward,random_reward,entropy,loss"
    return [[int(r.split(",")[0])] + [float(x) for x in r.split(",")[1:]] for r in lines[1:]]


def test_resume_continues_episode_count(tmp_path, stub_dirs):
    out = tmp_path / "out"
    reinforce.main(stub_args(stub_dirs, out, "--max-episodes", "20"), make_env=StubEnv)
    first = read_log(out)
    assert [r[0] for r in first] == list(range(1, 21))
    assert torch.load(out / "checkpoint.pt")["episode"] == 20
    # A run killed after its episode-20 checkpoint leaves later rows behind.
    with open(out / "train_log.csv", "a") as f:
        f.write("21,0.5,0,0,0,0\n")

    reinforce.main(stub_args(stub_dirs, out, "--max-episodes", "25", "--resume"), make_env=StubEnv)
    rows = read_log(out)
    assert [r[0] for r in rows] == list(range(1, 26))
    assert rows[:20] == first
    assert rows[20][1] >= rows[19][1]  # hours continue from the checkpoint
    assert torch.load(out / "checkpoint.pt")["episode"] == 25
    assert (out / "DONE").exists()

    # The resumed run continues exactly where an uninterrupted one would be.
    ref = tmp_path / "ref"
    reinforce.main(stub_args(stub_dirs, ref, "--max-episodes", "25"), make_env=StubEnv)
    ref_rows = read_log(ref)
    for a, b in zip(rows, ref_rows):
        assert a[0] == b[0] and a[2:] == pytest.approx(b[2:], rel=1e-6, abs=1e-9)


def test_max_hours_counts_checkpoint_hours(tmp_path, stub_dirs):
    out = tmp_path / "out"
    reinforce.main(stub_args(stub_dirs, out, "--max-episodes", "10"), make_env=StubEnv)
    ck = torch.load(out / "checkpoint.pt")
    ck["hours"] = 5.0
    torch.save(ck, out / "checkpoint.pt")
    reinforce.main(stub_args(stub_dirs, out, "--max-episodes", "30", "--max-hours", "3", "--resume"),
                   make_env=StubEnv)
    assert [r[0] for r in read_log(out)] == list(range(1, 11))
    assert (out / "DONE").exists()


def test_refuses_to_overwrite_without_resume(tmp_path, stub_dirs):
    out = tmp_path / "out"
    reinforce.main(stub_args(stub_dirs, out, "--max-episodes", "1"), make_env=StubEnv)
    with pytest.raises(SystemExit):
        reinforce.main(stub_args(stub_dirs, out, "--max-episodes", "1"), make_env=StubEnv)


def snapshot(out):
    return {f: (out / f).read_bytes() for f in ("checkpoint.pt", "train_log.csv", "val_log.csv", "DONE")}


@pytest.mark.parametrize("changed", [("--seed", "1"), ("--entropy-beta", "0.05")])
def test_resume_refuses_changed_arguments(tmp_path, stub_dirs, changed):
    out = tmp_path / "out"
    reinforce.main(stub_args(stub_dirs, out, "--max-episodes", "3"), make_env=StubEnv)
    before = snapshot(out)
    with pytest.raises(SystemExit, match=changed[0]):
        reinforce.main(stub_args(stub_dirs, out, "--max-episodes", "6", "--resume", *changed), make_env=StubEnv)
    assert snapshot(out) == before


@pytest.mark.parametrize("same_dir", ["as given", "trailing slash"])
def test_resume_with_only_max_episodes_changed_continues(tmp_path, stub_dirs, same_dir):
    out = tmp_path / "out"
    reinforce.main(stub_args(stub_dirs, out, "--max-episodes", "3"), make_env=StubEnv)
    dirs = stub_dirs if same_dir == "as given" else [d + "/" for d in stub_dirs]  # normalised on both sides
    reinforce.main(stub_args(dirs, out, "--max-episodes", "5", "--resume"), make_env=StubEnv)
    assert [r[0] for r in read_log(out)] == list(range(1, 6))
    ck = torch.load(out / "checkpoint.pt")
    assert ck["episode"] == 5
    assert ck["args"]["max_episodes"] == 3 and ck["args"]["resume"] is False  # the first invocation's, kept


def test_one_sample_stores_and_restores_the_moving_average_baseline(tmp_path, stub_dirs, monkeypatch):
    out = tmp_path / "out"
    reinforce.main(stub_args(stub_dirs, out, "--samples", "1", "--max-episodes", "12"), make_env=StubEnv)
    ck = torch.load(out / "checkpoint.pt")
    assert ck["episode"] == 12 and isinstance(ck["ema"], float)
    assert ck["val_seed"] == reinforce.VAL_SEED and ck["val_files"] == reinforce.validation_paths(stub_dirs[1])
    tc = json.loads((out / "weights.json").read_text())["train_config"]
    assert tc["samples"] == 1 and tc["threads"] == 4 == torch.get_num_threads()
    assert tc["val_seed"] == reinforce.VAL_SEED and tc["val_files"] == ck["val_files"]

    seen = []
    real = reinforce.train_iteration

    def spy(policy, env, opt, n_samples, beta, ema, generator, rng):
        seen.append(ema)
        return real(policy, env, opt, n_samples, beta, ema, generator, rng)

    monkeypatch.setattr(reinforce, "train_iteration", spy)
    reinforce.main(stub_args(stub_dirs, out, "--samples", "1", "--max-episodes", "15", "--resume"), make_env=StubEnv)
    monkeypatch.undo()
    assert seen[0] == ck["ema"]  # the first iteration after the resume starts from the stored baseline
    assert torch.load(out / "checkpoint.pt")["ema"] != ck["ema"]

    ref = tmp_path / "ref"  # and continues exactly where an uninterrupted run would be
    reinforce.main(stub_args(stub_dirs, ref, "--samples", "1", "--max-episodes", "15"), make_env=StubEnv)
    for a, b in zip(read_log(out), read_log(ref), strict=True):
        assert a[0] == b[0] and a[2:] == pytest.approx(b[2:], rel=1e-6, abs=1e-9)
    assert torch.load(out / "checkpoint.pt")["ema"] == pytest.approx(torch.load(ref / "checkpoint.pt")["ema"])


def test_threads_sets_torch_threads(tmp_path, stub_dirs):
    try:
        reinforce.main(stub_args(stub_dirs, tmp_path / "out", "--max-episodes", "1", "--threads", "2"),
                       make_env=StubEnv)
        assert torch.get_num_threads() == 2
    finally:
        torch.set_num_threads(4)
    assert json.loads((tmp_path / "out" / "weights.json").read_text())["train_config"]["threads"] == 2


def test_validation_list_covers_every_cell(tmp_path):
    for cell in ("10_20", "20_50"):
        (tmp_path / "nested" / cell).mkdir(parents=True)
        for i in range(20):
            (tmp_path / "nested" / cell / f"{i}.json").write_text("{}")
    paths = reinforce.validation_paths(str(tmp_path / "nested"))
    assert paths == reinforce.validation_paths(str(tmp_path / "nested"))
    cells = [Path(p).parent.name for p in paths]
    assert cells == ["10_20"] * 16 + ["20_50"] * 16  # 32 // 2 per cell, cells in sorted order
    names = sorted(f"{i}.json" for i in range(20))[:16]
    assert [Path(p).name for p in paths] == names * 2
    (tmp_path / "flat").mkdir()
    for i in range(40):
        (tmp_path / "flat" / f"{i}.json").write_text("{}")
    assert [Path(p).name for p in reinforce.validation_paths(str(tmp_path / "flat"))] == \
        sorted(f"{i}.json" for i in range(40))[:32]


class RecordingEnv(StubEnv):
    built = []

    def __init__(self, paths, seed, **kwargs):
        RecordingEnv.built.append(list(paths))
        super().__init__(paths, seed, **kwargs)


def test_main_validates_on_every_cell(tmp_path, stub_dirs):
    rng = np.random.default_rng(1)
    val = tmp_path / "val_nested"
    for cell, count in (("a", 33), ("b", 2)):  # the first 32 paths would all be in cell a
        (val / cell).mkdir(parents=True)
        for i in range(count):
            (val / cell / f"{i}.json").write_text(json.dumps(make_instance(rng, 2, 3)))
    RecordingEnv.built = []
    reinforce.main(stub_args([stub_dirs[0], str(val)], tmp_path / "out", "--max-episodes", "1"), make_env=RecordingEnv)
    (val_env,) = [paths for paths in RecordingEnv.built if str(val) in paths[0]]
    assert val_env == reinforce.validation_paths(str(val))
    assert sorted({Path(p).parent.name for p in val_env}) == ["a", "b"] and len(val_env) == 18


def test_init_sets_architecture_and_dropout(tmp_path, stub_dirs):
    torch.manual_seed(0)
    init = tmp_path / "v1.json"
    export_json(V1Encoder(d=16, heads=2, n_att=1), np.zeros(4), np.ones(4), str(init),
                extra={"train_config": {"dropout": 0.0}})
    bare = tmp_path / "v1_bare.json"
    export_json(V1Encoder(d=16, heads=2, n_att=1), np.zeros(4), np.ones(4), str(bare))

    def run(out, init_path, *extra):
        reinforce.main(["--train-instances-dir", stub_dirs[0], "--val-instances-dir", stub_dirs[1],
                        "--out", str(tmp_path / out), "--init", str(init_path), "--batch-envs", "2", "--horizon", "2",
                        "--max-episodes", "1", "--device", "cpu", *extra], make_env=StubEnv)
        args = torch.load(tmp_path / out / "checkpoint.pt")["args"]
        return args["dropout"], (args["d"], args["heads"], args["n_att"])

    assert run("from_init", init) == (0.0, (16, 2, 1))
    assert json.loads((tmp_path / "from_init" / "weights.json").read_text())["train_config"]["dropout"] == 0.0
    assert run("explicit", init, "--dropout", "0.3", "--d", "16") == (0.3, (16, 2, 1))
    assert run("no_train_config", bare) == (0.1, (16, 2, 1))
    with pytest.raises(SystemExit):
        run("contradicts", init, "--d", "32")


def test_real_env_two_episodes(tmp_path):
    pytest.importorskip(
        "vrppd_py",
        reason="vrppd_py is not built: run `.venv/bin/maturin develop --release -m crates/vrppd-py/Cargo.toml`",
    )
    dirs = []
    for split, cells in (("train", ["5_5", "4_5", "3_5"]), ("val", ["5_4", "4_4"])):
        d = tmp_path / split
        d.mkdir()
        for cell in cells:
            shutil.copy(sorted((FIXTURES / cell).glob("*.json"))[0], d / f"{cell}.json")
        dirs.append(str(d))
    out = tmp_path / "out"
    reinforce.main(["--train-instances-dir", dirs[0], "--val-instances-dir", dirs[1], "--out", str(out),
                    "--horizon", "5", "--batch-envs", "2", "--max-episodes", "2", "--samples", "2",
                    "--d", "16", "--heads", "2", "--n-att", "1", "--device", "cpu"])  # default env: vrppd_py.BatchEnv
    rows = read_log(out)
    assert [r[0] for r in rows] == [1, 2]
    assert all(math.isfinite(x) for r in rows for x in r)
    assert torch.load(out / "checkpoint.pt")["episode"] == 2
    assert (out / "val_log.csv").read_text().splitlines()[1].startswith("2,")  # validated after the last episode
    w = json.loads((out / "weights.json").read_text())
    assert w["model"] == "v1" and w["training"] == "reinforce"
    model, tok_mean, _tok_std = load_json(str(out / "weights.json"))
    assert model.config() == {"d": 16, "heads": 2, "n_att": 1, "ff_mult": 2} and len(tok_mean) == 4
    assert (out / "DONE").exists()
