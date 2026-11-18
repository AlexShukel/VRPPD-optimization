import json
import math
import random
from pathlib import Path

import pytest

vrppd_py = pytest.importorskip(
    "vrppd_py",
    reason="vrppd_py is not built: run `.venv/bin/maturin develop --release -m crates/vrppd-py/Cargo.toml`",
)

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "small-instances"
EARTH_RADIUS_KM = 6371.0


def instance_paths():
    return [str(sorted((FIXTURES / cell).glob("*.json"))[0]) for cell in ("5_5", "4_5")]


def load_instances():
    return [json.loads(Path(p).read_text()) for p in instance_paths()]


def make_env(seed=0, **kwargs):
    env = vrppd_py.BatchEnv(instance_paths(), seed, **kwargs)
    env.reset()
    return env


def one_set_each(rng, env):
    return [rng.sample(range(n), k) for n, k in zip(env.n_orders(), env.k())]


def sets_each(rng, env, per_env):
    return [[rng.sample(range(n), k) for _ in range(per_env)] for n, k in zip(env.n_orders(), env.k())]


def km(a, b):
    """Spherical law of cosines, as in vrppd-core."""
    lat1, lon1 = math.radians(a["latitude"]), math.radians(a["longitude"])
    lat2, lon2 = math.radians(b["latitude"]), math.radians(b["longitude"])
    v = math.sin(lat1) * math.sin(lat2) + math.cos(lat1) * math.cos(lat2) * math.cos(lon1 - lon2)
    return math.acos(max(-1.0, min(1.0, v))) * EARTH_RADIUS_KM


def total_distance(instance, routes):
    total = 0.0
    for vehicle, route in zip(instance["vehicles"], routes):
        prev = vehicle["startLocation"]
        for i, kind in route:
            here = instance["orders"][i]["pickupLocation" if kind == "pickup" else "deliveryLocation"]
            total += km(prev, here)
            prev = here
    return total


def assert_each_order_once(routes, n_orders):
    pickups, deliveries = {}, {}
    for v, route in enumerate(routes):
        for pos, (i, kind) in enumerate(route):
            assert kind in ("pickup", "delivery")
            seen = pickups if kind == "pickup" else deliveries
            assert i not in seen, f"order {i} has two {kind} stops"
            seen[i] = (v, pos)
    assert sorted(pickups) == sorted(deliveries) == list(range(n_orders))
    for i in range(n_orders):
        assert pickups[i][0] == deliveries[i][0] and pickups[i][1] < deliveries[i][1]


def test_sizes_match_instances():
    env = make_env()
    instances = load_instances()
    assert env.n_orders() == [len(d["orders"]) for d in instances]
    assert env.n_vehicles() == [len(d["vehicles"]) for d in instances]
    assert env.k() == [3, 3]  # max(3, round(0.15 * 5))
    assert [len(r) for r in env.routes()] == env.n_vehicles()


def test_evaluate_does_not_change_state():
    env = make_env(seed=1)
    rng = random.Random(0)
    for _ in range(5):
        env.step(one_set_each(rng, env))
    routes, objective = env.routes(), env.objective()
    removals = sets_each(rng, env, 6)
    first = env.evaluate(removals)
    second = env.evaluate(removals)
    assert env.routes() == routes
    assert env.objective() == objective
    assert first == second
    assert [len(r) for r in first] == [6, 6]
    assert all(math.isfinite(x) for row in first for x in row)


def test_empty_removal_gives_zero():
    env = make_env()
    assert env.evaluate([[[]], [[]]]) == [[0.0], [0.0]]
    out = env.evaluate([[[], [0, 1, 2]], [[1, 2, 3], []]])
    assert out[0][0] == 0.0 and out[1][1] == 0.0


def test_evaluate_matches_step_for_single_order():
    # Reinserting one order has no random repair order and never worsens the
    # solution, so the step is accepted and lands where evaluate predicted.
    env = make_env(seed=2)
    for i in range(5):
        before = env.objective()
        gain = env.evaluate([[[i]], [[4 - i]]])
        assert env.step([[i], [4 - i]]) == [True, True]
        for b, g, after in zip(before, gain, env.objective()):
            assert g[0] >= -1e-9
            assert after == pytest.approx(b * (1 - g[0] / 100), rel=1e-12, abs=1e-9)


def test_step_keeps_objective_consistent():
    env = make_env(seed=3, horizon=50)
    instances = load_instances()
    rng = random.Random(1)
    accepted = [env.step(one_set_each(rng, env)) for _ in range(50)]
    assert any(any(row) for row in accepted)
    for instance, routes, objective in zip(instances, env.routes(), env.objective()):
        assert_each_order_once(routes, len(instance["orders"]))
        assert abs(total_distance(instance, routes) - objective) <= 1e-6


def trajectory(evaluate_between):
    env = make_env(seed=7, horizon=30)
    rng, eval_rng = random.Random(2), random.Random(3)
    out = [(env.routes(), env.objective())]
    for step in range(40):
        if step == 30:
            env.reset()
            out.append((env.routes(), env.objective()))
        if evaluate_between:
            for _ in range(3):
                env.evaluate(sets_each(eval_rng, env, 4))
        accepted = env.step(one_set_each(rng, env))
        out.append((accepted, env.routes(), env.objective()))
    return out


def test_same_seed_same_trajectory():
    plain = trajectory(evaluate_between=False)
    assert trajectory(evaluate_between=False) == plain
    assert trajectory(evaluate_between=True) == plain


def test_threads_do_not_change_results():
    envs = [make_env(seed=6, threads=threads) for threads in (0, 1, 2)]
    rng = random.Random(4)
    for _ in range(10):
        removals = sets_each(rng, envs[0], 3)
        sets = one_set_each(rng, envs[0])
        results = [(env.evaluate(removals), env.step(sets), env.routes()) for env in envs]
        assert results[1] == results[0] and results[2] == results[0]


def test_invalid_order_index_raises_value_error():
    env, twin = make_env(seed=4), make_env(seed=4)
    n = env.n_orders()[0]
    bad = [
        [[[0]], [[n]]],  # index out of range
        [[[0]], [[-1]]],  # negative index
        [[[0]], [[1, 2, 1]]],  # duplicate inside one set
        [[[0]]],  # one list for two environments
    ]
    for removals in bad:
        with pytest.raises(ValueError):
            env.evaluate(removals)
        with pytest.raises(ValueError):
            env.step([sets[0] for sets in removals])
    # Failed steps apply nothing, not even to the valid first environment.
    rng = random.Random(5)
    for _ in range(10):
        sets = one_set_each(rng, env)
        assert env.step(sets) == twin.step(sets)
        assert env.routes() == twin.routes()


def test_constructor_rejects_bad_arguments(tmp_path):
    paths = instance_paths()
    with pytest.raises(ValueError):
        vrppd_py.BatchEnv([], 0)
    with pytest.raises(ValueError):
        vrppd_py.BatchEnv(paths, 0, horizon=0)
    with pytest.raises(ValueError):
        vrppd_py.BatchEnv(paths, 0, k_frac=0.0)
    with pytest.raises(OSError):
        vrppd_py.BatchEnv([str(tmp_path / "missing.json")], 0)
    broken = tmp_path / "broken.json"
    broken.write_text("{")
    with pytest.raises(ValueError):
        vrppd_py.BatchEnv([str(broken)], 0)
