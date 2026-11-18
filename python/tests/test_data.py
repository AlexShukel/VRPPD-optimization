import numpy as np
import pytest

from vrppd_nn.data import load_states, subsample_instances, zscore


def test_subsample_is_nested_and_deterministic(write_labels):
    states = load_states(write_labels("a.jsonl", 40, per_instance=2))
    small = subsample_instances(states, 5, seed=3)
    big = subsample_instances(states, 15, seed=3)
    assert {s.instance for s in small} < {s.instance for s in big}
    assert len({s.instance for s in small}) == 5 and len(small) == 10
    again = subsample_instances(states, 5, seed=3)
    assert [(s.instance, s.iteration) for s in small] == [(s.instance, s.iteration) for s in again]
    other = subsample_instances(states, 5, seed=4)
    assert {s.instance for s in other} != {s.instance for s in small}


def test_target_top_reads_top_share(write_labels):
    path = write_labels("a.jsonl", 3)
    mean = load_states(path, target="mean")
    top = load_states(path, target="top")
    assert not np.allclose(mean[0].z, top[0].z)
    assert top[0].mask.sum() == 11 and not top[0].mask[0]
    valid = top[0].z[top[0].mask]
    assert abs(valid.mean()) < 1e-9 and abs(valid.std() - 1) < 1e-9


def test_target_top_missing_field_raises(write_labels):
    path = write_labels("old.jsonl", 2, top=False)
    assert len(load_states(path)) == 2
    with pytest.raises(ValueError, match="top_share"):
        load_states(path, target="top")


def test_zero_spread_state_gives_zero_targets(write_labels):
    states = load_states(write_labels("a.jsonl", 3, flat_first=True))
    assert len(states) == 3
    assert np.all(states[0].z == 0.0)
    assert states[1].z.std() > 0


def test_tiny_spread_gives_exactly_zero():
    # the mean of three 0.1 is not exactly 0.1, so (s - mean) is about 1e-17, not 0
    z, mask = zscore(np.array([0.1, 0.1, 0.1, np.nan]))
    assert mask.tolist() == [True, True, True, False]
    assert np.all(z == 0.0)
