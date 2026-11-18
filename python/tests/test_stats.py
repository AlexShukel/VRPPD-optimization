import numpy as np
import pytest

from vrppd_nn import stats


def test_holm_example():
    assert stats.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_holm_caps_at_one_and_keeps_order():
    assert stats.holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])
    assert stats.holm([]) == []


def test_wilcoxon_identical_samples_gives_p_one():
    a = np.linspace(1.0, 2.0, 12)
    r = stats.paired_wilcoxon(a, a.copy())
    assert r["n"] == 12
    assert r["median_diff"] == 0.0
    assert r["p"] == 1.0


def test_wilcoxon_shifted_samples_small_p():
    rng = np.random.default_rng(0)
    b = rng.normal(100.0, 10.0, size=20)
    r = stats.paired_wilcoxon(b + 0.5, b)
    assert r["n"] == 20
    assert r["median_diff"] == pytest.approx(0.5)
    assert r["p"] < 0.001


def test_wilcoxon_drops_pairs_with_nan():
    a = np.array([1.0, 2.0, np.nan, 4.0])
    b = np.array([0.0, 1.0, 1.0, np.nan])
    r = stats.paired_wilcoxon(a, b)
    assert r["n"] == 2
    assert r["median_diff"] == 1.0
