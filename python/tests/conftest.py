import json

import numpy as np
import pytest


def make_line(rng, instance, iteration, n=12, top=True, flat=False):
    scores = [None if i == 0 else float(rng.normal()) for i in range(n)]
    d = {
        "instance": instance, "cell": "c", "iteration": iteration, "k": 3,
        "features": rng.normal(size=(n, 8)).tolist(),
        "scores": [1.0 if (flat and s is not None) else s for s in scores],
        "in_best": [bool(i < 3) for i in range(n)],
        "try_improvements": [0.1, 0.2],
        "objective": 1.0,
    }
    if top:
        d["top_share"] = [None if s is None else float(rng.uniform()) for s in scores]
    return d


@pytest.fixture
def write_labels(tmp_path):
    def _write(name, n_instances, per_instance=1, top=True, flat_first=False, seed=0):
        rng = np.random.default_rng(seed)
        path = tmp_path / name
        with open(path, "w") as f:
            for i in range(n_instances):
                for it in range(per_instance):
                    d = make_line(rng, f"inst{i:03d}", it, top=top, flat=flat_first and i == 0 and it == 0)
                    f.write(json.dumps(d) + "\n")
        return str(path)
    return _write
