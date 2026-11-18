import json
import os
import subprocess

import numpy as np
import pandas as pd
import pytest

from vrppd_nn import aggregate3

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CLASSICAL = ["random", "worst", "related", "adaptive"]
NEURAL = ["b0", "v1", "v2", "adaptive_n"]


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f)


def write_run(root, group, operator, cell, instance, seed, *, best=100.0, objective="distance", iters=50,
              budget_iters=50, budget_secs=None, host="h1", threads=4, cpu_list="none", tau=0.0):
    initial = best * 1.2
    totals = {"distance": best, "price": best * 2.0, "empty": best / 10.0}
    totals[objective] = best
    curve = [{"iteration": 0, "elapsed_ms": 0.0, "current": initial, "best": initial, "accepted": True},
             {"iteration": iters // 2, "elapsed_ms": 5.0, "current": best * 1.1, "best": best * 1.1, "accepted": True},
             {"iteration": iters, "elapsed_ms": 10.0, "current": best, "best": best, "accepted": True}]
    d = {"cell": cell, "instance": str(instance), "n_orders": 10, "n_vehicles": 10, "budget_iters": budget_iters,
         "budget_secs": budget_secs, "tau": tau, "host": host, "threads": threads, "cpu_list": cpu_list,
         "operator": operator, "objective": objective, "seed": seed, "k": 3, "initial": initial, "best": best,
         "best_distance": totals["distance"], "best_price": totals["price"], "best_empty": totals["empty"],
         "final_current": best, "iterations": iters, "elapsed_ms": 10.0, "accepted": iters // 2,
         "improved_current": 3, "new_best_count": 2, "best_iteration": iters, "best_elapsed_ms": 10.0,
         "curve": curve, "best_routes": [], "destroy_ms": 1.0, "repair_ms": 9.0, "valid": True}
    write_json(os.path.join(root, "lns", group, operator, cell, f"{instance}_s{seed}.json"), d)


def write_batch(root, batch, host="h1"):
    write_json(os.path.join(root, "batches", batch, "manifest.json"),
               {"batch": batch, "host": host, "cpu_model": "test cpu", "taskset": "", "threads": {"iters": 4, "timed": 4},
                "commit": "abc1234", "start": "2026-11-20T10:00:00", "end": "2026-11-20T10:05:00", "runs": 1})
    with open(os.path.join(root, "batches", batch, "DONE"), "w") as f:
        f.write("")


def write_b0_model(root, sub, name, *, width=64, depth=2, loss="mse", target="mean", seed=0, p_at_k=0.3,
                   train_instances=None, select_by="best_try"):
    d = os.path.join(root, "models", "b0", sub, name)
    write_json(os.path.join(d, "config.json"), {"width": width, "depth": depth, "loss": loss, "target": target,
                                                "seed": seed, "dropout": 0.1, "weight_decay": 1e-4,
                                                "train_instances": train_instances})
    write_json(os.path.join(d, "metrics.json"), {"spearman": 0.2, "precision_at_k": p_at_k, "precision_at_k_random": 0.15,
                                                 "best_epoch": 3, "epochs_run": 5, "best_val_loss": 0.9,
                                                 "train_seconds": 12.0, "n_params": 4801, "p_at_k_best_try": p_at_k,
                                                 "selection_value": p_at_k, "select_by": select_by,
                                                 "n_train_states": 100, "n_val_states": 20})
    write_json(os.path.join(d, "weights.json"), {"layers": []})
    with open(os.path.join(d, "loss_curve.csv"), "w") as f:
        f.write("epoch,train_loss,val_loss,val_spearman,val_p_at_k\n1,1.0,1.1,0.1,0.2\n2,0.9,1.0,0.2,0.3\n")
    return d


def make_b0_batch(root):
    for loss in ("mse", "rank"):
        for target in ("mean", "top"):
            write_b0_model(root, "grid", f"w64_d2_{loss}_{target}_s0", loss=loss, target=target,
                           p_at_k=0.4 if (loss, target) == ("rank", "top") else 0.3)
    best = "w64_d2_rank_top"
    for n in (2, 4):
        write_b0_model(root, "trainsize", f"{best}_n{n}_s0", loss="rank", target="top", train_instances=n)
    write_b0_model(root, "seeds", f"{best}_s1", loss="rank", target="top", seed=1, p_at_k=0.38)
    write_json(os.path.join(root, "models", "b0", "best.json"),
               {"best": f"grid/{best}_s0", "seeds": [f"grid/{best}_s0", f"seeds/{best}_s1"],
                "trainsize": {"2": f"trainsize/{best}_n2_s0", "4": f"trainsize/{best}_n4_s0", "6": f"grid/{best}_s0"}})
    write_batch(root, "b0")


def make_iters_batch(root, cell="10_10", n_instances=6, seeds=(1, 2), taus=None, done=True):
    for k, op in enumerate(CLASSICAL + NEURAL):
        for i in range(n_instances):
            for s in seeds:
                write_run(root, "iters5000", op, cell, i, s + 1000 * i, best=100.0 + i + k * 0.5 + 0.1 * s,
                          tau=(taus or {}).get(op, 0.0))
    if done:
        write_batch(root, f"iters-{cell}")
    else:
        os.makedirs(os.path.join(root, "batches", f"iters-{cell}"), exist_ok=True)
        with open(os.path.join(root, "batches", f"iters-{cell}", "STARTED"), "w") as f:
            f.write("h1 2026-11-20T10:00:00\n")


def test_selection_uses_only_validation_files(tmp_path):
    val = ["results/stage3/instances/val/10_20/0.json", "results/stage3/instances/val/20_50/0.json"]
    sel = {"models": {"b0": {"weights": "models/b0/grid/x/weights.json", "tau": 0.5, "instance_files": list(val)}},
           "v2": {"name": "v2-a", "tau": 0.25, "instance_files": list(val)},
           "best_model": {"name": "b0", "tau": 0.5, "instance_files": list(val)}}
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(sel))
    aggregate3.check_selection(str(path))
    sel["best_model"]["instance_files"].append("problem-gen/problems/10_20/0_1791122481373.json")
    path.write_text(json.dumps(sel))
    with pytest.raises(ValueError, match="problem-gen/problems"):
        aggregate3.check_selection(str(path))


def test_objective_column_matches_stage(tmp_path):
    root = str(tmp_path)
    write_run(root, "objective_price", "random", "10_10", 0, 1, objective="price")
    runs, _ = aggregate3.load_runs3(root)
    assert list(runs["objective"]) == ["price"]
    write_run(root, "objective_price", "adaptive", "10_10", 0, 1, objective="distance")
    with pytest.raises(ValueError, match="objective"):
        aggregate3.load_runs3(root)


def test_reported_best_must_match_its_objective_total(tmp_path):
    root = str(tmp_path)
    write_run(root, "iters5000", "random", "10_10", 0, 1, best=100.0)
    path = os.path.join(root, "lns", "iters5000", "random", "10_10", "0_s1.json")
    with open(path) as f:
        d = json.load(f)
    d["best_distance"] = 100.0 + 1e-9  # same solution, other maths library: tolerated
    write_json(path, d)
    aggregate3.load_runs3(root)
    d["best_distance"] = 101.0
    write_json(path, d)
    with pytest.raises(ValueError, match="best_distance"):
        aggregate3.load_runs3(root)


def test_timed_runs_share_cpu_setup(tmp_path):
    root = str(tmp_path)
    timed = dict(budget_iters=None, budget_secs=2.0, threads=6, cpu_list="0,2,4,6,8,10")
    write_run(root, "time60s", "random", "10_10", 0, 1, host="h1", **timed)
    write_run(root, "time60s", "v1", "10_10", 0, 1, host="h1", **timed)
    write_run(root, "time60s", "random", "10_20", 0, 1, host="h2", **timed)  # other class, other host: fine
    runs, _ = aggregate3.load_runs3(root)
    aggregate3.check_timed_runs(runs)
    for field, value in (("host", "h2"), ("threads", 4), ("cpu_list", "0,1,2,3,4,5")):
        changed = dict(timed, host="h1")
        changed[field] = value
        write_run(root, "time60s", "adaptive", "10_10", 0, 1, **changed)
        runs, _ = aggregate3.load_runs3(root)
        with pytest.raises(ValueError, match="10_10"):
            aggregate3.check_timed_runs(runs)
    os.remove(os.path.join(root, "lns", "time60s", "adaptive", "10_10", "0_s1.json"))
    refs = pd.DataFrame([{"cell": "10_10", "instance": "0", "objective": "distance", "host": "h2", "threads": 6,
                          "cpu_list": "0,2,4,6,8,10"}])
    runs, _ = aggregate3.load_runs3(root)
    with pytest.raises(ValueError, match="10_10"):
        aggregate3.check_timed_runs(runs, refs)


def test_partial_results_aggregate(tmp_path):
    root = str(tmp_path)
    make_b0_batch(root)
    make_iters_batch(root)
    aggregate3.main(["--root", root])

    with open(os.path.join(root, "summary.md")) as f:
        md = f.read()
    first_section = md.split("\n## ")[1]
    assert first_section.startswith("Coverage")
    coverage = {line.split("|")[1].strip(): line.split("|")[2].strip()
                for line in first_section.splitlines() if line.startswith("| ") and "---" not in line}
    assert coverage["b0"] == "done" and coverage["iters-10_10"] == "done"
    missing = [b for b in aggregate3.BATCHES if b not in ("b0", "iters-10_10")]
    assert all(coverage[b] == "missing" for b in missing)

    for name in ("runs.csv", "summary.csv", "stats.csv", "training_summary.csv"):
        assert os.path.exists(os.path.join(root, name)), name
    training = pd.read_csv(os.path.join(root, "training_summary.csv"))
    assert {"spearman", "p_at_k", "p_at_k_best_try"} <= set(training.columns)
    assert training.loc[training.is_best == True, "name"].tolist() == ["w64_d2_rank_top_s0"]  # noqa: E712
    summary = pd.read_csv(os.path.join(root, "summary.csv"))
    assert set(summary["operator"]) == set(CLASSICAL + NEURAL)
    assert set(summary["budget"]) == {"iters5000"}
    stats_df = pd.read_csv(os.path.join(root, "stats.csv"))
    assert set(stats_df["operator"]) == set(NEURAL) and set(stats_df["baseline"]) == {"random", "adaptive"}
    assert set(stats_df["cell"]) == {"10_10", "pooled"}
    assert stats_df["p_holm"].between(0, 1).all()

    tables = os.listdir(os.path.join(root, "tables"))
    assert "main_iters.tex" in tables and "training_b0.tex" in tables
    assert not any(t.startswith("main_time") for t in tables)
    figures = set(os.listdir(os.path.join(root, "figures")))
    assert {"loss_curves_b0.png", "trainsize.png", "search_curves_iters.png"} <= figures
    assert not figures & {"search_curves_time.png", "v2_training.png", "k_sweep.png", "large_classes.png",
                          "objectives.png", "loss_curves_v1.png"}


def test_selection_picks_lowest_mean_objective_per_model(tmp_path):
    root = str(tmp_path)
    for cell in ("10_20", "20_50"):
        os.makedirs(os.path.join(root, "instances", "val", cell))
        for i in range(2):
            open(os.path.join(root, "instances", "val", cell, f"{i}.json"), "w").close()
    # b0 is best at tau 0.5, v1 at tau 0.1; v2-b beats v2-a; v1 is the best model overall
    means = {("b0", "0.1"): 110, ("b0", "0.5"): 105, ("v1", "0.1"): 100, ("v1", "0.5"): 104,
             ("v2-a", "0.1"): 109, ("v2-a", "0.5"): 108, ("v2-b", "0.1"): 107, ("v2-b", "0.5"): 106}
    for (model, tau), value in means.items():
        for cell in ("10_20", "20_50"):
            for i in range(2):
                write_run(root, f"select-tau{tau}", model, cell, i, 1 + 1000 * i, best=value, tau=float(tau))
    weights = {m: f"models/{m}/weights.json" for m in ("b0", "v1", "v2-a", "v2-b")}
    sel = aggregate3.write_selection(root, os.path.join(root, "instances", "val"), weights)
    assert sel["models"]["b0"]["tau"] == 0.5 and sel["models"]["v1"]["tau"] == 0.1
    assert sel["v2"]["name"] == "v2-b" and sel["v2"]["tau"] == 0.5
    assert sel["best_model"]["name"] == "v1" and sel["best_model"]["operator"] == "v1"
    assert sel["best_model"]["weights"] == "models/v1/weights.json"
    assert len(sel["best_model"]["instance_files"]) == 4
    assert all("/instances/val/" in p for p in sel["best_model"]["instance_files"])
    with open(os.path.join(root, "selection", "selection.json")) as f:
        assert json.load(f) == sel


def test_batches_match_run_all_list():
    out = subprocess.run(["bash", os.path.join(REPO, "scripts", "stage3", "run_all.sh"), "--list"],
                         check=True, capture_output=True, text=True).stdout.split()
    assert out == aggregate3.BATCHES


def test_pick_best_ranks_by_selection_value_of_one_metric(tmp_path):
    root = str(tmp_path)
    dirs = [write_b0_model(root, "grid", f"m{i}", p_at_k=v) for i, v in enumerate((0.2, 0.5, 0.4))]
    assert aggregate3.pick_best(dirs + [os.path.join(root, "unfinished")]) == dirs[1]
    mixed = write_b0_model(root, "grid", "own_target", p_at_k=0.9, select_by="p_at_k")
    with pytest.raises(SystemExit, match="different metrics"):
        aggregate3.pick_best(dirs + [mixed])


def test_unfinished_batches_stay_out_and_old_outputs_are_cleared(tmp_path):
    root = str(tmp_path)
    make_iters_batch(root, "10_10")
    make_iters_batch(root, "10_20", done=False)  # still running: STARTED, no DONE
    for d, name in (("tables", "old.tex"), ("figures", "old.png")):
        os.makedirs(os.path.join(root, d))
        open(os.path.join(root, d, name), "w").close()
    aggregate3.main(["--root", root])
    with open(os.path.join(root, "summary.md")) as f:
        assert "| iters-10_20 | started | h1 |" in f.read()
    for name in ("runs.csv", "summary.csv", "stats.csv"):
        assert set(pd.read_csv(os.path.join(root, name))["cell"]) <= {"10_10", "pooled"}, name
    with open(os.path.join(root, "tables", "main_iters.tex")) as f:
        assert "10\\_20" not in f.read()
    assert "old.tex" not in os.listdir(os.path.join(root, "tables"))
    assert "old.png" not in os.listdir(os.path.join(root, "figures"))


def test_neural_runs_must_use_selected_tau(tmp_path):
    root = str(tmp_path)
    selection = {"models": {"b0": {"tau": 0.5}, "v1": {"tau": 0.25}}, "v2": {"name": "v2-c", "tau": 1.0},
                 "best_model": {"name": "v1", "operator": "v1", "tau": 0.25}}
    write_json(os.path.join(root, "selection", "selection.json"), selection)
    taus = {"b0": 0.5, "v1": 0.25, "v2": 1.0, "adaptive_n": 0.25}
    make_iters_batch(root, taus=taus)
    aggregate3.main(["--root", root])
    make_iters_batch(root, taus=dict(taus, v1=0.1))  # runs kept from an older selection
    with pytest.raises(ValueError, match="lns/iters5000/v1/10_10"):
        aggregate3.main(["--root", root])


def summary_frame(rows):
    base = {"kind": "iters", "budget": "iters5000", "objective": "distance", "k_frac": np.nan, "cell": "10_10",
            "operator": "random", "gap_best_pct": 0.0, "cea_best": 100.0, "best": 100.0, "iterations": 10,
            "elapsed_ms": 1.0, "iters_per_s": 1.0, "reached_1pct": 1.0, "iters_to_1pct": 1.0, "secs_to_1pct": 1.0,
            "accept_rate": 0.5, "destroy_share": 0.5, "valid": True}
    return pd.DataFrame([{**base, **r} for r in rows])


def test_summarise_separates_spread_over_instances_and_repeats():
    # instance 0: repeats 1, 2, 3 (mean 2, sd 1); instance 1: 5, 5, 5 (mean 5, sd 0)
    df = summary_frame([{"instance": str(i), "gap_cea_pct": g} for i, gaps in ((0, (1, 2, 3)), (1, (5, 5, 5)))
                        for g in gaps])
    r = aggregate3.summarise(df).iloc[0]
    assert r.runs == 6 and r.instances == 2
    assert r.mean_gap_cea_pct == pytest.approx(3.5)
    assert r.sd_inst_gap_cea_pct == pytest.approx(np.std([2, 5], ddof=1))
    assert r.sd_rep_gap_cea_pct == pytest.approx(0.5)


def test_paired_tests_pair_by_instance():
    def rows(op, instances, offset):
        return [{"operator": op, "instance": str(i), "gap_cea_pct": i + offset + 0.01 * rep}
                for i in instances for rep in (0, 1)]
    # b0 has instances 0..5, random 1..6, adaptive 0..6: b0 pairs with random on 1..5, with adaptive on 0..5
    df = summary_frame(rows("b0", range(6), 0.5) + rows("random", range(1, 7), 0.0) + rows("adaptive", range(7), 1.0))
    first = aggregate3.paired_tests(df)
    shuffled = aggregate3.paired_tests(df.sample(frac=1.0, random_state=3))
    pd.testing.assert_frame_equal(first, shuffled)
    got = first.set_index(["cell", "baseline"])
    assert got.loc[("10_10", "random"), "n"] == 5 and got.loc[("10_10", "random"), "n_nonzero"] == 5
    assert got.loc[("10_10", "random"), "median_diff"] == pytest.approx(0.5)
    assert got.loc[("10_10", "adaptive"), "n"] == 6
    assert got.loc[("10_10", "adaptive"), "median_diff"] == pytest.approx(-0.5)


def test_aggregate_refuses_wall_time_table_of_mixed_host_class(tmp_path):
    root = str(tmp_path)
    timed = dict(budget_iters=None, budget_secs=60.0, threads=6, cpu_list="0,2,4,6,8,10")
    for cell, hosts in (("10_10", {"random": "h1", "adaptive": "h2"}), ("10_20", {"random": "h1", "adaptive": "h1"})):
        for op, host in hosts.items():
            for i in range(3):
                write_run(root, "time60s", op, cell, i, 1 + 1000 * i, best=100.0 + i, host=host, **timed)
        write_batch(root, f"time-{cell}")
    aggregate3.main(["--root", root])
    with open(os.path.join(root, "summary.md")) as f:
        assert "Refused the wall-time table of class 10_10" in f.read()
    with open(os.path.join(root, "tables", "main_time.tex")) as f:
        table = f.read()
    assert "10\\_20" in table and "10\\_10" not in table
    assert set(pd.read_csv(os.path.join(root, "summary.csv"))["cell"]) == {"10_20"}


def write_v2_run(root, x, *, done=True):
    d = os.path.join(root, "models", "v2", x)
    write_json(os.path.join(d, "weights.json"), {
        "model": "v1", "config": {"d": 16, "heads": 2, "n_att": 1, "ff_mult": 2},
        "params": {"head.l2.weight": {"shape": [1, 16], "data": [0.0] * 16}},
        "training": "reinforce",
        "train_config": {"init": None if x == "b" else "models/v1/x/weights.json", "samples": 4, "entropy_beta": 0.01,
                         "episode": 10, "val_reward": 0.5, "dropout": 0.1, "seed": 0}})
    write_json(os.path.join(d, "DONE"), {"episode": 20, "hours": 0.5, "best_episode": 10, "best_val": 0.5})
    with open(os.path.join(d, "train_log.csv"), "w") as f:
        f.write("episode,hours,mean_reward,random_reward,entropy,loss\n1,0.1,1.0,0.5,2.0,0.1\n2,0.2,1.1,0.5,1.9,0.1\n")
    with open(os.path.join(d, "val_log.csv"), "w") as f:
        f.write("episode,hours,val_reward\n2,0.2,0.5\n")
    if done:
        write_batch(root, f"v2-{x}")


def test_training_summary_keeps_every_finished_v2_run(tmp_path):
    root = str(tmp_path)
    for x in ("a", "b", "c"):
        write_v2_run(root, x)
    write_v2_run(root, "d", done=False)  # its batch has no DONE yet
    done = {"v2-a", "v2-b", "v2-c"}
    assert aggregate3.training_summary(root, done)["name"].tolist() == ["v2-a", "v2-b", "v2-c"]
    aggregate3.main(["--root", root])
    assert pd.read_csv(os.path.join(root, "training_summary.csv"))["name"].tolist() == ["v2-a", "v2-b", "v2-c"]
    with open(os.path.join(root, "tables", "training_v2.tex")) as f:
        assert [line.split(" & ")[0] for line in f.read().splitlines()] == ["v2-a", "v2-b", "v2-c"]
