# Stage 2 (interim) — implementation notes for the report author

Branch `feat/neural-lns-stage2` of `AlexShukel/VRPPD-optimization`. Written 2026-11-06 by the implementing agent before the full pipeline was launched; the pipeline (`scripts/stage2/run_all.sh`) appends its final tables, timings and figure paths to the end of this file when it finishes (section "Pipeline results"). Every number in this file comes from an actual run; numbers from the tiny smoke configuration are marked as such and must not be reported as results.

## 1. What was implemented (file paths)

Rust crate `crates/vrppd-lns` (workspace member, edition 2024):

| file | content |
|---|---|
| `src/solution.rs` | `Context` (distance matrices, demands, static per-order data), order placements, removal, **delta-cost cheapest insertion** (`best_insertion_in_route`: O(1) distance delta per candidate position, capacity check via a running maximum of the load profile; pickup before delivery on the same route), `repair_greedy`, `removal_saving`. Unit-tested against the thesis' `vrppd_cea::insert_cheapest` (made `pub` for the test; it was `pub(crate)`): identical insertion costs on 10_10, 10_20 and 20_50 instances. |
| `src/destroy.rs` | `DestroyOperator` trait (`select`, `feedback`) and the operators `random`, `worst`, `related`, `adaptive`, `neural`; `build_operator(name, weights, tau, worst_p)`. |
| `src/lns.rs` | LNS driver `run_lns`: RSCIM start solution (`vrppd_cea::generate_rscim_seeded`), destroy → greedy repair (removed orders reinserted in random order) → SA acceptance; budgets by iterations and/or wall seconds; curve logging; `LnsResult` (JSON). |
| `src/features.rs` | the eight B0 features per order (raw km / counts, standardised at inference with the mean/std stored in the weights file). |
| `src/mlp.rs` | JSON weight loading + hand-written MLP forward pass (ReLU hidden layers, linear scalar output). Test `forward_matches_python_fixture` checks it against a numpy forward pass on 12 random inputs (`tests/fixtures/mlp_fixture_*.json`, produced by `python/vrppd_nn/make_fixture.py`), tolerance 1e-6. |
| `src/label.rs` | label generation: random-removal LNS, snapshot schedule, M random removal tries per state, per-order mean improvement, `LabelledState` (JSONL line). |
| `src/io.rs` | instance loading (thesis cells or any directory), rayon thread cap. |
| `src/bin/lns.rs` | CLI `lns --cell --operator --iters --seconds --seed --k --repeats --out [--weights --tau --worst-p --log-every --log-every-ms --max-instances --threads --force]`. One JSON result + one CSV search curve per run (`<instance>_s<seed>.json/.curve.csv`), runs in parallel over instances × repeats (rayon, hard cap 6 threads), existing results are skipped (idempotent). |
| `src/bin/label.rs` | CLI `label --cell <dir> --split --lns-iters --states --m --k --seed --schedule log|even --out <jsonl>`; parallel over instances (rayon, cap 6). |
| `src/bin/gen_instances.rs` | seeded generator: samples V vehicles and O orders without replacement from the same master pools (`problem-gen/data/orders_*.json`, `vehicles_*.json`) as the thesis TypeScript generator, with `rand_xoshiro`. |
| `src/bin/cea_ref.rs` | CEA reference values (best of N seeded CEA runs, thesis defaults, wall cap), sequential over instances, resumable. |

Python package `python/vrppd_nn` (torch 2.15 CPU, `torch.set_num_threads(4)`, Python 3.12 venv at `.venv`, deps in `python/requirements.txt` / `python/pyproject.toml`):

| file | content |
|---|---|
| `data.py` | JSONL loader → `State` objects; per-state z-score of the raw scores; padded mini-batches of states (features standardised with training-set mean/std). |
| `model.py` | `B0` MLP (depth hidden layers of `width` units, ReLU, dropout, linear output), JSON export (`layers[{weight,bias}]`, `feature_mean`, `feature_std`), numpy reference forward. |
| `train.py` | training CLI: MSE regression on z-scores or listwise ranking loss (ListNet top-one: cross-entropy between softmax of true z-scores and softmax of predictions, per state), Adam + weight decay, early stopping on validation loss (patience 10), CSV loss curve per epoch, `metrics.json`, `weights.json`, `val_predictions.jsonl`. |
| `metrics.py` | validation metrics: MSE on per-state z-scores (predictions z-scored per state too, so the ranking model gets a comparable number), mean per-state Spearman, precision@k (overlap of predicted and true top-k; random expectation k/n), recall of the best try's set. |
| `label_stats.py` | label-quality statistics (see §4). |
| `aggregate.py` | `summary.csv`, `runs.csv`, `training_summary.csv`, `summary.md` and the PNG figures. |

Pipeline: `scripts/stage2/run_all.sh` (bash, idempotent, each stage skipped if its output exists, logs to `results/stage2/pipeline.log`, timings to `results/stage2/timings.csv`, machine specs to `results/stage2/machine.txt`). `STAGE2_TINY=1` runs the whole chain on a 2-instance / 50-iteration / M=4 configuration into `results/stage2-tiny/` (used for the end-to-end smoke test; it completed in about 1 minute).

Repo: `README.md` has a new section; `.gitignore` ignores `.venv/`, the generated instances, labels and raw LNS run files (reproducible by seed); `results/stage2/summary.*`, `models/`, `figures/`, `reference/` and this file are meant to be committed after the pipeline finishes (the report author should `git add results/stage2` then — the trained weights are < 200 KB each).

## 2. Design decisions and deviations from the Stage 1 plan (with reasons)

1. **Everything is in Rust; no Python LNS, no Rust↔Python binding.** The plan said "a Python package for the LNS driver … plus a Rust-to-Python binding crate". The Stage 2 task asked for a Rust LNS crate and a pure-Rust MLP forward pass; this removes the per-iteration Python call that the plan listed as the main risk. Python is used only for training and analysis.
2. **Repair**: greedy cheapest insertion, removed orders reinserted in random order (as in Ropke & Pisinger's basic greedy). Own delta-cost implementation instead of the CEA's `insert_cheapest` (which clones and recalculates a whole route per candidate position) because the LNS runs millions of iterations; equality of the chosen insertion cost with the CEA implementation is unit-tested.
3. **Acceptance**: simulated annealing as in Ropke & Pisinger 2006: start temperature such that a solution 5 % worse than the start solution is accepted with probability 0.5 (T0 = 0.05·f0/ln 2), geometric cooling to 0.2 % of T0 at the end of the budget. Because runs have either an iteration or a time budget, the temperature is driven by `progress = max(iter/max_iters, elapsed/max_secs)`: T = T0 · 0.002^progress. Identical for every operator. A candidate equal to the current solution (removed orders reinserted at the same places) counts as accepted, which is why acceptance rates are high (≈ 0.8–0.95).
4. **Destroy size** k = round(0.15·n), min 3: k = 3 on 10×10 and 10×20, k = 8 on 20×50.
5. **Worst removal** uses the Ropke & Pisinger randomisation (pick the ⌊y^p·|L|⌋-th most expensive order, p = 3) instead of strictly the k most expensive, because a deterministic choice removes the same set every iteration and the greedy repair then rebuilds the same solution (the search stalls). This was observed with the deterministic neural top-k, see §6.
6. **Related removal**: random seed order + the k−1 orders with the smallest d(pickup, pickup) + d(delivery, delivery) to the seed (no time/load terms, no randomisation), as written in the plan.
7. **Adaptive**: roulette over random/worst/related, scores σ = (33, 9, 13) for new global best / better than current / accepted worse, segment length 100 iterations, reaction factor 0.1, weights floored at 0.05 so no operator dies.
8. **Neural operator**: computes all eight features for every order each iteration, runs the MLP, and removes the top-k (τ = 0, run as `neural`) or samples k orders without replacement from softmax(score/τ) with τ = 1 (run as `neural_tau`). Both variants are run in the pipeline because the smoke test showed that deterministic top-k stalls (§6).
9. **Labels**: states are taken from a random-removal LNS of 5 000 iterations with a **log-spaced snapshot schedule** (iterations round(5000^(i/20)), i = 1..20, i.e. 2, 4, 5, 8, 13, 20, 30, 46, 71, 108, 166, 254, 388, 595, 910, 1394, 2133, 3266, 5000 after de-duplication) instead of "every N iterations". Reason: the LNS converges in roughly 200 iterations on 10×20 and after about 1 500 iterations no random 3-removal improves the solution any more (smoke run on instance 0: share of improving tries 0.03 at iteration 250, 0.00 from iteration 1 500 on), so evenly spaced states would be ≈ 90 % pure noise. The log schedule still includes the late, converged states (the last snapshot is the final state). M = 300 tries per state (each order is in ≈ 0.15·M = 45 tries), 20 snapshots requested per instance = 19 distinct states after de-duplication. Raw score = mean improvement (drop in total distance, can be negative) over the tries containing the order; `in_best`, `counts` and all M try improvements are stored so that normalisation can be changed without relabelling.
10. **Training-set size**: 300 + 150 training and 30 + 15 validation instances (10×20 + 20×50) as requested for the interim, instead of 2 000 + 1 000 / 200 + 100 in the plan — a **deviation**. Note for Stage 3: label generation is cheap (40 states with M = 300 on two 10×20 instances took 0.1 s), so the full-size sets cost minutes, not hours; the reduction was a precaution, not a necessity.
11. **Batching for training**: mini-batches of 8 states (≈ 250–400 order rows; the plan said batch 256) so that the listwise loss can be computed per state; the same batching is used for the MSE loss.
12. **Model selection**: early stopping on validation loss within a configuration; across the grid (width {64,128} × depth {2,3} × loss {mse, rank}) the configuration with the highest validation Spearman is chosen (losses of the two objectives are not comparable), then retrained with seeds 1 and 2; the seed-0 weights drive the neural operator.
13. **Reference values**: best of **3** CEA runs (plan: 10) with the thesis defaults and a 60 s cap, 6 threads, sequential. The repo contained no stored thesis results for these cells (no `results/` directory), so they were recomputed. Gaps are reported against this CEA reference and also against the best value known for the instance (min over CEA and all LNS runs), because LNS beats CEA on several instances (§6).
14. **Repeats**: 5 per instance and operator (plan: 10).
15. **Wall-time budget**: 30 s per LNS run instead of 60 s, so that the 1 800 time-budget runs (6 operators × 3 cells × 20 instances × 5 repeats) fit in ≈ 2.5 h on 6 threads. With a cap of 6 parallel runs each run has its own core; CEA (multi-threaded) runs alone.
16. **Resource caps** as required: rayon 6 threads (`RAYON_NUM_THREADS=6` plus a hard cap in `io::init_threads`), torch 4 threads, CEA 6 threads, all stages sequential; `df` check (≥ 10 GB) before the big stages.
17. The thesis test instances (`problem-gen/problems/<cell>`) and the generated train/val instances are sampled from the same order/vehicle pools; individual orders can overlap between a training and a test instance, whole instances cannot in practice (random subsets of 11 166 orders and 10 000 vehicles). This is the same relation as between any two thesis instances.

## 3. Preprocessing and splitting as implemented

- Instances: `gen_instances` with seeds 20261001 (train 10×20, 300), 20261002 (train 20×50, 150), 20261003 (val 10×20, 30), 20261004 (val 20×50, 15) → `results/stage2/instances/{train,val}/{10_20,20_50}/<i>.json`.
- Labels: `label --lns-iters 5000 --states 20 --m 300 --schedule log`, seed = 1 + instance index (train) / 100001 + index (val); per-cell JSONL files concatenated to `labels/train.jsonl` and `labels/val.jsonl`. Expected sizes: 450 × 19 = 8 550 training states (≈ 5 700 × 20 + 2 850 × 50 = 256 500 order rows) and 855 validation states.
- Features are standardised with the **training-set** mean/std (stored in `weights.json`); targets are z-scored **within each state**; orders never sampled in any try (should not occur with M = 300) are masked out of loss and metrics.
- Validation = separate instances (not states of training instances), so the validation loss measures generalisation to unseen instances of the same two classes. The test cells are the thesis benchmark instances (never used for training); 10×10 is a class never seen in training.

## 4. Measurements from the smoke runs (not results; single instances, tiny configs)

Iteration speed of the classical operators (1 thread, 5 000 iterations, instance 0 of each cell, Mac mini M4 10-core): 10×10 ≈ 500 000–700 000 it/s, 10×20 ≈ 330 000–440 000 it/s, 20×50 ≈ 55 000–73 000 it/s; repair takes 70–90 % of the time, the destroy step the rest. The neural operator is 5–10× slower per iteration (features + MLP ≈ 90 % of the iteration time in the tiny run with a 64×2 network).

Quality: 5 000 LNS iterations (0.01–0.09 s) already reach the value of a 60 s CEA run on the probed instances (10×10 instance 0: both 2 240.67; 20×50 instance 0: related 9 179 vs CEA 9 212). The CEA converges in ≈ 1 s on 10×10 and uses the full 60 s on 20×50.

Label sparsity (two 10×20 instances, M = 300, even schedule): at iteration 250 only 3 % of random removal tries improved the solution, 92 % reproduced it exactly (improvement 0) and 5 % made it worse; from iteration 1 500 on, 0 % improved. This motivated the log schedule (decision 9) and is the central label-noise issue for the error analysis.

## 5. Machine

Apple Mac mini (M4, 10 cores, 16 GB; **not** the 14-core / 64 GB machine named in the plan), macOS, rustc 1.95, Python 3.12 + torch 2.15 (CPU). Exact strings in `results/stage2/machine.txt` once the pipeline has run. The machine also hosts Docker containers, hence the thread caps.

## 6. Initial error analysis (from the smoke runs; to be re-checked against the final tables below)

- **Deterministic top-k neural destroy stalls.** With τ = 0 the operator removes the same k orders every iteration; greedy repair puts them back in the same places; the solution never changes (tiny run: gap +10 % to +20 % after 88 000 iterations, acceptance rate 1.0 = the candidate always equals the current solution). This is a property of any deterministic destroy with a deterministic repair, not of the network. The sampled variant (τ = 1) does not have this problem. For the report this argues for score *sampling* (as in the plan's "or k orders sampled by score") or for a tie-breaking/tabu rule in Stage 3.
- **Iteration cost.** The neural operator recomputes all eight features and the MLP for all n orders every iteration; at 5–10× the cost of random removal it does 5–10× fewer iterations in the time budget. Since the classical LNS converges in well under a second on these cells, the time budget comparison will mostly show the converged values; the iteration budget (5 000) is the meaningful comparison for the decisions.
- **Label noise / sparsity.** Most random removals from a converged state change nothing; the per-order mean improvement is then dominated by the few negative tries, so the labels in late states mostly say "which orders are harmful to remove", not "which are useful". The z-score normalisation keeps the state in the loss, but its ranking information is weak. Expect low Spearman (≈ 0.2–0.4 on the smoke data) and precision@k only moderately above k/n.

## 7. Problems encountered

- `python3` on the machine is 3.15 without torch; a 3.12 venv was created with `uv` (installed with Homebrew).
- `cargo clippy` reports pre-existing warnings in `vrppd-core`, `vrppd-cea`, `vrppd-psa`, `vrppd-milp`, `vrppd-brute-force` (new lints in clippy 1.95: `manual_is_multiple_of`, `collapsible_if`); they were left untouched to keep the diff focused. `vrppd-lns` is warning-free.
- The pre-existing integration test `vrppd-benchmarks/tests/verify_small_instances` fails on `main` as well (it needs the Python module `ortools`, which is not installed); unrelated to this work. All other workspace tests pass (`cargo test --release --workspace` minus that test).
- `cargo fmt --all` would reformat untouched crates (import ordering of the 2024 style edition); only the new crate was formatted.

## 8. What remains for Stage 3

- V1 (learned encoder), V2 (REINFORCE fine-tuning), V3 (destroy size), generalisation to the large cells and the price/empty objectives.
- Full-size training sets (cheap, see decision 10), 10 CEA reference runs, 10 repeats, 60 s budget.
- Better labels: more informative targets for converged states (e.g. larger k for labelling, best-set indicator, or improvement relative to the best of the M tries), and a tabu/tie-break for deterministic top-k.
- Cheaper neural destroy: incremental feature updates (only routes touched by the last repair change) and batching the MLP over orders.
- Statistical tests over instances (Wilcoxon signed-rank), as planned for the final report.
