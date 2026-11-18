# VRPPD-optimization

## LNS with learned destroy operators (ANN course, Stage 2)

`crates/vrppd-lns` is a Large Neighbourhood Search for the DISTANCE objective:
RSCIM start solution, a pluggable destroy operator (`random`, `worst`, `related`,
`adaptive`, `neural`), greedy cheapest-insertion repair (pickup before delivery,
capacity feasible) and simulated-annealing acceptance. The `neural` operator
scores every order with a small MLP on eight hand-made features (pure-Rust
forward pass, weights loaded from JSON) and removes the top-k.

Binaries (all `cargo run --release -p vrppd-lns --bin <name> -- --help`):

- `lns` — run one operator on a cell: `lns --cell 10_20 --operator adaptive --iters 5000 --repeats 5 --out results/x`
- `label` — supervised labels (JSONL, one search state per line) from random-removal LNS states
- `gen_instances` — seeded instance generator sampling the same pools as `problem-gen`
- `cea_ref` — CEA reference values (best of N runs)

`python/vrppd_nn` (torch, CPU, 4 threads) trains the B0 MLP on the labels
(`python -m vrppd_nn.train`), reports label statistics (`label_stats`) and
aggregates results (`aggregate`). Environment: `uv venv --python 3.12 .venv &&
uv pip install --python .venv/bin/python -r python/requirements.txt`.

The whole Stage 2 experiment is `scripts/stage2/run_all.sh` (idempotent, each
stage skipped if its output exists; `STAGE2_TINY=1` runs a smoke config). It
writes to `results/stage2/` and appends the result tables to
`results/stage2/REPORT-NOTES.md`.

## Stage 3: reproduce

Prerequisites: Rust (stable), Python 3.12 with `uv`, and on the compute host a CUDA
build of torch. The test instances (`problem-gen/problems/`) and the generator pools
(`problem-gen/data/`) are not in git; copy them next to the code first.

One host file per machine, `scripts/stage3/hosts/<name>.env`, sets the CPU list and
thread counts. From the Mac, `scripts/stage3/remote.sh bootstrap` prepares the host
(clone, venv, build, tests) and `remote.sh run stage3 'scripts/stage3/run_all.sh'`
starts the experiment there. The experiment is a queue of resumable batches
(`run_all.sh --list`); restarting it skips finished work, `remote.sh pull` copies
finished batches back and `remote.sh watch` restarts the queue after a reboot. On a
single machine: `STAGE3_HOST=<name> scripts/stage3/run_all.sh`.
`SMOKE=1 scripts/stage3/run_all.sh` runs a tiny configuration in a few minutes.

Outputs land in `results/stage3/`: trained models (`models/`), the choice of model
and temperature made on validation instances (`selection/selection.json`), CEA
references (`reference/`), batch manifests (`batches/`) and raw runs (`lns/`, not in
git). `python -m vrppd_nn.aggregate3 --root results/stage3` writes `summary.md`,
`summary.csv`, `stats.csv`, `training_summary.csv`, `tables/` and `figures/`.
