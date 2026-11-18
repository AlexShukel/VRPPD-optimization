#!/usr/bin/env bash
# Stage 3 experiment (ANN course, final results) as a queue of small, resumable batches.
#
# Usage: STAGE3_HOST=<name> scripts/stage3/run_all.sh [--list] [--only <batch>] [--from <batch>]
#        SMOKE=1 scripts/stage3/run_all.sh         tiny configuration, writes results/stage3-smoke
#   --list          print the batch names in queue order and exit
#   --only <batch>  run that batch (and every prerequisite whose outputs are missing here)
#   --from <batch>  run the queue from that batch on
# STAGE3_HOST names scripts/stage3/hosts/<name>.env (TIMED_CPUS, TIMED_THREADS, ITER_THREADS).
# SMOKE=1 without STAGE3_HOST uses a local configuration (host local-smoke, 4 threads, no taskset).
# STAGE3_ROOT overrides the output directory.
#
# A batch is skipped when batches/<batch>/DONE exists. A finished batch runs again only when a
# later batch needs its outputs and they are missing (instances and labels are never pulled from
# a host, so a new host regenerates `data` from the seeds). Inside a batch every existing output
# is skipped: run files, reference records, models with metrics.json, V2 runs with DONE (V2 runs
# always resume). A batch writes batches/<batch>/manifest.json, then DONE, only after checking that
# every output it registered is complete (run files per directory, reference records, finish markers).
# Iteration-budget runs use ITER_THREADS. Wall-time work (60 s runs, CEA references) runs under
# `taskset -c $TIMED_CPUS` with TIMED_THREADS. A wall-time batch started on another host is refused,
# and so are wall-time runs whose class's CEA references (or earlier runs) come from another
# host, thread count or CPU list. One pipeline at a time: lock <root>/pipeline.pid.
set -euo pipefail
shopt -s nullglob inherit_errexit
cd "$(dirname "$0")/../.."
REPO="$(pwd)"

BATCHES=(data b0 v1 ref-10_10 ref-10_20 ref-20_50 v2-a v2-b v2-c v2-d v2-e select
  iters-10_10 iters-10_20 iters-20_50 time-10_10 time-10_20 time-20_50 ksweep
  large-30_100 large-50_200 large-100_500 obj-price obj-empty)

log() { echo "[$(date '+%F %T')] $*"; }
die() {
  echo "[$(date '+%F %T')] run_all: $*" >&2
  exit 1
}
is_batch() {
  local b
  for b in "${BATCHES[@]}"; do [[ $b == "$1" ]] && return 0; done
  return 1
}

ONLY="" FROM=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --list) printf '%s\n' "${BATCHES[@]}"; exit 0 ;;
    --only) ONLY="${2:?--only needs a batch}"; shift 2 ;;
    --from) FROM="${2:?--from needs a batch}"; shift 2 ;;
    *) die "unknown argument $1 (usage in the header of $0)" ;;
  esac
done
[[ -z $ONLY ]] || is_batch "$ONLY" || die "unknown batch $ONLY"
[[ -z $FROM ]] || is_batch "$FROM" || die "unknown batch $FROM"

# ---------------------------------------------------------------- host and configuration
SMOKE="${SMOKE:-0}"
if [[ -n "${STAGE3_HOST:-}" ]]; then
  HOST_FILE="scripts/stage3/hosts/$STAGE3_HOST.env"
  [[ -f $HOST_FILE ]] || die "no host file $HOST_FILE"
  # shellcheck disable=SC1090
  source "$HOST_FILE"
  HOST_NAME="$STAGE3_HOST"
elif [[ $SMOKE == 1 ]]; then
  HOST_NAME=local-smoke TIMED_CPUS="" TIMED_THREADS=4 ITER_THREADS=4
else
  die "set STAGE3_HOST=<name> (scripts/stage3/hosts/<name>.env), or SMOKE=1 for the local smoke configuration"
fi
TIMED_CPUS="${TIMED_CPUS:-}"
CPU_LIST="${TIMED_CPUS:-none}"
if [[ -n $TIMED_CPUS ]]; then
  command -v taskset > /dev/null || die "TIMED_CPUS is set but taskset is missing"
fi

if [[ $SMOKE == 1 ]]; then
  ROOT="${STAGE3_ROOT:-results/stage3-smoke}"
  N_TRAIN_A=4 N_TRAIN_B=2 N_VAL_A=3 N_VAL_B=2 VAL_SEARCH=2
  LABEL_ITERS=100 LABEL_STATES=4 LABEL_M=20
  EPOCHS=2 PATIENCE=2
  B0_WIDTHS="64" B0_DEPTHS="2" LOSSES="mse rank" TARGETS="mean top" TRAIN_SIZES="2 6" SEEDS="0 1"
  V1_DS="16" V1_NATTS="1 2" V1_DROPOUTS="0 0.1" V1_WDS="0 1e-4"
  V2_MAX_HOURS=0.03 V2_EXTRA="--batch-envs 2 --horizon 5 --max-episodes 2"
  SEL_TAUS="0.1 1.0" SEL_ITERS=100 SEL_REPEATS=1
  TEST_INST=2 REPEATS=2 ITERS=200 TIME_BUDGET=2
  REF_RUNS=1 REF_SECONDS=2 LARGE_REF_RUNS=1 OBJ_REF_RUNS=1
  KFRACS="0.05 0.25"
  LARGE_INST=1 LARGE_REPEATS=1 PROBE_ITERS=20
else
  ROOT="${STAGE3_ROOT:-results/stage3}"
  N_TRAIN_A=2000 N_TRAIN_B=1000 N_VAL_A=200 N_VAL_B=100 VAL_SEARCH=20
  # 20 log-spaced snapshots give 19 distinct states per instance (as in Stage 2)
  LABEL_ITERS=5000 LABEL_STATES=20 LABEL_M=1000
  EPOCHS=100 PATIENCE=10
  B0_WIDTHS="64 128 256" B0_DEPTHS="2 3" LOSSES="mse rank" TARGETS="mean top" TRAIN_SIZES="500 1000 3000" SEEDS="0 1 2"
  V1_DS="64 128" V1_NATTS="1 2" V1_DROPOUTS="0 0.1" V1_WDS="0 1e-4"
  V2_MAX_HOURS=3 V2_EXTRA=""
  SEL_TAUS="0.1 0.25 0.5 1.0" SEL_ITERS=5000 SEL_REPEATS=3
  TEST_INST=20 REPEATS=10 ITERS=5000 TIME_BUDGET=60
  REF_RUNS=10 REF_SECONDS=60 LARGE_REF_RUNS=3 OBJ_REF_RUNS=3
  KFRACS="0.05 0.10 0.15 0.25 0.35"
  LARGE_INST=10 LARGE_REPEATS=5 PROBE_ITERS=100
fi
V1_BASE_DROPOUT=0.1 V1_BASE_WD=1e-4 # the size grid's regularisation (train defaults)
LARGE_MAX_RUN_S=600 LARGE_CUT_ITERS=1000
TEST_CELLS="10_10 10_20 20_50"
OPERATORS="random worst related adaptive b0 v1 v2 adaptive_n"
PY="$REPO/.venv/bin/python"
BIN="$REPO/target/release"
INST="$ROOT/instances" LABELS="$ROOT/labels" MODELS="$ROOT/models" LNS="$ROOT/lns" REFS="$ROOT/reference"
export RAYON_NUM_THREADS=$((ITER_THREADS > TIMED_THREADS ? ITER_THREADS : TIMED_THREADS))
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

# ---------------------------------------------------------------- helpers
json_get() { # file dotted.key
  "$PY" -c 'import json, sys
v = json.load(open(sys.argv[1]))
for k in sys.argv[2].split("."):
    v = v[k]
print(v)' "$1" "$2"
}
cfg_fields() { "$PY" -c 'import json, sys; c = json.load(open(sys.argv[1])); print(*[c[k] for k in sys.argv[2:]])' "$@"; }
sel() { json_get "$ROOT/selection/selection.json" "$1"; }
pick_best() { "$PY" -m vrppd_nn.aggregate3 pick-best "$@"; }
count_json() {
  if [[ -d $1 ]]; then find "$1" -maxdepth 1 -name '*.json' | wc -l | tr -d ' '; else echo 0; fi
}
pin() { if [[ -n $TIMED_CPUS ]]; then taskset -c "$TIMED_CPUS" "$@"; else "$@"; fi; }
cpu_model() {
  if [[ -r /proc/cpuinfo ]]; then grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ *//'
  else sysctl -n machdep.cpu.brand_string 2> /dev/null || echo unknown; fi
}
git_commit() { echo "$(git rev-parse --short HEAD)$(git diff --quiet HEAD -- . 2> /dev/null || echo -dirty)"; }
check_disk() {
  local gb
  gb=$(df -Pk "$ROOT" | awk 'NR==2 {print int($4 / 1048576)}')
  ((gb >= 10)) || die "only ${gb} GB free under $ROOT"
}

# What a batch must leave behind ("runs|dir|n": exactly n run files, "refs|file|n|runs": n reference
# records of `runs` runs each, "file|path"); checked before DONE is written.
expect() {
  local IFS='|'
  EXPECT+=("$*")
}

verify_outputs() {
  local e kind path n runs missing=()
  for e in "${EXPECT[@]}"; do
    IFS='|' read -r kind path n runs <<< "$e"
    case $kind in
      runs) [[ $(count_json "$path") == "$n" ]] || missing+=("$path: $(count_json "$path") run files, expected $n") ;;
      file) [[ -f $path ]] || missing+=("$path: missing") ;;
      refs)
        "$PY" - "$path" "$n" "$runs" << 'EOF' || missing+=("$path: expected $n records of $runs runs")
import json, os, sys
path, n, runs = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
records = json.load(open(path)) if os.path.exists(path) else []
sys.exit(0 if len(records) == n and all(len(r["runs"]) == runs for r in records) else 1)
EOF
        ;;
    esac
  done
  if ((${#missing[@]})); then
    printf '  %s\n' "${missing[@]}" >&2
    die "batch $BATCH is incomplete (${#missing[@]} outputs above); DONE not written"
  fi
}

# Wall-time results of one class must share one setup: the class's CEA reference records and any
# run files already in the given directories must match this host, TIMED_THREADS and CPU_LIST.
check_timed_setup() { # reference-file run-dir...
  "$PY" - "$HOST_NAME" "$TIMED_THREADS" "$CPU_LIST" "$@" << 'EOF' || die "the wall-time files above do not match this setup (host $HOST_NAME, threads $TIMED_THREADS, cpus $CPU_LIST): rerun the class's references here (delete $1 and its batch directory under batches/, and the listed run directories), then restart"
import glob, json, os, sys
want = (sys.argv[1], int(sys.argv[2]), sys.argv[3])
ref, dirs = sys.argv[4], sys.argv[5:]
if not os.path.exists(ref):
    sys.exit(f"  {ref}: missing (a class's references must exist before its wall-time runs)")
found = [(f"{ref} instance {r['instance']}", r) for r in json.load(open(ref))]
found += [(p, json.load(open(p))) for d in dirs for p in sorted(glob.glob(os.path.join(d, "*.json")))]
bad = [f"  {name}: host={r.get('host', '')!r} threads={r.get('threads', 0)} cpus={r.get('cpu_list', '')!r}"
       for name, r in found if (r.get("host", ""), r.get("threads", 0), r.get("cpu_list", "")) != want]
if bad:
    more = [f"  ... {len(bad) - 10} more"] if len(bad) > 10 else []
    print("\n".join(bad[:10] + more), file=sys.stderr)
    sys.exit(1)
EOF
}

run_lns() { # out iters|timed lns-args (with --max-instances, optionally --repeats)
  local out=$1 mode=$2 n="" r=1 a prev=""
  shift 2
  for a in "$@"; do
    case $prev in --max-instances) n=$a ;; --repeats) r=$a ;; esac
    prev=$a
  done
  [[ -n $n ]] || die "run_lns $out: --max-instances missing"
  expect runs "$out" $((n * r))
  if [[ $mode == timed ]]; then
    pin "$BIN/lns" "$@" --out "$out" --threads "$TIMED_THREADS" --host-tag "$HOST_NAME" --cpu-list "$CPU_LIST"
  else
    "$BIN/lns" "$@" --out "$out" --threads "$ITER_THREADS" --host-tag "$HOST_NAME" --cpu-list none
  fi
  RUNS=$((RUNS + $(count_json "$out")))
}

run_ref() { # out cell runs instances objective
  expect refs "$1" "$4" "$3"
  pin "$BIN/cea_ref" --cell "$2" --objective "$5" --runs "$3" --seconds "$REF_SECONDS" --threads "$TIMED_THREADS" \
    --max-instances "$4" --out "$1" --host-tag "$HOST_NAME" --cpu-list "$CPU_LIST"
  RUNS=$((RUNS + $("$PY" -c 'import json, sys; print(len(json.load(open(sys.argv[1]))))' "$1")))
}

op_args() { # operator -> OP_ARGS (neural operators take weights and tau from selection.json)
  case $1 in
    random | worst | related | adaptive) OP_ARGS=(--operator "$1") ;;
    b0 | v1) OP_ARGS=(--operator neural --weights "$ROOT/$(sel "models.$1.weights")" --tau "$(sel "models.$1.tau")") ;;
    v2) OP_ARGS=(--operator neural --weights "$ROOT/$(sel v2.weights)" --tau "$(sel v2.tau)") ;;
    adaptive_n) OP_ARGS=(--operator adaptive_n --weights "$ROOT/$(sel best_model.weights)" --tau "$(sel best_model.tau)") ;;
    *) die "unknown operator $1" ;;
  esac
}

train_model() { # module out train-args...; models are ranked by precision at k against the best try,
  # which does not depend on the training target (mean or top)
  local module=$1 out=$2
  shift 2
  RUNS=$((RUNS + 1))
  expect file "$out/metrics.json"
  if [[ -f $out/metrics.json ]]; then return; fi
  mkdir -p "$out"
  "$PY" -m "vrppd_nn.$module" --train "$LABELS/train.jsonl" --val "$LABELS/val.jsonl" --out "$out" --device auto \
    --select-by best_try --epochs "$EPOCHS" --patience "$PATIENCE" "$@" > "$out/train.log" 2>&1 || {
    tail -5 "$out/train.log"
    die "training $out failed"
  }
  log "trained $out ($(grep -c '^epoch' "$out/train.log") epochs)"
}

write_best() { # path best seeds trainsize ("n=dir ...")
  expect file "$1"
  "$PY" - "$@" << 'EOF'
import json, os, sys
path, best, seeds, sizes = sys.argv[1:5]
d = {"best": best, "seeds": seeds.split()}
if sizes.strip():
    d["trainsize"] = dict(s.split("=", 1) for s in sizes.split())
with open(path + ".tmp", "w") as f:
    json.dump(d, f, indent=2)
os.replace(path + ".tmp", path)
EOF
}

# ---------------------------------------------------------------- batches
gen() { # split cell vehicles orders count seed
  local dir="$INST/$1/$2"
  if (($(count_json "$dir") >= $5)); then return; fi
  "$BIN/gen_instances" --vehicles "$3" --orders "$4" --count "$5" --seed "$6" --out "$dir"
}

batch_data() {
  gen train 10_20 10 20 "$N_TRAIN_A" 20261111
  gen train 20_50 20 50 "$N_TRAIN_B" 20261112
  gen val 10_20 10 20 "$N_VAL_A" 20261113
  gen val 20_50 20 50 "$N_VAL_B" 20261114
  RUNS=$((N_TRAIN_A + N_TRAIN_B + N_VAL_A + N_VAL_B))
  expect runs "$INST/train/10_20" "$N_TRAIN_A"
  expect runs "$INST/train/20_50" "$N_TRAIN_B"
  expect runs "$INST/val/10_20" "$N_VAL_A"
  expect runs "$INST/val/20_50" "$N_VAL_B"
  mkdir -p "$LABELS" "$ROOT/label_stats"
  local split cell seed
  for split in train val; do
    seed=$([[ $split == train ]] && echo 1 || echo 100001)
    if [[ ! -f $LABELS/$split.jsonl ]]; then
      for cell in 10_20 20_50; do
        [[ -f $LABELS/${split}_$cell.jsonl ]] || "$BIN/label" --cell "$INST/$split/$cell" --split "$split" \
          --lns-iters "$LABEL_ITERS" --states "$LABEL_STATES" --m "$LABEL_M" --top-frac 0.1 --seed "$seed" \
          --schedule log --out "$LABELS/${split}_$cell.jsonl" --threads "$ITER_THREADS"
      done
      # label writes instances in completion order; a fixed order (by instance, stable within one)
      # keeps training reproducible
      LC_ALL=C sort -s -t, -k1,1 "$LABELS/${split}_10_20.jsonl" "$LABELS/${split}_20_50.jsonl" > "$LABELS/$split.jsonl.tmp"
      mv "$LABELS/$split.jsonl.tmp" "$LABELS/$split.jsonl"
    fi
    expect file "$LABELS/$split.jsonl"
    expect file "$ROOT/label_stats/stats_$split.json"
    [[ -f $ROOT/label_stats/stats_$split.json ]] || "$PY" -m vrppd_nn.label_stats --labels "$LABELS/$split.jsonl" \
      --out "$ROOT/label_stats/stats_$split.json" --md "$ROOT/label_stats/stats_$split.md" > /dev/null
  done
}

batch_b0() {
  local grid="$MODELS/b0/grid" loss target w d n s best name cfg fields bw bd bl bt sizes="" seeds=""
  for loss in $LOSSES; do for target in $TARGETS; do for w in $B0_WIDTHS; do for d in $B0_DEPTHS; do
    train_model train "$grid/w${w}_d${d}_${loss}_${target}_s0" --width "$w" --depth "$d" --loss "$loss" --target "$target" --seed 0
  done; done; done; done
  best=$(pick_best "$grid"/*)
  name=$(basename "$best") cfg=${name%_s0}
  fields=$(cfg_fields "$best/config.json" width depth loss target)
  read -r bw bd bl bt <<< "$fields"
  log "best B0 configuration (validation selection_value): $cfg"
  for n in $TRAIN_SIZES; do
    if ((n >= N_TRAIN_A + N_TRAIN_B)); then # every instance: the grid run itself
      sizes+=" $n=grid/$name"
      continue
    fi
    train_model train "$MODELS/b0/trainsize/${cfg}_n${n}_s0" --width "$bw" --depth "$bd" --loss "$bl" --target "$bt" \
      --seed 0 --train-instances "$n"
    sizes+=" $n=trainsize/${cfg}_n${n}_s0"
  done
  for s in $SEEDS; do
    if [[ $s == 0 ]]; then
      seeds+=" grid/$name"
      continue
    fi
    train_model train "$MODELS/b0/seeds/${cfg}_s$s" --width "$bw" --depth "$bd" --loss "$bl" --target "$bt" --seed "$s"
    seeds+=" seeds/${cfg}_s$s"
  done
  write_best "$MODELS/b0/best.json" "grid/$name" "$seeds" "$sizes"
}

batch_v1() {
  local loss target d a p wd s best rel name cfg fields sd sa bd ba bp bwd seeds=""
  best=$(pick_best "$MODELS/b0/grid"/*)
  fields=$(cfg_fields "$best/config.json" loss target)
  read -r loss target <<< "$fields"
  v1_train() { train_model train_v1 "$1" --instances-root "$INST" --loss "$loss" --target "$target" "${@:2}"; }
  for d in $V1_DS; do for a in $V1_NATTS; do
    v1_train "$MODELS/v1/grid/d${d}_a${a}_do${V1_BASE_DROPOUT}_wd${V1_BASE_WD}_s0" --d "$d" --n-att "$a" --seed 0 \
      --dropout "$V1_BASE_DROPOUT" --weight-decay "$V1_BASE_WD"
  done; done
  best=$(pick_best "$MODELS/v1/grid"/*)
  fields=$(cfg_fields "$best/config.json" d n_att)
  read -r sd sa <<< "$fields"
  for p in $V1_DROPOUTS; do for wd in $V1_WDS; do
    [[ $p == "$V1_BASE_DROPOUT" && $wd == "$V1_BASE_WD" ]] && continue # the size-grid run
    v1_train "$MODELS/v1/reg/d${sd}_a${sa}_do${p}_wd${wd}_s0" --d "$sd" --n-att "$sa" --seed 0 --dropout "$p" --weight-decay "$wd"
  done; done
  best=$(pick_best "$MODELS/v1/grid"/* "$MODELS/v1/reg"/*)
  rel=${best#"$MODELS/v1/"} name=$(basename "$best") cfg=${name%_s0}
  fields=$(cfg_fields "$best/config.json" d n_att dropout weight_decay)
  read -r bd ba bp bwd <<< "$fields"
  log "best V1 configuration (validation selection_value): $rel"
  for s in $SEEDS; do
    if [[ $s == 0 ]]; then
      seeds+=" $rel"
      continue
    fi
    v1_train "$MODELS/v1/seeds/${cfg}_s$s" --d "$bd" --n-att "$ba" --seed "$s" --dropout "$bp" --weight-decay "$bwd"
    seeds+=" seeds/${cfg}_s$s"
  done
  write_best "$MODELS/v1/best.json" "$rel" "$seeds" ""
}

batch_v2() { # a | b | c | d | e
  local out="$MODELS/v2/$1" v1dir fields args=()
  v1dir="$MODELS/v1/$(json_get "$MODELS/v1/best.json" best)"
  RUNS=1
  expect file "$out/DONE"
  expect file "$out/weights.json"
  if [[ -f $out/DONE ]]; then return; fi
  case $1 in
    a) args=(--init "$v1dir/weights.json" --samples 4 --entropy-beta 0.01) ;;
    b) # from scratch with the best V1 architecture and dropout
      fields=$(cfg_fields "$v1dir/config.json" d heads n_att dropout)
      read -r -a args <<< "$fields"
      args=(--d "${args[0]}" --heads "${args[1]}" --n-att "${args[2]}" --dropout "${args[3]}" --samples 4 --entropy-beta 0.01) ;;
    c) args=(--init "$v1dir/weights.json" --samples 1 --entropy-beta 0.01) ;;
    d) args=(--init "$v1dir/weights.json" --samples 8 --entropy-beta 0.01) ;;
    e) args=(--init "$v1dir/weights.json" --samples 4 --entropy-beta 0) ;;
  esac
  mkdir -p "$out"
  # shellcheck disable=SC2086
  "$PY" -m vrppd_nn.reinforce --train-instances-dir "$INST/train" --val-instances-dir "$INST/val" --out "$out" \
    --max-hours "$V2_MAX_HOURS" --seed 0 --device auto --resume "${args[@]}" $V2_EXTRA >> "$out/train.log" 2>&1 || {
    tail -5 "$out/train.log"
    die "V2 run $out failed"
  }
  log "V2 run $1: $(cat "$out/DONE")"
}

batch_select() {
  local specs=() margs=() spec m tau cell x
  specs=("b0=models/b0/$(json_get "$MODELS/b0/best.json" best)/weights.json"
    "v1=models/v1/$(json_get "$MODELS/v1/best.json" best)/weights.json")
  for x in a b c d e; do specs+=("v2-$x=models/v2/$x/weights.json"); done
  for spec in "${specs[@]}"; do
    m=${spec%%=*}
    margs+=(--model "$spec")
    for tau in $SEL_TAUS; do for cell in 10_20 20_50; do
      run_lns "$LNS/select-tau$tau/$m/$cell" iters --cell "$INST/val/$cell" --max-instances "$VAL_SEARCH" \
        --repeats "$SEL_REPEATS" --iters "$SEL_ITERS" --log-every 0 --operator neural --weights "$ROOT/${spec#*=}" --tau "$tau"
    done; done
  done
  expect file "$ROOT/selection/selection.json"
  "$PY" -m vrppd_nn.aggregate3 select --root "$ROOT" --val-dir "$INST/val" "${margs[@]}"
}

batch_iters() { # cell
  local op
  for op in $OPERATORS; do
    op_args "$op"
    run_lns "$LNS/iters$ITERS/$op/$1" iters --cell "$1" --max-instances "$TEST_INST" --repeats "$REPEATS" \
      --iters "$ITERS" --log-every 10 "${OP_ARGS[@]}"
  done
}

batch_time() { # cell
  local op
  check_timed_setup "$REFS/cea_ref_$1.json" "$LNS/time${TIME_BUDGET}s"/*/"$1"
  for op in $OPERATORS; do
    op_args "$op"
    run_lns "$LNS/time${TIME_BUDGET}s/$op/$1" timed --cell "$1" --max-instances "$TEST_INST" --repeats "$REPEATS" \
      --iters 0 --seconds "$TIME_BUDGET" --log-every 0 --log-every-ms 100 "${OP_ARGS[@]}"
  done
}

batch_ref() { # cell; a resumed reference file must not mix setups either
  run_ref "$REFS/cea_ref_$1.json" "$1" "$REF_RUNS" "$TEST_INST" distance
  check_timed_setup "$REFS/cea_ref_$1.json"
}

batch_ksweep() {
  local k op cell best
  best=$(sel best_model.operator)
  for k in $KFRACS; do for op in random adaptive "$best"; do
    op_args "$op"
    for cell in $TEST_CELLS; do
      run_lns "$LNS/ksweep-k$k/$op/$cell" iters --cell "$cell" --max-instances "$TEST_INST" --repeats "$REPEATS" \
        --iters "$ITERS" --log-every 0 --k-frac "$k" "${OP_ARGS[@]}"
    done
  done; done
}

large_iters() { # cell operator -> iteration budget of the class (probe stored in probe.json, kept across restarts)
  local probe="$ROOT/batches/large-$1/probe.json" dir="$ROOT/batches/large-$1/probe"
  if [[ ! -f $probe ]]; then
    op_args "$2"
    "$BIN/lns" --cell "$1" --max-instances 1 --repeats 1 --iters "$PROBE_ITERS" --log-every 0 --threads 1 --force \
      --out "$dir" "${OP_ARGS[@]}" >&2
    "$PY" - "$dir" "$probe" "$2" "$ITERS" "$LARGE_MAX_RUN_S" "$LARGE_CUT_ITERS" << 'EOF'
import glob, json, os, sys
d, out, op = sys.argv[1:4]
full, limit, cut = int(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6])
with open(glob.glob(os.path.join(d, "*.json"))[0]) as f:
    run = json.load(f)
rate = 1000 * run["iterations"] / run["elapsed_ms"]
p = {"operator": op, "probe_iters": run["iterations"], "iters_per_s": rate, "full_iters": full,
     "projected_s": full / rate, "limit_s": limit, "iterations": full if full / rate <= limit else cut}
with open(out + ".tmp", "w") as f:
    json.dump(p, f, indent=2)
os.replace(out + ".tmp", out)
EOF
  fi
  json_get "$probe" iterations
}

batch_large() { # cell: references, then both budgets for random, adaptive, best_model and adaptive_n
  local best op iters=$ITERS
  best=$(sel best_model.operator)
  run_ref "$REFS/cea_ref_$1.json" "$1" "$LARGE_REF_RUNS" "$LARGE_INST" distance
  if [[ $1 == 100_500 ]]; then
    iters=$(large_iters "$1" "$best")
    expect file "$ROOT/batches/large-$1/probe.json"
    log "100_500 iteration budget: $iters ($(cat "$ROOT/batches/large-$1/probe.json" | tr -d '\n '))"
  fi
  for op in random adaptive "$best" adaptive_n; do
    op_args "$op"
    run_lns "$LNS/iters$iters/$op/$1" iters --cell "$1" --max-instances "$LARGE_INST" --repeats "$LARGE_REPEATS" \
      --iters "$iters" --log-every 10 "${OP_ARGS[@]}"
  done
  check_timed_setup "$REFS/cea_ref_$1.json" "$LNS/time${TIME_BUDGET}s"/*/"$1"
  for op in random adaptive "$best" adaptive_n; do
    op_args "$op"
    run_lns "$LNS/time${TIME_BUDGET}s/$op/$1" timed --cell "$1" --max-instances "$LARGE_INST" --repeats "$LARGE_REPEATS" \
      --iters 0 --seconds "$TIME_BUDGET" --log-every 0 --log-every-ms 100 "${OP_ARGS[@]}"
  done
}

batch_obj() { # price | empty: references, then the four operators on the small test classes
  local op cell best
  best=$(sel best_model.operator)
  for cell in $TEST_CELLS; do run_ref "$REFS/objective_$1/cea_ref_$cell.json" "$cell" "$OBJ_REF_RUNS" "$TEST_INST" "$1"; done
  for op in random adaptive "$best" adaptive_n; do
    op_args "$op"
    for cell in $TEST_CELLS; do
      run_lns "$LNS/objective_$1/$op/$cell" iters --cell "$cell" --objective "$1" --max-instances "$TEST_INST" \
        --repeats "$REPEATS" --iters "$ITERS" --log-every 0 "${OP_ARGS[@]}"
    done
  done
}

# ---------------------------------------------------------------- queue
is_done() { [[ -f $ROOT/batches/$1/DONE ]]; }
is_timed() { [[ $1 == ref-* || $1 == time-* || $1 == large-* || $1 == obj-* ]]; }

outputs_ok() { # what later batches read from a finished batch is present here
  local x
  case $1 in
    data) (($(count_json "$INST/train/10_20") >= N_TRAIN_A && $(count_json "$INST/train/20_50") >= N_TRAIN_B &&
      $(count_json "$INST/val/10_20") >= N_VAL_A && $(count_json "$INST/val/20_50") >= N_VAL_B)) &&
      [[ -f $LABELS/train.jsonl && -f $LABELS/val.jsonl ]] ;;
    b0 | v1) [[ -f $MODELS/$1/best.json ]] ;;
    v2-*) x=${1#v2-} && [[ -f $MODELS/v2/$x/DONE && -f $MODELS/v2/$x/weights.json ]] ;;
    ref-*) [[ -f $REFS/cea_ref_${1#ref-}.json ]] ;;
    select) [[ -f $ROOT/selection/selection.json ]] ;;
    *) return 0 ;;
  esac
}

prereqs() {
  case $1 in
    b0) echo data ;;
    v1) echo data b0 ;;
    v2-*) echo data v1 ;;
    select) echo data b0 v1 v2-a v2-b v2-c v2-d v2-e ;;
    time-*) echo select "ref-${1#time-}" ;; # wall-time runs need their class's references first
    iters-* | ksweep | large-* | obj-*) echo select ;;
  esac
}

run_batch() {
  local b=$1 d="$ROOT/batches/$1" first
  mkdir -p "$d"
  rm -f "$d/DONE"
  if [[ -f $d/STARTED ]]; then
    first=$(cut -d' ' -f1 < "$d/STARTED")
    if [[ $first != "$HOST_NAME" ]]; then
      is_timed "$b" && die "batch $b was started on host $first: wall-time results never mix machines; delete its outputs and $d, then run it again"
      echo "$HOST_NAME $(date '+%FT%T')" > "$d/STARTED"
    fi
  else
    echo "$HOST_NAME $(date '+%FT%T')" > "$d/STARTED"
  fi
  BATCH=$b BATCH_T0=$(date '+%FT%T') RUNS=0 EXPECT=()
  log "=== batch $b: start"
  case $b in
    data) batch_data ;;
    b0) batch_b0 ;;
    v1) batch_v1 ;;
    ref-*) batch_ref "${b#ref-}" ;;
    v2-*) batch_v2 "${b#v2-}" ;;
    select) batch_select ;;
    iters-*) batch_iters "${b#iters-}" ;;
    time-*) batch_time "${b#time-}" ;;
    ksweep) batch_ksweep ;;
    large-*) batch_large "${b#large-}" ;;
    obj-*) batch_obj "${b#obj-}" ;;
  esac
  verify_outputs
  "$PY" - "$d/manifest.json" "$b" "$HOST_NAME" "$(cpu_model)" "$TIMED_CPUS" "$ITER_THREADS" "$TIMED_THREADS" \
    "$(git_commit)" "$BATCH_T0" "$(date '+%FT%T')" "$RUNS" "$SMOKE" << 'EOF'
import json, os, sys
path, batch, host, cpu, taskset, it, tt, commit, start, end, runs, smoke = sys.argv[1:]
m = {"batch": batch, "host": host, "cpu_model": cpu, "taskset": taskset,
     "threads": {"iterations": int(it), "timed": int(tt)}, "commit": commit, "start": start, "end": end,
     "runs": int(runs), "smoke": smoke == "1"}
with open(path + ".tmp", "w") as f:
    json.dump(m, f, indent=2)
os.replace(path + ".tmp", path)
EOF
  touch "$d/DONE"
  log "=== batch $b: done ($RUNS runs)"
  check_disk
}

ensure_prereqs() {
  local p
  for p in $(prereqs "$1"); do
    if is_done "$p" && outputs_ok "$p"; then continue; fi
    log "batch $1 needs $p (not done, or its outputs are missing here)"
    ensure_prereqs "$p"
    run_batch "$p"
  done
}

mkdir -p "$ROOT"
LOCK="$ROOT/pipeline.pid"
if [[ -f $LOCK ]]; then
  old=$(cat "$LOCK" 2> /dev/null || true)
  if [[ -n $old ]] && kill -0 "$old" 2> /dev/null && ps -p "$old" -o command= 2> /dev/null | grep -q run_all.sh; then
    die "a pipeline (pid $old) holds $LOCK"
  fi
  log "removing stale lock $LOCK (pid ${old:-?})"
fi
echo $$ > "$LOCK.tmp" && mv "$LOCK.tmp" "$LOCK"
trap 'if [[ "$(cat "$LOCK" 2> /dev/null)" == "$$" ]]; then rm -f "$LOCK"; fi' EXIT

log "pipeline start: root=$ROOT host=$HOST_NAME smoke=$SMOKE pid=$$ iter_threads=$ITER_THREADS timed_threads=$TIMED_THREADS taskset=$CPU_LIST"
check_disk
cargo build --release -p vrppd-lns 2>&1 | tail -1

if [[ -n $ONLY ]]; then
  QUEUE=("$ONLY")
else
  QUEUE=() on=$([[ -z $FROM ]] && echo 1 || echo 0)
  for b in "${BATCHES[@]}"; do
    [[ $b == "$FROM" ]] && on=1
    ((on)) && QUEUE+=("$b")
  done
fi
for b in "${QUEUE[@]}"; do
  if is_done "$b"; then
    log "skip $b (done)"
    continue
  fi
  ensure_prereqs "$b"
  run_batch "$b"
done
log "queue finished"
if [[ $SMOKE == 1 ]]; then "$PY" -m vrppd_nn.aggregate3 --root "$ROOT"; fi
