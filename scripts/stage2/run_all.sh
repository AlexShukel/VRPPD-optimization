#!/usr/bin/env bash
# Stage 2 experiment pipeline (ANN course, interim report).
# Idempotent: every stage is skipped when its output already exists.
# Resource caps: rayon 6 threads, torch 4 threads, CEA 6 threads, stages run sequentially.
#
# Usage:  nohup scripts/stage2/run_all.sh > results/stage2/pipeline.log 2>&1 &
# Tiny smoke config: STAGE2_TINY=1 scripts/stage2/run_all.sh   (writes to results/stage2-tiny)
set -euo pipefail
cd "$(dirname "$0")/../.."
REPO="$(pwd)"

export RAYON_NUM_THREADS=6
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

TINY="${STAGE2_TINY:-0}"
if [[ "$TINY" == "1" ]]; then
  ROOT="${STAGE2_ROOT:-results/stage2-tiny}"
  N_TRAIN_A=2; N_TRAIN_B=2; N_VAL_A=2; N_VAL_B=2
  LABEL_ITERS=50; LABEL_STATES=3; LABEL_M=4
  EPOCHS=2; PATIENCE=2
  LNS_ITERS=50; LNS_SECONDS=1; REPEATS=1; MAX_INST=2
  CEA_RUNS=1; CEA_SECONDS=2
  GRID_WIDTHS="64"; GRID_DEPTHS="2"; GRID_LOSSES="mse rank"; BEST_SEEDS="0 1"
else
  ROOT="${STAGE2_ROOT:-results/stage2}"
  N_TRAIN_A=${N_TRAIN_A:-300}; N_TRAIN_B=${N_TRAIN_B:-150}; N_VAL_A=${N_VAL_A:-30}; N_VAL_B=${N_VAL_B:-15}
  LABEL_ITERS=${LABEL_ITERS:-5000}; LABEL_STATES=${LABEL_STATES:-20}; LABEL_M=${LABEL_M:-300}
  EPOCHS=${EPOCHS:-100}; PATIENCE=${PATIENCE:-10}
  LNS_ITERS=${LNS_ITERS:-5000}; LNS_SECONDS=${LNS_SECONDS:-30}; REPEATS=${REPEATS:-5}; MAX_INST=${MAX_INST:-20}
  CEA_RUNS=${CEA_RUNS:-3}; CEA_SECONDS=${CEA_SECONDS:-60}
  GRID_WIDTHS="64 128"; GRID_DEPTHS="2 3"; GRID_LOSSES="mse rank"; BEST_SEEDS="0 1 2"
fi
TEST_CELLS="10_10 10_20 20_50"
CLASSICAL_OPS="random worst related adaptive"
PY="$REPO/.venv/bin/python"
BIN="$REPO/target/release"
mkdir -p "$ROOT"
TIMINGS="$ROOT/timings.csv"
[[ -f "$TIMINGS" ]] || echo "stage,seconds,finished_at" > "$TIMINGS"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
stage_start() { STAGE_T0=$(date +%s); log "=== stage $1: start"; }
stage_end() { local s=$(( $(date +%s) - STAGE_T0 )); echo "$1,$s,$(date '+%Y-%m-%d %H:%M:%S')" >> "$TIMINGS"; log "=== stage $1: done in ${s}s"; }
check_disk() {
  local avail_gb
  avail_gb=$(df -g / | awk 'NR==2 {print $4}')
  if (( avail_gb < 10 )); then log "ABORT: only ${avail_gb} GB free on /"; exit 1; fi
}

log "pipeline start; root=$ROOT tiny=$TINY pid=$$ host=$(hostname) cores=$(sysctl -n hw.ncpu) mem_gb=$(( $(sysctl -n hw.memsize) / 1073741824 ))"
echo $$ > "$ROOT/pipeline.pid"
check_disk

# machine specs (for the notes)
{
  echo "hostname: $(hostname)"; echo "date: $(date)"; echo "os: $(sw_vers -productName) $(sw_vers -productVersion)";
  echo "cpu: $(sysctl -n machdep.cpu.brand_string) ($(sysctl -n hw.ncpu) cores)"; echo "memory_gb: $(( $(sysctl -n hw.memsize) / 1073741824 ))";
  echo "rustc: $(rustc --version)"; echo "python: $($PY --version) torch $($PY -c 'import torch;print(torch.__version__)')";
  echo "git: $(git rev-parse --short HEAD) ($(git rev-parse --abbrev-ref HEAD))";
  echo "caps: RAYON_NUM_THREADS=$RAYON_NUM_THREADS torch threads=4 CEA threads=6";
  echo "config: train=$N_TRAIN_A+$N_TRAIN_B val=$N_VAL_A+$N_VAL_B label_iters=$LABEL_ITERS states=$LABEL_STATES M=$LABEL_M lns_iters=$LNS_ITERS lns_seconds=$LNS_SECONDS repeats=$REPEATS cea_runs=$CEA_RUNS cea_seconds=$CEA_SECONDS";
} > "$ROOT/machine.txt"

# 0. build
stage_start build
cargo build --release -p vrppd-lns 2>&1 | tail -1
stage_end build

# a. instances
stage_start instances
gen() { # split cell V O count seed
  local dir="$ROOT/instances/$1/$2"
  if [[ -d "$dir" && $(ls "$dir"/*.json 2>/dev/null | wc -l) -ge $5 ]]; then log "skip $dir"; return; fi
  "$BIN/gen_instances" --vehicles "$3" --orders "$4" --count "$5" --seed "$6" --out "$dir"
}
gen train 10_20 10 20 "$N_TRAIN_A" 20261001
gen train 20_50 20 50 "$N_TRAIN_B" 20261002
gen val   10_20 10 20 "$N_VAL_A"   20261003
gen val   20_50 20 50 "$N_VAL_B"   20261004
stage_end instances

# d. labels
stage_start labels
mkdir -p "$ROOT/labels"
for split in train val; do
  out="$ROOT/labels/$split.jsonl"
  if [[ -f "$out" ]]; then log "skip $out"; continue; fi
  for cell in 10_20 20_50; do
    part="$ROOT/labels/${split}_${cell}.jsonl"
    if [[ ! -f "$part" ]]; then
      "$BIN/label" --cell "$ROOT/instances/$split/$cell" --split "$split" --lns-iters "$LABEL_ITERS" \
        --states "$LABEL_STATES" --m "$LABEL_M" --seed "$([[ $split == train ]] && echo 1 || echo 100001)" --schedule log --out "$part" --threads 6
    fi
  done
  cat "$ROOT/labels/${split}_10_20.jsonl" "$ROOT/labels/${split}_20_50.jsonl" > "$out.tmp" && mv "$out.tmp" "$out"
done
for split in train val; do
  if [[ ! -f "$ROOT/labels/stats_$split.json" ]]; then
    (cd python && "$PY" -m vrppd_nn.label_stats --labels "$REPO/$ROOT/labels/$split.jsonl" --out "$REPO/$ROOT/labels/stats_$split.json" --md "$REPO/$ROOT/labels/stats_$split.md")
  fi
done
check_disk
stage_end labels

# e. train B0 grid
stage_start train_grid
mkdir -p "$ROOT/models/grid"
for loss in $GRID_LOSSES; do for w in $GRID_WIDTHS; do for d in $GRID_DEPTHS; do
  out="$ROOT/models/grid/w${w}_d${d}_${loss}_s0"
  if [[ -f "$out/metrics.json" ]]; then log "skip $out"; continue; fi
  (cd python && "$PY" -m vrppd_nn.train --train "$REPO/$ROOT/labels/train.jsonl" --val "$REPO/$ROOT/labels/val.jsonl" \
     --out "$REPO/$out" --width "$w" --depth "$d" --loss "$loss" --seed 0 --epochs "$EPOCHS" --patience "$PATIENCE" > "$REPO/$out.log" 2>&1 || { tail -5 "$REPO/$out.log"; exit 1; })
  log "trained $out: $(tail -1 "$out.log" | cut -c1-200)"
done; done; done
stage_end train_grid

# e'. best config x 3 seeds (selection by validation Spearman, see REPORT-NOTES)
stage_start train_best
if [[ ! -f "$ROOT/models/best_config.json" ]]; then
  "$PY" - "$ROOT/models/grid" "$ROOT/models/best_config.json" <<'EOF'
import glob, json, os, sys
best = None
for d in sorted(glob.glob(os.path.join(sys.argv[1], "*"))):
    mp = os.path.join(d, "metrics.json")
    if not os.path.exists(mp): continue
    m = json.load(open(mp)); c = json.load(open(os.path.join(d, "config.json")))
    row = {"name": os.path.basename(d), "width": c["width"], "depth": c["depth"], "loss": c["loss"], "val_spearman": m["spearman"], "val_p_at_k": m["precision_at_k"], "val_loss": m["best_val_loss"]}
    if best is None or row["val_spearman"] > best["val_spearman"]: best = row
json.dump(best, open(sys.argv[2], "w"), indent=2); print("best config:", best)
EOF
fi
BW=$("$PY" -c "import json;print(json.load(open('$ROOT/models/best_config.json'))['width'])")
BD=$("$PY" -c "import json;print(json.load(open('$ROOT/models/best_config.json'))['depth'])")
BL=$("$PY" -c "import json;print(json.load(open('$ROOT/models/best_config.json'))['loss'])")
mkdir -p "$ROOT/models/best"
for s in $BEST_SEEDS; do
  out="$ROOT/models/best/w${BW}_d${BD}_${BL}_s${s}"
  if [[ -f "$out/metrics.json" ]]; then log "skip $out"; continue; fi
  if [[ "$s" == "0" && -f "$ROOT/models/grid/w${BW}_d${BD}_${BL}_s0/metrics.json" ]]; then
    cp -r "$ROOT/models/grid/w${BW}_d${BD}_${BL}_s0" "$out"; continue
  fi
  (cd python && "$PY" -m vrppd_nn.train --train "$REPO/$ROOT/labels/train.jsonl" --val "$REPO/$ROOT/labels/val.jsonl" \
     --out "$REPO/$out" --width "$BW" --depth "$BD" --loss "$BL" --seed "$s" --epochs "$EPOCHS" --patience "$PATIENCE" > "$REPO/$out.log" 2>&1 || { tail -5 "$REPO/$out.log"; exit 1; })
  log "trained $out: $(tail -1 "$out.log" | cut -c1-200)"
done
# the operator uses the seed-0 weights of the best config
cp "$ROOT/models/best/w${BW}_d${BD}_${BL}_s0/weights.json" "$ROOT/models/b0_best.json"
WEIGHTS="$REPO/$ROOT/models/b0_best.json"
stage_end train_best

run_lns() { # budget_name op cell extra-args...
  local budget="$1" op="$2" cell="$3"; shift 3
  local out="$ROOT/lns/$budget/$op/$cell"
  local expect=$(( MAX_INST * REPEATS ))
  if [[ -d "$out" && $(ls "$out"/*.json 2>/dev/null | wc -l) -ge $expect ]]; then log "skip $out"; return; fi
  "$BIN/lns" --cell "$cell" --operator "${op%%_*}" --repeats "$REPEATS" --max-instances "$MAX_INST" --seed 1 --out "$out" --threads 6 "$@"
}

# c1/f1. iteration budget (fast)
stage_start lns_iters
for cell in $TEST_CELLS; do
  for op in $CLASSICAL_OPS; do run_lns "iters${LNS_ITERS}" "$op" "$cell" --iters "$LNS_ITERS" --log-every 10; done
  run_lns "iters${LNS_ITERS}" neural "$cell" --iters "$LNS_ITERS" --log-every 10 --weights "$WEIGHTS" --tau 0
  run_lns "iters${LNS_ITERS}" neural_tau "$cell" --iters "$LNS_ITERS" --log-every 10 --weights "$WEIGHTS" --tau 1
done
stage_end lns_iters

# b. CEA reference values (sequential; CEA uses 6 threads)
stage_start cea_reference
mkdir -p "$ROOT/reference"
for cell in $TEST_CELLS; do
  out="$ROOT/reference/cea_ref_$cell.json"
  if [[ -f "$out" && $("$PY" -c "import json;print(len(json.load(open('$out'))))") -ge "$MAX_INST" ]]; then log "skip $out"; continue; fi
  "$BIN/cea_ref" --cell "$cell" --runs "$CEA_RUNS" --seconds "$CEA_SECONDS" --threads 6 --max-instances "$MAX_INST" --out "$out"
done
stage_end cea_reference

# c2/f2. wall-time budget
stage_start lns_time
for cell in $TEST_CELLS; do
  for op in $CLASSICAL_OPS; do run_lns "time${LNS_SECONDS}s" "$op" "$cell" --iters 0 --seconds "$LNS_SECONDS" --log-every 0 --log-every-ms 100; done
  run_lns "time${LNS_SECONDS}s" neural "$cell" --iters 0 --seconds "$LNS_SECONDS" --log-every 0 --log-every-ms 100 --weights "$WEIGHTS" --tau 0
  run_lns "time${LNS_SECONDS}s" neural_tau "$cell" --iters 0 --seconds "$LNS_SECONDS" --log-every 0 --log-every-ms 100 --weights "$WEIGHTS" --tau 1
done
check_disk
stage_end lns_time

# g. aggregate + append to the notes
stage_start aggregate
(cd python && "$PY" -m vrppd_nn.aggregate --root "$REPO/$ROOT" > /dev/null)
NOTES="$ROOT/REPORT-NOTES.md"
{
  echo
  echo "---"
  echo
  echo "# Pipeline results (appended automatically by scripts/stage2/run_all.sh on $(date '+%Y-%m-%d %H:%M'))"
  echo
  echo "## Stage timings (seconds)"
  echo
  echo '```'
  cat "$TIMINGS"
  echo '```'
  echo
  echo "## Machine"
  echo
  echo '```'
  cat "$ROOT/machine.txt"
  echo '```'
  echo
  cat "$ROOT/summary.md"
  echo
  echo "## Figures"
  echo
  for f in "$ROOT"/figures/*.png; do echo "- \`$f\`"; done
} >> "$NOTES"
stage_end aggregate
log "pipeline finished"
