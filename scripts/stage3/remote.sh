#!/usr/bin/env bash
# Drive the Stage 3 compute host from the Mac.
# Usage: remote.sh [--host <name>] bootstrap | sync [--force] | push-state | run <name> '<command>'
#                  | status | pull | watch | keepalive
# bootstrap pushes the branch first (never main). Host settings live in scripts/stage3/hosts/<name>.env (default pingwin, or STAGE3_HOST).
# run takes the command as ONE string (quote it); it runs in tmux session <name> with STAGE3_HOST exported.
# SMOKE=1 makes status, pull, push-state and watch use results/stage3-smoke and the smoke queue
# (notifications prefixed "[stage3 test]"); WATCH_INTERVAL sets watch's period in seconds (default 600).
# sync refuses while a pipeline (run_all.sh) holds a lock under results/ on the host: stop the queue
# first, or pass --force. watch gives up restarting the queue (one notification) after 3 restarts
# without a newly finished batch; a finished batch resets the count.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/../.."
LOCAL_REPO="$(pwd)"

HOST="${STAGE3_HOST:-pingwin}"
if [[ "${1:-}" == "--host" ]]; then
  HOST="${2:?--host needs a name}"
  shift 2
fi
HOST_FILE="$HERE/hosts/$HOST.env"
[[ -f "$HOST_FILE" ]] || { echo "unknown host $HOST" >&2; exit 1; }
# shellcheck disable=SC1090
source "$HOST_FILE"

BRANCH=feat/neural-lns-stage3
REPO_URL=https://github.com/AlexShukel/VRPPD-optimization.git
REMOTE_PATH='export PATH=$HOME/.cargo/bin:$HOME/.local/bin:$PATH'
NOTIFY="$HOME/Git/halo/scripts/notify-telegram.sh"
if [[ "${SMOKE:-0}" == 1 ]]; then
  ROOT=results/stage3-smoke QUEUE="SMOKE=1 scripts/stage3/run_all.sh" NOTIFY_PREFIX="[stage3 test] "
else
  ROOT=results/stage3 QUEUE="scripts/stage3/run_all.sh" NOTIFY_PREFIX=""
fi
WATCH_INTERVAL="${WATCH_INTERVAL:-600}"
MAX_RESTARTS=3 # watch: restarts without a newly finished batch before it gives up
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=15)
# Run in the repository on the host: prints each pipeline lock under results/ that a live
# run_all.sh holds (the check run_all.sh makes on its own lock), nothing when none does.
# shellcheck disable=SC2016 # expanded by the host's shell
LIVE_PIPELINES='for f in results/*/pipeline.pid; do
  [ -f "$f" ] || continue
  pid=$(cat "$f" 2>/dev/null || true)
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && ps -p "$pid" -o command= 2>/dev/null | grep -q run_all.sh; then
    echo "$f (pid $pid)"
  fi
done'

rssh() { ssh "${SSH_OPTS[@]}" "$SSH_TARGET" "$@"; }
rsh_e() { echo "ssh ${SSH_OPTS[*]}"; }
notify() { [[ -x "$NOTIFY" ]] && "$NOTIFY" "$NOTIFY_PREFIX$1" >/dev/null 2>&1 || true; }
# Inputs that are not in git: test instances, and the generator pools gen_instances samples from.
rsync_inputs() {
  rsync -a -e "$(rsh_e)" problem-gen/problems/ "$SSH_TARGET:$REPO_DIR/problem-gen/problems/"
  rsync -a -e "$(rsh_e)" problem-gen/data/ "$SSH_TARGET:$REPO_DIR/problem-gen/data/"
}

cmd_bootstrap() {
  # The server can only check out the branch once it is on GitHub (never pushes main).
  git push origin "$BRANCH"
  # The unit tests read instances from problem-gen/problems (not in git): clone, rsync the inputs, then build and test.
  rssh "REPO_URL=$REPO_URL REPO_DIR=$REPO_DIR BRANCH=$BRANCH bash -s" <<'REMOTE'
set -euo pipefail
[[ -d "$REPO_DIR/.git" ]] || git clone "$REPO_URL" "$REPO_DIR"
cd "$REPO_DIR"
git fetch origin
git checkout "$BRANCH"
git reset --hard "origin/$BRANCH"
REMOTE
  rsync_inputs
  rssh "REPO_URL=$REPO_URL REPO_DIR=$REPO_DIR BRANCH=$BRANCH TORCH_INDEX_URL=$TORCH_INDEX_URL REQUIRE_CUDA=${REQUIRE_CUDA:-1} bash -s" <<'REMOTE'
set -euo pipefail
export PATH=$HOME/.cargo/bin:$HOME/.local/bin:$PATH
cd "$REPO_DIR"
cargo build --release -p vrppd-lns
[[ -d .venv ]] || uv venv .venv --python 3.12
# torch comes from the CUDA wheel index (uv's default first-index strategy: PyPI cannot override it).
uv pip install -p .venv/bin/python --index-url "$TORCH_INDEX_URL" --extra-index-url https://pypi.org/simple torch
uv pip install -p .venv/bin/python -e python pytest maturin
if [[ -f crates/vrppd-py/Cargo.toml ]]; then
  VIRTUAL_ENV="$PWD/.venv" .venv/bin/maturin develop --release -m crates/vrppd-py/Cargo.toml
else
  echo "bootstrap: crates/vrppd-py missing, skipping binding build"
fi
cargo test --release -p vrppd-lns
if [[ -d python/tests ]]; then
  (cd python && ../.venv/bin/python -m pytest tests -q)
else
  echo "bootstrap: python/tests missing, skipping Python tests"
fi
REQUIRE_CUDA="$REQUIRE_CUDA" .venv/bin/python -c "
import os, torch
ok = torch.cuda.is_available()
print(ok, torch.cuda.get_device_name(0) if ok else 'no GPU')
assert ok or os.environ['REQUIRE_CUDA'] != '1', 'REQUIRE_CUDA=1 but torch.cuda.is_available() is False'
"
REMOTE
}

cmd_sync() {
  local live
  case "${1:-}" in
    "")
      # A running pipeline would continue on the new code (its batches record one commit).
      live="$(rssh "cd $REPO_DIR && $LIVE_PIPELINES")" || {
        echo "sync: cannot check $HOST for a running pipeline (sync --force skips the check)" >&2
        exit 1
      }
      if [[ -n "$live" ]]; then
        echo "sync: a pipeline is running on $HOST ($live): stop the queue first, or pass --force" >&2
        exit 1
      fi ;;
    --force) echo "sync: --force: not checking $HOST for a running pipeline" >&2 ;;
    *) echo "usage: remote.sh sync [--force]" >&2; exit 1 ;;
  esac
  git push origin "$BRANCH"
  rssh "$REMOTE_PATH; cd $REPO_DIR && git fetch && git checkout $BRANCH && git reset --hard origin/$BRANCH"
  rsync_inputs
}

cmd_push_state() {
  [[ -d $ROOT ]] || { echo "no $ROOT to push" >&2; exit 1; }
  rssh "mkdir -p $REPO_DIR/$ROOT"
  rsync -a -e "$(rsh_e)" "$ROOT/" "$SSH_TARGET:$REPO_DIR/$ROOT/"
}

cmd_run() {
  local name="${1:?run <name> '<command>'}"
  [[ $# -eq 2 ]] || { echo "run <name> '<command>': the command is one (quoted) string" >&2; exit 1; }
  local inner="$REMOTE_PATH; export STAGE3_HOST=$HOST; cd $REPO_DIR; $2"
  rssh "tmux new-session -d -s $(printf %q "$name") bash -c $(printf %q "$inner")"
  echo "started tmux session $name on $HOST"
}

cmd_status() {
  rssh "tmux ls 2>&1 || true; echo '--- pipeline.log'; tail -n 20 $REPO_DIR/$ROOT/pipeline.log 2>&1 || true; echo '---'; uptime"
}

cmd_pull() {
  mkdir -p "$ROOT"
  # instances/ and labels/ at the top of the results tree are regenerated from seeds on any host
  rsync -a --partial -e "$(rsh_e)" --exclude '/instances/' --exclude '/labels/' \
    "$SSH_TARGET:$REPO_DIR/$ROOT/" "$ROOT/"
}

done_batches() {
  [[ -d $ROOT/batches ]] || return 0
  local f
  for f in "$ROOT"/batches/*/DONE; do
    [[ -e "$f" ]] && basename "$(dirname "$f")"
  done | sort
}

cmd_watch() {
  local last_restart=0 restarts=0 gave_up=0 known finished now b remaining
  known="$(done_batches)"
  while true; do
    cmd_pull || echo "watch: pull failed" >&2
    finished="$(comm -13 <(echo "$known") <(done_batches))"
    for b in $finished; do
      notify "stage3 batch done: $b ($HOST)"
    done
    if [[ -n "$finished" ]]; then # the queue makes progress: restarts count from zero again
      restarts=0 gave_up=0
    fi
    known="$(done_batches)"
    now=$(date +%s)
    if ! rssh true 2>/dev/null; then
      echo "watch: $HOST unreachable at $(date '+%F %T')" >&2
    elif ! rssh "tmux has-session -t stage3" 2>/dev/null; then
      # remaining = every batch of the queue (run_all.sh --list) without a DONE in the pulled copy
      remaining=""
      if rssh "test -x $REPO_DIR/scripts/stage3/run_all.sh" 2>/dev/null; then
        remaining="$(comm -23 <(rssh "cd $REPO_DIR && scripts/stage3/run_all.sh --list" 2>/dev/null | sort) <(done_batches))"
      fi
      if [[ -n "$remaining" ]] && (( now - last_restart >= WATCH_INTERVAL )); then
        if (( restarts >= MAX_RESTARTS )); then
          if (( ! gave_up )); then
            gave_up=1
            echo "watch: gave up restarting stage3 on $HOST at $(date '+%F %T')" >&2
            notify "stage3 watch gave up: $MAX_RESTARTS restarts on $HOST without a finished batch; see $ROOT/pipeline.log there"
          fi
        else
          last_restart=$now
          if cmd_run stage3 "mkdir -p $ROOT && $QUEUE >> $ROOT/pipeline.log 2>&1"; then
            restarts=$((restarts + 1))
            notify "stage3 queue restarted on $HOST ($(echo "$remaining" | wc -l | tr -d ' ') batches remain; restart $restarts of $MAX_RESTARTS without a finished batch)"
          else
            echo "watch: restart of stage3 failed on $HOST" >&2
          fi
        fi
      fi
    fi
    sleep "$WATCH_INTERVAL"
  done
}

cmd_keepalive() {
  while true; do
    ssh "${SSH_OPTS[@]}" -N -o ServerAliveInterval=30 "$SSH_TARGET" || true
    sleep 10
  done
}

sub="${1:-}"
[[ -n "$sub" ]] || { sed -n '2,5p' "$0" >&2; exit 1; }
shift
case "$sub" in
  bootstrap) cmd_bootstrap ;;
  sync) cmd_sync "$@" ;;
  push-state) cmd_push_state ;;
  run) cmd_run "$@" ;;
  status) cmd_status ;;
  pull) cmd_pull ;;
  watch) cmd_watch ;;
  keepalive) cmd_keepalive ;;
  *) echo "unknown subcommand $sub" >&2; exit 1 ;;
esac
