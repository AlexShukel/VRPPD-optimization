#!/usr/bin/env bash
# Drive the Stage 3 compute host from the Mac.
# Usage: remote.sh [--host <name>] bootstrap | sync | push-state | run <name> <command...>
#                  | status | pull | watch | keepalive
# bootstrap pushes the branch first (never main). Host settings live in scripts/stage3/hosts/<name>.env (default pingwin, or STAGE3_HOST).
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
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=15)

rssh() { ssh "${SSH_OPTS[@]}" "$SSH_TARGET" "$@"; }
rsh_e() { echo "ssh ${SSH_OPTS[*]}"; }
notify() { [[ -x "$NOTIFY" ]] && "$NOTIFY" "$1" >/dev/null 2>&1 || true; }

cmd_bootstrap() {
  # The server can only check out the branch once it is on GitHub (never pushes main).
  git push origin "$BRANCH"
  # The unit tests read instances from problem-gen/problems (not in git): clone, rsync them, then build and test.
  rssh "REPO_URL=$REPO_URL REPO_DIR=$REPO_DIR BRANCH=$BRANCH bash -s" <<'REMOTE'
set -euo pipefail
[[ -d "$REPO_DIR/.git" ]] || git clone "$REPO_URL" "$REPO_DIR"
cd "$REPO_DIR"
git fetch origin
git checkout "$BRANCH"
git reset --hard "origin/$BRANCH"
REMOTE
  rsync -a -e "$(rsh_e)" problem-gen/problems/ "$SSH_TARGET:$REPO_DIR/problem-gen/problems/"
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
  git push origin "$BRANCH"
  rssh "$REMOTE_PATH; cd $REPO_DIR && git fetch && git checkout $BRANCH && git reset --hard origin/$BRANCH"
  rsync -a -e "$(rsh_e)" problem-gen/problems/ "$SSH_TARGET:$REPO_DIR/problem-gen/problems/"
}

cmd_push_state() {
  [[ -d results/stage3 ]] || { echo "no results/stage3 to push" >&2; exit 1; }
  rssh "mkdir -p $REPO_DIR/results/stage3"
  rsync -a -e "$(rsh_e)" results/stage3/ "$SSH_TARGET:$REPO_DIR/results/stage3/"
}

cmd_run() {
  local name="${1:?run <name> <command...>}"; shift
  [[ $# -gt 0 ]] || { echo "run <name> <command...>" >&2; exit 1; }
  local inner="$REMOTE_PATH; cd $REPO_DIR; $*"
  rssh "tmux new-session -d -s $(printf %q "$name") bash -c $(printf %q "$inner")"
  echo "started tmux session $name on $HOST"
}

cmd_status() {
  rssh "tmux ls 2>&1 || true; echo '--- pipeline.log'; tail -n 20 $REPO_DIR/results/stage3/pipeline.log 2>&1 || true; echo '---'; uptime"
}

cmd_pull() {
  mkdir -p results/stage3
  rsync -a -e "$(rsh_e)" --exclude 'instances/' --exclude 'labels/' \
    "$SSH_TARGET:$REPO_DIR/results/stage3/" results/stage3/
}

done_batches() {
  [[ -d results/stage3/batches ]] || return 0
  local f
  for f in results/stage3/batches/*/DONE; do
    [[ -e "$f" ]] && basename "$(dirname "$f")"
  done | sort
}

cmd_watch() {
  local last_restart=0 known now b remaining
  known="$(done_batches)"
  while true; do
    cmd_pull || echo "watch: pull failed" >&2
    for b in $(comm -13 <(echo "$known") <(done_batches)); do
      notify "stage3 batch done: $b ($HOST)"
    done
    known="$(done_batches)"
    now=$(date +%s)
    if ! rssh true 2>/dev/null; then
      echo "watch: $HOST unreachable at $(date '+%F %T')" >&2
    elif ! rssh "tmux has-session -t stage3" 2>/dev/null; then
      remaining=""
      if rssh "test -x $REPO_DIR/scripts/stage3/run_all.sh" 2>/dev/null; then
        remaining="$(rssh "$REMOTE_PATH; cd $REPO_DIR && scripts/stage3/run_all.sh --list" 2>/dev/null || true)"
      fi
      if [[ -n "$remaining" ]] && (( now - last_restart >= 600 )); then
        last_restart=$now
        if cmd_run stage3 "scripts/stage3/run_all.sh >> results/stage3/pipeline.log 2>&1"; then
          notify "stage3 queue restarted on $HOST ($(echo "$remaining" | wc -l | tr -d ' ') batches remain)"
        else
          echo "watch: restart of stage3 failed on $HOST" >&2
        fi
      fi
    fi
    sleep 600
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
  sync) cmd_sync ;;
  push-state) cmd_push_state ;;
  run) cmd_run "$@" ;;
  status) cmd_status ;;
  pull) cmd_pull ;;
  watch) cmd_watch ;;
  keepalive) cmd_keepalive ;;
  *) echo "unknown subcommand $sub" >&2; exit 1 ;;
esac
