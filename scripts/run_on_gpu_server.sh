#!/usr/bin/env bash
# Run any `meda` command on a GPU server of the local network (e.g. 192.168.x.x),
# or on this machine when no server is reachable or it has no GPU.
#
#   GPU_SERVER=student@192.168.1.50 bash scripts/run_on_gpu_server.sh \
#       meda train -c configs/training/paper_30x30_healthy.yaml
#
#   bash scripts/run_on_gpu_server.sh status     # GPUs of the server, running jobs
#   bash scripts/run_on_gpu_server.sh fetch      # copy the server's runs/ back here
#
# What it does for a command:
#   1. checks that the server answers over ssh and has an NVIDIA GPU
#      (`nvidia-smi`); if not, it runs the command locally instead, where
#      `meda` still uses a local GPU if this machine has one;
#   2. copies this repository (code and configs, not runs/ or .git) to
#      $REMOTE_DIR on the server;
#   3. creates a virtualenv there on first use (it reuses the server's
#      CUDA-enabled PyTorch if one is installed system-wide) and installs
#      the package;
#   4. runs the command there. `meda` picks the server's GPU with the most
#      free memory by itself; set MEDA_DEVICE=cuda:1 to choose one;
#   5. copies the server's runs/ folder (checkpoints, logs, plots) back into
#      this repository's runs/ folder.
#
# DETACH=1 starts the command in the background on the server (it survives
# a closed laptop or ssh session) and returns at once; its log goes to
# runs/remote_logs/ there. Use `status` to follow it and `fetch` to get the
# results later.
#
# Settings (environment variables, or put them in scripts/gpu_server.env):
#   GPU_SERVER   user@host of the server, e.g. student@192.168.1.50 (required
#                for remote runs; with ssh keys set up: ssh-copy-id user@host)
#   REMOTE_DIR   folder on the server           (default: meda-gnn-routing, in $HOME)
#   REMOTE_PY    python on the server           (default: python3)
#   SSH_OPTS     extra ssh options, e.g. "-p 2222 -i ~/.ssh/lab_key"
#   DETACH       1 = run in the background on the server
#   MEDA_DEVICE  forwarded to the server, e.g. cuda:1
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
[[ -f scripts/gpu_server.env ]] && source scripts/gpu_server.env

GPU_SERVER=${GPU_SERVER:-}
REMOTE_DIR=${REMOTE_DIR:-meda-gnn-routing}
REMOTE_PY=${REMOTE_PY:-python3}
SSH=${SSH:-ssh}
read -r -a SSH_ARGS <<< "-o BatchMode=yes -o ConnectTimeout=${SSH_TIMEOUT:-8} ${SSH_OPTS:-}"

remote() { "$SSH" "${SSH_ARGS[@]}" "$GPU_SERVER" "$@"; }

usage() { sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

run_locally() {
  echo "== running on this machine: $*" >&2
  exec "$@"
}

server_has_gpu() {
  [[ -n "$GPU_SERVER" ]] || { echo "== GPU_SERVER is not set" >&2; return 1; }
  if ! remote true 2>/dev/null; then
    echo "== cannot reach $GPU_SERVER over ssh" >&2; return 1
  fi
  if ! remote "nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader" 2>/dev/null; then
    echo "== $GPU_SERVER has no NVIDIA GPU (nvidia-smi failed)" >&2; return 1
  fi
}

upload() {
  echo "== copying the repository to $GPU_SERVER:$REMOTE_DIR" >&2
  tar czf - --exclude=./.git --exclude=./runs --exclude=./.venv --exclude='*/__pycache__' \
      --exclude=./scripts/gpu_server.env . \
    | remote "mkdir -p '$REMOTE_DIR' && tar xzf - -C '$REMOTE_DIR'"
}

setup_remote_env() {
  remote "cd '$REMOTE_DIR' && if [ ! -x .venv/bin/python ]; then
      echo '== creating .venv on the server (first run)' >&2
      $REMOTE_PY -m venv --system-site-packages .venv &&
      .venv/bin/python -m pip install -q --upgrade pip &&
      .venv/bin/python -c 'import torch' 2>/dev/null || .venv/bin/python -m pip install -q torch
    fi && .venv/bin/python -m pip install -q -e . &&
    .venv/bin/python -c 'import torch; print(\"== server torch\", torch.__version__, \"CUDA\", torch.cuda.is_available())' >&2"
}

fetch() {
  echo "== copying $GPU_SERVER:$REMOTE_DIR/runs back to $ROOT/runs" >&2
  mkdir -p runs
  remote "cd '$REMOTE_DIR' && mkdir -p runs && tar czf - runs" | tar xzf - -C "$ROOT"
}

quote() { printf '%q ' "$@"; }

case "${1:-}" in
  ""|-h|--help) usage ;;
  status)
    server_has_gpu || exit 1
    remote "cd '$REMOTE_DIR' 2>/dev/null && ls -t runs/remote_logs/*.log 2>/dev/null | head -1 | xargs -r tail -n 5; pgrep -af '[m]eda (train|curriculum|evaluate|compare|bioassay|render)' || echo '(no meda job running)'"
    exit ;;
  fetch)
    [[ -n "$GPU_SERVER" ]] || { echo "set GPU_SERVER=user@host" >&2; exit 1; }
    fetch; exit ;;
esac

if ! server_has_gpu; then
  run_locally "$@"
fi

upload
setup_remote_env
CMD="cd '$REMOTE_DIR' && source .venv/bin/activate && ${MEDA_DEVICE:+MEDA_DEVICE=$MEDA_DEVICE }$(quote "$@")"
if [[ "${DETACH:-0}" == "1" ]]; then
  LOG="runs/remote_logs/$(date +%Y%m%d_%H%M%S).log"
  remote "mkdir -p '$REMOTE_DIR/runs/remote_logs' && nohup bash -c $(printf '%q' "$CMD") > '$REMOTE_DIR/$LOG' 2>&1 < /dev/null &"
  echo "== started in the background on $GPU_SERVER; log: $REMOTE_DIR/$LOG" >&2
  echo "   follow it with: bash scripts/run_on_gpu_server.sh status" >&2
  echo "   get results with: bash scripts/run_on_gpu_server.sh fetch" >&2
  exit 0
fi
echo "== running on $GPU_SERVER: $*" >&2
status=0
remote "$CMD" || status=$?
fetch
exit $status
