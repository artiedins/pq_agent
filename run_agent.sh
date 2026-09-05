#!/usr/bin/env bash

# Launch agent.py inside bubblewrap. The project is writable at /workspace;
# harness code is read-only at /agent. Network access and host Chrome are shared.
# This is filesystem/process isolation, not isolation from all host capabilities.
# PQ_MODEL selects the registry entry, PQ_API_KEY its provider key, and
# PQ_PLAYWRIGHT=0 disables browser tools. No other agent configuration is required.

set -euo pipefail

# Host-specific capabilities are opt-in. Edit these settings for this machine.
ENABLE_AUDIO=0
ENABLE_MUSIC=0
ENABLE_HF_CACHE=0
HF_OFFLINE=0

AGENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ $# -eq 0 ]; then
    PROJECT_DIR="$(pwd)"
elif [ $# -eq 1 ]; then
    PROJECT_DIR="$(cd "$1" && pwd)"
else
    echo "usage: bash run_agent.sh [project_dir]" >&2
    exit 1
fi

if ! command -v bwrap &>/dev/null; then
    echo "error: bwrap not found. install with: sudo apt install bubblewrap" >&2
    exit 1
fi

ENV_ARGS=()
for name in PQ_MODEL PQ_API_KEY PQ_PLAYWRIGHT; do
    if [ -n "${!name:-}" ]; then
        ENV_ARGS+=(--setenv "$name" "${!name}")
    fi
done

EXTRA_ARGS=()
if [ "$ENABLE_AUDIO" = 1 ]; then
    # The entire runtime directory exposes PipeWire, microphone and dbus access.
    RUNTIME_DIR="/run/user/$(id -u)"
    EXTRA_ARGS+=(--dev-bind-try /dev/snd /dev/snd
                 --bind-try "$RUNTIME_DIR" "$RUNTIME_DIR"
                 --setenv XDG_RUNTIME_DIR "$RUNTIME_DIR")
fi
if [ "$ENABLE_MUSIC" = 1 ]; then
    EXTRA_ARGS+=(--ro-bind-try "${HOME}/Music" /music --setenv MUSIC_ROOT /music)
fi
if [ "$ENABLE_HF_CACHE" = 1 ]; then
    # Read-only host models; downloads need a writable cache selected by the task.
    EXTRA_ARGS+=(--ro-bind-try "${HOME}/.cache/huggingface" /hf-cache
                 --setenv HF_HOME /hf-cache
                 --setenv HUGGINGFACE_HUB_CACHE /hf-cache/hub
                 --setenv TRANSFORMERS_CACHE /hf-cache/hub
                 --setenv HF_MODULES_CACHE /tmp/hf_modules)
fi
if [ "$HF_OFFLINE" = 1 ]; then
    EXTRA_ARGS+=(--setenv HF_HUB_OFFLINE 1)
fi

echo "agent dir  : $AGENT_DIR"
echo "project dir: $PROJECT_DIR"
echo "model      : ${PQ_MODEL:-(default)}"
echo "playwright : ${PQ_PLAYWRIGHT:-1 (default)}"
echo "host extras: audio=$ENABLE_AUDIO music=$ENABLE_MUSIC hf_cache=$ENABLE_HF_CACHE hf_offline=$HF_OFFLINE"

# No Playwright browser-cache bind is needed: the server connects over CDP and
# does not launch a downloaded browser. Private temporary files remain writable.
exec bwrap \
  --ro-bind /usr /usr \
  --ro-bind-try /bin /bin \
  --ro-bind-try /lib /lib \
  --ro-bind-try /lib64 /lib64 \
  --ro-bind-try /sbin /sbin \
  --ro-bind /etc /etc \
  --proc /proc \
  --dev /dev \
  --tmpfs /dev/shm \
  --tmpfs /tmp \
  --tmpfs /home \
  --tmpfs /root \
  --tmpfs /run \
  --ro-bind-try /run/systemd/resolve /run/systemd/resolve \
  --ro-bind "$AGENT_DIR" /agent \
  --bind "$PROJECT_DIR" /workspace \
  --tmpfs /workspace/.pq \
  --unshare-pid \
  --unshare-ipc \
  --unshare-uts \
  --die-with-parent \
  --new-session \
  --clearenv \
  --setenv PATH /usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  --setenv HOME /tmp \
  --setenv TMPDIR /tmp \
  "${EXTRA_ARGS[@]}" \
  "${ENV_ARGS[@]}" \
  --chdir /workspace \
  -- \
  python3 -u /agent/agent.py
