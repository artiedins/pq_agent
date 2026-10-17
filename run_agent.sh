#!/usr/bin/env bash

# Launch agent.py inside bubblewrap. The project is writable at /workspace;
# harness code is read-only at /agent. Network access and host Chrome are shared.
# GPU device nodes are exposed when ENABLE_GPU=1 (no-op on hosts without them).
# This is filesystem/process isolation, not isolation from all host capabilities.
# PQ_MODEL selects the registry entry, PQ_API_KEY its provider key, and
# PQ_PLAYWRIGHT the web backend: 1 (default) = headed Chrome via the Node MCP
# server, 2 = lightweight HTTP tools (web_lite.py; for low-memory hosts, no
# Node/browser needed), 0 = no built-in web tools. No other agent configuration
# is required.

set -euo pipefail

# Host-specific capabilities are opt-in. Edit these settings for this machine.
# The interpreter is the system python3; harness deps (requests, bs4, flowmark)
# must be importable from it. On PEP 668 hosts (Debian 12+/Raspberry Pi OS
# Bookworm+) that means `sudo pip3 install --break-system-packages ...`, which
# lands in /usr/local/lib/python3.X/dist-packages, inside the /usr bind below.
ENABLE_AUDIO=0
ENABLE_MUSIC=0
ENABLE_HF_CACHE=0
HF_OFFLINE=0
# GPU access: bind the NVIDIA character devices and the DRM render nodes into the
# sandbox, which bwrap's --dev does not provide. Nothing is bound when the nodes
# are absent, so this stays correct on a host without a GPU. The KMS master nodes
# (/dev/dri/card*) are deliberately left out: they would let the sandbox take over
# the display, and nothing here needs them.
ENABLE_GPU=1

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
if [ "$ENABLE_GPU" = 1 ]; then
    # Globs expand against the host /dev before bwrap starts; a pattern with no
    # match stays literal and the -e test drops it.
    for dev in /dev/nvidia[0-9]* /dev/nvidiactl /dev/nvidia-modeset /dev/nvidia-uvm \
               /dev/nvidia-uvm-tools /dev/nvidia-caps /dev/dri/render*; do
        [ -e "$dev" ] && EXTRA_ARGS+=(--dev-bind "$dev" "$dev")
    done
fi

echo "agent dir  : $AGENT_DIR"
echo "project dir: $PROJECT_DIR"
echo "model      : ${PQ_MODEL:-(default)}"
echo "web mode   : ${PQ_PLAYWRIGHT:-1 (default: 1=Chrome browser, 2=lite HTTP, 0=none)}"
echo "host extras: audio=$ENABLE_AUDIO music=$ENABLE_MUSIC hf_cache=$ENABLE_HF_CACHE hf_offline=$HF_OFFLINE gpu=$ENABLE_GPU"

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
