#!/usr/bin/env bash

# Launch agent.py inside bubblewrap. The project is writable at /workspace;
# harness code is read-only at /agent. Network access and host Chrome are shared.
# GPU device nodes are exposed when ENABLE_GPU=1 (no-op on hosts without them).
# This is filesystem/process isolation, not isolation from all host capabilities.
# PQ_MODEL selects the registry entry, PQ_API_KEY its provider key, and
# PQ_PLAYWRIGHT the web backend: 1 (default) = the shared headed Chrome, driven
# by openCode browse when it is installed and by the Node CDP server otherwise;
# 2 = lightweight HTTP tools (web_lite.py; for low-memory hosts, no
# Node/browser needed); 0 = no built-in web tools. No other agent
# configuration is required.

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
for name in PQ_MODEL PQ_API_KEY PQ_PLAYWRIGHT PQ_BROWSER_BACKEND; do
    if [ -n "${!name:-}" ]; then
        ENV_ARGS+=(--setenv "$name" "${!name}")
    fi
done

# OpenCode browse: the CLI, its XDG state, and one shared screenshot folder are
# bound into the sandbox so the relay started inside it sees the pairing the
# operator made on the host. Pairing tokens live in ~/.local/share/browse, so
# binding that folder is what makes an already-paired Chrome work here.
# AUTO_UPDATE/UPDATE_CHECK are off because the binary is read-only inside, and
# MCP_DIRECT skips the supervisor that would try to swap it. Every path is
# optional: a host without browse (or PQ_BROWSER_BACKEND=legacy) falls back to
# mcp_server.js over CDP, and a Raspberry Pi never gets here at all.
BROWSE_BIN="${HOME}/.local/bin/browse"
BROWSE_FILES_DIR="/tmp/pq_browse_files"
mkdir -p -m 700 "$BROWSE_FILES_DIR"
ENV_ARGS+=(--setenv OPENCODE_BROWSER_FILE_ROOTS "/workspace:$BROWSE_FILES_DIR"
           --setenv OPENCODE_BROWSER_AUTO_UPDATE "off"
           --setenv OPENCODE_BROWSER_UPDATE_CHECK "0"
           --setenv OPENCODE_BROWSER_MCP_DIRECT "1")

# Preflight: agent.py reports which backend it actually chose, but that line
# scrolls away, and "web tools fell back to CDP" is the one failure worth
# reading after the fact. These paths are host paths, so this is the only place
# that can see them at all.
if [ -x "$BROWSE_BIN" ]; then
    echo "browse bin : $BROWSE_BIN ($("$BROWSE_BIN" --version 2>/dev/null | head -1))"
else
    echo "browse bin : absent at $BROWSE_BIN; web mode 1 falls back to mcp_server.js over CDP"
fi
if [ -f "${HOME}/.local/share/browse/browser-tokens.json" ]; then
    echo "browse pair: state in ${HOME}/.local/share/browse"
else
    echo "browse pair: no tokens in ${HOME}/.local/share/browse; run 'browse pair' on the host"
fi

# A relay left running on the host is joined rather than started, and two
# problems follow: a relay from a newer binary rejects this session's client
# with a version mismatch, and screenshots are written under the host's /tmp,
# which the sandbox cannot read. Stop it so this session starts its own inside.
# The cost is that any other browse user on the host loses its relay too.
# Matching on /proc/<pid>/comm, not just the command line: a bare pkill -f also
# matches any shell whose arguments contain "browse serve", including a caller.
stopped_relay=0
for pid in $(pgrep -f "browse serve" 2>/dev/null || true); do
    [ "$pid" = "$$" ] && continue
    if [ "$(cat "/proc/$pid/comm" 2>/dev/null || true)" = "browse" ]; then
        kill "$pid" 2>/dev/null && stopped_relay=1
    fi
done
if [ "$stopped_relay" = 1 ]; then
    echo "browse relay: stopped the host-side relay so this session owns it"
    sleep 1
fi

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
  --ro-bind-try "$BROWSE_BIN" /tmp/.local/bin/browse \
  --bind "$BROWSE_FILES_DIR" "$BROWSE_FILES_DIR" \
  --bind-try "${HOME}/.config/browse" /tmp/.config/browse \
  --bind-try "${HOME}/.local/share/browse" /tmp/.local/share/browse \
  --bind-try "${HOME}/.local/state/browse" /tmp/.local/state/browse \
  --bind-try "${HOME}/.cache/browse" /tmp/.cache/browse \
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
  --setenv PATH /tmp/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  --setenv HOME /tmp \
  --setenv TMPDIR /tmp \
  "${EXTRA_ARGS[@]}" \
  "${ENV_ARGS[@]}" \
  --chdir /workspace \
  -- \
  python3 -u /agent/agent.py
