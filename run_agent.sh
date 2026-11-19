#!/usr/bin/env bash

# Launch agent.py inside bubblewrap (or directly on the host when USE_BWRAP=0).
# Under bubblewrap, the project directory is writable at /workspace and harness
# code is read-only at /agent. Network access and host Chrome (when PQ_PLAYWRIGHT=1)
# are shared; GPU device and DRM render nodes are bound automatically when present
# on the host (/dev/dri/card* KMS master nodes remain excluded).
#
# Environment variables forwarded into the sandbox:
#   PQ_API_KEY            Provider API key (required unless DEBUG_PROMPTS=1; passed
#                         via inherited fd into bwrap so it never appears in ps/cmdline)
#   PQ_MODEL              Registry key in agent.py (default: go-dsv41f)
#   PQ_PLAYWRIGHT         1 (default) = shared headed Chrome via browse (auto-falls
#                         back to web_lite.py if browse is unavailable);
#                         2 = lightweight HTTP tools (web_lite.py); 0 = no web tools
#   PQ_OPENCODE           1 = OpenCode v2 headers, billing flag, and prompt framing;
#                         0 (default) = pq-agent identity
#   PQ_BROWSE_ASSIST      1 = operator is watching Chrome for interactive handoff();
#                         0 (default) = unassisted run, handoff() disabled
#   PQ_PREVIOUS_SESSIONS  1 (default) = archive prior task_report/ into previous_sessions/;
#                         0 = wipe task_report/ cleanly at startup (daemon mode)
#   PQ_LIVE_INBOX         1 = poll .inbox/*.md mid-run and wake sleep early;
#                         0 (default) = standard batch run
#
# A project directory containing a .pq directory has that directory screened behind
# an empty tmpfs inside bubblewrap (and is refused under USE_BWRAP=0).

set -euo pipefail

AGENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ $# -eq 0 ]; then
    PROJECT_DIR="$(pwd)"
elif [ $# -eq 1 ]; then
    PROJECT_DIR="$(cd "$1" && pwd)"
else
    echo "usage: bash run_agent.sh [project_dir]" >&2
    exit 1
fi

case "${PQ_PLAYWRIGHT:-1}" in
    1|true|TRUE|True|yes|YES|Yes) WANT_BROWSE=1 ;;
    *) WANT_BROWSE=0 ;;
esac

BROWSE_BIN="${HOME}/.local/bin/browse"
BROWSE_FILES_DIR="/tmp/pq_browse_files"
BROWSE_PORT="${OPENCODE_BROWSER_PORT:-19988}"

relay_port_ready() {
    python3 - "$BROWSE_PORT" <<'PY' >/dev/null 2>&1
import socket, sys
port = int(sys.argv[1])
with socket.create_connection(("127.0.0.1", port), timeout=0.5):
    pass
PY
}

relay_matches_host() {
    python3 - "$BROWSE_PORT" "$1" "$BROWSE_FILES_DIR" <<'PY' >/dev/null 2>&1
import http.client, json, sys
port, want_ver, want_roots = int(sys.argv[1]), sys.argv[2], sys.argv[3]
conn = http.client.HTTPConnection("127.0.0.1", port, timeout=0.5)
conn.request("GET", "/version", headers={"Host": f"127.0.0.1:{port}"})
resp = conn.getresponse()
data = json.loads(resp.read().decode("utf-8"))
conn.close()
if resp.status != 200 or data.get("version") != want_ver:
    sys.exit(1)
pid = int(data.get("pid") or 0)
if pid <= 0:
    sys.exit(1)
env_bytes = open(f"/proc/{pid}/environ", "rb").read().split(b"\0")
if (f"OPENCODE_BROWSER_FILE_ROOTS={want_roots}").encode("utf-8") not in env_bytes:
    sys.exit(1)
PY
}

# Ensure a single persistent host-owned `browse serve` daemon is running outside
# any agent's bwrap PID namespace. That lets HOME_DAEMON and multiple concurrent
# harness runs across different working directories share Chrome in parallel
# (each agent gets an isolated `mcp-<id>` tab group) without Agent 1's exit
# tearing down the relay for Agent 2, and restricts relay file roots strictly to
# /tmp/pq_browse_files so no sandboxed agent can read host paths through the relay.
ensure_host_browse_relay() {
    mkdir -p -m 700 "$BROWSE_FILES_DIR"
    if [ ! -x "$BROWSE_BIN" ]; then
        echo "browse bin : absent at $BROWSE_BIN; web mode 1 will fall back to web_lite.py"
        return 0
    fi
    local browse_ver
    browse_ver="$("$BROWSE_BIN" --version 2>/dev/null | head -1 || true)"
    echo "browse bin : $BROWSE_BIN ($browse_ver)"
    if [ -f "${HOME}/.local/share/browse/browser-tokens.json" ]; then
        echo "browse pair: state in ${HOME}/.local/share/browse"
    else
        echo "browse pair: no tokens in ${HOME}/.local/share/browse; run 'browse pair' on the host"
    fi

    (
        flock -w 10 8 || exit 0
        local ver_file="$BROWSE_FILES_DIR/.relay.version"

        if [ -n "$browse_ver" ] && relay_matches_host "$browse_ver"; then
            printf '%s\n' "$browse_ver" >"$ver_file"
            echo "browse relay: reusing shared host relay on 127.0.0.1:$BROWSE_PORT ($browse_ver)"
            exit 0
        fi

        # Stop an existing relay only if it is unresponsive, missing file-root lockdown, or from another version.
        for pid in $(pgrep -f "browse serve" 2>/dev/null || true); do
            [ "$pid" = "$$" ] && continue
            if [ "$(cat "/proc/$pid/comm" 2>/dev/null || true)" = "browse" ]; then
                kill "$pid" 2>/dev/null || true
            fi
        done
        local release_wait=0
        while [ "$release_wait" -lt 15 ] && relay_port_ready; do
            sleep 0.2
            release_wait=$((release_wait + 1))
        done

        nohup env \
            HOME="$HOME" \
            PATH="$PATH" \
            TMPDIR="$BROWSE_FILES_DIR" \
            OPENCODE_BROWSER_PORT="$BROWSE_PORT" \
            OPENCODE_BROWSER_FILE_ROOTS="$BROWSE_FILES_DIR" \
            OPENCODE_BROWSER_AUTO_UPDATE="off" \
            OPENCODE_BROWSER_UPDATE_CHECK="0" \
            "$BROWSE_BIN" serve </dev/null >>"$BROWSE_FILES_DIR/relay.log" 2>&1 &

        local waited=0
        while [ "$waited" -lt 25 ]; do
            if relay_port_ready; then
                printf '%s\n' "$browse_ver" >"$ver_file"
                echo "browse relay: started shared host relay on 127.0.0.1:$BROWSE_PORT ($browse_ver)"
                exit 0
            fi
            sleep 0.2
            waited=$((waited + 1))
        done
        echo "browse relay: host relay did not bind 127.0.0.1:$BROWSE_PORT within 5s (see $BROWSE_FILES_DIR/relay.log)"
    ) 8>"$BROWSE_FILES_DIR/.relay.lock"
}

if [ "${USE_BWRAP:-1}" = "0" ]; then
    if [ -e "$PROJECT_DIR/.pq" ] || [ -L "$PROJECT_DIR/.pq" ]; then
        echo "error: $PROJECT_DIR/.pq is present and cannot be screened when USE_BWRAP=0" >&2
        exit 1
    fi
    if [ "$WANT_BROWSE" = 1 ]; then
        ensure_host_browse_relay
        export OPENCODE_BROWSER_PORT="$BROWSE_PORT"
        export OPENCODE_BROWSER_AUTOSTART="0"
        export OPENCODE_BROWSER_FILE_ROOTS="$BROWSE_FILES_DIR"
        export OPENCODE_BROWSER_AUTO_UPDATE="off"
        export OPENCODE_BROWSER_UPDATE_CHECK="0"
    fi
    echo "agent dir  : $AGENT_DIR"
    echo "project dir: $PROJECT_DIR"
    echo "sandbox    : disabled (USE_BWRAP=0)"
    echo "model      : ${PQ_MODEL:-(default)}"
    echo "web mode   : ${PQ_PLAYWRIGHT:-1 (default: 1=Chrome browse, 2=lite HTTP, 0=none)}"
    cd "$PROJECT_DIR"
    exec python3 -u "$AGENT_DIR/agent.py"
fi

if ! command -v bwrap &>/dev/null; then
    echo "error: bwrap not found. install with: sudo apt install bubblewrap" >&2
    exit 1
fi

ENV_ARGS=()
for name in PQ_MODEL PQ_PLAYWRIGHT PQ_OPENCODE PQ_BROWSE_ASSIST \
            PQ_PREVIOUS_SESSIONS PQ_LIVE_INBOX DEBUG_PROMPTS; do
    if [ -n "${!name:-}" ]; then
        ENV_ARGS+=(--setenv "$name" "${!name}")
    fi
done

# Pass PQ_API_KEY through an inherited pipe fd rather than --setenv on bwrap's
# argv so the secret never appears in /proc/1/cmdline inside the sandbox.
if [ -n "${PQ_API_KEY:-}" ]; then
    exec 9<<<"${PQ_API_KEY}"
    unset PQ_API_KEY
    ENV_ARGS+=(--setenv PQ_API_KEY_FD 9)
fi

# OpenCode browse: gated strictly on PQ_PLAYWRIGHT=1 so non-browser runs (0 or 2)
# never mount Chrome pairing tokens (~/.local/share/browse/browser-tokens.json)
# and never start a host relay.
BROWSE_ARGS=()
if [ "$WANT_BROWSE" = 1 ]; then
    ensure_host_browse_relay
    ENV_ARGS+=(--setenv PQ_HOST_PID "$$"
               --setenv OPENCODE_BROWSER_PORT "$BROWSE_PORT"
               --setenv OPENCODE_BROWSER_AUTOSTART "0"
               --setenv OPENCODE_BROWSER_FILE_ROOTS "$BROWSE_FILES_DIR"
               --setenv OPENCODE_BROWSER_AUTO_UPDATE "off"
               --setenv OPENCODE_BROWSER_UPDATE_CHECK "0")

    BROWSE_ARGS+=(--ro-bind-try "$BROWSE_BIN" /tmp/.local/bin/browse
                  --bind "$BROWSE_FILES_DIR" "$BROWSE_FILES_DIR"
                  --bind-try "${HOME}/.config/browse" /tmp/.config/browse
                  --bind-try "${HOME}/.local/share/browse" /tmp/.local/share/browse
                  --bind-try "${HOME}/.local/state/browse" /tmp/.local/state/browse
                  --bind-try "${HOME}/.cache/browse" /tmp/.cache/browse)
fi

# GPU access: auto-bind NVIDIA character devices and DRM render nodes whenever
# present on the host. CPU-only hosts bind nothing. KMS display-master nodes
# (/dev/dri/card*) are excluded so the sandbox cannot take over the display.
GPU_ARGS=()
gpu_bound=0
for dev in /dev/nvidia[0-9]* /dev/nvidiactl /dev/nvidia-modeset /dev/nvidia-uvm \
           /dev/nvidia-uvm-tools /dev/nvidia-caps /dev/dri/render*; do
    if [ -e "$dev" ]; then
        GPU_ARGS+=(--dev-bind "$dev" "$dev")
        gpu_bound=1
    fi
done

# .pq holds operator-only material (held-out checks) that the agent must not read.
# Shadow it with an empty tmpfs when present; refuse symlinks or plain files.
PQ_ARGS=()
if [ -L "$PROJECT_DIR/.pq" ]; then
    echo "error: $PROJECT_DIR/.pq is a symlink and cannot be screened; use a real directory" >&2
    exit 1
elif [ -e "$PROJECT_DIR/.pq" ] && [ ! -d "$PROJECT_DIR/.pq" ]; then
    echo "error: $PROJECT_DIR/.pq is not a directory and cannot be screened" >&2
    exit 1
elif [ -d "$PROJECT_DIR/.pq" ]; then
    PQ_ARGS+=(--tmpfs /workspace/.pq)
fi

echo "agent dir  : $AGENT_DIR"
echo "project dir: $PROJECT_DIR"
echo "model      : ${PQ_MODEL:-(default)}"
echo "web mode   : ${PQ_PLAYWRIGHT:-1 (default: 1=Chrome browse, 2=lite HTTP, 0=none)} (assist=${PQ_BROWSE_ASSIST:-0})"
if [ "${PQ_OPENCODE:-0}" = "1" ]; then
    echo "identity   : opencode (PQ_OPENCODE=1)"
else
    echo "identity   : pq-agent (PQ_OPENCODE=${PQ_OPENCODE:-0})"
fi
echo "session    : previous_sessions=${PQ_PREVIOUS_SESSIONS:-1} live_inbox=${PQ_LIVE_INBOX:-0} gpu=$gpu_bound"
if [ ${#PQ_ARGS[@]} -gt 0 ]; then
    echo ".pq        : present in the project, screened behind an empty tmpfs"
fi

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
  "${BROWSE_ARGS[@]}" \
  --tmpfs /home \
  --tmpfs /root \
  --tmpfs /run \
  --ro-bind-try /run/systemd/resolve /run/systemd/resolve \
  --ro-bind "$AGENT_DIR" /agent \
  --bind "$PROJECT_DIR" /workspace \
  "${PQ_ARGS[@]}" \
  --unshare-pid \
  --unshare-ipc \
  --unshare-uts \
  --die-with-parent \
  --new-session \
  --clearenv \
  --setenv PATH /tmp/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  --setenv HOME /tmp \
  --setenv TMPDIR /tmp \
  "${GPU_ARGS[@]}" \
  "${ENV_ARGS[@]}" \
  --chdir /workspace \
  -- \
  python3 -u /agent/agent.py
