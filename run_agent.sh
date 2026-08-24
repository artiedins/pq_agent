#!/usr/bin/env bash

# run_agent.sh - launch agent.py inside a bubblewrap sandbox
#
# usage: bash run_agent.sh [project_dir]
#        PQ_MODEL=glm52 PQ_PLAYWRIGHT=0 bash run_agent.sh [project_dir]
#
# if project_dir is omitted, the current working directory is used.
#
# env vars the agent needs:
#   PQ_MODEL          - which model to use (default: go-dsv4p)
#   PQ_API_KEY        - API key for OpenRouter or OpenCode (go_*/zen_*); put the
#                       provider key that matches PQ_MODEL in this one var
#   PQ_PLAYWRIGHT     - 1 to enable web search via headed Chrome, 0 to disable
#
# the agent code lives here (read-only inside sandbox at /agent)
# the project dir is where the agent reads and writes (read-write at /workspace)
#
# protects: $HOME entirely invisible, only project dir is writable
# .pq is shadowed with an empty tmpfs so the agent cannot see harness files
# allows: full network (playwright needs it, local models need localhost)
# playwright browser cache read-only
# sound: binds the host /dev/snd (direct ALSA) and the user session's
#        PipeWire/PulseAudio sockets (/run/user/UID) so the agent's players
#        can actually be heard through the machine's speakers/headphones

set -euo pipefail

AGENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ $# -eq 0 ]; then
    PROJECT_DIR="$(pwd)"
elif [ $# -eq 1 ]; then
    PROJECT_DIR="$(cd "$1" && pwd)"
else
    echo "usage: bash run_agent.sh [project_dir]" >&2
    echo "  if project_dir is omitted, the current working directory is used." >&2
    exit 1
fi

PW_CACHE="${HOME}/.cache/ms-playwright"

# audio: the logged-in desktop session's runtime dir holds the PipeWire and
# PulseAudio sockets (plus the dbus session bus). Binding it into the sandbox
# lets mpv/pactl reach the same sound server the user hears through. Note this
# also exposes the microphone to the sandbox - acceptable on a home music box.
# When no desktop session is logged in the dir is absent and --bind-try skips
# it; mpv then falls back to direct ALSA via the /dev/snd bind below.
RUNTIME_DIR="/run/user/$(id -u)"

if ! command -v bwrap &>/dev/null; then
    echo "error: bwrap not found. install with: sudo apt install bubblewrap" >&2
    exit 1
fi

# build the list of env vars to pass into the sandbox.
# agent.py handles defaults and validation; bwrap --clearenv strips the rest.
ENV_ARGS=()
if [ -n "${PQ_MODEL:-}" ]; then
    ENV_ARGS+=(--setenv PQ_MODEL "$PQ_MODEL")
fi
if [ -n "${PQ_API_KEY:-}" ]; then
    ENV_ARGS+=(--setenv PQ_API_KEY "$PQ_API_KEY")
fi
if [ -n "${PQ_PLAYWRIGHT:-}" ]; then
    ENV_ARGS+=(--setenv PQ_PLAYWRIGHT "$PQ_PLAYWRIGHT")
fi

echo "agent dir  : $AGENT_DIR"
echo "project dir: $PROJECT_DIR"
echo "model      : ${PQ_MODEL:-(default)}"
echo "playwright : ${PQ_PLAYWRIGHT:-1 (default)}"
echo ""

exec bwrap \
  --ro-bind /usr /usr \
  --ro-bind-try /bin /bin \
  --ro-bind-try /lib /lib \
  --ro-bind-try /lib64 /lib64 \
  --ro-bind-try /sbin /sbin \
  --ro-bind /etc /etc \
  --proc /proc \
  --dev /dev \
  `# sound: direct ALSA access to the host audio hardware (headphones/speakers)` \
  --dev-bind-try /dev/snd /dev/snd \
  --tmpfs /dev/shm \
  --tmpfs /tmp \
  --tmpfs /home \
  --tmpfs /root \
  --tmpfs /run \
  `# sound: host user-session sockets (PipeWire/PulseAudio/dbus); absent when
   # no desktop session is logged in - mpv then falls back to ALSA` \
  --bind-try "$RUNTIME_DIR" "$RUNTIME_DIR" \
  --ro-bind-try /run/systemd/resolve /run/systemd/resolve \
  `# agent code is read-only - the agent cannot modify itself` \
  --ro-bind "$AGENT_DIR" /agent \
  `# project dir is the only writable location` \
  --bind "$PROJECT_DIR" /workspace \
  `# shadow .pq with empty tmpfs so agent cannot read harness files` \
  --tmpfs /workspace/.pq \
  --ro-bind-try "$PW_CACHE" /pw-cache \
  `# HF caches commonly used by huggingface_hub / transformers` \
   --ro-bind-try "${HOME}/.cache/huggingface" /hf-cache \
  `# music library: host ~/Music is bound read-only at /music - the agent plays
   # from it but can never edit it, and the files stay out of the workdir` \
  --ro-bind-try "${HOME}/Music" /music \
  --unshare-pid \
  --unshare-ipc \
  --unshare-uts \
  --die-with-parent \
  --new-session \
  --clearenv \
  --setenv PATH /usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  --setenv HOME /tmp \
  --setenv TMPDIR /tmp \
  --setenv XDG_RUNTIME_DIR "$RUNTIME_DIR" \
  --setenv PLAYWRIGHT_BROWSERS_PATH /pw-cache \
  --setenv AGENT_DIR /agent \
  --setenv HF_HOME /hf-cache \
  --setenv HUGGINGFACE_HUB_CACHE /hf-cache/hub \
  --setenv TRANSFORMERS_CACHE /hf-cache/hub \
  --setenv HF_HUB_OFFLINE 1 \
  --setenv HF_MODULES_CACHE /tmp/hf_modules \
  --setenv MUSIC_ROOT /music \
  "${ENV_ARGS[@]}" \
  --chdir /workspace \
  -- \
  python3 -u /agent/agent.py

