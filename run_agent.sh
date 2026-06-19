#!/usr/bin/env bash

# run_agent.sh - launch agent.py inside a bubblewrap sandbox
#
# usage: bash run_agent.sh [project_dir]
#
# if project_dir is omitted, the current working directory is used.
#
# the agent code lives here (read-only inside sandbox at /agent)
# the project dir is where the agent reads and writes (read-write at /workspace)
#
# protects: $HOME entirely invisible, only project dir is writable
# .pq is shadowed with an empty tmpfs so the agent cannot see harness files
# allows: full network (playwright needs it), playwright browser cache read-only
#
# OPENROUTER_API_KEY is the only LLM key needed inside the sandbox.

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

: "${OPENROUTER_API_KEY:?OPENROUTER_API_KEY must be set in the environment}"

PW_CACHE="${HOME}/.cache/ms-playwright"

if ! command -v bwrap &>/dev/null; then
    echo "error: bwrap not found. install with: sudo apt install bubblewrap" >&2
    exit 1
fi

echo "agent dir  : $AGENT_DIR"
echo "project dir: $PROJECT_DIR"
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
  --tmpfs /dev/shm \
  --tmpfs /tmp \
  --tmpfs /home \
  --tmpfs /root \
  --tmpfs /run \
  --ro-bind-try /run/systemd/resolve /run/systemd/resolve \
  `# agent code is read-only - the agent cannot modify itself` \
  --ro-bind "$AGENT_DIR" /agent \
  `# project dir is the only writable location` \
  --bind "$PROJECT_DIR" /workspace \
  `# shadow .pq with empty tmpfs so agent cannot read harness files` \
  --tmpfs /workspace/.pq \
  --ro-bind-try "$PW_CACHE" /pw-cache \
  --unshare-pid \
  --unshare-ipc \
  --unshare-uts \
  --die-with-parent \
  --new-session \
  --clearenv \
  --setenv PATH /usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  --setenv HOME /tmp \
  --setenv TMPDIR /tmp \
  --setenv OPENROUTER_API_KEY "$OPENROUTER_API_KEY" \
  --setenv PLAYWRIGHT_BROWSERS_PATH /pw-cache \
  --setenv AGENT_DIR /agent \
  --chdir /workspace \
  -- \
  python3 /agent/agent.py
