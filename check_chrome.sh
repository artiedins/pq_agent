#!/usr/bin/env bash
set -euo pipefail

CHROME_BIN="/opt/google/chrome/chrome"
CHROME_DIR="${HOME}/.chrome-research"
BROWSE_BIN="${HOME}/.local/bin/browse"
DEBUG_PORT=9222
DEBUG_URL="http://127.0.0.1:${DEBUG_PORT}/json/version"

FORCE_KILL=0

usage() {
    echo "Usage: $(basename "$0") [-k|--kill|kill|-r|--restart|restart]"
    echo "  (no args)   Start Chrome if it is not running or not responding on port ${DEBUG_PORT}."
    echo "  -k, --kill  Kill any running Chrome instance and restart it cleanly."
}

case "${1:-}" in
    "")
        ;;
    -k|--kill|kill|-r|--restart|restart)
        FORCE_KILL=1
        ;;
    -h|--help|help)
        usage
        exit 0
        ;;
    *)
        echo "Unknown option: $1" >&2
        usage >&2
        exit 1
        ;;
esac

UID_NUM="$(id -u)"
export DISPLAY="${DISPLAY:-:0}"
export XAUTHORITY="${XAUTHORITY:-${HOME}/.Xauthority}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/${UID_NUM}}"
if [ -S "/run/user/${UID_NUM}/bus" ]; then
    export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=/run/user/${UID_NUM}/bus}"
fi

find_chrome_pids() {
    pgrep -u "${UID_NUM}" -f "${CHROME_BIN}.*--user-data-dir=${CHROME_DIR}" || true
}

is_chrome_healthy() {
    local pids
    pids="$(find_chrome_pids)"
    if [ -z "${pids}" ]; then
        return 1
    fi
    curl -fsS --max-time 3 "${DEBUG_URL}" >/dev/null 2>&1
}

stop_chrome() {
    local pids
    pids="$(find_chrome_pids)"
    if [ -n "${pids}" ]; then
        echo "Stopping Chrome processes for ${CHROME_DIR}..."
        for pid in ${pids}; do
            kill -TERM "${pid}" 2>/dev/null || true
        done
        for _ in $(seq 1 15); do
            pids="$(find_chrome_pids)"
            if [ -z "${pids}" ]; then
                break
            fi
            sleep 0.3
        done
        pids="$(find_chrome_pids)"
        if [ -n "${pids}" ]; then
            echo "Force-killing remaining Chrome processes..."
            for pid in ${pids}; do
                kill -KILL "${pid}" 2>/dev/null || true
            done
            sleep 0.5
        fi
    fi
    rm -f "${CHROME_DIR}/SingletonLock" "${CHROME_DIR}/SingletonSocket" "${CHROME_DIR}/SingletonCookie"
}

clear_crash_state() {
    local prefs="${CHROME_DIR}/Default/Preferences"
    if [ -f "${prefs}" ]; then
        python3 - "${prefs}" <<'PY' 2>/dev/null || true
import json, sys
path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
prof = data.setdefault("profile", {})
prof["exit_type"] = "Normal"
prof["exited_cleanly"] = True
with open(path, "w", encoding="utf-8") as f:
    json.dump(data, f, separators=(",", ":"))
PY
    fi
}

start_chrome() {
    clear_crash_state
    local cmd="google-chrome --remote-debugging-port=${DEBUG_PORT} --user-data-dir=${CHROME_DIR} --no-first-run --no-default-browser-check"
    local i3_sock=""
    if [ -d "/run/user/${UID_NUM}/i3" ]; then
        i3_sock="$(ls -t "/run/user/${UID_NUM}/i3/"ipc-socket.* 2>/dev/null | head -n 1 || true)"
    fi
    echo "Starting Chrome on DISPLAY=${DISPLAY} (port ${DEBUG_PORT})..."
    if [ -n "${i3_sock}" ] && command -v i3-msg >/dev/null 2>&1; then
        I3SOCK="${i3_sock}" i3-msg -q "exec --no-startup-id ${cmd}" >/dev/null 2>&1 || \
            nohup google-chrome --remote-debugging-port="${DEBUG_PORT}" --user-data-dir="${CHROME_DIR}" --no-first-run --no-default-browser-check >/dev/null 2>&1 &
    else
        nohup google-chrome --remote-debugging-port="${DEBUG_PORT}" --user-data-dir="${CHROME_DIR}" --no-first-run --no-default-browser-check >/dev/null 2>&1 &
    fi

    for _ in $(seq 1 30); do
        if is_chrome_healthy; then
            local ver
            ver="$(curl -fsS --max-time 2 "${DEBUG_URL}" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("Browser","unknown"))' 2>/dev/null || echo "running")"
            echo "Chrome is up and responding on 127.0.0.1:${DEBUG_PORT} (${ver})."
            report_browse_status 10
            return 0
        fi
        sleep 0.5
    done

    echo "ERROR: Chrome failed to respond on ${DEBUG_URL} within 15 seconds." >&2
    return 1
}

report_browse_status() {
    local max_wait="${1:-1}"
    if [ ! -x "${BROWSE_BIN}" ]; then
        return 0
    fi
    local ext_line=""
    for _ in $(seq 1 "${max_wait}"); do
        ext_line="$("${BROWSE_BIN}" status 2>/dev/null | grep -E "^Extension:" || true)"
        if echo "${ext_line}" | grep -q "connected"; then
            break
        fi
        sleep 0.5
    done
    if [ -n "${ext_line}" ]; then
        echo "browse ${ext_line}"
    fi
}

main() {
    if [ "${FORCE_KILL}" -eq 1 ]; then
        stop_chrome
        start_chrome
        exit 0
    fi

    if is_chrome_healthy; then
        local ver
        ver="$(curl -fsS --max-time 2 "${DEBUG_URL}" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("Browser","unknown"))' 2>/dev/null || echo "running")"
        echo "Chrome is already running and healthy (${ver} on 127.0.0.1:${DEBUG_PORT})."
        report_browse_status 1
        exit 0
    fi

    if [ -n "$(find_chrome_pids)" ]; then
        echo "Chrome process found but not responding on 127.0.0.1:${DEBUG_PORT}; restarting..."
        stop_chrome
    fi
    start_chrome
}

main
