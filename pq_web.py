#!/usr/bin/env python3
# pq_web.py - web front end for pq_minder.py
#
# Run from your project workspace:
#     cd /your/project
#     python3 /path/to/pq_agent/pq_web.py
#
# Then open the printed URL in your browser.
#
# This file holds the server logic: Flask routes, runner thread, task driver.
# The HTML/CSS/JS template lives in pq_web_html.py - imported here as a single
# constant. Splitting them keeps this file diff-friendly across feature work.
#
# Extra requirement beyond pq_minder.py:
#     pip install flask

import os
import sys
import json
import socket
import threading
import time
import traceback
import subprocess
import logging

# silence werkzeug access log before Flask starts
logging.getLogger("werkzeug").setLevel(logging.ERROR)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from flask import Flask, jsonify, request, Response

from pq_minder import (
    validate_task_inputs,
    read_queue,
    stage_workspace,
    unstage_workspace,
    collect_task_report,
    call_judge_turn,
    build_first_judge_text,
    build_followup_judge_text,
    build_content_items,
    format_feedback_for_agent,
    save_agent_initial_prompt,
    save_judge_input,
    read_text,
    write_text,
    load_json,
    write_json,
    tail_file,
    make_escalation_verdict,
    check_api_keys,
    JUDGE_PROVIDER_OPTIONS,
    AGENT_MODEL,
    MAX_ATTEMPTS,
    LOG_TAIL_LINES,
    LOW_CONF_THRESHOLD,
    AGENT_TIMEOUT_S,
)

from pq_web_html import HTML

# Web UI default judge override. pq_minder.py defaults to Claude Opus 4.7 for CLI
# use; the web UI defaults to Kimi K2.6 (cheaper, still strong). Change this one
# line to flip the web default - the CLI default is independent.
WEB_DEFAULT_JUDGE_PROVIDER = "kimi"

WORKSPACE = os.path.abspath(os.getcwd())
HARNESS_DIR = os.path.join(WORKSPACE, ".pq")
PORT = 5000

app = Flask(__name__)
_orig_stdout = sys.stdout

# shared state (all mutations under _lock)

_lock = threading.RLock()
_stop_event = threading.Event()
_runner_thread = None

_console = []
_runner_status = "idle"
_current_task = None
_current_run = None
_run_start_time = None
_error = None

# user-selected judge provider; passed into each run. Agent is hardcoded.
_judge_provider = WEB_DEFAULT_JUDGE_PROVIDER


# stdout tee: captures print() output into _console


class _Tee:
    def __init__(self, original):
        self._orig = original
        self._buf = ""

    def write(self, data):
        self._orig.write(data)
        self._orig.flush()
        self._buf += data
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            _append_line(line.rstrip("\r"))

    def flush(self):
        self._orig.flush()

    def fileno(self):
        return self._orig.fileno()

    def isatty(self):
        return False


def _append_line(line):
    stripped = line.rstrip("\r")
    with _lock:
        if stripped.strip() or not _console or _console[-1].strip():
            _console.append(stripped)


def _cap(text):
    for line in str(text).split("\n"):
        _append_line(line)


# modified run_agent that feeds subprocess output into _console


def _run_agent_web(workspace_dir, run_dir):
    agent_script = os.path.join(_SCRIPT_DIR, "run_agent.sh")
    if not os.path.exists(agent_script):
        raise RuntimeError("run_agent.sh not found: " + agent_script)

    log_path = os.path.join(run_dir, "stdout.log")
    _cap("Starting agent...")

    run_start = time.time()
    timed_out = False

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("[pq_minder] workspace=" + workspace_dir + "\n")
        f.write("[pq_minder] started=" + time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
        f.flush()

        proc = subprocess.Popen(
            ["bash", agent_script, workspace_dir],
            cwd=_SCRIPT_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        def _tee():
            for raw in proc.stdout:
                _append_line(raw.rstrip("\n\r"))
                f.write(raw)
                f.flush()

        reader = threading.Thread(target=_tee, daemon=True)
        reader.start()
        reader.join(timeout=AGENT_TIMEOUT_S)

        if reader.is_alive():
            proc.kill()
            reader.join()
            timed_out = True
            rc = 124
            msg = "[pq_minder] agent timed out after " + str(AGENT_TIMEOUT_S) + "s"
            _cap(msg)
            f.write(msg + "\n")
        else:
            rc = proc.wait()

        elapsed = int(time.time() - run_start)
        summary = "[pq_minder] elapsed_s=" + str(elapsed) + " exit=" + str(rc)
        _cap(summary)
        f.write(summary + "\n")

    return rc, timed_out, elapsed


# task runner


def _run_task(harness_dir, workspace_dir, task_id, judge_provider):
    global _current_task, _current_run, _run_start_time

    task_dir = os.path.join(harness_dir, "tasks", task_id)
    if not os.path.exists(task_dir):
        raise RuntimeError("task directory not found: " + task_dir)

    validate_task_inputs(harness_dir, task_dir)

    task_json_path = os.path.join(task_dir, "task.json")
    meta = load_json(task_json_path, {"status": "open", "attempts": 0})
    status = meta.get("status")

    if status == "passed":
        _cap("Skip " + task_id + " (already passed)")
        return "passed"

    if status == "escalated":
        _cap("[ESCALATE] " + task_id + " is awaiting human review")
        return "escalated"

    if status == "blocked":
        _cap("Task " + task_id + " was blocked - resetting to open")
        meta["status"] = "open"

    attempts_done = int(meta.get("attempts") or 0)
    if attempts_done >= MAX_ATTEMPTS:
        _cap("Task " + task_id + " exhausted attempts - escalating")
        meta["status"] = "escalated"
        write_json(task_json_path, meta)
        return "escalated"

    rubric = read_text(os.path.join(task_dir, "q.md"))
    judge_messages = []
    consecutive_empty = 0
    prev_issues = None
    retry_note = None
    prior_verdict = None
    agent_was_told = None

    final_status = "failed"

    for attempt_num in range(attempts_done + 1, MAX_ATTEMPTS + 1):
        if _stop_event.is_set():
            meta["attempts"] = attempt_num - 1
            meta["status"] = "interrupted"
            write_json(task_json_path, meta)
            return "interrupted"

        run_dir = os.path.join(task_dir, "runs", "run_" + str(attempt_num))
        os.makedirs(run_dir, exist_ok=True)

        with _lock:
            _current_task = task_id
            _current_run = "run_" + str(attempt_num)
            _run_start_time = time.time()

        _cap("")
        _cap("Task " + task_id + " attempt " + str(attempt_num) + "/" + str(MAX_ATTEMPTS) + " [judge: " + judge_provider + " | agent: " + AGENT_MODEL + "]")

        stage_workspace(harness_dir, task_dir, workspace_dir, attempt_num)

        # save staged prompt immediately after staging, before agent runs
        save_agent_initial_prompt(run_dir, workspace_dir, task_id, attempt_num)

        run_start = time.time()
        try:
            agent_rc, agent_timed_out, agent_elapsed = _run_agent_web(workspace_dir, run_dir)
        finally:
            unstage_workspace(workspace_dir)

        if agent_timed_out:
            verdict = make_escalation_verdict(
                "agent timed out after " + str(AGENT_TIMEOUT_S) + "s",
                "Agent timed out. The task may be too broad or the agent is looping.",
                "agent_timeout",
                run_start,
                agent_elapsed,
            )
            write_json(os.path.join(run_dir, "verdict.json"), verdict)
            meta["attempts"] = attempt_num
            meta["status"] = "escalated"
            write_json(task_json_path, meta)
            _cap("  [escalate] agent timed out - human review required")
            final_status = "escalated"
            break

        report_text, image_paths = collect_task_report(workspace_dir)
        log_tail = tail_file(os.path.join(run_dir, "stdout.log"), LOG_TAIL_LINES)

        if not report_text and not image_paths:
            consecutive_empty += 1
            _cap("  task_report/ is empty (consecutive: " + str(consecutive_empty) + ")")
            if consecutive_empty >= 2:
                verdict = make_escalation_verdict(
                    "no output for " + str(consecutive_empty) + " consecutive attempts",
                    "Agent consistently produces no output. Verify task instructions and tooling.",
                    "consecutive_empty_reports",
                    run_start,
                    agent_elapsed,
                )
                write_json(os.path.join(run_dir, "verdict.json"), verdict)
                meta["attempts"] = attempt_num
                meta["status"] = "escalated"
                write_json(task_json_path, meta)
                _cap("  [escalate] consecutive empty reports - human review required")
                final_status = "escalated"
                break
        else:
            consecutive_empty = 0
            if image_paths:
                _cap("  judge: " + str(len(image_paths)) + " image(s): " + str([os.path.basename(p) for p in image_paths]))

        if not judge_messages:
            judge_text = build_first_judge_text(rubric, report_text, log_tail, AGENT_MODEL)
        else:
            judge_text = build_followup_judge_text(
                report_text,
                log_tail,
                attempt_num,
                AGENT_MODEL,
                prior_verdict=prior_verdict,
                agent_was_told=agent_was_told,
                retry_note=retry_note,
            )
            retry_note = None

        judge_messages.append({"role": "user", "content": build_content_items(judge_text, image_paths)})

        # save complete judge conversation before calling, for diffable audit trail
        save_judge_input(run_dir, judge_messages, task_id, attempt_num)

        _cap("  calling judge (" + judge_provider + ")...")

        try:
            verdict_raw, verdict = call_judge_turn(judge_messages, judge_provider)
        except Exception as e:
            _cap("  [escalate] judge call/parse failed: " + str(e))
            verdict = make_escalation_verdict(
                "judge failed: " + str(e),
                "Judge call failed. Check q.md and judge model configuration.",
                "judge_failure",
                run_start,
                agent_elapsed,
            )
            write_json(os.path.join(run_dir, "verdict.json"), verdict)
            meta["attempts"] = attempt_num
            meta["status"] = "escalated"
            write_json(task_json_path, meta)
            final_status = "escalated"
            break

        judge_messages.append({"role": "assistant", "content": verdict_raw})
        verdict["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(run_start))
        verdict["elapsed_s"] = agent_elapsed
        verdict["agent_model"] = AGENT_MODEL
        verdict["judge_provider"] = judge_provider
        write_json(os.path.join(run_dir, "verdict.json"), verdict)
        meta["attempts"] = attempt_num
        write_json(task_json_path, meta)

        conf = float(verdict.get("confidence", 0))
        _cap("Verdict: status=" + str(verdict.get("status")) + " confidence={:.2f}".format(conf))
        if verdict.get("issues"):
            _cap("Issues: " + str(verdict["issues"]))

        if verdict.get("status") == "passed" and conf >= LOW_CONF_THRESHOLD:
            meta["status"] = "passed"
            write_json(task_json_path, meta)
            _cap("Task " + task_id + " PASSED on attempt " + str(attempt_num))
            final_status = "passed"
            break

        if verdict.get("status") == "passed":
            _cap("  [warn] low-confidence pass ({:.2f} < {:.2f}) - treating as failed".format(conf, LOW_CONF_THRESHOLD))
            retry_note = 'Prior verdict was "passed" but confidence {:.2f} < required {:.2f}. ' "Please evaluate more decisively.".format(conf, LOW_CONF_THRESHOLD)
            verdict["original_status"] = "passed"
            verdict["status"] = "failed"
            if not verdict.get("issues"):
                verdict["issues"] = ["pass confidence below threshold ({:.2f} < {:.2f})".format(conf, LOW_CONF_THRESHOLD)]
            write_json(os.path.join(run_dir, "verdict.json"), verdict)
            prior_verdict = verdict
            agent_was_told = format_feedback_for_agent(verdict, attempt_num)
            prev_issues = sorted(verdict.get("issues", []))
            if attempt_num == MAX_ATTEMPTS:
                _cap("  [escalate] low-confidence pass on final attempt - human review required")
                verdict["escalation_reason"] = "low_confidence_pass_on_final_attempt"
                write_json(os.path.join(run_dir, "verdict.json"), verdict)
                meta["status"] = "escalated"
                write_json(task_json_path, meta)
                final_status = "escalated"
                break
            continue

        curr_issues = sorted(verdict.get("issues", []))
        if prev_issues is not None and curr_issues and curr_issues == prev_issues:
            _cap("  [escalate] identical issues on consecutive verdicts - agent is stuck")
            verdict["escalation_reason"] = "repeated_issues"
            write_json(os.path.join(run_dir, "verdict.json"), verdict)
            meta["status"] = "escalated"
            write_json(task_json_path, meta)
            final_status = "escalated"
            break

        prior_verdict = verdict
        agent_was_told = format_feedback_for_agent(verdict, attempt_num)
        prev_issues = curr_issues

        if attempt_num == MAX_ATTEMPTS:
            _cap("  [escalate] all " + str(MAX_ATTEMPTS) + " attempts exhausted without passing")
            verdict["escalation_reason"] = "max_attempts_exhausted"
            write_json(os.path.join(run_dir, "verdict.json"), verdict)
            meta["status"] = "escalated"
            write_json(task_json_path, meta)
            final_status = "escalated"
            break

    return final_status


# queue runner thread


def _queue_runner():
    global _runner_status, _current_task, _current_run, _error

    sys.stdout = _Tee(_orig_stdout)

    with _lock:
        jp = _judge_provider

    try:
        check_api_keys(judge_provider=jp)
    except RuntimeError as e:
        with _lock:
            _runner_status = "stopped"
            _error = str(e)
        _cap("[ERROR] " + str(e))
        sys.stdout = _orig_stdout
        return

    with _lock:
        _runner_status = "running"
        _error = None

    try:
        queue = read_queue(HARNESS_DIR)
        _cap("Queue: " + str(len(queue)) + " tasks")
        _cap("Judge: " + jp)
        _cap("Agent: " + AGENT_MODEL)
        _cap("")

        for task_id in queue:
            if _stop_event.is_set():
                _cap("[STOPPED] Queue halted by user")
                break
            # capture current judge selection at start of each task so a
            # mid-queue change takes effect on the next task rather than mid-task
            with _lock:
                jp = _judge_provider
            try:
                status = _run_task(HARNESS_DIR, WORKSPACE, task_id, jp)
            except Exception as e:
                _cap("[ERROR] task " + task_id + ": " + str(e))
                _cap(traceback.format_exc())
                with _lock:
                    _error = str(e)
                break
            if status in ("escalated", "interrupted"):
                if status == "escalated":
                    _cap("[ESCALATE] " + task_id + " requires human review.")
                    _cap("[ESCALATE] Use [FLAG:PASS] to accept or [RETRY] to retry from scratch.")
                break
        else:
            _cap("")
            _cap("All tasks complete.")

    except Exception as e:
        with _lock:
            _error = str(e)
        _cap("[ERROR] " + str(e))
        _cap(traceback.format_exc())
    finally:
        sys.stdout = _orig_stdout
        with _lock:
            _runner_status = "stopped"
            _current_task = None
            _current_run = None


# helpers


def _load_tasks():
    try:
        queue = read_queue(HARNESS_DIR)
    except Exception:
        return []
    tasks = []
    for i, task_id in enumerate(queue):
        task_dir = os.path.join(HARNESS_DIR, "tasks", task_id)
        task_json_path = os.path.join(task_dir, "task.json")
        meta = load_json(task_json_path, {"status": "open", "attempts": 0})
        p_path = os.path.join(task_dir, "p.md")
        q_path = os.path.join(task_dir, "q.md")
        p_text = read_text(p_path).strip() if os.path.exists(p_path) else ""
        q_text = read_text(q_path).strip() if os.path.exists(q_path) else ""
        last_verdict = {}
        attempts = int(meta.get("attempts") or 0)
        for n in range(attempts, 0, -1):
            vp = os.path.join(task_dir, "runs", "run_" + str(n), "verdict.json")
            v = load_json(vp, {})
            if v:
                last_verdict = v
                break
        tasks.append(
            {
                "id": task_id,
                "index": i + 1,
                "status": meta.get("status", "open"),
                "attempts": attempts,
                "p_text": p_text,
                "q_text": q_text,
                "last_verdict": last_verdict,
            }
        )
    return tasks


def _get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "0.0.0.0"


# Flask routes


@app.route("/")
def index():
    return HTML


@app.route("/api/state")
def api_state():
    with _lock:
        rs = _runner_status
        ct = _current_task
        cr = _current_run
        jp = _judge_provider
        rst = _run_start_time
        err = _error
    elapsed = int(time.time() - rst) if rst and rs == "running" else 0
    return jsonify(
        {
            "runner_status": rs,
            "current_task": ct,
            "current_run": cr,
            "judge_provider": jp,
            "agent_model": AGENT_MODEL,
            "elapsed": elapsed,
            "error": err,
            "tasks": _load_tasks(),
            "max_attempts": MAX_ATTEMPTS,
        }
    )


@app.route("/api/judge", methods=["GET", "POST"])
def api_judge():
    global _judge_provider
    if request.method == "POST":
        with _lock:
            if _runner_status == "running":
                return jsonify({"error": "cannot change judge while running"}), 400
        data = request.json or {}
        new_judge = data.get("judge")
        valid = [o["key"] for o in JUDGE_PROVIDER_OPTIONS]
        with _lock:
            if new_judge is not None:
                if new_judge not in valid:
                    return jsonify({"error": "invalid judge provider: " + new_judge}), 400
                _judge_provider = new_judge
            jp = _judge_provider
        return jsonify({"judge": jp})
    with _lock:
        jp = _judge_provider
    return jsonify(
        {
            "judge": jp,
            "judge_options": JUDGE_PROVIDER_OPTIONS,
        }
    )


@app.route("/api/console/stream")
def api_console_stream():
    def generate():
        sent = 0
        try:
            while True:
                with _lock:
                    chunk = _console[sent:]
                    sent += len(chunk)
                for line in chunk:
                    yield "data: " + json.dumps(line) + "\n\n"
                if not chunk:
                    time.sleep(0.25)
        except GeneratorExit:
            pass

    resp = Response(generate(), mimetype="text/event-stream")
    resp.headers["Cache-Control"] = "no-cache"
    resp.headers["X-Accel-Buffering"] = "no"
    resp.headers["Connection"] = "keep-alive"
    return resp


@app.route("/api/start", methods=["POST"])
def api_start():
    global _runner_thread, _runner_status
    with _lock:
        if _runner_status == "running":
            return jsonify({"error": "already running"}), 400
        _stop_event.clear()
        _console.clear()
        _runner_status = "idle"

    _runner_thread = threading.Thread(target=_queue_runner, daemon=True, name="pq-runner")
    _runner_thread.start()
    return jsonify({"ok": True})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    global _runner_status
    _stop_event.set()
    with _lock:
        if _runner_status == "running":
            _runner_status = "stopping"
    return jsonify({"ok": True})


@app.route("/api/task/<task_id>/retry", methods=["POST"])
def api_retry(task_id):
    task_json_path = os.path.join(HARNESS_DIR, "tasks", task_id, "task.json")
    meta = load_json(task_json_path, {"status": "open", "attempts": 0})
    meta["status"] = "open"
    meta["attempts"] = 0
    write_json(task_json_path, meta)
    return jsonify({"ok": True})


@app.route("/api/task/<task_id>/pass", methods=["POST"])
def api_pass(task_id):
    task_json_path = os.path.join(HARNESS_DIR, "tasks", task_id, "task.json")
    meta = load_json(task_json_path, {"status": "open", "attempts": 0})
    meta["status"] = "passed"
    write_json(task_json_path, meta)
    return jsonify({"ok": True})


@app.route("/api/task/<task_id>/p", methods=["GET", "POST"])
def api_p(task_id):
    p_path = os.path.join(HARNESS_DIR, "tasks", task_id, "p.md")
    if request.method == "POST":
        write_text(p_path, request.json.get("content", ""))
        return jsonify({"ok": True})
    return jsonify({"content": read_text(p_path) if os.path.exists(p_path) else ""})


@app.route("/api/task/<task_id>/q", methods=["GET", "POST"])
def api_q(task_id):
    q_path = os.path.join(HARNESS_DIR, "tasks", task_id, "q.md")
    if request.method == "POST":
        write_text(q_path, request.json.get("content", ""))
        return jsonify({"ok": True})
    return jsonify({"content": read_text(q_path) if os.path.exists(q_path) else ""})


# entry point

if __name__ == "__main__":
    if not os.path.exists(HARNESS_DIR):
        print("ERROR: .pq harness directory not found in: " + WORKSPACE)
        print("Run pq_web.py from your project workspace directory.")
        sys.exit(1)

    local_ip = _get_local_ip()

    print("PQ_MINDER web")
    print("  workspace : " + WORKSPACE)
    print("  harness   : " + HARNESS_DIR)
    print("  judge     : " + WEB_DEFAULT_JUDGE_PROVIDER + " (web default)")
    print("  agent     : " + AGENT_MODEL)
    print("")
    print("  http://localhost:" + str(PORT))
    print("  http://" + local_ip + ":" + str(PORT))
    print("")
    print("  Ctrl+C to stop.")
    print("")

    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True, use_reloader=False)
