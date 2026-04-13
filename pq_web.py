#!/usr/bin/env python3

import os
import sys
import json
import socket
import threading
import time
import traceback
import subprocess
import logging

# Code style:
# - No type hinting
# - No doc strings
# - No triple quoted multi-line strings
# - No comments with repeated characters for visual page breaks like # ---
# - No non-ascii characters
# - No command line argument processing
# - No global variables unless making them local increases complexity
# - Yes strategic inline comments enhancing rapid code comprehension by real humans
# - Yes if __name__ == "__main__": main()

# Run-in-Directory structure (all paths relative to workspace root):
# <workspace>/                      run pq_minder.py from here
#   .pq/                            harness dir, hidden from agent via bubblewrap tmpfs
#     queue.txt                     task_ids one per line, in dependency order; # = comment
#     project.md                    project-wide context injected into every agent run
#     tasks/
#       <task_id>/
#         p.md                      task prompt (must exist and be non-empty)
#         q.md                      rubric for judge (must exist)
#         task.json                 persists attempt count and status across runs
#         runs/
#           run_<n>/
#             stdout.log            full agent stdout/stderr captured by pq_minder
#             verdict.json          judge output: status, confidence, issues, feedback
#   task_report/                    agent writes ALL output here (md, jpg, png only)
#     *.md                          agent self-report(s); concatenated for judge
#     *.jpg / *.png                 images produced by agent, downsampled before sending


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
    format_feedback_for_agent,
    read_text,
    write_text,
    load_json,
    write_json,
    tail_file,
    make_escalation_verdict,
    DEFAULT_AGENT_MODEL,
    VALID_AGENT_MODELS,
    MAX_ATTEMPTS,
    LOG_TAIL_LINES,
    LOW_CONF_THRESHOLD,
    AGENT_TIMEOUT_S,
    JUDGE_MODEL,
    build_content_items,
    check_api_keys,
)

WORKSPACE = os.path.abspath(os.getcwd())
HARNESS_DIR = os.path.join(WORKSPACE, ".pq")
PORT = 5000

app = Flask(__name__)
_orig_stdout = sys.stdout

# ── shared state (all mutations under _lock) ──────────────────────────────────

_lock = threading.RLock()
_stop_event = threading.Event()
_runner_thread = None

_console = []  # list[str] streamed to browser via SSE
_runner_status = "idle"
_current_task = None
_current_run = None
_current_model = DEFAULT_AGENT_MODEL
_run_start_time = None
_error = None


# ── stdout tee: captures print() output into _console ─────────────────────────


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
    """Append one line, suppressing runs of consecutive blank lines."""
    stripped = line.rstrip("\r")
    with _lock:
        if stripped.strip() or not _console or _console[-1].strip():
            _console.append(stripped)


def _cap(text):
    """Append a (possibly multi-line) string to the console buffer."""
    for line in str(text).split("\n"):
        _append_line(line)


# ── modified run_agent that feeds subprocess output into _console ──────────────


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


# ── task runner (mirrors pq_minder.run_task) ──────────────────────────────────


def _run_task(harness_dir, workspace_dir, task_id):
    global _current_task, _current_run, _current_model, _run_start_time

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
    current_agent_model = DEFAULT_AGENT_MODEL

    with _lock:
        _current_model = current_agent_model

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
            _current_model = current_agent_model
            _run_start_time = time.time()

        _cap("")
        _cap("Task " + task_id + " attempt " + str(attempt_num) + "/" + str(MAX_ATTEMPTS) + " [model: " + current_agent_model + "]")

        stage_workspace(harness_dir, task_dir, workspace_dir, attempt_num, current_agent_model)
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
            judge_text = build_first_judge_text(rubric, report_text, log_tail, current_agent_model)
        else:
            judge_text = build_followup_judge_text(
                report_text,
                log_tail,
                attempt_num,
                current_agent_model,
                prior_verdict=prior_verdict,
                agent_was_told=agent_was_told,
                retry_note=retry_note,
            )
            retry_note = None

        judge_messages.append({"role": "user", "content": build_content_items(judge_text, image_paths)})
        _cap("  calling judge (" + JUDGE_MODEL + ")...")

        try:
            verdict_raw, verdict = call_judge_turn(judge_messages)
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
        verdict["agent_model"] = current_agent_model
        write_json(os.path.join(run_dir, "verdict.json"), verdict)
        meta["attempts"] = attempt_num
        write_json(task_json_path, meta)

        conf = float(verdict.get("confidence", 0))
        _cap("Verdict: status=" + str(verdict.get("status")) + " confidence={:.2f}".format(conf))
        if verdict.get("issues"):
            _cap("Issues: " + str(verdict["issues"]))

        req_model = verdict.get("next_model")
        if req_model and req_model in VALID_AGENT_MODELS and req_model != current_agent_model:
            _cap("  [model switch] " + current_agent_model + " -> " + req_model)
            current_agent_model = req_model
            with _lock:
                _current_model = current_agent_model
        elif req_model and req_model == current_agent_model:
            _cap("  [model keep] judge confirmed " + current_agent_model)

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


# ── queue runner thread ────────────────────────────────────────────────────────


def _queue_runner():
    global _runner_status, _current_task, _current_run, _error

    sys.stdout = _Tee(_orig_stdout)

    try:
        check_api_keys()
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
        _cap("Queue: " + str(len(queue)) + " tasks | judge: " + JUDGE_MODEL)
        _cap("Agent default: " + DEFAULT_AGENT_MODEL)
        _cap("")

        for task_id in queue:
            if _stop_event.is_set():
                _cap("[STOPPED] Queue halted by user")
                break
            try:
                status = _run_task(HARNESS_DIR, WORKSPACE, task_id)
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


# ── helpers ───────────────────────────────────────────────────────────────────


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


# ── Flask routes ──────────────────────────────────────────────────────────────


@app.route("/")
def index():
    return _HTML


@app.route("/api/state")
def api_state():
    with _lock:
        rs = _runner_status
        ct = _current_task
        cr = _current_run
        cm = _current_model
        rst = _run_start_time
        err = _error
    elapsed = int(time.time() - rst) if rst and rs == "running" else 0
    return jsonify(
        {
            "runner_status": rs,
            "current_task": ct,
            "current_run": cr,
            "current_model": cm,
            "elapsed": elapsed,
            "error": err,
            "tasks": _load_tasks(),
            "judge": JUDGE_MODEL,
            "valid_models": list(VALID_AGENT_MODELS),
            "max_attempts": MAX_ATTEMPTS,
        }
    )


@app.route("/api/console/stream")
def api_console_stream():
    """SSE endpoint — one persistent connection replaces all console polling."""

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


# ── HTML single-file app ──────────────────────────────────────────────────────

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PQ MINDER</title>
<style>
*{box-sizing:border-box;margin:0;padding:0;}
html,body{height:100vh;overflow:hidden;background:#111;font-family:'Courier New',Courier,monospace;font-size:12px;color:#888;}
.app{height:100vh;display:flex;flex-direction:column;}

/* titlebar */
.tb{background:#181818;border-bottom:2px solid #272727;padding:5px 14px;display:flex;justify-content:space-between;align-items:center;flex-shrink:0;}
.tname{color:#eee;letter-spacing:.1em;font-size:13px;}
.tctrl{display:flex;gap:8px;align-items:center;}
.tstat{color:#666;font-size:11px;}

/* buttons */
.btn{border:1px solid #383838;font-family:inherit;font-size:11px;padding:2px 12px;cursor:pointer;background:transparent;color:#888;}
.btn:hover{border-color:#666;color:#ccc;}
.btn:disabled{opacity:.28;cursor:not-allowed;}
.btn-s{border-color:#2a4a2a;color:#5bd878;background:#0d1a0d;}
.btn-s:not(:disabled):hover{background:#162816;}
.btn-x{border-color:#4a2020;color:#d06060;background:#1a0d0d;}
.btn-x:not(:disabled):hover{background:#2e1616;}

/* layout */
.body{flex:1;display:grid;grid-template-columns:420px 1fr;min-height:0;overflow:hidden;}

/* left panel */
.left{border-right:2px solid #202020;display:flex;flex-direction:column;overflow:hidden;}
.sh{background:#141414;border-bottom:1px solid #1e1e1e;padding:3px 12px;display:flex;justify-content:space-between;align-items:center;flex-shrink:0;}
.shl{color:#666;font-size:10px;letter-spacing:.05em;}
.shr{color:#555;font-size:10px;}

/* queue */
.qwrap{flex:0 0 auto;overflow-y:auto;max-height:200px;}
.qcols{display:grid;grid-template-columns:22px 1fr 68px 42px;padding:3px 12px;color:#363636;font-size:10px;border-bottom:1px solid #181818;}
.qrow{display:grid;grid-template-columns:22px 1fr 68px 42px;padding:5px 12px;border-bottom:1px solid #1c1c1c;align-items:center;cursor:pointer;border-left:3px solid #222;}
.qrow:hover{background:#151515;}
.qrow.sel{background:#171717;}
.sp{border-left-color:#1D9E75 !important;}
.sr{border-left-color:#EF9F27 !important;background:#0f0e08 !important;}
.se{border-left-color:#b04040 !important;}
.si{border-left-color:#666 !important;}
.so{border-left-color:#2a2a2a !important;}
.qnum{color:#4a4a4a;}
.qnm{color:#ddd;font-size:11px;}
.qnmd{color:#888;font-size:11px;}
.qat{color:#666;font-size:10px;}
.ql{font-size:10px;}
.lp{color:#2bbf8a;}
.lr{color:#EF9F27;}
.le{color:#cc6060;}
.li{color:#888;}
.lo{color:#555;}
.blink{animation:blink .6s step-end infinite;}
@keyframes blink{50%{opacity:0;}}

/* detail */
.dsh{background:#141414;border-top:2px solid #202020;border-bottom:1px solid #1e1e1e;padding:3px 12px;flex-shrink:0;}
.dsh span{color:#666;font-size:10px;letter-spacing:.04em;}
.det{flex:1;overflow-y:auto;padding:10px 14px 16px;}
.demp{color:#3a3a3a;font-size:11px;}
.dl{color:#777;font-size:10px;letter-spacing:.06em;margin-top:11px;margin-bottom:3px;}
.dl:first-child{margin-top:0;}
.dv{color:#aaa;font-size:11px;line-height:1.5;word-break:break-word;}
.da{display:flex;gap:5px;flex-wrap:wrap;margin-top:12px;}
.db{border:1px solid #333;color:#888;font-size:10px;padding:2px 9px;font-family:inherit;cursor:pointer;background:transparent;}
.db:hover{border-color:#666;color:#ccc;}
.dbr{border-color:#5c3800 !important;color:#dfa030 !important;}
.dbr:hover{background:#1a0e00 !important;}
.dbp{border-color:#1a3a22 !important;color:#3aaa60 !important;}
.dbp:hover{background:#0a1810 !important;}
.vb{margin-top:10px;border-top:1px solid #1e1e1e;padding-top:8px;font-size:11px;line-height:1.7;}
.vk{color:#777;}
.vp{color:#2bbf8a;}
.vf{color:#cc6060;}
.ve{color:#EF9F27;}
.vv{color:#aaa;}
.vi{color:#c07828;display:block;}
.vfb{color:#70b070;display:block;}
.vesc{color:#cc6060;display:block;}

/* console */
.right{display:flex;flex-direction:column;background:#0a0a0a;}
.ch{background:#0d0d0d;border-bottom:1px solid #1a1a1a;padding:3px 12px;display:flex;justify-content:space-between;flex-shrink:0;}
.ch span{color:#333;font-size:10px;}
.chtask{color:#555 !important;}
.cbody{flex:1;overflow-y:auto;padding:7px 14px 10px;min-height:0;}
.cbody div{font-size:11px;line-height:1.5;white-space:pre-wrap;word-break:break-all;min-height:1em;}
.cn{color:#52c068;}
.ch2{color:#80de98;}
.cd{color:#3a6644;}
.ca{color:#e09828;}
.cr{color:#d06060;}
.cb{color:#5a90cc;}
.cw{color:#bbb;}
.cs{color:#223a2c;}

/* footer */
.foot{background:#0c0c0c;border-top:1px solid #1a1a1a;padding:3px 14px;display:flex;justify-content:space-between;color:#3a3a3a;font-size:10px;flex-shrink:0;}

/* modal */
.mbg{display:none;position:fixed;inset:0;background:rgba(0,0,0,.92);z-index:100;align-items:center;justify-content:center;}
.mbg.open{display:flex;}
.mbox{background:#111;border:1px solid #303030;width:620px;max-height:80vh;display:flex;flex-direction:column;}
.mhdr{padding:7px 14px;border-bottom:1px solid #222;display:flex;justify-content:space-between;align-items:center;}
.mttl{color:#999;font-size:12px;}
.mcls{color:#555;cursor:pointer;font-size:17px;line-height:1;}
.mcls:hover{color:#ccc;}
.mbdy{flex:1;padding:8px;display:flex;flex-direction:column;overflow:hidden;}
.mbdy textarea{flex:1;min-height:280px;background:#080808;color:#aaa;border:1px solid #222;font-family:inherit;font-size:11px;padding:8px;resize:none;outline:none;line-height:1.55;}
.mbdy textarea:focus{border-color:#363636;}
.mft{padding:7px 8px;display:flex;gap:6px;justify-content:flex-end;border-top:1px solid #1a1a1a;}
.mb{border:1px solid #2a2a2a;color:#777;font-size:11px;padding:3px 12px;font-family:inherit;cursor:pointer;background:transparent;}
.mb:hover{border-color:#555;color:#ccc;}
.mbs{border-color:#2a4a2a !important;color:#5bd878 !important;background:#0d1a0d !important;}
.mbs:hover{background:#162816 !important;}

::-webkit-scrollbar{width:5px;}
::-webkit-scrollbar-track{background:#0a0a0a;}
::-webkit-scrollbar-thumb{background:#222;}
::-webkit-scrollbar-thumb:hover{background:#333;}
</style>
</head>
<body>
<div class="app">

  <div class="tb">
    <span class="tname">PQ_MINDER</span>
    <div class="tctrl">
      <button class="btn btn-s" id="btn-start" onclick="doStart()">START</button>
      <button class="btn btn-x" id="btn-stop" onclick="doStop()" disabled>STOP</button>
      <span class="tstat" id="tstat">IDLE</span>
    </div>
  </div>

  <div class="body">
    <div class="left">
      <div class="sh">
        <span class="shl" id="qcount">QUEUE [0]</span>
        <span class="shr" id="qsum"></span>
      </div>
      <div class="qcols"><span>#</span><span>TASK_ID</span><span>STATUS</span><span>ATT</span></div>
      <div class="qwrap" id="qlist"></div>
      <div class="dsh"><span id="dsht">SELECTED: —</span></div>
      <div class="det" id="det"><div class="demp">click a task to view details</div></div>
    </div>

    <div class="right">
      <div class="ch">
        <span>CONSOLE [LIVE]</span>
        <span class="chtask" id="ctask">—</span>
      </div>
      <div class="cbody" id="cbody"></div>
    </div>
  </div>

  <div class="foot">
    <span id="fagent">agent: —</span>
    <span id="fjudge">judge: —</span>
    <span id="felapsed">elapsed: —</span>
    <span id="flines">lines: 0</span>
  </div>
</div>

<div class="mbg" id="modal" onclick="if(event.target===this)closeModal()">
  <div class="mbox">
    <div class="mhdr">
      <span class="mttl" id="mttl">EDIT</span>
      <span class="mcls" onclick="closeModal()">&#xd7;</span>
    </div>
    <div class="mbdy"><textarea id="mta" spellcheck="false"></textarea></div>
    <div class="mft">
      <button class="mb" onclick="closeModal()">CANCEL</button>
      <button class="mb mbs" onclick="saveModal()">SAVE</button>
    </div>
  </div>
</div>

<script>
var selTask=null, lastState=null, modalUrl=null, es=null, lineCount=0;

var SLBL={open:'[PEND]',passed:'[PASS]',failed:'[FAIL]',escalated:'[ESC!]',interrupted:'[INT]',blocked:'[BLKD]',stopping:'[STOP]'};
var SCSS={open:'so',passed:'sp',failed:'se',escalated:'se',interrupted:'si',blocked:'so',stopping:'si'};
var SLC={open:'lo',passed:'lp',failed:'le',escalated:'le',interrupted:'li',blocked:'lo',stopping:'li'};

function esc(s){return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
function pad(n){return n<10?'0'+n:''+n;}
function fmtT(s){var h=Math.floor(s/3600),m=Math.floor((s%3600)/60),ss=s%60;return(h?pad(h)+':':'')+pad(m)+':'+pad(ss);}

function colorLine(r){
  if(!r||!r.trim()) return '<span class="cs">&nbsp;</span>';
  var t=esc(r);
  if(/\[tool call\]/i.test(r))    return '<span class="ca">'+t+'</span>';
  if(/\[escalate\]/i.test(r)||/\[ERROR\]/i.test(r)) return '<span class="cr">'+t+'</span>';
  if(/\bPASSED\b/.test(r)||/passed on attempt/i.test(r)||/All tasks complete/i.test(r)) return '<span class="ch2">'+t+'</span>';
  if(/\[warn\]/i.test(r)||/\[model switch\]/i.test(r)||/\[ESCALATE\]/i.test(r)||/\[STOPPED\]/i.test(r)||/low-confidence/i.test(r)) return '<span class="ca">'+t+'</span>';
  if(/calling judge/i.test(r)||/RESPONSE.*model=/i.test(r)||/JUDGE PROMPT/i.test(r)) return '<span class="cb">'+t+'</span>';
  if(/^Verdict:/i.test(r)) return '<span class="cw">'+t+'</span>';
  if(/^Issues:/i.test(r)) return '<span class="ca">'+t+'</span>';
  if(/^Task\s+\S+\s+attempt/i.test(r)) return '<span class="ch2">'+t+'</span>';
  if(/\[pq_minder\]/.test(r)||/^Starting agent/.test(r)||/^MCP/.test(r)||/^Agent model/.test(r)||/^Queue:/.test(r)||/^Agent default/.test(r)) return '<span class="cd">'+t+'</span>';
  return '<span class="cn">'+t+'</span>';
}

function appendLine(raw){
  var cb=document.getElementById('cbody');
  var atBot=(cb.scrollHeight-cb.scrollTop-cb.clientHeight)<70;
  var d=document.createElement('div');
  d.innerHTML=colorLine(raw);
  cb.appendChild(d);
  lineCount++;
  document.getElementById('flines').textContent='lines: '+lineCount;
  if(atBot) cb.scrollTop=cb.scrollHeight;
}

function connectSSE(){
  if(es){es.close();es=null;}
  es=new EventSource('/api/console/stream');
  es.onmessage=function(e){appendLine(JSON.parse(e.data));};
  es.onerror=function(){};
}

function pollState(){
  fetch('/api/state').then(function(r){return r.json();}).then(function(s){
    lastState=s;
    renderStatus(s);
    renderQueue(s);
    renderFooter(s);
    if(selTask){var t=s.tasks.find(function(x){return x.id===selTask;});if(t)renderDetail(t,s);}
  }).catch(function(){});
}

function renderStatus(s){
  var rs=s.runner_status,tasks=s.tasks||[];
  var np=tasks.filter(function(t){return t.status==='passed';}).length;
  var ne=tasks.filter(function(t){return t.status==='escalated';}).length;
  var lbl='';
  if(rs==='running')       lbl='<span style="color:#EF9F27">RUNNING</span>';
  else if(rs==='stopping') lbl='<span style="color:#d06060">STOPPING</span>';
  else if(rs==='stopped')  lbl=s.error?'<span style="color:#d06060">ERROR</span>':'<span style="color:#555">STOPPED</span>';
  else                     lbl='<span style="color:#444">IDLE</span>';
  if(s.current_task&&rs==='running')
    lbl+=' | <span style="color:#666">'+esc(s.current_task)+(s.current_run?' / '+esc(s.current_run):'')+'</span>';
  lbl+=' &nbsp;|&nbsp; TASKS:'+tasks.length+' PASSED:'+np;
  if(ne) lbl+=' ESC:<span style="color:#d06060">'+ne+'</span>';
  document.getElementById('tstat').innerHTML=lbl;
  document.getElementById('btn-start').disabled=(rs==='running'||rs==='stopping');
  document.getElementById('btn-stop').disabled=(rs!=='running');
  document.getElementById('ctask').textContent=
    (s.current_task&&rs==='running')?(s.current_task+(s.current_run?' / '+s.current_run:'')):'—';
}

function renderQueue(s){
  var tasks=s.tasks||[], html='';
  tasks.forEach(function(t){
    var isRun=(s.current_task===t.id&&s.runner_status==='running');
    var sc=isRun?'sr':(SCSS[t.status]||'so');
    var lc=isRun?'lr':(SLC[t.status]||'lo');
    var lbl=isRun?'[RUN]':(SLBL[t.status]||'[???]');
    var nc=(t.status==='passed')?'qnm':'qnmd';
    var sel=(selTask===t.id)?' sel':'';
    html+='<div class="qrow '+sc+sel+'" data-id="'+esc(t.id)+'">';
    html+='<span class="qnum">'+t.index+'</span>';
    html+='<span class="'+nc+'">'+esc(t.id)+'</span>';
    html+='<span class="ql '+lc+'">'+lbl+(isRun?'<span class="blink"> &#9646;</span>':'')+'</span>';
    html+='<span class="qat">'+t.attempts+'/'+(s.max_attempts||3)+'</span>';
    html+='</div>';
  });
  var el=document.getElementById('qlist');
  el.innerHTML=html;
  el.querySelectorAll('.qrow').forEach(function(r){
    r.addEventListener('click',function(){selectTask(this.dataset.id);});
  });
  document.getElementById('qcount').textContent='QUEUE ['+tasks.length+']';
  var np=tasks.filter(function(t){return t.status==='passed';}).length;
  var ne=tasks.filter(function(t){return t.status==='escalated';}).length;
  var sum=np+'/'+tasks.length+' PASSED';
  if(ne) sum+=' | '+ne+' ESC';
  document.getElementById('qsum').textContent=sum;
}

function selectTask(id){
  selTask=id;
  if(!lastState) return;
  var t=lastState.tasks.find(function(x){return x.id===id;});
  if(t) renderDetail(t,lastState);
}

function renderDetail(t,s){
  var isRun=(s.current_task===t.id&&s.runner_status==='running');
  document.getElementById('dsht').textContent='SELECTED: '+t.id;
  var h='';
  h+='<div class="dl">P.MD</div>';
  h+='<div class="dv">'+esc(t.p_text.length>240?t.p_text.substring(0,240)+'\u2026':t.p_text)+'</div>';
  h+='<div class="dl">Q.MD</div>';
  h+='<div class="dv">'+esc(t.q_text.length>170?t.q_text.substring(0,170)+'\u2026':t.q_text)+'</div>';
  h+='<div class="da">';
  h+='<button class="db" onclick="editFile(\''+t.id+'\',\'p\')">[EDIT P]</button>';
  h+='<button class="db" onclick="editFile(\''+t.id+'\',\'q\')">[EDIT Q]</button>';
  if(!isRun){
    h+='<button class="db dbr" onclick="doRetry(\''+t.id+'\')">[RETRY]</button>';
    if(t.status!=='passed')
      h+='<button class="db dbp" onclick="doPass(\''+t.id+'\')">[FLAG:PASS]</button>';
  }
  h+='</div>';
  var v=t.last_verdict;
  if(v&&v.status){
    var vc=(v.status==='passed')?'vp':(v.escalation_reason?'ve':'vf');
    h+='<div class="vb">';
    h+='<span class="vk">VERDICT: </span><span class="'+vc+'">'+v.status.toUpperCase()+'</span>';
    if(v.confidence!==undefined) h+=' <span class="vk">conf=</span><span class="vv">'+parseFloat(v.confidence).toFixed(2)+'</span>';
    if(v.agent_model) h+=' <span class="vk">model=</span><span class="vv">'+esc(v.agent_model.split('/').pop())+'</span>';
    if(v.elapsed_s)   h+=' <span class="vk">t=</span><span class="vv">'+fmtT(v.elapsed_s)+'</span>';
    if(v.issues&&v.issues.length) v.issues.forEach(function(i){h+='<span class="vi">&#8627; '+esc(i)+'</span>';});
    if(v.feedback) h+='<span class="vfb">feedback: '+esc(v.feedback)+'</span>';
    if(v.escalation_reason) h+='<span class="vesc">esc: '+esc(v.escalation_reason)+'</span>';
    h+='</div>';
  }
  document.getElementById('det').innerHTML=h;
}

function renderFooter(s){
  document.getElementById('fagent').textContent='agent: '+(s.current_model||'—');
  document.getElementById('fjudge').textContent='judge: '+(s.judge||'—');
  document.getElementById('felapsed').textContent=
    (s.runner_status==='running'&&s.elapsed)?'elapsed: '+fmtT(s.elapsed):'elapsed: —';
}

function doStart(){
  fetch('/api/start',{method:'POST'}).then(function(){
    document.getElementById('cbody').innerHTML='';
    lineCount=0;
    document.getElementById('flines').textContent='lines: 0';
    connectSSE();
    pollState();
  });
}
function doStop(){fetch('/api/stop',{method:'POST'}).then(pollState);}

function doRetry(id){
  if(!confirm('Reset "'+id+'" to open (attempts=0)?')) return;
  fetch('/api/task/'+id+'/retry',{method:'POST'}).then(pollState);
}
function doPass(id){
  if(!confirm('Mark "'+id+'" as PASSED?')) return;
  fetch('/api/task/'+id+'/pass',{method:'POST'}).then(pollState);
}

function editFile(id,type){
  modalUrl='/api/task/'+id+'/'+type;
  document.getElementById('mttl').textContent='EDIT '+type.toUpperCase()+'.MD \u2014 '+id;
  fetch('/api/task/'+id+'/'+type).then(function(r){return r.json();}).then(function(d){
    document.getElementById('mta').value=d.content||'';
    document.getElementById('modal').classList.add('open');
    setTimeout(function(){document.getElementById('mta').focus();},40);
  });
}
function closeModal(){document.getElementById('modal').classList.remove('open');}
function saveModal(){
  var c=document.getElementById('mta').value;
  fetch(modalUrl,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({content:c})})
    .then(function(){closeModal();pollState();});
}

document.addEventListener('keydown',function(e){if(e.key==='Escape')closeModal();});

// only one poll (state, 2s) + one persistent SSE connection for console
setInterval(pollState, 2000);
pollState();
connectSSE();
</script>
</body>
</html>"""

# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not os.path.exists(HARNESS_DIR):
        print("ERROR: .pq harness directory not found in: " + WORKSPACE)
        print("Run pq_web.py from your project workspace directory.")
        sys.exit(1)

    local_ip = _get_local_ip()

    print("PQ_MINDER web")
    print("  workspace : " + WORKSPACE)
    print("  harness   : " + HARNESS_DIR)
    print("  judge     : " + JUDGE_MODEL)
    print("  agent     : " + DEFAULT_AGENT_MODEL)
    print("")
    print("  http://localhost:" + str(PORT))
    print("  http://" + local_ip + ":" + str(PORT))
    print("")
    print("  Ctrl+C to stop.")
    print("")

    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True, use_reloader=False)
