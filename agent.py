import os
import sys
import json
import zlib
import select
import signal
import subprocess
import tempfile
import requests
import time
import random
import tiktoken
import re
import urllib.parse

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

# Harness is targeted at flash/mimo class models (cheap, fast, weaker at exotic
# formats and long-horizon discipline). Design choices that follow from that:
# - search_web is a single tool (encode + navigate + extract) so the model never
#   hand-builds DDG URLs
# - str_replace exists alongside hashline edit_file because small models are
#   trained heavily on search/replace style edits
# - soft tool-call budget with injected wrap-up notices, since cheap models are
#   bad at time estimation
# - reasoning effort is high at plan and wrap-up, medium in the middle
#
# DeepSeek V4 thinking-mode REQUIRES the prior assistant turn's reasoning state
# (reasoning_details, reasoning, or reasoning_content) to be passed back when
# that turn included a tool_call - omitting it causes a 400 from the upstream
# DeepSeek provider through OpenRouter. We rely on appending the assistant
# message dict from the response verbatim into the conversation, which carries
# all returned fields through to the next request. This append-verbatim rule is
# harmless for providers that don't need it, so it stays regardless of MODEL.
# MiniMax M2.x/M3 "highly recommend preserving reasoning between turns" and
# Kimi K2.7 Code mandates it - both are handled by the same append-verbatim.


ALL_MODELS = {
    "ds-v4-pro": "deepseek/deepseek-v4-pro",
    "kimi-k2.7-code": "moonshotai/kimi-k2.7-code",
    "glm-5.1": "z-ai/glm-5.1",
    "gemma-4-31b": "google/gemma-4-31b-it",
    "minimax-m3": "minimax/minimax-m3",
    "kimi-k2.6": "moonshotai/kimi-k2.6",
    "step-3.7-flash": "stepfun/step-3.7-flash",
    "ds-v4-flash": "deepseek/deepseek-v4-flash",
}


MODEL_ID = "ds-v4-flash"


# NOTE: WE ALWAYS WANT TO APPEND :exacto, I the user accept any consequences of this decision
MODEL = ALL_MODELS[MODEL_ID] + ":exacto"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Per-model tuning for best results through OpenRouter's unified API.
# Models not listed here use defaults: effort-based reasoning sandwich, 8000
# max_tokens. Key findings from provider docs and OpenRouter model pages:
#
# reasoning_mode controls how the reasoning param is constructed per-call:
#   "effort"     (default) - send reasoning.effort = high/medium/etc
#   "always_on"  - model always thinks; omit reasoning param entirely so
#                  the provider doesn't reject or misinterpret it
#   "disabled"   - send reasoning.enabled = false; model works best without
#                  thinking in agentic tool-calling loops
#
# Kimi K2.7 Code: always-on thinking that cannot be disabled. Kimi's docs
# mandate max_tokens >= 16000 so reasoning_content + content aren't truncated.
# The model also requires preserving reasoning_content across all turns, which
# the append-verbatim pattern already provides.
#
# MiMo V2.x: Xiaomi's integration guidance explicitly says "turn off reasoning
# mode for the best and fastest performance" when using agentic tools.
#
# DeepSeek V4: supports reasoning.effort "high" and "xhigh" (xhigh maps to
# max reasoning). The default effort sandwich works well here.
#
# MiniMax M3/M2.x: supports reasoning via OpenRouter's unified API. MiniMax
# "highly recommends preserving reasoning between turns" - already handled.
#
# Gemma 4 (26b, 31b): configurable reasoning/thinking mode via OpenRouter.
# Standard effort sandwich works. 256K context, fine with 150K compaction.
#
# All other models use OpenRouter's normalized reasoning.effort and work fine
# with the default effort sandwich.

MODEL_OVERRIDES = {
    # Kimi: always-on thinking, higher max_tokens mandatory
    "kimi-k2.7-code": {"max_tokens": 16000, "reasoning_mode": "always_on"},
    "kimi-k2.6": {"max_tokens": 16000},
    # MiMo: reasoning off for agentic tool-calling per Xiaomi guidance
    "mimo-v2.5": {"reasoning_mode": "disabled"},
    "mimo-v2.5-pro": {"reasoning_mode": "disabled"},
}


def _mcfg(key, default=None):
    return MODEL_OVERRIDES.get(MODEL_ID, {}).get(key, default)


# Reasoning sandwich: spend thinking where it pays. High effort on the first
# turn (planning) and again once the wrap-up threshold is crossed (verification
# and report writing); medium for the mechanical middle turns. Set any of these
# to None to disable the reasoning param for that phase.
REASONING_EFFORT_PLAN = "high"
REASONING_EFFORT_WORK = "medium"
REASONING_EFFORT_WRAPUP = "high"
REASONING_EFFORT_COMPACTION = "high"

# Soft tool-call budget. The model is told the budget in the system prompt and
# gets injected notices as it approaches and exceeds it. There is no hard stop;
# pq_minder's wall clock remains the only hard limit.
MAX_STEPS_SUGGESTION = 80
WRAPUP_WARN_AT = 60

# Playwright page extracts are the only tool results we truncate; everything
# else enters the conversation untouched. End-truncation is intentional here:
# page content front-loads the useful part.
MAX_PLAYWRIGHT_RESULT_TOKENS = 9000

# If the model ends its turn without having written task_report/report.md we
# nudge it instead of exiting, up to this many times, so a forgetful final turn
# doesn't burn an entire pq_minder attempt.
MAX_REPORT_RESCUES = 3

# Adaptive output boost: on first truncation, max_tokens is doubled up to this
# cap. At DS V4 Flash rates ($0.18/M output), 32000 reserves ~$0.006 per
# request through OpenRouter - negligible. For truly large files the
# decomposition nudge kicks in on the second truncation.
MAX_OUTPUT_BOOST = 32000

AGENT_DIR = os.environ.get("AGENT_DIR", os.path.dirname(os.path.abspath(__file__)))
WORKSPACE = os.path.abspath(os.getcwd())

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    sys.exit("Error: OPENROUTER_API_KEY environment variable is not set.")

HEADERS = {
    "Authorization": "Bearer " + OPENROUTER_API_KEY,
    "Content-Type": "application/json",
}


_enc = tiktoken.get_encoding("cl100k_base")


def ts():
    return time.strftime("[%H:%M:%S] ")


def truncate_playwright_text(text):
    # single-pass end truncation with one notice appended, replacing the old
    # iterative trim loop that could stack garbled notices on top of each other
    toks = _enc.encode(text)
    if len(toks) <= MAX_PLAYWRIGHT_RESULT_TOKENS:
        return text
    head = _enc.decode(toks[:MAX_PLAYWRIGHT_RESULT_TOKENS])
    return head + "\n**USER AGENT HARNESS NOTICE:** This content has been trimmed due to excessive length. Please use this if you can, otherwise find another way to achieve your goal.\n"


def est_messages_tokens(messages):
    # pure estimator - never mutates messages. Used only to decide when to
    # compact; real token counts from the API overwrite the estimate each turn.
    tokens = 3  # reply priming overhead
    for msg in messages:
        tokens += 3  # per-message framing
        for key, val in msg.items():
            if val is None:
                continue
            if isinstance(val, str):
                tokens += len(_enc.encode(val))
            else:
                tokens += len(_enc.encode(json.dumps(val)))
    return tokens


def post_with_retry(payload):
    for attempt in range(9):
        if attempt > 0:
            p = attempt - 1
            delay = random.uniform(2**p, 2 ** (p + 1))
            time.sleep(delay)
        try:
            resp = requests.post(OPENROUTER_URL, headers=HEADERS, json=payload, timeout=60)
        except requests.exceptions.Timeout:
            if attempt < 8:
                print(ts() + "  [error] request timed out, retrying (attempt " + str(attempt + 1) + "/8)...")
                continue
            raise
        except requests.exceptions.ConnectionError as e:
            # covers ChunkedEncodingError, broken pipes, reset connections.
            # these are transient network faults, not OpenRouter account errors.
            if attempt < 8:
                print(ts() + "  [error] connection error, retrying (attempt " + str(attempt + 1) + "/8): " + str(e)[:120])
                continue
            raise
        # retry all 5xx, not just 503 - OpenRouter throws 502/520/524 regularly
        if (resp.status_code == 429 or resp.status_code >= 500) and attempt < 8:
            print(ts() + "  [error] " + str(resp.status_code) + " transient error, retrying (attempt " + str(attempt + 1) + "/8)...")
            continue
        if not resp.ok:
            print(ts() + "\n[error] status=" + str(resp.status_code))
            for key, val in resp.headers.items():
                print("  " + key + ": " + val)
            body_preview = resp.text[:300].replace("\n", " ").strip()
            if body_preview:
                print("  body: " + body_preview)
        resp.raise_for_status()
        # Validate the body parses as JSON before declaring success. OpenRouter
        # occasionally returns 200 OK with truncated or SSE-style bodies (we've
        # seen this most often immediately after compaction, when payloads peak).
        # Treat parse failures like transient 429/503s and retry rather than
        # letting .json() crash the run at the call site. ValueError covers both
        # json.JSONDecodeError and requests.exceptions.JSONDecodeError.
        try:
            data = resp.json()
        except ValueError as e:
            if attempt < 8:
                body_preview = resp.text[:200].replace("\n", " ").strip()
                print(ts() + "  [error] response body not valid JSON, retrying (attempt " + str(attempt + 1) + "/8): " + str(e))
                if body_preview:
                    print("  body: " + body_preview)
                continue
            raise
        # OpenRouter can return 200 OK with an error body instead of choices.
        # This happens on provider-side failures, content moderation, and
        # transient upstream errors. Permanent errors (auth, billing) arrive
        # with non-200 status and are caught by raise_for_status above.
        if "choices" not in data and "error" in data:
            err = data["error"]
            code = err.get("code", 0)
            msg = err.get("message", "unknown error")
            # 4xx-class errors from the error body are permanent (bad request,
            # auth, billing) - surface them immediately so the user sees the
            # raw OpenRouter message. Provider errors (5xx) and content filter
            # issues after generation are retryable.
            if isinstance(code, int) and 400 <= code < 500 and code != 429:
                raise RuntimeError("OpenRouter error " + str(code) + ": " + msg)
            if attempt < 8:
                print(ts() + "  [error] response has error instead of choices (code=" + str(code) + "), retrying (attempt " + str(attempt + 1) + "/8): " + msg[:120])
                continue
            raise RuntimeError("OpenRouter error after retries: " + str(code) + ": " + msg)
        break
    return resp


def post_compaction(payload):
    try:
        resp = post_with_retry(payload)
        resp.raise_for_status()
        return resp
    except requests.exceptions.HTTPError as e:
        # defensive fallback: if the server rejects the reasoning param, retry without it.
        # with the round-trip working this should rarely fire, but keeping it as a safety
        # net for future provider quirks. the warning will be visible in logs.
        if e.response.status_code == 400 and "reasoning" in payload:
            print(ts() + "  [warn] reasoning param rejected, retrying without it...")
            payload = {k: v for k, v in payload.items() if k != "reasoning"}
            resp = post_with_retry(payload)
            resp.raise_for_status()
            return resp
        raise


def extract_compaction_summary(raw_msg):
    summary = raw_msg.get("content")
    if not summary or not isinstance(summary, str) or summary.strip().lower() in ("", "none", "yes"):
        # fall back through OpenRouter reasoning shapes: plain string fields first,
        # then the structured reasoning_details array (sequence of {type, text, ...} blocks)
        summary = raw_msg.get("reasoning") or raw_msg.get("reasoning_content")
        if not summary:
            details = raw_msg.get("reasoning_details") or []
            parts = []
            for d in details:
                if isinstance(d, dict):
                    t = d.get("text") or d.get("content")
                    if t:
                        parts.append(t)
            if parts:
                summary = "\n".join(parts)
    if summary:
        summary = summary.strip()
    return summary or None


def build_reasoning_param(effort):
    # construct the reasoning dict for the payload based on per-model config.
    # returns None when no reasoning param should be sent.
    rmode = _mcfg("reasoning_mode", "effort")
    if rmode == "always_on":
        # model always thinks internally; sending effort is unnecessary and
        # some providers reject it. omit the param entirely.
        return None
    if rmode == "disabled":
        # model performs best without thinking in agentic loops
        return {"enabled": False}
    # default "effort" mode
    if effort:
        return {"effort": effort}
    return None


def chat(messages, tools, new_messages, state, session_messages, effort):
    # 150K is a conservative cap. Long-context quality degrades well before
    # nominal limits, and compaction itself is a fragile operation - bigger
    # headroom = fewer compaction events = fewer crashes.
    MAX_CONTEXT_LENGTH = 150000

    new_prompt_tokens = est_messages_tokens(new_messages)
    new_prompt_tokens = int(round(0.2221 * (new_prompt_tokens**1.1866)))
    pre_prompt_total_context = state["last_post_tokens"] + new_prompt_tokens

    if pre_prompt_total_context > MAX_CONTEXT_LENGTH:
        print(ts() + "PERFORM COMPACTION", flush=True)
        state["compaction_count"] += 1
        compaction_prompt = (
            "The agent context limit was reached and this session is being compacted. "
            "The conversation above is the full committed session history. "
            "Write a compact plain-text summary covering:\n"
            "1. What actions were taken and whether they succeeded or failed.\n"
            "2. Current state of any modified files (exact filenames, key content or structure).\n"
            "3. Any discovered facts relevant to completing the task (e.g. specific data values found).\n"
            "4. Immediate next step.\n"
            "Include any prior compaction summaries in condensed form.\n"
            "Do NOT summarize the system prompt or initial task instructions - those are provided fresh.\n"
            "Be terse. If this summary exceeds 9000 tokens it will be clipped.\n"
            "Minimize thinking and output summary content.\n"
        )
        max_tok = _mcfg("max_tokens", 16000)
        compaction_payload = {
            "model": MODEL,
            "max_tokens": max_tok,
            "messages": messages + new_messages + [{"role": "user", "content": compaction_prompt}],
        }
        rparam = build_reasoning_param(REASONING_EFFORT_COMPACTION)
        if rparam:
            compaction_payload["reasoning"] = rparam
        new_messages.clear()

        resp_json = post_compaction(compaction_payload).json()
        raw_msg = resp_json["choices"][0]["message"]
        summary = extract_compaction_summary(raw_msg)

        print()
        print("-" * 80)
        if summary:
            print(summary)
        else:
            print(raw_msg)
        print("-" * 80)
        print()

        if not summary:
            raise Exception("compaction returned no usable summary")

        # enforce the clip promised in the compaction prompt
        toks = _enc.encode(summary)
        if len(toks) > 9000:
            summary = _enc.decode(toks[:9000]) + "\n[summary clipped at 9000 tokens]"

        summary_msg = {"role": "user", "content": "[context compacted] Session summary:\n" + summary}

        new_session = list(session_messages) + [summary_msg]
        messages.clear()
        messages += new_session

    else:
        pct = 100 * pre_prompt_total_context / MAX_CONTEXT_LENGTH
        warn = " [!]" if pct > 80 else ""
        print(ts() + "ctx={} ({:.1f}%){}".format(pre_prompt_total_context, pct, warn), flush=True)

    messages += new_messages
    new_messages.clear()

    max_tok = state.get("max_tokens_override") or _mcfg("max_tokens", 16000)
    payload = {
        "model": MODEL,
        "tools": tools,
        "tool_choice": "auto",
        "max_tokens": max_tok,
        "messages": messages,
    }
    rparam = build_reasoning_param(effort)
    if rparam:
        payload["reasoning"] = rparam

    data = post_with_retry(payload).json()

    state["last_post_tokens"] = int(data.get("usage", {}).get("prompt_tokens", 0))

    return data


# mcp helpers


def mcp_send(mcp, method, params, notify=False):
    if notify:
        msg = {"jsonrpc": "2.0", "method": method, "params": params}
        mcp["proc"].stdin.write(json.dumps(msg) + "\n")
        mcp["proc"].stdin.flush()
        return None
    mcp["id"] += 1
    msg = {"jsonrpc": "2.0", "id": mcp["id"], "method": method, "params": params}
    mcp["proc"].stdin.write(json.dumps(msg) + "\n")
    mcp["proc"].stdin.flush()
    return mcp["id"]


def mcp_recv(mcp, expected_id):
    while True:
        ready, _, _ = select.select([mcp["proc"].stdout], [], [], 60)
        if not ready:
            raise TimeoutError("MCP recv timed out after 60s")
        line = mcp["proc"].stdout.readline()
        if not line:
            raise RuntimeError("MCP server closed unexpectedly")
        msg = json.loads(line)
        if msg.get("id") == expected_id:
            if "error" in msg:
                raise RuntimeError("MCP error: " + str(msg["error"]))
            return msg["result"]


def mcp_call(mcp, method, params):
    return mcp_recv(mcp, mcp_send(mcp, method, params))


def _mcp_handshake(mcp):
    mcp_call(
        mcp,
        "initialize",
        {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "agent", "version": "0.1"},
        },
    )
    mcp_send(mcp, "notifications/initialized", {}, notify=True)


def start_mcp():
    proc = subprocess.Popen(
        ["node", os.path.join(AGENT_DIR, "mcp_server.js")],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        text=True,
        bufsize=1,
    )
    time.sleep(1)
    if proc.poll() is not None:
        sys.exit("MCP server exited immediately with code " + str(proc.returncode))
    mcp = {"proc": proc, "id": 0}
    _mcp_handshake(mcp)
    return mcp


def restart_mcp(mcp):
    mcp["proc"].terminate()
    time.sleep(1)
    proc = subprocess.Popen(
        ["node", os.path.join(AGENT_DIR, "mcp_server.js")],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        text=True,
        bufsize=1,
    )
    time.sleep(1)
    if proc.poll() is not None:
        raise RuntimeError("MCP server failed to restart, exit code " + str(proc.returncode))
    mcp["proc"] = proc
    mcp["id"] = 0
    _mcp_handshake(mcp)


def call_playwright(mcp, name, arguments):
    # shared retry/restart wrapper for all playwright-backed tools; returns
    # the extracted text already truncated to the playwright result cap
    for attempt in range(9):
        if attempt > 0:
            delay = random.uniform(2 ** (attempt - 1), 2**attempt)
            print(ts() + "  [mcp retry " + str(attempt) + "/8] waiting " + "{:.1f}".format(delay) + "s then restarting mcp...")
            time.sleep(delay)
            restart_mcp(mcp)
        try:
            result = mcp_call(mcp, "tools/call", {"name": name, "arguments": arguments})
            text = "\n".join(b["text"] for b in result.get("content", []) if b.get("type") == "text")
            return truncate_playwright_text(text)
        except TimeoutError as e:
            if attempt == 8:
                raise
            ctx = arguments.get("url", name)
            print(ts() + "  [mcp timeout] attempt " + str(attempt + 1) + "/9 on " + ctx[:80] + ": " + str(e))


# hashline helpers


def line_hash(line):
    return "{:02x}".format(zlib.crc32(line.rstrip().encode("utf-8")) % 256)


def render_hashlines(lines):
    return "\n".join(str(i) + ":" + line_hash(l) + "|" + l.rstrip() for i, l in enumerate(lines, 1))


def render_hashline_region(lines, center, radius=5):
    # fresh anchors for a window around a failed anchor, so the model can
    # self-correct without spending a turn on a full read_file
    lo = max(1, center - radius)
    hi = min(len(lines), center + radius)
    return "\n".join(str(i) + ":" + line_hash(lines[i - 1]) + "|" + lines[i - 1].rstrip() for i in range(lo, hi + 1))


def parse_anchor(anchor):
    parts = anchor.split(":")
    if len(parts) != 2:
        raise ValueError("bad anchor format: " + repr(anchor) + " - expected LINENUM:HASH e.g. '5:a3'")
    return int(parts[0]), parts[1]


def safe_path(filename):
    target = os.path.abspath(os.path.join(WORKSPACE, filename))
    if not target.startswith(WORKSPACE):
        raise ValueError("path '" + filename + "' resolves outside workspace")
    return target


# file tool implementations


def tool_write_file(filename, content):
    # guard against the model accidentally writing raw hashline output
    hashline_re = re.compile(r"^\d+:[0-9a-f]{2}\|")
    bad_lines = [l for l in content.splitlines() if hashline_re.match(l)]
    if bad_lines:
        sample = bad_lines[0]
        return "Error: content looks like raw read_file output (e.g. '" + sample + "'). Strip the 'LINENUM:HASH|' prefixes before writing."

    # enforce task_report/ restrictions: if writing into task_report/, only md/jpg/png allowed
    norm = filename.replace("\\", "/")
    if norm.startswith("task_report/") or norm.startswith("./task_report/"):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in (".md", ".jpg", ".jpeg", ".png"):
            return "Error: task_report/ only accepts .md, .jpg, and .png files. Got: " + ext

    target = safe_path(filename)
    parent = os.path.dirname(target)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        f.write(content)
    lines = content.splitlines()
    rel = os.path.relpath(target, WORKSPACE)
    print(ts() + "  [tool call] write_file: " + rel + " (" + str(len(lines)) + " lines)")
    return "Written " + str(os.stat(target).st_size) + " bytes to " + rel


def tool_read_file(filename):
    target = safe_path(filename)
    if not os.path.exists(target):
        return "Error: file not found: " + filename
    with open(target, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    rel = os.path.relpath(target, WORKSPACE)
    print(ts() + "  [tool call] read_file: " + rel + " (" + str(len(lines)) + " lines)")
    return render_hashlines(lines)


def tool_str_replace(filename, old_str, new_str):
    target = safe_path(filename)
    if not os.path.exists(target):
        return "Error: file not found: " + filename + " - use write_file to create it first"
    with open(target, "r", encoding="utf-8") as f:
        content = f.read()

    count = content.count(old_str)
    if count == 0:
        return "Error: old_str not found in " + filename + ". Match must be exact including whitespace and indentation. Use read_file to see the current content."
    if count > 1:
        return "Error: old_str appears " + str(count) + " times in " + filename + " - include more surrounding lines so it matches exactly once."

    new_content = content.replace(old_str, new_str, 1)
    with open(target, "w", encoding="utf-8") as f:
        f.write(new_content)

    rel = os.path.relpath(target, WORKSPACE)
    n_lines = len(new_content.splitlines())
    print(ts() + "  [tool call] str_replace: " + rel + " (file now " + str(n_lines) + " lines)")
    return "Replaced 1 occurrence in " + rel + ". File now has " + str(n_lines) + " lines."


def tool_edit_file(filename, start_anchor, end_anchor, new_text):
    target = safe_path(filename)
    if not os.path.exists(target):
        return "Error: file not found: " + filename + " - use write_file to create it first"
    with open(target, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    start_line, start_hash = parse_anchor(start_anchor)
    end_line, end_hash = parse_anchor(end_anchor)

    if start_line < 1 or start_line > len(lines):
        return "Error: start_anchor line " + str(start_line) + " out of range (file has " + str(len(lines)) + " lines)"
    if end_line < start_line or end_line > len(lines):
        return "Error: end_anchor line " + str(end_line) + " out of range"

    # on hash mismatch, return fresh anchors for the region so the model can
    # retry immediately instead of burning a turn on read_file
    actual_start = line_hash(lines[start_line - 1])
    if actual_start != start_hash:
        return (
            "Error: start_anchor hash mismatch at line " + str(start_line) + ": expected " + start_hash + ", got " + actual_start + ". "
            "Current content near line " + str(start_line) + " with fresh anchors:\n" + render_hashline_region(lines, start_line) + "\n"
            "Retry using these anchors, or use str_replace instead."
        )

    actual_end = line_hash(lines[end_line - 1])
    if actual_end != end_hash:
        return (
            "Error: end_anchor hash mismatch at line " + str(end_line) + ": expected " + end_hash + ", got " + actual_end + ". "
            "Current content near line " + str(end_line) + " with fresh anchors:\n" + render_hashline_region(lines, end_line) + "\n"
            "Retry using these anchors, or use str_replace instead."
        )

    new_lines = new_text.splitlines()
    lines[start_line - 1 : end_line] = new_lines
    with open(target, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    rel = os.path.relpath(target, WORKSPACE)
    print(ts() + "  [tool call] edit_file: " + rel + " replaced lines " + str(start_line) + "-" + str(end_line) + " with " + str(len(new_lines)) + " lines")
    return render_hashlines(lines)


def tool_run_command(command):
    # scrub the API key from the child environment; commands the model runs
    # have no legitimate need for it. note this is best-effort (same-user
    # processes can still read /proc), but it stops accidental leaks via env
    # dumps in debug output that would then flow back into the conversation.
    child_env = {k: v for k, v in os.environ.items() if k != "OPENROUTER_API_KEY"}
    # file-backed output instead of pipes. pipes block on close, so a
    # backgrounded child ("python3 -m http.server &") keeps communicate()
    # stuck for the full timeout even though the shell exited instantly.
    # with temp files, wait() returns as soon as the shell exits and we read
    # whatever was written. backgrounded children keep writing to the
    # (unlinked) temp file harmlessly.
    #
    # start_new_session gives the shell its own process group so killpg on
    # timeout reaps backgrounded children that would otherwise accumulate.
    # on normal exit the group is left alone - the model may need the server.
    out_f = tempfile.TemporaryFile()
    err_f = tempfile.TemporaryFile()
    proc = subprocess.Popen(
        command,
        shell=True,
        cwd=WORKSPACE,
        stdout=out_f,
        stderr=err_f,
        env=child_env,
        start_new_session=True,
    )
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        # kill the entire process group - this reaps backgrounded children
        # that would otherwise survive the shell's death and accumulate
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            pass
        proc.kill()
        proc.wait()
        out_f.seek(0)
        err_f.seek(0)
        partial = out_f.read().decode("utf-8", "replace") + err_f.read().decode("utf-8", "replace")
        out_f.close()
        err_f.close()
        print(ts() + "  [tool call] run_command: TIMED OUT | " + command[:80])
        return partial + "\n[error: command timed out after 30s; any partial output is shown above]"
    out_f.seek(0)
    err_f.seek(0)
    output = out_f.read().decode("utf-8", "replace") + err_f.read().decode("utf-8", "replace")
    out_f.close()
    err_f.close()
    nonempty = [l for l in output.strip().splitlines() if l.strip()]
    preview = (" | " + nonempty[0][:100]) if nonempty else ""
    print(ts() + "  [tool call] run_command: exit " + str(proc.returncode) + " | " + command[:60] + preview)
    return output + "\n[exit code: " + str(proc.returncode) + "]"


def tool_search_web(mcp, query):
    # one tool = one search: encode the query, navigate DDG plain HTML, extract
    # markdown. Collapsing the navigate+extract dance into a single call removes
    # the URL-encoding and sequencing failure modes that trip up small models.
    url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote_plus(query)
    call_playwright(mcp, "playwright_navigate", {"url": url})
    text = call_playwright(mcp, "playwright_extract_content", {})
    print(ts() + "[tool call] search_web: " + query[:80] + " | " + str(len(text)) + " chars")
    return text


# tool dispatcher


def dispatch_tool(mcp, name, arguments):
    if name == "search_web":
        return tool_search_web(mcp, arguments["query"])

    if name == "playwright_navigate":
        text = call_playwright(mcp, name, arguments)
        print(ts() + "[tool call] playwright_navigate: " + arguments.get("url", "")[:120])
        return text

    if name == "playwright_extract_content":
        text = call_playwright(mcp, name, arguments)
        preview = text[:200].replace("\n", " ").strip()
        print(ts() + "[tool call] playwright_extract_content: " + str(len(text)) + " chars | " + preview[:100])
        return text

    if name == "write_file":
        return tool_write_file(arguments["filename"], arguments["content"])
    if name == "read_file":
        return tool_read_file(arguments["filename"])
    if name == "str_replace":
        return tool_str_replace(arguments["filename"], arguments["old_str"], arguments["new_str"])
    if name == "edit_file":
        return tool_edit_file(
            arguments["filename"],
            arguments["start_anchor"],
            arguments["end_anchor"],
            arguments["new_text"],
        )
    if name == "run_command":
        return tool_run_command(arguments["command"])

    return "Unknown tool: " + name


def get_env_snapshot():
    cmd = "echo '=PWD=' && pwd && echo '=LS=' && ls -1 && echo '=PY=' && python3 --version 2>&1"
    try:
        r = subprocess.run(cmd, shell=True, cwd=WORKSPACE, capture_output=True, text=True, timeout=5)
        return "[workspace snapshot]\n" + r.stdout.strip()
    except Exception:
        return ""


def read_p():
    p_path = os.path.join(WORKSPACE, "p.md")
    if not os.path.exists(p_path):
        sys.exit("Error: no p.md found in " + WORKSPACE)
    with open(p_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def read_project():
    # project.md is optional at agent level; pq_minder validates it exists before staging
    project_path = os.path.join(WORKSPACE, "project.md")
    if not os.path.exists(project_path):
        return ""
    with open(project_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def make_tools():
    return [
        {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "Search the web and get results back as markdown. Provide a plain text query e.g. 'python csv parsing example'. Use this for ALL web searches - do not build search URLs yourself.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "Plain text search query"}},
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "playwright_navigate",
                "description": "Navigate the browser to a specific URL and return the page title. Use this for visiting known URLs (e.g. links found in search results). For searches use the search_web tool instead.",
                "parameters": {
                    "type": "object",
                    "properties": {"url": {"type": "string"}},
                    "required": ["url"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "playwright_extract_content",
                "description": "Extract the current browser page as clean markdown.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "selector": {
                            "type": "string",
                            "description": "Optional CSS selector to scope extraction e.g. 'main'",
                        }
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Create or overwrite a file with the given content. Use a relative path e.g. 'solution.py'. You MUST use this tool to create new files - do not write file content in your reply. Also the best choice when rewriting most or all of a file (under about 200 lines) since you avoid anchor/match complexity.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filename": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["filename", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file. Returns each line prefixed with a hashline anchor in the format LINENUM:HASH|content e.g. '3:a4|some text here'. You MUST call read_file before editing a file with str_replace or edit_file.",
                "parameters": {
                    "type": "object",
                    "properties": {"filename": {"type": "string"}},
                    "required": ["filename"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "str_replace",
                "description": "Edit a file by replacing one exact occurrence of old_str with new_str. old_str must match the file content exactly (including whitespace and indentation, WITHOUT the LINENUM:HASH| prefixes from read_file) and must appear exactly once - include enough surrounding lines to make it unique. Best for small targeted edits (a few lines). Call read_file first to see current content.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filename": {"type": "string"},
                        "old_str": {"type": "string", "description": "Exact text to replace, must appear exactly once in the file"},
                        "new_str": {"type": "string", "description": "Replacement text"},
                    },
                    "required": ["filename", "old_str", "new_str"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "edit_file",
                "description": "Edit a file by line range using hashline anchors from a previous read_file call. Replaces the lines from start_anchor to end_anchor (inclusive) with new_text. Returns the updated file with new anchors so you can chain further edits. Best for replacing larger blocks of code (5+ lines) where you want precise line-range control.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filename": {"type": "string"},
                        "start_anchor": {"type": "string", "description": "LINENUM:HASH of the first line to replace e.g. '5:a3'"},
                        "end_anchor": {"type": "string", "description": "LINENUM:HASH of the last line to replace e.g. '7:f1'"},
                        "new_text": {"type": "string", "description": "Replacement text. Use newlines for multiple lines."},
                    },
                    "required": ["filename", "start_anchor", "end_anchor", "new_text"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_command",
                "description": "Run a shell command in the workspace and return its output. Timeout is 30 seconds.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Shell command to run e.g. 'python3 solution.py'"},
                    },
                    "required": ["command"],
                },
            },
        },
    ]


def make_system_prompt():
    # four concerns: tool rules, work budget, verification, and finishing protocol
    return (
        "You are an autonomous agent with browser, shell, and file tools.\n\n"
        "Tool rules:\n"
        "1. Always use tools for file operations and commands - never output file contents in your reply.\n"
        "2. To edit a file: call read_file first, then choose the best edit tool for the job:\n"
        "   - str_replace: for small targeted changes (a few lines).\n"
        "   - edit_file: for replacing larger blocks using line-range anchors.\n"
        "   - write_file: for rewriting the whole file (best when changing most of it, or files under ~200 lines).\n"
        "3. edit_file anchors are LINENUM:HASH strings - copy them exactly from read_file output.\n"
        "4. For web searches use the search_web tool with a plain text query.\n\n"
        "Work budget:\n"
        "Aim to finish within about " + str(MAX_STEPS_SUGGESTION) + " tool calls. This is a soft target, "
        "not a hard cutoff - if you find yourself over budget, prioritize finishing and reporting over polish. "
        "You will receive a notice when you should start wrapping up.\n\n"
        "Verification:\n"
        "Before writing your report, verify your work by actually running it: execute your code, re-read final files, "
        "re-check computed values. Include the real observed output in your report. "
        "A report that claims success without demonstrated verification is incomplete.\n\n"
        "When the task is complete, use write_file to create task_report/report.md containing:\n"
        "1. A step-by-step summary of what you did.\n"
        "2. Key decisions and why you made them.\n"
        "3. Anything you are uncertain about.\n"
        "4. Anything about the environment or tool calling that you seemed to unnecessarily struggle with.\n"
        "5. Your assessment of whether the task succeeded, including the verification evidence you observed.\n"
        "6. You may create or copy images (.jpg or .png, and no larger than 1200 pixels please) to task_report/ if the task requires those for evaluation.\n"
        "7. Markdown or images in task_report/ are only for evaluation and will be deleted after your work is evaluated.\n"
        "After writing task_report/report.md, reply with one short sentence confirming completion.\n"
    )


def write_stats(state, start_time):
    elapsed_minutes = (time.time() - start_time) / 60.0
    stats_dir = os.path.join(WORKSPACE, "task_report")
    os.makedirs(stats_dir, exist_ok=True)
    stats_path = os.path.join(stats_dir, "stats.yaml")
    ec = state["edit_counts"]
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write("model_id: " + MODEL_ID + "\n")
        f.write("final_context_tokens: " + str(state["last_post_tokens"]) + "\n")
        f.write("compaction_count: " + str(state["compaction_count"]) + "\n")
        f.write("elapsed_minutes: " + "{:.2f}".format(elapsed_minutes) + "\n")
        f.write("edit_counts:\n")
        f.write("  write_file: " + str(ec["write_file"]) + "\n")
        f.write("  str_replace: " + str(ec["str_replace"]) + "\n")
        f.write("  edit_file: " + str(ec["edit_file"]) + "\n")
    # print a summary so the operator can see edit method preferences at a glance
    total_edits = ec["write_file"] + ec["str_replace"] + ec["edit_file"]
    if total_edits > 0:
        print(ts() + "Edit methods used: write_file=" + str(ec["write_file"]) + " str_replace=" + str(ec["str_replace"]) + " edit_file=" + str(ec["edit_file"]) + " (total=" + str(total_edits) + ")")
    print(ts() + "Stats written to task_report/stats.yaml")


def main():
    start_time = time.time()

    print(ts() + "Agent model: " + MODEL, flush=True)
    rmode = _mcfg("reasoning_mode", "effort")
    if rmode != "effort":
        print(ts() + "Reasoning mode: " + rmode, flush=True)

    tools = make_tools()

    # three-part session context: (1) system rules, (2) project context, (3) this task
    # project.md and p.md are staged into workspace root by pq_minder before this runs
    task_prompt = read_p()
    project_text = read_project()
    system_prompt = make_system_prompt()
    snapshot = get_env_snapshot()

    # combine project context and task into one user message, clearly labeled
    initial_content = ""
    if project_text:
        initial_content += "## Project Context\n\n" + project_text + "\n\n"
    initial_content += "## Task\n\n" + task_prompt
    if snapshot:
        initial_content += "\n\n" + snapshot

    state = {"last_post_tokens": 949, "compaction_count": 0, "edit_counts": {"write_file": 0, "str_replace": 0, "edit_file": 0}, "truncation_streak": 0}

    session_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": initial_content},
    ]

    messages = []
    new_messages = list(session_messages)

    print(ts() + "MCP server starting...")
    mcp = start_mcp()
    print(ts() + "MCP ready.\n")

    print(ts() + "Starting agent loop...\n")

    tool_calls_done = 0
    warned_wrapup = False
    warned_over = False
    report_rescues = 0

    try:
        while True:
            # reasoning sandwich: high at plan (first turn) and wrap-up, medium between
            if tool_calls_done == 0:
                effort = REASONING_EFFORT_PLAN
            elif tool_calls_done >= WRAPUP_WARN_AT:
                effort = REASONING_EFFORT_WRAPUP
            else:
                effort = REASONING_EFFORT_WORK

            response = chat(messages, tools, new_messages, state, session_messages, effort)
            choice = response["choices"][0]
            msg = choice["message"]
            finish = choice["finish_reason"]
            # CRITICAL: msg may include reasoning_details / reasoning / reasoning_content fields
            # when thinking mode is on. Appending the dict verbatim preserves them so they
            # round-trip on the next request. Do NOT strip these fields - DeepSeek 400s if a
            # prior tool-call turn's reasoning state is missing on the follow-up. MiniMax and
            # Kimi K2.7 Code also require preserved reasoning across turns.
            new_messages.append(msg)

            # branch on the presence of tool_calls rather than finish_reason: some
            # providers report tool calls under finish_reason "stop", and a "length"
            # finish can still carry complete earlier tool calls
            tool_calls = msg.get("tool_calls") or []

            if tool_calls:
                # model produced parseable tool calls - reset any truncation state
                state["truncation_streak"] = 0
                for tc in tool_calls:
                    fn_name = tc["function"]["name"]
                    # two layers of cheap-model error recovery:
                    # layer 1: malformed JSON in tool call arguments
                    # layer 2: valid JSON but wrong/missing parameter keys
                    # both return the error as a tool result so the model self-corrects
                    try:
                        fn_args = json.loads(tc["function"]["arguments"])
                    except (ValueError, TypeError) as e:
                        print(ts() + "  [tool call] " + fn_name + ": MALFORMED ARGUMENTS")
                        tool_result = "Error: tool call arguments were not valid JSON (" + str(e) + "). Re-issue the call with corrected, complete JSON arguments."
                    else:
                        try:
                            tool_result = dispatch_tool(mcp, fn_name, fn_args)
                        except KeyError as e:
                            print(ts() + "  [tool call] " + fn_name + ": MISSING PARAMETER " + str(e))
                            tool_result = "Error: tool call missing required parameter " + str(e) + ". Check the tool definition and re-issue with the correct parameter names."
                        except TypeError as e:
                            print(ts() + "  [tool call] " + fn_name + ": BAD PARAMETER TYPE")
                            tool_result = "Error: tool call parameter type error (" + str(e) + "). Re-issue with correct argument types."
                    # track file editing method usage for observability
                    if fn_name in state["edit_counts"]:
                        state["edit_counts"][fn_name] += 1
                    new_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": tool_result,
                        }
                    )
                    tool_calls_done += 1

                # soft budget notices, each injected at most once
                if not warned_wrapup and tool_calls_done >= WRAPUP_WARN_AT:
                    warned_wrapup = True
                    new_messages.append(
                        {
                            "role": "user",
                            "content": "[harness notice] You have used "
                            + str(tool_calls_done)
                            + " of a suggested "
                            + str(MAX_STEPS_SUGGESTION)
                            + " tool calls. Start wrapping up: verify what you have built and write task_report/report.md soon.",
                        }
                    )
                if not warned_over and tool_calls_done >= MAX_STEPS_SUGGESTION:
                    warned_over = True
                    new_messages.append(
                        {
                            "role": "user",
                            "content": "[harness notice] You have exceeded the suggested budget of "
                            + str(MAX_STEPS_SUGGESTION)
                            + " tool calls. Finish now: verify what you have, write task_report/report.md, and reply with your completion confirmation.",
                        }
                    )
                continue

            if finish == "length":
                # reply was cut off by max_tokens mid-thought (or mid-tool-call).
                # strategy: first truncation -> boost max_tokens and retry.
                # subsequent truncations -> keep boost, tell model to decompose
                # large writes into skeleton + incremental edits.
                state["truncation_streak"] += 1
                streak = state["truncation_streak"]

                if streak == 1:
                    # first truncation: double max_tokens up to the boost cap
                    base = _mcfg("max_tokens", 16000)
                    boosted = min(base * 2, MAX_OUTPUT_BOOST)
                    state["max_tokens_override"] = boosted
                    print(ts() + "  [warn] reply truncated at max_tokens, boosting output to " + str(boosted) + " and nudging model")
                    new_messages.append(
                        {
                            "role": "user",
                            "content": "[harness notice] Your previous reply was cut off by the output token limit. "
                            "The output budget has been increased. Continue from where you left off. "
                            "If you were issuing a tool call, re-issue it completely. "
                            "Tip: if a file is very large, write a skeleton version first with write_file, "
                            "then add remaining sections with str_replace.",
                        }
                    )
                else:
                    # second+ truncation: the boosted budget still wasn't enough.
                    # the file is genuinely too large for a single tool call.
                    print(ts() + "  [warn] reply truncated again (streak=" + str(streak) + "), sending decomposition nudge")
                    new_messages.append(
                        {
                            "role": "user",
                            "content": "[harness notice] Your reply was cut off again by the output token limit. "
                            "The file you are trying to write is too large for a single tool call. "
                            "You MUST split it: use write_file to create a skeleton or partial version first, "
                            "then use str_replace to add the remaining sections one at a time. "
                            "Do NOT attempt to write the entire file in one call.",
                        }
                    )
                continue

            # model produced a final text reply - make sure the report actually exists
            # before accepting it, otherwise this whole attempt is wasted
            report_path = os.path.join(WORKSPACE, "task_report", "report.md")
            if not os.path.exists(report_path) and report_rescues < MAX_REPORT_RESCUES:
                report_rescues += 1
                print(ts() + "  [warn] model stopped without task_report/report.md - rescue " + str(report_rescues) + "/" + str(MAX_REPORT_RESCUES))
                new_messages.append(
                    {
                        "role": "user",
                        "content": "[harness notice] You ended your turn but task_report/report.md does not exist. Use write_file to create task_report/report.md now (summary, key decisions, uncertainties, success assessment with verification evidence), then reply with one short confirmation sentence.",
                    }
                )
                continue

            print(ts() + "\n[done] " + str(msg["content"]))
            break

    finally:
        # guaranteed cleanup - runs after normal exit AND unhandled exceptions.
        # OOM/SIGKILL bypass finally (nothing can catch those), but every
        # Python-level crash is covered: API failures, MCP pipe breaks, etc.
        try:
            write_stats(state, start_time)
        except Exception as e:
            print(ts() + "[warn] failed to write stats: " + str(e))
        try:
            mcp["proc"].stdin.close()
            mcp["proc"].terminate()
        except Exception:
            pass


if __name__ == "__main__":
    main()
