import os
import sys
import json
import zlib
import select
import subprocess
import requests
import time
from pathlib import Path
import random
import tiktoken
import re

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

# Config

SLOW_MODE = False
BUMP_OVER_LIMIT_MSGS = False

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    sys.exit("Error: GOOGLE_API_KEY environment variable is not set.")

AGENT_DIR = os.environ.get("AGENT_DIR", os.path.dirname(os.path.abspath(__file__)))

MODEL = "gemini-3.1-flash-lite-preview"

OPENROUTER_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
HEADERS = {
    "Authorization": "Bearer " + GOOGLE_API_KEY,
    "Content-Type": "application/json",
}

WORKSPACE = os.path.abspath(os.getcwd())

_enc = tiktoken.get_encoding("cl100k_base")


def shorten_content_with_notice(m):
    m = m[:-2000]
    m += "\n**USER AGENT HARNESS NOTICE:** This content has been trimmed due to excessive length. Please use this if you can, otherwise find another way to achieve your goal.\n"
    return m


def filter_msgs_and_est_tokens(messages):
    tokens = 3  # reply priming

    MAX_MSG_TOKENS = 9000

    for i, msg in enumerate(messages):
        tokens += 3

        shorten_loops = 0
        msg_tokens = MAX_MSG_TOKENS + 100
        while msg_tokens > MAX_MSG_TOKENS:
            shorten_loops += 1
            msg_tokens = 0
            for key, val in msg.items():
                if val is None:
                    continue
                if isinstance(val, str):
                    msg_tokens += len(_enc.encode(val))
                else:
                    msg_tokens += len(_enc.encode(json.dumps(val)))

            if msg_tokens > MAX_MSG_TOKENS:
                if "content" in msg and isinstance(msg["content"], str) and len(msg["content"]) > 15000 and shorten_loops < 30000:
                    messages[i]["content"] = shorten_content_with_notice(msg["content"])
                else:
                    for key, val in msg.items():
                        if val is None:
                            continue
                        if isinstance(val, str):
                            print("AAAA", key, len(val), len(_enc.encode(val)))
                        else:
                            print("BBBB", key, len(json.dumps(val)), len(_enc.encode(json.dumps(val))))
                    raise Exception("fix this")

        if shorten_loops > 1:
            print("SHORTENED:", len(messages[i]["content"]), re.sub(r"\s+", " ", messages[i]["content"][:180]), flush=True)

        tokens += msg_tokens

    return tokens


def post_with_retry(payload):
    for attempt in range(9):
        if attempt > 0 or SLOW_MODE:
            p = attempt if SLOW_MODE else attempt - 1
            delay = random.uniform(2**p, 2 ** (p + 1))
            time.sleep(delay)
        try:
            resp = requests.post(OPENROUTER_URL, headers=HEADERS, json=payload, timeout=60)
        except requests.exceptions.Timeout:
            if attempt < 8:
                print("  [error] request timed out, retrying (attempt " + str(attempt + 1) + "/8)...")
                continue
            raise
        if resp.status_code == 429 and attempt < 8:
            print("  [error] 429 rate limit, retrying (attempt " + str(attempt + 1) + "/8)...")
            continue
        if not resp.ok:
            print("\n[error] status=" + str(resp.status_code))
            for key, val in resp.headers.items():
                print("  " + key + ": " + val)
        resp.raise_for_status()
        break
    return resp


def post_compaction(payload):
    try:
        resp = post_with_retry(payload)
        resp.raise_for_status()
        return resp
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 400 and "reasoning" in payload:
            print("  [warn] reasoning param rejected, retrying without it...")
            payload = {k: v for k, v in payload.items() if k != "reasoning"}
            resp = post_with_retry(payload)
            resp.raise_for_status()
            return resp
        raise


def extract_compaction_summary(raw_msg):
    summary = raw_msg.get("content")
    if not summary or not isinstance(summary, str) or summary.strip().lower() in ("", "none", "yes"):
        summary = raw_msg.get("reasoning") or raw_msg.get("reasoning_content")
    if summary:
        summary = summary.strip()
    return summary or None


def chat(messages, tools, new_messages, state, session_messages):
    MAX_CONTEXT_LENGTH = 130000

    new_prompt_tokens = filter_msgs_and_est_tokens(new_messages)
    new_prompt_tokens = int(round(0.2221 * (new_prompt_tokens**1.1866)))
    pre_prompt_total_context = state["last_post_tokens"] + new_prompt_tokens

    if pre_prompt_total_context > MAX_CONTEXT_LENGTH:
        print("PERFORM COMPACTION", flush=True)

        if BUMP_OVER_LIMIT_MSGS:
            compaction_prompt = (
                "The agent context limit was reached and this session is being compacted. "
                "The conversation above is the full committed session history. "
                "Write a compact plain-text summary covering:\n"
                "1. What actions were taken and whether they succeeded or failed.\n"
                "2. Current state of any modified files (exact filenames, key content or structure).\n"
                "3. Any discovered facts relevant to completing the task (e.g. specific data values found).\n"
                "4. What tool was running at the moment of compaction, "
                "and that its result (or further prompts, or nothing) will follow immediately after this summary.\n"
                "Include any prior compaction summaries in condensed form.\n"
                "Do NOT summarize the system prompt or initial task instructions - those are provided fresh.\n"
                "Be terse. If this summary exceeds 9000 tokens it will be clipped.\n"
                "Minimize thinking and output summary content.\n"
            )
            compaction_payload = {
                "model": MODEL,
                "messages": messages + [{"role": "user", "content": compaction_prompt}],
            }
        else:
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
            compaction_payload = {
                "model": MODEL,
                "messages": messages + new_messages + [{"role": "user", "content": compaction_prompt}],
            }
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

        summary_msg = {"role": "user", "content": "[context compacted] Session summary:\n" + summary}

        new_session = list(session_messages) + [summary_msg]
        filter_msgs_and_est_tokens(new_session)
        messages.clear()
        messages += new_session

    else:
        print("Sending CONTEXT", pre_prompt_total_context, "which is {:.1f}% of max context".format(100 * pre_prompt_total_context / MAX_CONTEXT_LENGTH))

    messages += new_messages
    new_messages.clear()

    payload = {
        "model": MODEL,
        "tools": tools,
        "tool_choice": "auto",
        "messages": messages,
    }

    data = post_with_retry(payload).json()

    state["last_post_tokens"] = int(data.get("usage", {}).get("prompt_tokens", 0))
    print("------ MSGS", len(messages), "-------- CONTEXT", state["last_post_tokens"])

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


# hashline helpers


def line_hash(line):
    return "{:02x}".format(zlib.crc32(line.rstrip().encode("utf-8")) % 256)


def render_hashlines(lines):
    return "\n".join(str(i) + ":" + line_hash(l) + "|" + l.rstrip() for i, l in enumerate(lines, 1))


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
    hashline_re = re.compile(r"^\d+:[0-9a-f]{2}\|")
    bad_lines = [l for l in content.splitlines() if hashline_re.match(l)]
    if bad_lines:
        sample = bad_lines[0]
        return "Error: content looks like raw read_file output (e.g. '" + sample + "'). Strip the 'LINENUM:HASH|' prefixes from each line before writing."
    target = safe_path(filename)
    parent = os.path.dirname(target)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        f.write(content)
    lines = content.splitlines()
    rel = os.path.relpath(target, WORKSPACE)
    print("  write_file: " + rel + " (" + str(len(lines)) + " lines)")
    return "Written " + str(os.stat(target).st_size) + " bytes to " + rel


def tool_read_file(filename):
    target = safe_path(filename)
    if not os.path.exists(target):
        return "Error: file not found: " + filename
    with open(target, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    rel = os.path.relpath(target, WORKSPACE)
    print("  read_file: " + rel + " (" + str(len(lines)) + " lines)")
    return render_hashlines(lines)


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

    actual_start = line_hash(lines[start_line - 1])
    if actual_start != start_hash:
        return "Error: start_anchor hash mismatch at line " + str(start_line) + ": expected " + start_hash + ", got " + actual_start + " - re-read the file first"

    actual_end = line_hash(lines[end_line - 1])
    if actual_end != end_hash:
        return "Error: end_anchor hash mismatch at line " + str(end_line) + ": expected " + end_hash + ", got " + actual_end + " - re-read the file first"

    new_lines = new_text.splitlines()
    lines[start_line - 1 : end_line] = new_lines
    with open(target, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    rel = os.path.relpath(target, WORKSPACE)
    print("  edit_file: " + rel + " replaced lines " + str(start_line) + "-" + str(end_line) + " with " + str(len(new_lines)) + " lines")

    return render_hashlines(lines)


def tool_run_command(command):
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=WORKSPACE,
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return "[error: command timed out after 30s]"
    print("  run_command: " + command[:80] + " -> exit " + str(result.returncode))
    return output + "\n[exit code: " + str(result.returncode) + "]"


# tool dispatcher


def dispatch_tool(mcp, name, arguments):
    if name in ("playwright_navigate", "playwright_extract_content"):
        for attempt in range(9):
            if attempt > 0:
                delay = random.uniform(2 ** (attempt - 1), 2**attempt)
                print("  [mcp retry " + str(attempt) + "/8] waiting " + "{:.1f}".format(delay) + "s then restarting mcp...")
                time.sleep(delay)
                restart_mcp(mcp)
            try:
                result = mcp_call(mcp, "tools/call", {"name": name, "arguments": arguments})
                text = "\n".join(b["text"] for b in result.get("content", []) if b.get("type") == "text")
                if name == "playwright_navigate":
                    print("  playwright_navigate: " + arguments.get("url", "")[:80])
                else:
                    preview = text[:300].replace("\n", " ").strip()
                    print("  playwright_extract_content: " + str(len(text)) + " chars")
                    print("  [debug preview] " + preview)
                return text
            except TimeoutError as e:
                if attempt == 8:
                    raise
                print("  [mcp timeout] attempt " + str(attempt + 1) + "/9: " + str(e))

    if name == "write_file":
        return tool_write_file(arguments["filename"], arguments["content"])

    if name == "read_file":
        return tool_read_file(arguments["filename"])

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
    # project.md is optional at agent level - pq_minder validates it exists before staging
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
                "name": "playwright_navigate",
                "description": "Navigate the browser to a URL and return the page title. Use DDG plain HTML (https://html.duckduckgo.com/html/?q=...) for searches.",
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
                "description": "Create or overwrite a file with the given content. Use a relative path e.g. 'solution.py'. You MUST use this tool to create files - do not write file content in your reply.",
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
                "description": "Read a file. Returns each line prefixed with a hashline anchor in the format LINENUM:HASH|content e.g. '3:a4|some text here'. You MUST call read_file before calling edit_file to get valid anchors.",
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
                "name": "edit_file",
                "description": "Edit a file using hashline anchors from a previous read_file call. Replaces the lines from start_anchor to end_anchor (inclusive) with new_text. Anchors are LINENUM:HASH strings e.g. '5:a3' - copy them exactly from read_file output. To insert after a line: set both anchors to that line and include it in new_text followed by the new content. Returns the updated file content with new hashline anchors. You MUST use this tool to edit files - do not output edited file content in your reply.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filename": {"type": "string"},
                        "start_anchor": {
                            "type": "string",
                            "description": "LINENUM:HASH of the first line to replace e.g. '5:a3'",
                        },
                        "end_anchor": {
                            "type": "string",
                            "description": "LINENUM:HASH of the last line to replace e.g. '7:f1'",
                        },
                        "new_text": {
                            "type": "string",
                            "description": "Replacement text. Use newlines for multiple lines.",
                        },
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
                        "command": {
                            "type": "string",
                            "description": "Shell command to run e.g. 'python3 q.py'",
                        }
                    },
                    "required": ["command"],
                },
            },
        },
    ]


def make_system_prompt():
    # three concerns: tool rules, image output constraints, and the finishing protocol
    return (
        "You are an autonomous agent with browser, shell, and file tools.\n\n"
        "Tool rules:\n"
        "1. Always use tools for file operations and commands - never output file contents in your reply.\n"
        "2. Always call read_file before edit_file to get current line anchors.\n"
        "3. Anchors are LINENUM:HASH strings - copy them exactly from read_file output.\n"
        "4. For web searches use DuckDuckGo plain HTML: https://html.duckduckgo.com/html/?q=<query>\n\n"
        "Image output rules (when necessary for task completion):\n"
        "- Only produce image files with .jpg or .png extension. No other formats are accepted.\n"
        "- Images must be no larger than 1200 pixels on the longest side.\n\n"
        "Finishing:\n"
        "When the task is complete, use write_file to create report.md containing:\n"
        "1. A step-by-step summary of what you did.\n"
        "2. Key decisions and why you made them.\n"
        "3. Anything you are uncertain about.\n"
        "4. Your assessment of whether the task succeeded.\n"
        "After writing report.md, reply with one short sentence confirming completion."
    )


def main():
    tools = make_tools()

    # three-part context: (1) system rules, (2) project context, (3) task
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

    state = {"last_post_tokens": 949}

    session_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": initial_content},
    ]

    messages = []
    new_messages = list(session_messages)

    print("MCP server starting...")
    mcp = start_mcp()
    print("MCP ready.\n")

    print("Starting agent loop...\n")

    while True:
        response = chat(messages, tools, new_messages, state, session_messages)
        choice = response["choices"][0]
        msg = choice["message"]
        finish = choice["finish_reason"]
        new_messages.append(msg)

        if finish == "tool_calls":
            for tc in msg["tool_calls"]:
                fn_name = tc["function"]["name"]
                fn_args = json.loads(tc["function"]["arguments"])
                print("[tool call] " + fn_name)
                tool_result = dispatch_tool(mcp, fn_name, fn_args)
                new_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": tool_result,
                    }
                )
        else:
            print("\n[done] " + str(msg["content"]))
            break

    mcp["proc"].stdin.close()
    mcp["proc"].terminate()


if __name__ == "__main__":
    main()
