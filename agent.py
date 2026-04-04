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
# - No comments with repeated characters for visual page breaks
# - No non-ascii characters
# - No command line argument processing
# - No global variables unless making them local increases complexity
# - Yes inline comments for showing intent without ceremony
# - Yes if __name__ == "__main__": main()


# Config

SLOW_MODE = True
USE_SYSTEM_PROMPT = True  # use system role otherwise user role for first msg

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    sys.exit("Error: OPENROUTER_API_KEY environment variable is not set.")

AGENT_DIR = os.environ.get("AGENT_DIR", str(Path(__file__).parent.resolve()))

MODEL = "qwen/qwen3.6-plus:free"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
}

WORKSPACE = Path.cwd().resolve()

# cl100k_base is a good approximation for most modern models
# loaded once; tiktoken caches the vocab file on disk after first download
_enc = tiktoken.get_encoding("cl100k_base")


def shorten_content_with_notice(m):
    m = m[:-2000]
    m += "\n**USER AGENT HARNESS NOTICE:** This content has been trimmed due to excessive length. Please use this if you can, otherwise find another way to achieve your goal.\n"
    return m


def filter_msgs_and_est_tokens(messages):
    # 3 tokens overhead per message (role framing), 3 for reply priming
    tokens = 3  # reply priming

    MAX_MSG_TOKENS = 9000

    for i, msg in enumerate(messages):
        tokens += 3  # per-message overhead

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


def chat(messages, tools, new_messages, state):
    MAX_CONTEXT_LENGTH = 128000

    # pre = last known accurate count + estimated tokens for the newest message only
    # (the newest message is the only one openrouter has not yet seen)
    # We use a correction factor to make tiktoken token count more accurate - DO NOT CHANGE THIS

    new_prompt_tokens = filter_msgs_and_est_tokens(new_messages)
    new_prompt_tokens = int(round(0.2221 * (new_prompt_tokens**1.1866)))
    pre_prompt_total_context = state["last_post_tokens"] + new_prompt_tokens

    if pre_prompt_total_context > MAX_CONTEXT_LENGTH:
        print("PERFORM COMPACTION", flush=True)
        raise Exception("need compaction mechanization")
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

    # 0 is the initial request, 1-8 are the retries
    for attempt in range(9):
        # determine if we need to sleep before this request
        if attempt > 0 or SLOW_MODE:
            p = attempt if SLOW_MODE else attempt - 1
            delay = random.uniform(2**p, 2 ** (p + 1))
            time.sleep(delay)

        try:
            resp = requests.post(OPENROUTER_URL, headers=HEADERS, json=payload, timeout=60)
        except requests.exceptions.Timeout:
            if attempt < 8:
                print(f"  [error] request timed out, retrying (attempt {attempt + 1}/8)...")
                continue
            raise

        # if rate limited, loop around to try again
        if resp.status_code == 429 and attempt < 8:
            print(f"  [error] 429 rate limit, retrying (attempt {attempt + 1}/8)...")
            continue

        # on any other failure or if we exhaust our retries, dump headers
        if not resp.ok:
            print(f"\n[error] status={resp.status_code}")
            for key, val in resp.headers.items():
                print(f"  {key}: {val}")

        resp.raise_for_status()
        break

    data = resp.json()

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
                raise RuntimeError(f"MCP error: {msg['error']}")
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
        sys.exit(f"MCP server exited immediately with code {proc.returncode}")
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
        raise RuntimeError(f"MCP server failed to restart, exit code {proc.returncode}")
    mcp["proc"] = proc
    mcp["id"] = 0
    _mcp_handshake(mcp)


# hashline helpers


def line_hash(line):
    return f"{zlib.crc32(line.rstrip().encode('utf-8')) % 256:02x}"


def render_hashlines(lines):
    return "\n".join(f"{i}:{line_hash(l)}|{l.rstrip()}" for i, l in enumerate(lines, 1))


def parse_anchor(anchor):
    parts = anchor.split(":")
    if len(parts) != 2:
        raise ValueError(f"bad anchor format: {anchor!r} - expected LINENUM:HASH e.g. '5:a3'")
    return int(parts[0]), parts[1]


def safe_path(filename):
    target = (WORKSPACE / filename).resolve()
    if not str(target).startswith(str(WORKSPACE)):
        raise ValueError(f"path '{filename}' resolves outside workspace")
    return target


# file tool implementations


def tool_write_file(filename, content):
    target = safe_path(filename)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    lines = content.splitlines()
    print(f"  write_file: {target.relative_to(WORKSPACE)} ({len(lines)} lines)")
    return f"Written {target.stat().st_size} bytes to {target.relative_to(WORKSPACE)}"


def tool_read_file(filename):
    target = safe_path(filename)
    if not target.exists():
        return f"Error: file not found: {filename}"
    lines = target.read_text(encoding="utf-8").splitlines()
    print(f"  read_file: {target.relative_to(WORKSPACE)} ({len(lines)} lines)")
    return render_hashlines(lines)


def tool_edit_file(filename, start_anchor, end_anchor, new_text):
    target = safe_path(filename)
    if not target.exists():
        return f"Error: file not found: {filename} - use write_file to create it first"
    lines = target.read_text(encoding="utf-8").splitlines()

    start_line, start_hash = parse_anchor(start_anchor)
    end_line, end_hash = parse_anchor(end_anchor)

    if start_line < 1 or start_line > len(lines):
        return f"Error: start_anchor line {start_line} out of range (file has {len(lines)} lines)"
    if end_line < start_line or end_line > len(lines):
        return f"Error: end_anchor line {end_line} out of range"

    actual_start = line_hash(lines[start_line - 1])
    if actual_start != start_hash:
        return f"Error: start_anchor hash mismatch at line {start_line}: " f"expected {start_hash}, got {actual_start} - re-read the file first"

    actual_end = line_hash(lines[end_line - 1])
    if actual_end != end_hash:
        return f"Error: end_anchor hash mismatch at line {end_line}: " f"expected {end_hash}, got {actual_end} - re-read the file first"

    new_lines = new_text.splitlines()
    lines[start_line - 1 : end_line] = new_lines
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"  edit_file: {target.relative_to(WORKSPACE)} " f"replaced lines {start_line}-{end_line} with {len(new_lines)} lines")

    return render_hashlines(lines)


# tool dispatcher


def dispatch_tool(mcp, name, arguments):
    if name in ("playwright_navigate", "playwright_extract_content"):
        for attempt in range(9):
            if attempt > 0:
                delay = random.uniform(2 ** (attempt - 1), 2**attempt)
                print(f"  [mcp retry {attempt}/8] waiting {delay:.1f}s then restarting mcp...")
                time.sleep(delay)
                restart_mcp(mcp)
            try:
                result = mcp_call(mcp, "tools/call", {"name": name, "arguments": arguments})
                text = "\n".join(b["text"] for b in result.get("content", []) if b.get("type") == "text")
                if name == "playwright_navigate":
                    print(f"  playwright_navigate: {arguments.get('url', '')[:80]}")
                else:
                    print(f"  playwright_extract_content: {len(text)} chars")
                return text
            except TimeoutError as e:
                if attempt == 8:
                    raise
                print(f"  [mcp timeout] attempt {attempt + 1}/9: {e}")

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

    return f"Unknown tool: {name}"


def main():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "playwright_navigate",
                "description": ("Navigate the browser to a URL and return the page title. " "Use DDG plain HTML (https://html.duckduckgo.com/html/?q=...) for searches."),
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
                "description": (
                    "Create or overwrite a file with the given content. "
                    "Use a relative path e.g. 'inflation.md'. "
                    "You MUST use this tool to create files - do not write file content in your reply."
                ),
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
                "description": (
                    "Read a file. Returns each line prefixed with a hashline anchor "
                    "in the format LINENUM:HASH|content e.g. '3:a4|some text here'. "
                    "You MUST call read_file before calling edit_file to get valid anchors."
                ),
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
                "description": (
                    "Edit a file using hashline anchors from a previous read_file call. "
                    "Replaces the lines from start_anchor to end_anchor (inclusive) with new_text. "
                    "Anchors are LINENUM:HASH strings e.g. '5:a3' - copy them exactly from read_file output. "
                    "To insert after a line: set both anchors to that line and include it in new_text followed by the new content. "
                    "Returns the updated file content with new hashline anchors. "
                    "You MUST use this tool to edit files - do not output edited file content in your reply."
                ),
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
    ]

    system_prompt = (
        "You are a research assistant with browser access and file tools.\n\n"
        "IMPORTANT: You MUST use tools for ALL file operations. "
        "Never output file contents or table data directly in your reply text. "
        "Always use write_file, read_file, or edit_file instead.\n\n"
        "Hashline editing rules:\n"
        "1. Always call read_file before edit_file to get current line anchors.\n"
        "2. Anchors are LINENUM:HASH strings - copy them exactly from read_file output.\n"
        "3. edit_file returns updated hashline content - use it for chained edits.\n\n"
        "Browsing rules:\n"
        "1. Use DuckDuckGo plain HTML for searches: "
        "https://html.duckduckgo.com/html/?q=<query>\n"
        "2. After finding what you need, stop browsing and proceed with the task.\n\n"
        "When fully done, reply with one short confirmation sentence only."
    )

    initial_prompt = (
        "Do the following steps in order using your tools:\n"
        "1. Read inflation.md to see what years are already in the table.\n"
        "2. Identify the earliest year currently in the file.\n"
        "3. Search the web for the US annual inflation rate for the year "
        "   immediately before that earliest year.\n"
        "4. Use edit_file to insert a new row for that year into the table. "
        "   The table must be in chronological order oldest-to-newest "
        "   (earliest year at the top of the data rows, most recent at the bottom).\n"
        "   If the current table is in the wrong order, reorder it with edit_file first.\n"
        "5. Reply with one short confirmation sentence naming the year you added."
    )

    # last confirmed post-call token count from openrouter
    # used as the base for the next pre-call estimate
    # estimates are more accurate if not started at zero
    state = {"last_post_tokens": 949}

    # ONLY add to messages inside chat() because we want to filter first
    messages = []
    new_messages = []

    if system_prompt:
        if USE_SYSTEM_PROMPT:
            new_messages.append({"role": "system", "content": system_prompt})
        else:
            new_messages.append({"role": "user", "content": system_prompt})

    if initial_prompt:
        new_messages.append({"role": "user", "content": initial_prompt})

    print("MCP server alive, handshaking...")
    mcp = start_mcp()
    print("MCP handshake complete.\n")

    print("Starting agent loop...\n")

    while True:
        response = chat(messages, tools, new_messages, state)
        choice = response["choices"][0]
        msg = choice["message"]
        finish = choice["finish_reason"]
        new_messages.append(msg)

        if finish == "tool_calls":
            for tc in msg["tool_calls"]:
                fn_name = tc["function"]["name"]
                fn_args = json.loads(tc["function"]["arguments"])
                print(f"[tool call] {fn_name}")
                tool_result = dispatch_tool(mcp, fn_name, fn_args)
                new_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": tool_result,
                    }
                )
        else:
            print(f"\n[done] {msg['content']}")
            break

    mcp["proc"].stdin.close()
    mcp["proc"].terminate()


if __name__ == "__main__":
    main()
