import os
import sys
import json
import subprocess
import requests
import time
from pathlib import Path


# Code style:
# - No type hinting
# - No doc strings
# - No triple quoted multi-line strings
# - No comments with repeated characters for visual page breaks
# - No non-ascii characters
# - No command line argument processing
# - Yes inline comments for showing intent without ceremony


# Config
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    sys.exit("Error: OPENROUTER_API_KEY environment variable is not set.")

# AGENT_DIR is set by run_agent.sh so we can find mcp_server.js
# regardless of which project dir we're working in
AGENT_DIR = os.environ.get("AGENT_DIR", str(Path(__file__).parent.resolve()))

MODEL = "qwen/qwen3.6-plus-preview:free"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
}

# workspace root is cwd - run_agent.sh sets this to the project dir via --chdir
# all file writes by the agent must stay inside here
WORKSPACE = Path.cwd().resolve()


def chat(messages, tools):
    payload = {
        "model": MODEL,
        "tools": tools,
        "tool_choice": "auto",
        "messages": messages,
    }
    resp = requests.post(OPENROUTER_URL, headers=HEADERS, json=payload)
    resp.raise_for_status()
    return resp.json()


# MCP server - pipe stderr to OUR stderr so node crash reasons are visible
mcp_proc = subprocess.Popen(
    ["node", os.path.join(AGENT_DIR, "mcp_server.js")],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=sys.stderr,
    text=True,
    bufsize=1,
)

time.sleep(1)
if mcp_proc.poll() is not None:
    sys.exit(f"MCP server exited immediately with code {mcp_proc.returncode}")

print("MCP server process is alive, proceeding with handshake...")

# JSON-RPC 2.0 helpers
_mcp_id = 0


def mcp_send(method, params, notify=False):
    global _mcp_id
    if notify:
        msg = {"jsonrpc": "2.0", "method": method, "params": params}
        mcp_proc.stdin.write(json.dumps(msg) + "\n")
        mcp_proc.stdin.flush()
        return None
    _mcp_id += 1
    msg = {"jsonrpc": "2.0", "id": _mcp_id, "method": method, "params": params}
    mcp_proc.stdin.write(json.dumps(msg) + "\n")
    mcp_proc.stdin.flush()
    return _mcp_id


def mcp_recv(expected_id):
    while True:
        line = mcp_proc.stdout.readline()
        if not line:
            raise RuntimeError("MCP server closed unexpectedly")
        msg = json.loads(line)
        if msg.get("id") == expected_id:
            if "error" in msg:
                raise RuntimeError(f"MCP error: {msg['error']}")
            return msg["result"]


def mcp_call(method, params):
    req_id = mcp_send(method, params)
    return mcp_recv(req_id)


# MCP handshake
mcp_call(
    "initialize",
    {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "agent", "version": "0.1"},
    },
)
mcp_send("notifications/initialized", {}, notify=True)

print("MCP handshake complete.\n")


# write_file tool implementation
# returns a result string the agent sees; raises on any security violation
def tool_write_file(filename, content):
    # resolve to absolute path and confirm it stays inside WORKSPACE
    # this blocks path traversal attacks like ../../etc/passwd
    target = (WORKSPACE / filename).resolve()
    if not str(target).startswith(str(WORKSPACE)):
        return f"Error: path '{filename}' resolves outside workspace - write refused"

    # create any intermediate directories the agent requested
    target.parent.mkdir(parents=True, exist_ok=True)

    target.write_text(content, encoding="utf-8")
    return f"Written {target.stat().st_size} bytes to {target.relative_to(WORKSPACE)}"


# Tool definitions
tools = [
    {
        "type": "function",
        "function": {
            "name": "playwright_navigate",
            "description": (
                "Navigate the headed Chrome browser to a URL and return the page title. "
                "Use DDG plain HTML (https://html.duckduckgo.com/html/?q=...) for searches."
            ),
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string", "description": "Full URL to navigate to"}},
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "playwright_extract_content",
            "description": (
                "Extract the current page as clean markdown. "
                "Optionally scope to a CSS selector such as 'main' or 'article'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "selector": {
                        "type": "string",
                        "description": "Optional CSS selector to scope extraction",
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
                "Write text content to a file inside the workspace. "
                "Use a relative path such as 'report.md' or 'output/summary.md'. "
                "Creates parent directories if needed. "
                "Overwrites the file if it already exists."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Relative path for the file, e.g. 'flash_attention.md'",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full text content to write to the file",
                    },
                },
                "required": ["filename", "content"],
            },
        },
    },
]


# Tool dispatcher
def dispatch_tool(name, arguments):
    if name in ("playwright_navigate", "playwright_extract_content"):
        result = mcp_call("tools/call", {"name": name, "arguments": arguments})
        content_blocks = result.get("content", [])
        text = "\n".join(block["text"] for block in content_blocks if block.get("type") == "text")
        print(f"\n[MCP <- {name}] {text[:120]}...")
        return text

    if name == "write_file":
        result = tool_write_file(arguments["filename"], arguments["content"])
        print(f"\n[write_file] {result}")
        return result

    return f"Unknown tool: {name}"


# System prompt
system_prompt = (
    "You are a rigorous research assistant with access to a headed Chrome browser "
    "via Playwright tools, and a write_file tool to save your results.\n\n"
    "Browsing rules:\n"
    "1. Always start searches with DuckDuckGo plain HTML: "
    "   https://html.duckduckgo.com/html/?q=<query>\n"
    "2. Run 2-4 meaningfully distinct queries before visiting any result.\n"
    "3. After gathering results, write a clear, well-structured markdown report "
    "   using the write_file tool. The filename should be descriptive, e.g. 'flash_attention.md'.\n"
    "4. Cite sources in the markdown report.\n"
    "5. After writing the file, respond with a single short confirmation message "
    "   and nothing else - do not reproduce the report content in your reply."
)

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "Please research 'flash attention' and save a concise technical summary as a markdown file."},
]

# Agentic loop
print("Starting agent loop...\n")

while True:
    response = chat(messages, tools)

    choice = response["choices"][0]
    msg = choice["message"]
    finish = choice["finish_reason"]

    messages.append(msg)

    if finish == "tool_calls":
        for tc in msg["tool_calls"]:
            fn_name = tc["function"]["name"]
            fn_args = json.loads(tc["function"]["arguments"])

            print(f"[Agent -> tool call] {fn_name}({list(fn_args.keys())})")

            tool_result = dispatch_tool(fn_name, fn_args)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": tool_result,
                }
            )

    else:
        print(f"\n[Agent done] {msg['content']}")
        break

mcp_proc.stdin.close()
mcp_proc.terminate()
