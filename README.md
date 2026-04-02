# pq_agent

A sandboxed LLM agent harness for software development tasks.

The name is a nod to [Hoare triples](https://en.wikipedia.org/wiki/Hoare_logic): the agent takes a **P** (prompt describing the task) and a **Q** (validator that checks the result). Right now it is a research agent with a browser; the P/Q structure is the intended direction.

## What it does

Runs an agentic loop against a free Qwen model via OpenRouter. The agent has access to:

- A Playwright-controlled Chrome browser for web research
- A `write_file` tool that writes results into the project directory

The agent process runs inside a [bubblewrap](https://github.com/containers/bubblewrap) sandbox. The sandbox gives the agent read-write access to the target project directory only. The rest of the filesystem — home directory, SSH keys, other projects, dotfiles — is invisible.

## Architecture

```
run_agent.sh <project_dir>
    |
    +-- bwrap sandbox
          |
          +-- /agent      (read-only)  ~/Projects/agent
          +-- /workspace  (read-write) <project_dir>
          +-- /pw-cache   (read-only)  ~/.cache/ms-playwright
          |
          +-- python3 agent.py
                |
                +-- node mcp_server.js   (Playwright MCP, stdio JSON-RPC)
                +-- OpenRouter API       (Qwen model, outbound HTTPS)
```

## Dependencies

- `bubblewrap` — `sudo apt install bubblewrap uidmap`
- `node` — for the Playwright MCP server
- `python3` with `requests` — `pip install requests`
- Playwright MCP — `npm install` (see `package.json`)
- An [OpenRouter](https://openrouter.ai) API key

## Setup

```bash
git clone https://github.com/yourname/pq_agent ~/Projects/pq_agent
cd ~/Projects/pq_agent
npm install
playwright install chromium
```

Set your API key in the environment (add to `~/.bashrc` or similar):

```bash
export OPENROUTER_API_KEY=your_key_here
```


## Usage

Create a project directory and run the agent against it:

```bash
mkdir ~/Projects/my_research
bash ~/Projects/pq_agent/run_agent.sh ~/Projects/my_research
```

The agent will write its output into `~/Projects/my_research`.

## Security model

The bubblewrap sandbox provides filesystem isolation:

| Path | Inside sandbox |
|---|---|
| `<project_dir>` | read-write (mounted at `/workspace`) |
| `~/.cache/ms-playwright` | read-only (mounted at `/pw-cache`) |
| `/usr`, `/etc`, `/bin` etc. | read-only |
| `$HOME`, `~/.ssh`, other projects | not visible |
| `/tmp`, `/run`, `/home`, `/root` | empty tmpfs |

