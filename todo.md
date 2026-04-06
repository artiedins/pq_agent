# pq_agent harness development plan

Design philosophy: provide the minimal set of harness features that all frontier models
will be increasingly effective using. Assume models in 2026-2027 are post-trained to
work well against a stable, simple harness contract. Complexity that compensates for
current model weaknesses is technical debt.

---

## Project layout convention

    ~/Projects/pq_agent/          harness - never edited per-project
        agent.py
        run_agent.sh
        mcp_server.js
        logsift_p.md              source p.md for logsift project
        logsift_q.py              source q.py for logsift project
        encoding_p.md             source p.md for encoding project
        encoding_q.py             source q.py for encoding project
        setup_logsift.sh          creates ~/Projects/logsift/
        setup_encoding.sh         creates ~/Projects/encoding/

    ~/Projects/XXXX/              project workspace - one per task
        p.md                      task prompt (human-readable markdown)
        q.py                      verification oracle (always run as python3 q.py)
        [task data and outputs]

The P/Q naming is loosely inspired by Hoare triples. P is the prompt - what you would
hand a human contractor. Q is the postcondition - the automated test that defines done.
They are separate files so each can be read and edited independently.

The system prompt in agent.py establishes the q.py convention universally: when the
agent believes the task is complete it runs python3 q.py and fixes failures until it
prints OK. p.md never mentions q.py - it is purely a task description.

---

## Known issues / backlog (fix when they bite)

- **Playwright output size**: content returned by `playwright_extract_content` may be
  larger than expected. Check by printing `len(text)` in the playwright_extract_content
  branch of `dispatch_tool`. If routinely above ~8000 chars, add a truncation pass
  with a notice similar to `shorten_content_with_notice`. Low priority until a task
  actually fails because of it.

- **Hashline codes leaking into output files**: guard in `tool_write_file` returns an
  error rather than silently stripping. The model must fix it. Monitor whether models
  hit this in practice.

- **Agent can overwrite q.py**: observed in the wild - the model wrote a simpler
  replacement q.py that happened to still pass. The oracle is the harness's ground
  truth and must not be replaceable by the agent. Fix: add a protected-files check
  in `tool_write_file`:
  ```python
  PROTECTED = {"q.py", "p.md"}
  if Path(filename).name in PROTECTED:
      return f"Error: {filename} is read-only in this workspace."
  ```
  p.md is less critical since agent.py reads it once before the loop and uses that
  copy for the session - a mid-run overwrite has no effect on the current run. q.py
  has no such protection and must be explicitly locked. Do this soon.

---

## Completed

### Environment snapshot before agent loop ✓
`get_env_snapshot()` runs pwd, ls, and python3 --version in the workspace before the
first LLM call and appends the result to the initial prompt. Eliminates 2-5 wasted
exploration turns. Silent failure if it errors.

### `run_command` tool ✓
Runs a shell command in the workspace, 30s timeout, 8000-char output cap, combined
stdout+stderr plus exit code. Models can use full unix pipelines.

### Qwen3.6 compaction fallback ✓
`extract_compaction_summary()` checks `reasoning` and `reasoning_content` fields if
`content` is empty, handling models that return summaries in the reasoning trace.

### Hashline write guard ✓
`tool_write_file` rejects content with raw `LINENUM:HASH|` prefixes and returns an
error asking the model to strip them.

### P/Q project layout ✓
- `agent.py` reads `p.md` from the workspace as the initial prompt.
- System prompt is hardcoded in `agent.py` — same for all tasks.
- All tools always loaded (playwright + write/read/edit_file + run_command).
- MCP always started — overhead is negligible, universality is worth it.
- `task_config.py` concept retired.
- Project files are `p.md` (prompt) and `q.py` (oracle) only.
- Setup scripts live in pq_agent, copy p.md and q.py into the workspace.

---

## Near term

### Browsing / inflation task
Write `inflation_p.md` and `inflation_q.py`, and a `setup_inflation.sh` that creates
the workspace with an `inflation.md` stub. This validates the MCP (playwright) path
end-to-end under the new layout.

`inflation_q.py` should verify:
- `inflation.md` exists
- Contains a markdown table
- The table has at least one row more than the stub
- Rows are in chronological order

### MODEL_OVERRIDE env var
Add to agent.py config section:
```python
MODEL = os.environ.get("MODEL_OVERRIDE", "google/gemini-3.1-flash-lite-preview")
```
No other changes needed. Unlocks model sweeps without editing any file.

### Model sweep runner
A shell script that runs `run_agent.sh` with different models against the same
project directory and records pass/fail per model. Requires MODEL_OVERRIDE above.
Shape:

```bash
#!/usr/bin/env bash
# usage: bash sweep.sh ~/Projects/encoding
MODELS=(
  "google/gemini-3.1-flash-lite-preview"
  "qwen/qwen3.6-plus:free"
)
for model in "${MODELS[@]}"; do
  echo "--- $model"
  MODEL_OVERRIDE="$model" ./run_agent.sh "$1"
done
```

---

## Medium term

### Structured result capture
Have q.py write a `result.json` in addition to printing OK/FAIL. Fields:
`{"status": "ok"|"fail", "detail": "..."}`. The sweep runner reads this instead of
parsing stdout. Useful once multiple models run the same task and you want automated
comparison. Convention: q.py always writes result.json before exiting, agent does not
need to know about it.

### Per-tool output length limits on playwright
Add a 12000-char ceiling on `playwright_extract_content` output. Check actual sizes
first (see known issues backlog).

---

## Long term / nice to have

- **`run_agent.sh` defaults to cwd**: currently requires a project dir argument.
  Change so that running `./run_agent.sh` with no argument uses the current directory.
  The one-argument form stays for calling from other scripts.

- **Multiple tasks in one run**: allow `p.md` to describe a sequence of sub-tasks.
  Probably not needed — individual focused tasks with their own q.py are cleaner and
  easier to debug.

- **Investigate hashline alternatives**: the `LINENUM:HASH|content` format works but
  causes occasional model confusion. By 2027 most models will handle unified diff or
  line-number-only references reliably. Revisit then — do not change it now.

---

## Do not do

### Complexity that compensates for current model weakness (will be obsolete by 2027)

**Forced `analysis` + `plan` fields in tool schema.**
Scaffolded CoT baked into the tool contract. By 2027 frontier models reason before
acting without structural coercion. Costs tokens for no gain with capable models.

**Double-confirmation `task_complete` with a checklist.**
Addresses a model reliability gap that is closing. The q.py convention handles
verification cleanly without adding a round-trip.

**Two-stage draft-verification retrieval.**
Adds a full LLM round-trip per query. Only useful when the model cannot express its
own uncertainty. Post-trained 2027 models will handle this natively.

**Label-primed contrastive pairs for classification.**
~200 lines of task-specific retrieval logic. Irrelevant to this harness's direction.

**Domain routing with lexical gates.**
Brittle and task-specific. Irrelevant for a general-purpose harness.

**Filesystem-based meta-optimization loop.**
Requires Claude Code as the proposer and 10M tokens per iteration. Not appropriate as
an ongoing development tool; use human judgment and this plan.

### Patterns that add noise

**Aggressive context pruning.**
Dropping old messages or tool results to save tokens loses coherence. The current
compaction approach (summarise the whole session when near the limit) is the right
trade-off.

**Scalar-only reward signals for any self-improvement loop.**
If you ever add a feedback layer, it must have access to the full trace — not just
pass/fail counts.

**Injecting repeated prompt reminders mid-conversation.**
One clear instruction in the system prompt is enough. Repeating "remember: use tools"
wastes context tokens and trains the model to ignore boilerplate.
