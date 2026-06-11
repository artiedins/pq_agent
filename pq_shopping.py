#!/usr/bin/env python3

import os
import sys
import json
import shutil
from datetime import datetime

# llm_client lives next to this script; sys.path is configured below so we
# can import it whether pq_shopping is run from its own directory or another.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from llm_client import call_llm

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


# Setup-time provider options. ds4_flash is excluded because it's the agent
# model and we want a stronger model designing the research workflow.
SETUP_PROVIDER_OPTIONS = [
    {"key": "anthropic", "label": "Claude Opus 4.8"},
    {"key": "qwen", "label": "Qwen3.6-Plus"},
    {"key": "glm", "label": "GLM-5.1"},
    {"key": "ds4_pro", "label": "DeepSeek V4 Pro"},
]
DEFAULT_SETUP_PROVIDER = "anthropic"

DEFAULT_LOCATION = "Gardena, CA, USA"
DEFAULT_PRICE_SENSITIVITY = 8


def ask(prompt, default=None):
    if default is not None:
        sys.stdout.write(prompt + " (default: " + str(default) + "): ")
    else:
        sys.stdout.write(prompt + ": ")
    sys.stdout.flush()
    val = sys.stdin.readline().strip()
    if not val and default is not None:
        return str(default)
    return val


def ask_provider():
    # show numbered list and accept either the number or the key as input
    print("\nWhich model should design the research tasks?")
    for i, opt in enumerate(SETUP_PROVIDER_OPTIONS, 1):
        marker = " (default)" if opt["key"] == DEFAULT_SETUP_PROVIDER else ""
        print("  " + str(i) + ") " + opt["label"] + " [" + opt["key"] + "]" + marker)
    raw = ask("Choice", DEFAULT_SETUP_PROVIDER)
    raw = raw.strip().lower()
    # numeric index
    if raw.isdigit():
        idx = int(raw) - 1
        if 0 <= idx < len(SETUP_PROVIDER_OPTIONS):
            return SETUP_PROVIDER_OPTIONS[idx]["key"]
    # key match
    for opt in SETUP_PROVIDER_OPTIONS:
        if raw == opt["key"]:
            return opt["key"]
    print("  (unrecognized choice, using default " + DEFAULT_SETUP_PROVIDER + ")")
    return DEFAULT_SETUP_PROVIDER


def price_description(n):
    if n <= 2:
        return "budget-focused: minimum cost, good enough is fine"
    if n <= 4:
        return "value-focused: balance cost and quality, avoid overspending"
    if n <= 6:
        return "quality-leaning: willing to pay more for meaningful improvements"
    if n <= 8:
        return "quality-focused: price is secondary to getting the right product"
    return "best-in-class: price is not a concern, only the best will do"


def build_meta_prompt(what, location, sensitivity):
    current_date_str = datetime.now().strftime("%Y-%m-%d (%A)")
    parts = [
        "You are designing a multi-step autonomous consumer research workflow.\n\n",
        "## System Overview\n\n",
        "The workflow uses a task queue where an AI agent (DeepSeek V4 Flash with reasoning) executes\n",
        "each task using browser and file tools, then a judge model evaluates the output against a\n",
        "rubric. Tasks run in order. Files written to the workspace persist between tasks - this is\n",
        "the mechanism by which later tasks build on earlier ones.\n\n",
        "The agent has these tools:\n",
        "- playwright_navigate(url): navigate browser to a URL, returns page title\n",
        "- playwright_extract_content(selector?): extract current page as clean markdown;\n",
        "  always pass a CSS selector to target relevant content and reduce noise\n",
        "- write_file(filename, content): write a file to the workspace (creates parent dirs)\n",
        "- read_file(filename): read a file from the workspace\n",
        "- run_command(command): run a shell command in the workspace (30s timeout)\n\n",
        "## Source Priorities and Search Patterns\n\n",
        "The agent should use these sources in approximate priority order:\n\n",
        "1. Specialist review sites with documented testing methodology - whichever is most\n",
        "   relevant to the product category. Examples by category:\n",
        "   - Electronics/audio: Wirecutter, RTINGS (https://www.rtings.com)\n",
        "   - Outdoor gear: OutdoorGearLab (https://www.outdoorgearlab.com)\n",
        "   - Kitchen/cooking: America's Test Kitchen, Wirecutter\n",
        "   - Tools/home: Wirecutter, This Old House\n",
        "   - General: The Strategist (nymag.com/strategist), Reviewed (reviewed.usatoday.com)\n",
        "   Search these directly by navigating to their sites and searching the product.\n\n",
        "2. Hacker News - high signal from technically sophisticated users:\n",
        "   Stories: https://hn.algolia.com/?q=QUERY&dateRange=pastYear&type=story\n",
        "   Comments (often most valuable): https://hn.algolia.com/?q=QUERY&type=comment&dateRange=pastYear\n",
        "   Use selector `.comment-text` when extracting HN comment threads.\n\n",
        "3. Reddit - real user experiences, long-term ownership, frank discussion:\n",
        "   Search a subreddit: https://old.reddit.com/r/SUBREDDIT/search?q=TOPIC&sort=top&restrict_sr=on&t=year\n",
        "   Browse top posts: https://old.reddit.com/r/SUBREDDIT/top/?t=year\n",
        "   Use selector `.commentarea` for comments, `.search-result` for search result pages.\n",
        "   IMPORTANT: Do not rapid-fire Reddit requests. Browse at a natural human pace.\n",
        "   To discover relevant subreddits: search DDG for '[product] site:reddit.com subreddit'\n\n",
        "4. YouTube - for products where visual demonstration matters:\n",
        "   https://www.youtube.com/results?search_query=QUERY\n",
        "   Titles and video descriptions are extractable. No video playback needed.\n\n",
        "5. DuckDuckGo plain HTML as a starting point for any search:\n",
        "   https://html.duckduckgo.com/html/?q=QUERY\n\n",
        "6. Direct retailer sites for current pricing:\n",
        "   Amazon product pages, specialty retailer sites (B&H, REI, etc.)\n\n",
        "Research quality standards:\n",
        "- Avoid content older than 5 years unless the category is very stable\n",
        "- Avoid SEO content farms: identifiable by generic headlines, heavy ads, thin content\n",
        "- Long-term ownership reports (6+ months of actual use) outweigh first impressions\n",
        "- Look for recurring patterns across multiple independent sources - these are reliable signals\n",
        "- Write summaries to disk frequently; do not accumulate many sources in memory at once\n",
        "- When a source is low credibility, note that in your summary\n",
        "- For pricing: note whether prices vary by retailer and whether there are deal patterns\n\n",
        "## Inter-Task File Conventions\n\n",
        "Tasks share a workspace. Use the research/ subdirectory for intermediate notes:\n\n",
        "Task 1 writes TWO things:\n",
        "  research/landscape.md   - full research notes: product map, price tiers, candidate list,\n",
        "                            key decision criteria, sources consulted with URLs\n",
        "  task_report/report.md   - brief evidence log FOR THE JUDGE: confirms landscape.md was\n",
        "                            written, lists all named products found, all source URLs used,\n",
        "                            and 2-3 sentences on each price tier identified\n\n",
        "Task 2 writes TWO things:\n",
        "  research/deep_[name].md - one file per contender researched (use short product name)\n",
        "                            each file: real user quotes, long-term ownership notes,\n",
        "                            common complaints, workarounds, use/care tips, sources\n",
        "  task_report/report.md   - brief evidence log FOR THE JUDGE: confirms deep files were\n",
        "                            written, lists which products were researched, key finding\n",
        "                            per product, all source URLs used\n\n",
        "Final task writes ONE thing:\n",
        "  task_report/report.md   - the complete buyer's guide (this IS the deliverable)\n\n",
        "CRITICAL: The judge can only see task_report/*.md. For tasks 1 and 2, the\n",
        "task_report/report.md must include enough evidence (product names, source URLs, key\n",
        "findings) that the judge can verify real research was done - not just claimed.\n\n",
        "## Buyer Profile\n\n",
        "- Current date: " + current_date_str + "\n",
        "- Location: " + location + "\n",
        "- Price sensitivity: " + str(sensitivity) + "/10 (" + price_description(sensitivity) + ")\n",
        "- Shopping request: " + what + "\n\n",
        "## Final Report Structure\n\n",
        "The final task_report/report.md must follow this structure:\n",
        "1. TL;DR (2-3 sentences): clear recommendation naming the specific product and price range\n",
        "2. Why this product: 3-5 concrete reasons matched to this buyer's actual context\n",
        "3. Where to buy: retailer name(s), expected price range, any deal/timing notes\n",
        "4. Honest cons: 1-3 real downsides (this builds trust and helps set expectations)\n",
        "5. Runner-up (if applicable): a clear second choice with 1-2 sentences explaining who it's for\n",
        "6. Practical tips: 2-4 tips on use, setup, care, or break-in that owners find valuable\n\n",
        "The report should be scannable and under 1200 words. It is written FOR the buyer to read,\n",
        "not for the judge. Avoid hedging language. Make a real recommendation.\n\n",
        "## Task Design Requirements\n\n",
        "Design 3 or 4 sequential tasks. Choose 3 for straightforward purchases.\n",
        "Choose 4 if the product category warrants separating: (a) pricing/availability research or\n",
        "(b) hands-on/technical spec analysis as its own step before synthesis.\n\n",
        "For each task:\n\n",
        "p.md requirements (the agent reads this and executes it):\n",
        "- Open with one sentence of context: what has been done before this task\n",
        "- For tasks 2+: explicitly name which files to read_file first before starting research\n",
        "- Specify which sources to prioritize and in what order for this specific product\n",
        "- Include example URLs with the actual product terms filled in (not abstract placeholders)\n",
        "  e.g. 'https://hn.algolia.com/?q=best+standing+desk&type=comment&dateRange=pastYear'\n",
        "- Name specific subreddits likely to be relevant for this product category\n",
        "- Give a rough research scope: approximately how many distinct sources to consult\n",
        "- State exactly which files to write and where (research/ vs task_report/)\n",
        "- For tasks 1-2: remind the agent that task_report/report.md is an evidence log for\n",
        "  the judge, not a final report - it must include source URLs and named products\n",
        "- For the final task: specify the report structure from the section above\n",
        "- Write as instructions to a capable but literal AI assistant\n\n",
        "q.md requirements (the judge reads this to evaluate the agent's work):\n",
        "- Use only concrete, checkable criteria\n",
        "- For tasks 1-2: the judge reads task_report/report.md to verify; criteria must be\n",
        "  checkable from that document alone (product names present, URLs cited, files confirmed)\n",
        "- Require minimum source counts with URLs: e.g. 'at least 4 distinct source URLs must\n",
        "  appear in the report'\n",
        "- Include at least 2 explicit fail conditions with the word FAIL in them\n",
        "- For task 1: require named candidates across at least 3 price tiers\n",
        "- For task 2: require evidence of long-term user experience (not just spec sheets)\n",
        "  and require common complaints/failure modes to be present\n",
        "- For final task: require named product + price range + retailer + at least 3 reasons\n",
        "  + at least 2 practical tips + at least 1 honest con\n",
        "- For final task: require the recommendation to reference the buyer's specific context\n",
        "  (not a generic recommendation that ignores the buyer profile)\n\n",
        "## Output Format\n\n",
        "Return ONLY a valid JSON object. No markdown fences. No preamble. No text outside the JSON.\n",
        "{\n",
        '  "project_md": "2-4 sentences describing the shopping goal, buyer profile, and research approach.",\n',
        '  "tasks": [\n',
        '    {"id": "task_001", "p": "full p.md content", "q": "full q.md content"},\n',
        '    {"id": "task_002", "p": "...", "q": "..."},\n',
        "    ...\n",
        "  ]\n",
        "}\n",
        "Task IDs must be task_001, task_002, etc. in order.",
    ]
    return "".join(parts)


def generate_tasks(what, location, sensitivity, provider):
    prompt = build_meta_prompt(what, location, sensitivity)

    print("\nGenerating research tasks via " + provider + "...")

    # call_llm handles retries, rate limits, and provider-specific quirks.
    # Returns a string; we strip any markdown fences before json.loads.
    text = call_llm(prompt, provider=provider).strip()

    if text.startswith("```"):
        # robust fence stripping: drop first line and last fence line
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines.pop(0)
        if lines and lines[-1].startswith("```"):
            lines.pop(-1)
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        sys.exit("Error: model returned invalid JSON: " + str(e) + "\nRaw response begins:\n" + text[:400])

    if "tasks" not in data or not data["tasks"]:
        sys.exit("Error: model returned no tasks in JSON")
    for i, task in enumerate(data["tasks"]):
        for key in ("id", "p", "q"):
            if key not in task or not str(task[key]).strip():
                sys.exit("Error: task[" + str(i) + "] missing required key: " + key)

    return data


def write_pq_structure(data, workspace_dir):
    harness_dir = os.path.join(workspace_dir, ".pq")
    os.makedirs(harness_dir, exist_ok=True)
    os.makedirs(os.path.join(harness_dir, "tasks"), exist_ok=True)

    task_ids = [t["id"] for t in data["tasks"]]

    with open(os.path.join(harness_dir, "queue.txt"), "w", encoding="utf-8") as f:
        f.write("# generated by pq_shopping.py\n")
        f.write("\n".join(task_ids) + "\n")

    with open(os.path.join(harness_dir, "project.md"), "w", encoding="utf-8") as f:
        f.write(data["project_md"])

    for task in data["tasks"]:
        task_dir = os.path.join(harness_dir, "tasks", task["id"])
        os.makedirs(task_dir, exist_ok=True)
        os.makedirs(os.path.join(task_dir, "runs"), exist_ok=True)
        with open(os.path.join(task_dir, "p.md"), "w", encoding="utf-8") as f:
            f.write(task["p"])
        with open(os.path.join(task_dir, "q.md"), "w", encoding="utf-8") as f:
            f.write(task["q"])

    return task_ids


def print_task_preview(harness_dir, task_ids):
    for tid in task_ids:
        p_path = os.path.join(harness_dir, "tasks", tid, "p.md")
        q_path = os.path.join(harness_dir, "tasks", tid, "q.md")
        p_preview = ""
        q_lines = 0
        if os.path.exists(p_path):
            with open(p_path, encoding="utf-8") as f:
                p_preview = f.readline().strip()
        if os.path.exists(q_path):
            with open(q_path, encoding="utf-8") as f:
                q_lines = len(f.readlines())
        print("  " + tid + ": " + p_preview[:72])
        if q_lines:
            print("           rubric: " + str(q_lines) + " lines")


def main():
    workspace_dir = os.getcwd()
    harness_dir = os.path.join(workspace_dir, ".pq")

    if os.path.exists(harness_dir):
        sys.stdout.write(".pq directory already exists here. Overwrite? (y/N): ")
        sys.stdout.flush()
        ans = sys.stdin.readline().strip().lower()
        if ans != "y":
            sys.exit("Aborted.")
        shutil.rmtree(harness_dir)

    print("")
    what = ask("What do you want to buy?")
    if not what.strip():
        sys.exit("Error: please describe what you want to buy.")

    location = ask("Where are you located?", DEFAULT_LOCATION)

    sensitivity_raw = ask(
        "Price sensitivity 0 (cheapest that works) to 10 (only the best)",
        str(DEFAULT_PRICE_SENSITIVITY),
    )
    try:
        sensitivity = max(0, min(10, int(sensitivity_raw)))
    except ValueError:
        sensitivity = DEFAULT_PRICE_SENSITIVITY
        print("  (invalid input, using default " + str(DEFAULT_PRICE_SENSITIVITY) + ")")

    provider = ask_provider()

    print("")
    print("What        : " + what)
    print("Location    : " + location)
    print("Sensitivity : " + str(sensitivity) + "/10 (" + price_description(sensitivity) + ")")
    print("Designer    : " + provider)

    data = generate_tasks(what, location, sensitivity, provider)

    task_ids = write_pq_structure(data, workspace_dir)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    minder_path = os.path.join(script_dir, "pq_minder.py")

    print("")
    print("Created .pq/ with " + str(len(task_ids)) + " research task(s):")
    print_task_preview(harness_dir, task_ids)
    print("")
    print("To run the research queue:")
    print("  python3 " + minder_path)
    print("")
    print("Or with the web UI:")
    print("  python3 " + os.path.join(script_dir, "pq_web.py"))
    print("")
    print("Final report will appear in task_report/report.md after the last task passes.")


if __name__ == "__main__":
    main()
