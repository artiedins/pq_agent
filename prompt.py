#!/usr/bin/python3


import os
import sys


def get_language_tag(filename):
    """Determine the markdown language tag based on file extension."""
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".py":
        return "python"
    elif ext == ".sh":
        return "bash"
    return "text"


def create_markdown_context(files):
    """Reads files and formats them into a Markdown string."""
    lines = []
    lines.append("# Codebase Context\n")
    lines.append("The following files are provided as context for follow-on tasks:\n")

    for filename in files:
        lines.append(f"## `{filename}`")

        if os.path.exists(filename):
            lang = get_language_tag(filename)
            lines.append(f"```{lang}")

            try:
                with open(filename, "r", encoding="utf-8") as f:
                    content = f.read().rstrip()  # Remove trailing newlines
                    # If file is empty, just put a placeholder
                    if not content:
                        content = "# (Empty file)"
                    lines.append(content)
            except Exception as e:
                lines.append(f"# Error reading file: {e}")

            lines.append("```\n")
        else:
            lines.append("> **Note:** File not found in the current directory.\n")

    return "\n".join(lines)


if __name__ == "__main__":
    # The specific files requested
    target_files = ["agent.py", "run_agent.sh", "pq_minder.py", "pq_web.py", "llm_client.py"]

    markdown_output = create_markdown_context(target_files)

    # Print to stdout so it can be easily copied or piped
    print(markdown_output)
