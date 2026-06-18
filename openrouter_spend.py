#!/usr/bin/env python3
"""
Query OpenRouter's Beta Analytics API for per-model spend in the last hour.

Requires a MANAGEMENT key (not a regular inference key).
Create one at: https://openrouter.ai/settings/management-keys

Set the env var:
    export OPENROUTER_MGMT_KEY="your-management-key-here"

Usage:
    python openrouter_spend.py
    python openrouter_spend.py --hours 4      # look back 4 hours instead of 1
"""

import json
import os
import sys
import urllib.request
import urllib.error
from datetime import datetime, timedelta, timezone
from argparse import ArgumentParser

API_URL = "https://openrouter.ai/api/v1/analytics/query"


def query_spend(api_key: str, hours: int = 1) -> dict:
    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=hours)

    payload = {
        "metrics": [
            "total_usage",
            "request_count",
            "tokens_total",
            "tokens_prompt",
            "tokens_completion",
        ],
        "dimensions": ["model"],
        "order_by": {"field": "total_usage", "direction": "desc"},
        "time_range": {
            "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "limit": 100,
    }

    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        API_URL,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        if e.code == 403:
            print(
                "ERROR 403: Forbidden. Make sure you're using a MANAGEMENT key, " "not a regular inference key.\n" "Create one at: https://openrouter.ai/settings/management-keys",
                file=sys.stderr,
            )
        else:
            print(f"HTTP {e.code}: {body}", file=sys.stderr)
        sys.exit(1)


def print_results(result: dict, hours: int) -> None:
    rows = result.get("data", {}).get("data", [])
    meta = result.get("data", {}).get("metadata", {})

    if not rows:
        print(f"No usage recorded in the last {hours} hour(s).")
        return

    # Normalize values — the API may return some metrics as strings
    for row in rows:
        for key in ("total_usage", "request_count", "tokens_total", "tokens_prompt", "tokens_completion"):
            val = row.get(key)
            if val is not None:
                row[key] = float(val)

    # Header
    print(f"\n{'=' * 72}")
    print(f"  OpenRouter spend by model — last {hours} hour(s)")
    print(f"{'=' * 72}\n")

    # Column widths
    model_w = max(len(r.get("model", "")) for r in rows)
    model_w = max(model_w, 5)

    header = f"  {'Model':<{model_w}}  {'Spend':>10}  {'Reqs':>7}  " f"{'Prompt Tok':>12}  {'Compl Tok':>12}  {'Total Tok':>12}"
    print(header)
    print(f"  {'-' * (len(header) - 2)}")

    total_spend = 0.0
    for row in rows:
        model = row.get("model", "unknown")
        spend = row.get("total_usage", 0)
        reqs = int(row.get("request_count", 0))
        tok_prompt = int(row.get("tokens_prompt", 0))
        tok_compl = int(row.get("tokens_completion", 0))
        tok_total = int(row.get("tokens_total", 0))
        total_spend += spend

        print(f"  {model:<{model_w}}  ${spend:>9.4f}  {reqs:>7,}  " f"{tok_prompt:>12,}  {tok_compl:>12,}  {tok_total:>12,}")

    print(f"  {'-' * (len(header) - 2)}")
    print(f"  {'TOTAL':<{model_w}}  ${total_spend:>9.4f}")
    print()

    if meta.get("truncated"):
        print("  ⚠  Results were truncated. Increase --limit or narrow the window.\n")


def main():
    parser = ArgumentParser(description="OpenRouter per-model spend in the last N hours")
    parser.add_argument(
        "--hours",
        type=int,
        default=1,
        help="Look-back window in hours (default: 1)",
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_MGMT_KEY", "").strip()
    if not api_key:
        print(
            "ERROR: Set OPENROUTER_MGMT_KEY to your management key.\n" "  export OPENROUTER_MGMT_KEY='your-management-key'\n" "Create one at: https://openrouter.ai/settings/management-keys",
            file=sys.stderr,
        )
        sys.exit(1)

    result = query_spend(api_key, hours=args.hours)
    print_results(result, hours=args.hours)


if __name__ == "__main__":
    main()
