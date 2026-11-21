#!/usr/bin/env python3
import csv
from dataclasses import dataclass
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import socket


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    root_dir: Path = Path(__file__).resolve().parent
    web_dir: Path = Path(__file__).resolve().parent / "web"


MODEL_CATALOG = [
    ("openai/gpt-6.2-astra", "GPT-6.2 Astra", "OpenAI", "openai", "standard"),
    ("anthropic/claude-fable-5.2", "Claude Fable 5.2", "Anthropic", "anthropic", "standard"),
    ("google/gemini-4-carbon", "Gemini 4 Carbon", "Google", "google", "standard"),
    ("openai/gpt-6.2-sol", "GPT-6.2 Sol", "OpenAI", "openai", "standard"),
    ("google/gemini-4-neon", "Gemini 4 Neon", "Google", "google", "standard"),
    ("anthropic/claude-opus-5.5", "Claude Opus 5.5", "Anthropic", "anthropic", "standard"),
    ("openai/gpt-6.1-terra", "GPT-6.1 Terra", "OpenAI", "openai", "standard"),
    ("google/gemini-4-hydrogen", "Gemini 4 Hydrogen", "Google", "google", "standard"),
    ("kimi-k3.1-pro", "Kimi K3.1 Pro", "Moonshot AI", "other", "standard"),
    ("meta/muse-wonder", "Muse Wonder", "Meta", "meta", "standard"),
    ("anthropic/claude-sonnet-5.5", "Claude Sonnet 5.5", "Anthropic", "anthropic", "standard"),
    ("google/gemini-3.8-flash", "Gemini 3.8 Flash", "Google", "google", "standard"),
    ("deepseek/deepseek-v4.1-flash-1017", "DeepSeek V4.1 Flash", "DeepSeek", "deepseek", "standard"),
    ("meta/muse-spark-1.2-contributor", "Muse Spark 1.2", "Meta", "meta", "standard"),
    ("baidu/ernie-5.5-flash-preview", "ERNIE 5.5 Flash", "Baidu", "other", "standard"),
    ("openai/gpt-5.6-terra", "GPT-5.6 Terra", "OpenAI", "openai", "govcloud"),
    ("x-ai/grok-4.7", "Grok 4.7", "xAI", "xai", "standard"),
    ("mimo-v2.6-pro", "MiMo V2.6 Pro", "Xiaomi", "other", "standard"),
    ("z-ai/glm-5.3", "GLM-5.3", "Z.ai", "other", "standard"),
    ("openai/gpt-6-luna", "GPT-6 Luna", "OpenAI", "openai", "standard"),
    ("openai/gpt-5.6-luna", "GPT-5.6 Luna", "OpenAI", "openai", "govcloud"),
    ("qwen4-max", "Qwen 4 Max", "Alibaba Qwen", "qwen", "standard"),
    ("nex-agi/nex-n2.6", "Nex N2.6", "Nex-AGI", "other", "standard"),
    ("z-ai/glm-5.3-flash", "GLM-5.3 Flash", "Z.ai", "other", "standard"),
    ("ricursiveai/ripple-65b-preview", "Ripple 65B", "Ricursive AI", "other", "onprem_open"),
    ("thinkingmachines/inkling-2-small:free", "Inkling 2 Small", "Thinking Machines", "other", "onprem_open"),
    ("meta/muse-spark-open", "Muse Spark Open", "Meta", "meta", "onprem_open"),
]


def load_metric_csv(path):
    records = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) < 4:
                continue
            date_str, slug, provider, value = row[0].strip(), row[1].strip(), row[2].strip(), row[3].strip()
            if slug not in records:
                records[slug] = {
                    "date": date_str.replace("_", "-"),
                    "provider": provider,
                    "value": float(value),
                }
    return records


def build_benchmark_payload(root_dir):
    adev = load_metric_csv(root_dir / "adev_bench.csv")
    tbs = load_metric_csv(root_dir / "tbs01_bench.csv")
    trivia = load_metric_csv(root_dir / "trivia_bench.csv")
    cost = load_metric_csv(root_dir / "cost_data.csv")

    models = []
    dates = []
    for slug, name, lab, lab_key, tier in MODEL_CATALOG:
        if slug not in adev or slug not in tbs or slug not in trivia or slug not in cost:
            continue
        adev_val = adev[slug]["value"]
        tbs_val = tbs[slug]["value"]
        trivia_val = trivia[slug]["value"]
        cost_val = cost[slug]["value"]
        composite = (adev_val + tbs_val + trivia_val) / 3.0
        eval_date = adev[slug]["date"]
        dates.append(eval_date)
        models.append(
            {
                "slug": slug,
                "name": name,
                "lab": lab,
                "labKey": lab_key,
                "tier": tier,
                "provider": adev[slug]["provider"],
                "date": eval_date,
                "adev": round(adev_val, 4),
                "tbs": round(tbs_val, 4),
                "trivia": round(trivia_val, 4),
                "composite": round(composite, 4),
                "cost": round(cost_val, 4),
            }
        )

    models.sort(key=lambda item: (-item["composite"], item["cost"]))
    for index, item in enumerate(models, start=1):
        item["rank"] = index

    # Paid Pareto frontier: non-dominated models with positive reported cost.
    paid = sorted((m for m in models if m["cost"] > 0), key=lambda m: (m["cost"], -m["composite"]))
    frontier_slugs = []
    best_score = -1.0
    for item in paid:
        if item["composite"] > best_score:
            frontier_slugs.append(item["slug"])
            best_score = item["composite"]

    return {
        "meta": {
            "suite": "Composite Intelligence Benchmark",
            "version": "Nov 2026",
            "dateRange": [min(dates), max(dates)] if dates else [],
            "modelCount": len(models),
            "benchmarks": ["ADEV Bench", "TB Science (TBS01)", "Trivia Bench"],
            "costUnit": "USD per suite run",
        },
        "frontier": frontier_slugs,
        "models": models,
    }


def detect_external_ip():
    # A UDP connect discovers the outbound route interface without transmitting packets.
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            ip = sock.getsockname()[0]
            if ip and not ip.startswith("127."):
                return ip
    except OSError:
        pass
    try:
        ip = socket.gethostbyname(socket.gethostname())
        if ip and not ip.startswith("127."):
            return ip
    except OSError:
        pass
    return "127.0.0.1"


class BenchmarkHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


def make_handler(config):
    class BenchmarkHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(config.web_dir), **kwargs)

        def end_headers(self):
            self.send_header("Cache-Control", "no-cache")
            super().end_headers()

        def do_GET(self):
            clean_path = self.path.split("?", 1)[0]
            if clean_path == "/api/benchmarks.json":
                payload = json.dumps(build_benchmark_payload(config.root_dir), indent=2).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            super().do_GET()

        def log_message(self, format_str, *args):
            print(f"[{self.log_date_time_string()}] {self.address_string()} - {format_str % args}")

    return BenchmarkHandler


def bind_server(config):
    handler = make_handler(config)
    for offset in range(20):
        port = config.port + offset
        try:
            return BenchmarkHTTPServer((config.host, port), handler), port
        except OSError:
            continue
    raise OSError(f"Could not bind to any port in range {config.port}-{config.port + 19}")


def main():
    config = ServerConfig()
    server, port = bind_server(config)
    external_ip = detect_external_ip()
    print(f"Serving benchmark plots from {config.web_dir}", flush=True)
    print(f"External URL: http://{external_ip}:{port}/", flush=True)
    print(f"Bound to:     http://{config.host}:{port}/", flush=True)
    print("Press Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
