#!/usr/bin/env python3

import os
import random
import sys
import time

# Code style:
# - No type hinting
# - No doc strings
# - No triple quoted multi-line strings
# - No comments with repeated characters for visual page breaks like # ---
# - No non-ascii characters
# - No global variables unless making them local increases complexity
# - Yes strategic inline comments enhancing rapid code comprehension by real humans


# Default provider selection. Edit this single line to switch backends for
# testing. Supported values: "anthropic", "qwen", "glm", "kimi", "ds4_pro", "ds4_flash".
DEFAULT_PROVIDER = "anthropic"

MAX_RETRIES = 8


# Anthropic / Opus 4.7 specifics (as of 2026-04):
# - budget_tokens is removed; only thinking: {"type": "adaptive"} is valid.
# - thinking display defaults to "omitted" on 4.7, so thinking blocks are not
#   returned in the response unless we explicitly ask for "summarized". Since
#   we want plain text output, omitted is exactly what we want.
# - temperature/top_p/top_k return 400 if set to non-default values.
# - effort controls token spend. "medium" is the right tradeoff for synthesis
#   tasks: enough deliberation to think hard, not so much that it over-elaborates.
ANTHROPIC_MODEL = "claude-opus-4-6"
ANTHROPIC_MAX_TOKENS = 8192 * 2
ANTHROPIC_EFFORT = "medium"


# Qwen3.6-Plus specifics (as of 2026-04):
# - Hosted by Alibaba Cloud Model Studio (DashScope). Plus tier is API-only.
# - Recommended Python integration is the OpenAI-compatible endpoint, so we
#   reuse the openai SDK rather than pulling in the dashscope package.
# - International endpoint is Singapore. HRV_QWEN_BASE_URL env var lets you
#   override without editing code (Beijing/US-Virginia/HK are alternatives,
#   but API keys are region-specific and not interchangeable).
# - Hybrid thinking mode is enabled by default for qwen3.6-plus, but we pass
#   enable_thinking=True explicitly so behavior stays deterministic if Alibaba
#   flips the default later. Reasoning content comes back in
#   message.reasoning_content (not in .content), so the final answer is clean.
QWEN_MODEL = "qwen3.6-plus"
QWEN_BASE_URL = os.environ.get(
    "HRV_QWEN_BASE_URL",
    "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
)
QWEN_MAX_TOKENS = 32000
QWEN_ENABLE_THINKING = True


# OpenRouter-backed providers (glm, kimi, ds4_pro, ds4_flash) all share the same
# transport: OpenAI-compatible API at openrouter.ai with the unified
# "reasoning" param riding in extra_body. Differences are just the model slug.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MAX_TOKENS = 32000
OPENROUTER_REASONING_ENABLED = True
OPENROUTER_APP_URL = os.environ.get("HRV_OPENROUTER_APP_URL", "")
OPENROUTER_APP_TITLE = os.environ.get("HRV_OPENROUTER_APP_TITLE", "llm-client")

# :exacto is OpenRouter's quality-routing shortcut. Safe on all four; it's a
# no-op for models with a single provider but doesn't hurt. Kimi K2.6 supports
# reasoning via the same unified reasoning param as the other openrouter
# providers - if the underlying provider doesn't honor it, OpenRouter ignores
# the field silently rather than erroring.
GLM_MODEL = "z-ai/glm-5.1:exacto"
KIMI_MODEL = "moonshotai/kimi-k2.6:exacto"
DS4_PRO_MODEL = "deepseek/deepseek-v4-pro:exacto"
DS4_FLASH_MODEL = "deepseek/deepseek-v4-flash:exacto"


def _sleep_with_jitter(attempt):
    # Exponential backoff with jitter: 1-2s, 2-4s, 4-8s, 8-16s...
    delay = random.uniform(2 ** (attempt - 1), 2**attempt)
    print("  [retry " + str(attempt) + "/" + str(MAX_RETRIES - 1) + "] waiting " + "{:.1f}".format(delay) + "s...", file=sys.stderr)
    time.sleep(delay)


def _convert_messages_for_anthropic(messages):
    # Anthropic's API expects system content as a top-level param and image
    # content items in {type:"image", source:{type:"base64",...}} format.
    # OpenAI/OpenRouter format uses image_url with a data URI string. We
    # accept the OpenAI shape on input and convert here so callers only ever
    # build one format.
    system_text = None
    converted = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if role == "system":
            # Concatenate system messages if there are multiple (rare).
            if isinstance(content, str):
                system_text = (system_text + "\n\n" + content) if system_text else content
            continue
        if isinstance(content, str):
            converted.append({"role": role, "content": content})
            continue
        # content is a list of items - convert image_url -> image
        new_items = []
        for item in content:
            t = item.get("type")
            if t == "text":
                new_items.append({"type": "text", "text": item.get("text", "")})
            elif t == "image_url":
                url = item.get("image_url", {}).get("url", "")
                # data URI parsing: data:image/png;base64,XXXXX
                if url.startswith("data:") and ";base64," in url:
                    head, b64 = url.split(";base64,", 1)
                    media_type = head[len("data:") :]
                    new_items.append(
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": media_type, "data": b64},
                        }
                    )
                else:
                    # URL images are also legal in Anthropic format
                    new_items.append({"type": "image", "source": {"type": "url", "url": url}})
            elif t == "image":
                # already in Anthropic format - pass through
                new_items.append(item)
        converted.append({"role": role, "content": new_items})
    return system_text, converted


def _call_anthropic(prompt=None, messages=None):
    # Two call modes: a plain string prompt (single user turn) or a pre-built
    # message list (multi-turn, possibly multimodal). The structured path is
    # what pq_minder/pq_web use for the judge.
    import anthropic  # deferred; only needed on this call path

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("ERROR: provider='anthropic' but ANTHROPIC_API_KEY is not set")

    client = anthropic.Anthropic(api_key=api_key)

    if messages is None:
        messages = [{"role": "user", "content": prompt}]
    system_text, anth_messages = _convert_messages_for_anthropic(messages)

    print("# calling " + ANTHROPIC_MODEL + " (effort=" + ANTHROPIC_EFFORT + ", adaptive thinking, display omitted)...", file=sys.stderr)

    kwargs = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": ANTHROPIC_MAX_TOKENS,
        "messages": anth_messages,
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": ANTHROPIC_EFFORT},
    }
    if system_text:
        kwargs["system"] = system_text

    response = None
    for attempt in range(MAX_RETRIES):
        if attempt > 0:
            _sleep_with_jitter(attempt)
        try:
            response = client.messages.create(**kwargs)
            break
        except anthropic.RateLimitError:
            if attempt < MAX_RETRIES - 1:
                print("  [error] 429 rate limit, retrying...", file=sys.stderr)
                continue
            raise
        except anthropic.APIConnectionError as e:
            if attempt < MAX_RETRIES - 1:
                print("  [error] connection error, retrying: " + str(e), file=sys.stderr)
                continue
            raise
        except anthropic.APIStatusError as e:
            if e.status_code >= 500 and attempt < MAX_RETRIES - 1:
                print("  [error] " + str(e.status_code) + " server error, retrying...", file=sys.stderr)
                continue
            raise
    if response is None:
        raise RuntimeError("_call_anthropic: exhausted retries")

    # Extract only text blocks. Thinking blocks are omitted by default on
    # Opus 4.7, but we filter defensively in case that ever changes.
    text = "".join(b.text for b in response.content if b.type == "text").strip()

    usage = {
        "prompt_tokens": response.usage.input_tokens,
        "completion_tokens": response.usage.output_tokens,
    }
    print("# input_tokens=" + str(usage["prompt_tokens"]) + " output_tokens=" + str(usage["completion_tokens"]), file=sys.stderr)
    if getattr(response, "stop_reason", None) == "max_tokens":
        print("# WARNING: hit max_tokens; output likely truncated", file=sys.stderr)

    return text, usage


def _call_qwen(prompt=None, messages=None):
    # Alibaba's Model Studio exposes an OpenAI-compatible endpoint, so we
    # reuse the openai SDK with a base_url override.
    import openai

    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        sys.exit("ERROR: provider='qwen' but DASHSCOPE_API_KEY is not set")

    if messages is None:
        messages = [{"role": "user", "content": prompt}]

    client = openai.OpenAI(api_key=api_key, base_url=QWEN_BASE_URL)

    print("# calling " + QWEN_MODEL + " via " + QWEN_BASE_URL + " (thinking=" + ("on" if QWEN_ENABLE_THINKING else "off") + ")...", file=sys.stderr)

    response = None
    for attempt in range(MAX_RETRIES):
        if attempt > 0:
            _sleep_with_jitter(attempt)
        try:
            response = client.chat.completions.create(
                model=QWEN_MODEL,
                messages=messages,
                max_tokens=QWEN_MAX_TOKENS,
                # Qwen-specific toggles ride in extra_body - the OpenAI schema
                # has no field for these. enable_thinking is explicit to lock
                # behavior; defaults on Alibaba's side have changed before.
                extra_body={"enable_thinking": QWEN_ENABLE_THINKING},
            )
            break
        except openai.RateLimitError:
            if attempt < MAX_RETRIES - 1:
                print("  [error] 429 rate limit, retrying...", file=sys.stderr)
                continue
            raise
        except openai.APIConnectionError as e:
            if attempt < MAX_RETRIES - 1:
                print("  [error] connection error, retrying: " + str(e), file=sys.stderr)
                continue
            raise
        except openai.APIStatusError as e:
            if e.status_code >= 500 and attempt < MAX_RETRIES - 1:
                print("  [error] " + str(e.status_code) + " server error, retrying...", file=sys.stderr)
                continue
            raise
    if response is None:
        raise RuntimeError("_call_qwen: exhausted retries")

    # Reasoning goes to message.reasoning_content, final answer to
    # message.content - so .content is already clean.
    choice = response.choices[0]
    text = (choice.message.content or "").strip()
    usage = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
    }
    print("# prompt_tokens=" + str(usage["prompt_tokens"]) + " completion_tokens=" + str(usage["completion_tokens"]), file=sys.stderr)
    if choice.finish_reason == "length":
        print("# WARNING: hit max_tokens; output likely truncated", file=sys.stderr)

    return text, usage


def _call_openrouter(model_slug, prompt=None, messages=None):
    # Shared OpenRouter transport for glm, kimi, ds4_pro, ds4_flash. All share
    # the unified reasoning param via extra_body and identical retry logic;
    # only the model slug differs.
    import openai

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("ERROR: OpenRouter provider selected but OPENROUTER_API_KEY is not set")

    if messages is None:
        messages = [{"role": "user", "content": prompt}]

    # OpenRouter's attribution headers are optional. Populating them lets the
    # request show up on their leaderboards. Empty values get dropped by httpx.
    default_headers = {}
    if OPENROUTER_APP_URL:
        default_headers["HTTP-Referer"] = OPENROUTER_APP_URL
    if OPENROUTER_APP_TITLE:
        default_headers["X-Title"] = OPENROUTER_APP_TITLE

    client = openai.OpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        default_headers=default_headers or None,
    )

    print("# calling " + model_slug + " via openrouter (reasoning=" + ("on" if OPENROUTER_REASONING_ENABLED else "off") + ")...", file=sys.stderr)

    response = None
    for attempt in range(MAX_RETRIES):
        if attempt > 0:
            _sleep_with_jitter(attempt)
        try:
            response = client.chat.completions.create(
                model=model_slug,
                messages=messages,
                max_tokens=OPENROUTER_MAX_TOKENS,
                # OpenRouter's unified reasoning param: {"enabled": bool}
                # toggles thinking across providers without us having to know
                # each backend's native dial. Rides in extra_body because the
                # OpenAI schema has no "reasoning" field.
                extra_body={"reasoning": {"enabled": OPENROUTER_REASONING_ENABLED}},
            )
            break
        except openai.RateLimitError:
            if attempt < MAX_RETRIES - 1:
                print("  [error] 429 rate limit, retrying...", file=sys.stderr)
                continue
            raise
        except openai.APIConnectionError as e:
            if attempt < MAX_RETRIES - 1:
                print("  [error] connection error, retrying: " + str(e), file=sys.stderr)
                continue
            raise
        except openai.APIStatusError as e:
            if e.status_code >= 500 and attempt < MAX_RETRIES - 1:
                print("  [error] " + str(e.status_code) + " server error, retrying...", file=sys.stderr)
                continue
            raise
    if response is None:
        raise RuntimeError("_call_openrouter: exhausted retries")

    # OpenRouter puts reasoning in message.reasoning (distinct from content).
    choice = response.choices[0]
    text = (choice.message.content or "").strip()
    usage = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
    }
    print("# prompt_tokens=" + str(usage["prompt_tokens"]) + " completion_tokens=" + str(usage["completion_tokens"]), file=sys.stderr)
    if choice.finish_reason == "length":
        print("# WARNING: hit max_tokens; output likely truncated", file=sys.stderr)

    return text, usage


def _call_glm(prompt=None, messages=None):
    return _call_openrouter(GLM_MODEL, prompt=prompt, messages=messages)


def _call_kimi(prompt=None, messages=None):
    return _call_openrouter(KIMI_MODEL, prompt=prompt, messages=messages)


def _call_ds4_pro(prompt=None, messages=None):
    return _call_openrouter(DS4_PRO_MODEL, prompt=prompt, messages=messages)


def _call_ds4_flash(prompt=None, messages=None):
    return _call_openrouter(DS4_FLASH_MODEL, prompt=prompt, messages=messages)


# Dispatch table. Adding a provider = one entry here + one _call_* function.
_PROVIDERS = {
    "anthropic": _call_anthropic,
    "qwen": _call_qwen,
    "glm": _call_glm,
    "kimi": _call_kimi,
    "ds4_pro": _call_ds4_pro,
    "ds4_flash": _call_ds4_flash,
}


# Friendly labels for UI display. Order here is the order shown to humans.
PROVIDER_LABELS = [
    {"key": "anthropic", "label": "Claude Opus 4.7"},
    {"key": "qwen", "label": "Qwen3.6-Plus"},
    {"key": "glm", "label": "GLM-5.1"},
    {"key": "kimi", "label": "Kimi K2.6"},
    {"key": "ds4_pro", "label": "DeepSeek V4 Pro"},
    {"key": "ds4_flash", "label": "DeepSeek V4 Flash"},
]


def call_llm(prompt, provider=None):
    # Simple string-in / string-out entry point. Used by callers that don't
    # need vision or multi-turn (e.g. pq_shopping setup, other projects).
    name = provider or DEFAULT_PROVIDER
    if name not in _PROVIDERS:
        raise ValueError("unknown provider: " + repr(name) + " (known: " + str(sorted(_PROVIDERS)) + ")")
    text, _usage = _PROVIDERS[name](prompt=prompt)
    return text


def call_llm_messages(messages, provider=None):
    # Structured entry point for multi-turn and multimodal. Caller passes a
    # list of OpenAI-format messages (image content items use image_url with
    # data URIs); returns (text, usage_dict) so callers can log token counts.
    # This is what the judge uses.
    name = provider or DEFAULT_PROVIDER
    if name not in _PROVIDERS:
        raise ValueError("unknown provider: " + repr(name) + " (known: " + str(sorted(_PROVIDERS)) + ")")
    return _PROVIDERS[name](messages=messages)
