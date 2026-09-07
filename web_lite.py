#!/usr/bin/env python3

# web_lite.py: the PQ_PLAYWRIGHT=2 web backend. Plain HTTP with static HTML
# extraction, no Node, no browser, no MCP. Built for low-memory hosts
# (Raspberry Pi Zero class); see previous_sessions/001_report.md for the plan
# this implements. Public entry points return model-facing text and never
# raise WebLiteError, mirroring the MCP backend where a tool error is an
# ordinary result the model can route around.
#
# Design constraints from the plan:
# - Never import agent.py (no provider headers, no credentials, no coupling);
#   the token estimator is a local copy.
# - Bodies are streamed with byte caps BEFORE buffering: a context limit is
#   not a download limit, and on a 512 MB board the fetch itself must be
#   bounded. Raw fetch caps at 4 MiB, parsed pages at 1 MiB.
# - Requests' connect/read timeouts are not wall-clock limits (per-address
#   attempts multiply elapsed time), so every call runs under a main-thread
#   SIGALRM watchdog in addition to socket timeouts. A watchdog is not process
#   isolation: a stuck native call could outlive it, an accepted limitation of
#   the in-process design.
# - Search: DuckDuckGo HTML by default, at most one Brave fallback, sequential
#   with a stagger; engines get a browser UA (an explicit, tested compatibility
#   override - DDG serves 202 challenge pages to library-default UAs), all
#   other destinations get the honest pq-agent UA.
# - Spill files live in /tmp/pq_web_lite (separate from mode 1's /tmp/pq_fetch)
#   under a 12 MiB per-run budget. /tmp is tmpfs under run_agent.sh: spill
#   bounds RAM, it is not disk storage. Handles already handed to the model are
#   never deleted; budget exhaustion is reported, not failed silently.

import os
import re
import sys
import time
import signal
import threading
import urllib.parse
from dataclasses import dataclass, field

import requests
from bs4 import BeautifulSoup, NavigableString, Tag
from soupsieve import SelectorSyntaxError

# Same identity agent.py sends to model providers: RFC 9110 product token with
# repo and contact. Wikimedia etiquette requires exactly this shape, and the
# python-requests default UA is blocked outright (phabricator T400119).
HONEST_UA = "pq-agent/1.0 (+https://github.com/artiedins/pq_agent; contact artiedins)"
# Search engines are the one destination class where the honest UA fails
# outright (DDG 202 challenges to library UAs, tested 2026-09-07 from this
# network). Browser UA here only; never send it to content sites.
BROWSER_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) " "Chrome/140.0.0.0 Safari/537.36"


@dataclass
class WebLiteSettings:
    # Proposed starting limits from the plan, not measured guarantees. One
    # dataclass so the whole policy is readable at a glance and adjustable per
    # host after real-device measurement.
    connect_timeout: float = 5.0
    read_timeout: float = 10.0
    fetch_watchdog: float = 45.0
    search_watchdog: float = 90.0
    max_html_bytes: int = 1 * 1024 * 1024
    max_fetch_bytes: int = 4 * 1024 * 1024
    max_redirects: int = 5
    search_stagger: float = 0.4
    result_token_cap: int = 5000
    spill_dir: str = "/tmp/pq_web_lite"
    spill_budget_bytes: int = 12 * 1024 * 1024
    retry_after_max_sleep: float = 5.0
    extra: dict = field(default_factory=dict)


class WebLiteError(Exception):
    # an ordinary tool-level failure (bad URL, HTTP error, bad selector);
    # public entry points catch these and render them as "Error: ..." text
    pass


_TOKEN_PAT = re.compile(r"(?:'s|'t|'re|'ve|'m|'ll|'d)| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+", re.S)


def est_tokens(text):
    # condensed copy of agent.py's estimator (web_lite must not import agent.py)
    if not text:
        return 0
    n = 0
    for match in _TOKEN_PAT.finditer(text):
        chunk = match.group()
        if chunk[0].isspace() and not chunk.strip():
            n += 1 if len(chunk) <= 2 else max(1, len(chunk) // 14)
            continue
        body = chunk[1:] if chunk[0] == " " else chunk
        if not body:
            n += 1
        elif body[0].isdigit():
            n += max(1, -(-len(body) // 3))
        elif body[0].isalpha():
            n += 1 if len(body) <= 6 else 1 + (len(body) - 6) // 5
        else:
            n += max(1, -(-len(body) // 3))
    n += (len(text.encode("utf-8")) - len(text)) // 2
    return n


def _chars_for_tokens(text, token_cap):
    est = est_tokens(text)
    if est <= token_cap:
        return None
    return max(1, len(text) * token_cap // est)


def _truncate(text, token_cap, spill_path=None):
    # head+tail like agent.py's truncate_playwright_text, but the guidance names
    # tools that exist in mode 2 (never playwright_*).
    budget = _chars_for_tokens(text, token_cap)
    if budget is None:
        return text
    head_n = (budget * 2) // 3
    tail_n = budget - head_n
    head = text[:head_n]
    tail = text[-tail_n:]
    total = est_tokens(text)
    elided = total - est_tokens(head) - est_tokens(tail)
    if spill_path:
        guidance = (
            "The full body is saved to " + spill_path + " (outside the workspace). "
            "Treat this path as a handle to the full value: query it with read or bash (grep/jq/sed); "
            "do not re-fetch the URL expecting the middle in context."
        )
    else:
        guidance = "Re-fetching the same URL returns this same truncated view. Scope the page with " "web_read plus a CSS selector, or use a different source."
    return (
        head
        + "\n[str(len_tokens~"
        + str(total)
        + ", head="
        + str(est_tokens(head))
        + ", tail="
        + str(est_tokens(tail))
        + "): "
        + str(elided)
        + " tokens elided from the middle of this page. "
        + guidance
        + "]\n"
        + tail
    )


def _log(msg):
    # agent.py tee's stdout to the run log; prefix matches its [mcp] style
    print(time.strftime("[%H:%M:%S] ") + "  [web_lite] " + msg, flush=True)


# SIGALRM watchdog: requests' timeouts are per-socket, not wall-clock, and a
# stalled DNS/connect chain can outlive them. Stack-aware so a per-query
# watchdog can sit inside a whole-batch watchdog without clobbering the outer
# timer. Only usable from the main thread (the same thread agent.py dispatches
# tools in); anywhere else it degrades to a no-op.
_ALARM_STACK = []


def _alarm_handler(signum, frame):
    raise TimeoutError("web_lite whole-call watchdog expired")


class _Watchdog:
    def __init__(self, seconds):
        self.seconds = seconds
        self._pushed = False

    def __enter__(self):
        try:
            if threading.current_thread().name != "MainThread":
                return self
            if not _ALARM_STACK:
                signal.signal(signal.SIGALRM, _alarm_handler)
            _ALARM_STACK.append(time.monotonic() + self.seconds)
            self._pushed = True
            signal.setitimer(signal.ITIMER_REAL, self.seconds)
        except (ValueError, OSError):
            pass
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self._pushed:
            return False
        self._pushed = False
        try:
            _ALARM_STACK.pop()
            if _ALARM_STACK:
                remaining = _ALARM_STACK[-1] - time.monotonic()
                signal.setitimer(signal.ITIMER_REAL, max(0.01, remaining))
            else:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, signal.SIG_DFL)
        except (ValueError, OSError, IndexError):
            pass
        return False


class _Response:
    # plain container: status, final url, content type, decoded text body,
    # whether the transfer completed inside the byte cap, Retry-After for 429.
    # The requests Response object never escapes _request().
    def __init__(self, status, final_url, content_type, body, complete, retry_after=None, note=""):
        self.status = status
        self.final_url = final_url
        self.content_type = content_type
        self.body = body
        self.complete = complete
        self.retry_after = retry_after
        self.note = note


class WebLite:
    def __init__(self, settings=None):
        self.s = settings or WebLiteSettings()
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": HONEST_UA,
                # identity everywhere: bounded decompression is not implemented,
                # so an unexpected compressed body is rejected explicitly rather
                # than expanded unbounded and measured afterwards
                "Accept-Encoding": "identity",
                "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,text/plain;q=0.8,*/*;q=0.5",
            }
        )
        self.session.max_redirects = self.s.max_redirects
        # one connection per host kept alive, tiny pool: a Pi Zero has one core
        # and a USB-attached NIC; parallel sockets buy nothing and cost RAM
        adapter = requests.adapters.HTTPAdapter(pool_connections=2, pool_maxsize=2, max_retries=0)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        self._spill_seq = 0
        self._spill_bytes = 0
        self._last_engine_hit = {}

    def close(self):
        try:
            self.session.close()
        except Exception:
            pass

    # ---------- low-level HTTP ----------

    def _request(self, url, max_bytes, headers=None, watchdog=None):
        # manual redirect walk (not allow_redirects=True) so every hop's scheme
        # is validated and the hop count is enforced exactly
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise WebLiteError("only http:// and https:// URLs are supported (got '" + url[:120] + "')")
        current = url
        with _Watchdog(watchdog if watchdog is not None else self.s.fetch_watchdog):
            for _hop in range(self.s.max_redirects + 1):
                resp = None
                try:
                    resp = self.session.get(
                        current,
                        stream=True,
                        allow_redirects=False,
                        timeout=(self.s.connect_timeout, self.s.read_timeout),
                        headers=headers,
                    )
                    if resp.status_code in (301, 302, 303, 307, 308):
                        loc = (resp.headers.get("Location") or "").strip()
                        if not loc:
                            raise WebLiteError("redirect without a Location header from " + current[:120])
                        nxt = urllib.parse.urljoin(current, loc)
                        nxt_parsed = urllib.parse.urlsplit(nxt)
                        if nxt_parsed.scheme not in ("http", "https") or not nxt_parsed.netloc:
                            raise WebLiteError("redirect to a non-http(s) target: " + nxt[:120])
                        current = nxt
                        continue
                    enc = (resp.headers.get("Content-Encoding") or "").strip().lower()
                    if enc and enc != "identity":
                        raise WebLiteError("unexpected Content-Encoding '" + enc + "' (identity was requested); " "this backend does not decompress")
                    body, complete = self._read_capped(resp, max_bytes)
                    content_type = resp.headers.get("Content-Type", "") or "unknown"
                    # get_encoding_from_headers is what requests itself uses to
                    # set resp.encoding; apparent_encoding is avoided because it
                    # buffers the whole body first
                    charset = requests.utils.get_encoding_from_headers(resp.headers)
                    if charset is None and body:
                        # meta charset sniff in the first 2 KiB
                        m = re.search(rb'charset\s*=\s*["\']?\s*([\w.:-]+)', body[:2048])
                        charset = m.group(1).decode("ascii", "replace") if m else "utf-8"
                    try:
                        text = body.decode(charset or "utf-8", errors="replace")
                    except (LookupError, UnicodeError):
                        text = body.decode("utf-8", errors="replace")
                    note = ""
                    if not complete:
                        note = "transfer stopped at the " + str(max_bytes) + "-byte cap; this body is PARTIAL and the true size is unknown"
                    return _Response(
                        resp.status_code,
                        current,
                        content_type,
                        text,
                        complete,
                        retry_after=resp.headers.get("Retry-After"),
                        note=note,
                    )
                finally:
                    if resp is not None:
                        resp.close()
            raise WebLiteError("too many redirects (>" + str(self.s.max_redirects) + " hops): " + url[:120])

    def _read_capped(self, resp, max_bytes):
        # stream and cap before buffering; enforced even when Content-Length is
        # missing or false (chunked transfer). Stops as soon as the cap is
        # exceeded, so a 4 MiB cap never materializes a 400 MiB body.
        chunks = []
        total = 0
        over = False
        for chunk in resp.iter_content(chunk_size=16384):
            total += len(chunk)
            if total > max_bytes:
                keep = len(chunk) - (total - max_bytes)
                if keep:
                    chunks.append(chunk[:keep])
                over = True
                break
            chunks.append(chunk)
        body = b"".join(chunks)
        if over:
            return body, False
        cl = resp.headers.get("Content-Length")
        if cl is not None:
            # a false Content-Length with a stream that ended early is a
            # partial body, not a complete one
            try:
                if int(cl) > len(body):
                    return body, False
            except ValueError:
                pass
        return body, True

    # ---------- spill ----------

    def _spill(self, text, url):
        # returns (path_or_None, error_note). Files already handed to the model
        # are never deleted; budget exhaustion is reported to the caller.
        size = len(text.encode("utf-8"))
        if self._spill_bytes + size > self.s.spill_budget_bytes:
            return None, ("spill storage budget (" + str(self.s.spill_budget_bytes // (1024 * 1024)) + " MiB under " + self.s.spill_dir + ") exhausted; only the truncated view below is available")
        try:
            os.makedirs(self.s.spill_dir, exist_ok=True)
            self._spill_seq += 1
            slug = re.sub(r"[^a-zA-Z0-9]+", "_", url.split("?")[0])[-40:].strip("_") or "page"
            path = os.path.join(self.s.spill_dir, str(self._spill_seq) + "_" + slug + ".txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            self._spill_bytes += size
            return path, ""
        except OSError as e:
            return None, "spill write failed (" + str(e)[:120] + ")"

    # ---------- public tools ----------

    def search(self, queries, engine=None):
        # Sequential batch, one result block per query, mirroring the MCP
        # backend's output format so models see the same shape in both modes.
        # A failed query never loses the others; the engine actually used (and
        # any fallback reason) is named in the block header.
        name = engine if engine in ("ddg", "brave") else "ddg"
        blocks = []
        with _Watchdog(self.s.search_watchdog):
            for i, q in enumerate(queries[:4]):
                if i > 0:
                    # same-instant bursts raise the DDG 202 rate (mcp_server.js
                    # v2.4 lesson); the stagger is politeness, not latency tuning
                    time.sleep(self.s.search_stagger)
                with _Watchdog(self.s.fetch_watchdog):
                    blocks.append(self._search_with_fallback(name, str(q)))
        return "\n\n".join(blocks)

    def _search_with_fallback(self, name, q):
        primary = self._search_one(name, q)
        # results is not None => either real results or a recognizable
        # no-results page: final either way, no fallback. An explicit engine
        # override is respected as-is (only ddg ever falls back).
        if name != "ddg" or primary[0] is not None:
            return self._format_block(q, name, primary)
        fallback = self._search_one("brave", q)
        if fallback[0]:
            return self._format_block(q, "brave", fallback, why=primary[1] or "empty ddg answer")
        if fallback[0] is not None:
            # brave parsed a genuine no-results page
            return self._format_block(q, "brave", fallback, why="after ddg: " + (primary[1] or "no results"))
        # ddg failed AND the brave retry failed: one honest failure block
        combined = (
            None,
            (primary[1] or "failed") + "; brave fallback also failed: " + (fallback[1] or "failed"),
            True,
        )
        return self._format_block(q, name, combined)

    def _search_one(self, name, q):
        # returns (results_or_None, status_note, hard_failure)
        eng = ENGINES[name]
        try:
            # per-engine spacing beyond the batch stagger, so repeat calls to
            # one engine do not burst
            last = self._last_engine_hit.get(name, 0.0)
            wait = self.s.search_stagger - (time.monotonic() - last)
            if wait > 0:
                time.sleep(wait)
            self._last_engine_hit[name] = time.monotonic()
            resp = self._request(eng["url"](q), self.s.max_html_bytes, headers={"User-Agent": BROWSER_UA})
        except WebLiteError as e:
            _log("search " + name + " for '" + q[:50] + "' failed: " + str(e)[:120])
            return None, str(e), True
        except (TimeoutError, requests.exceptions.RequestException, OSError) as e:
            _log("search " + name + " for '" + q[:50] + "' failed: " + type(e).__name__ + ": " + str(e)[:120])
            return None, type(e).__name__ + ": " + str(e)[:150], True
        if resp.status == 429:
            return self._rate_limited(name, q, resp.retry_after)
        if resp.status == 202:
            # DDG's intermittent bot-challenge anomaly page, seen through real
            # Chrome too (mcp_server.js v2.4 note); not a result page
            return None, "bot-challenge page (HTTP 202)", True
        if resp.status >= 400:
            return None, "HTTP " + str(resp.status), True
        soup = BeautifulSoup(resp.body, "html.parser")
        results = eng["parse"](soup)
        results = [r for r in results if r["url"].startswith(("http://", "https://"))]
        seen = set()
        deduped = []
        for r in results:
            if r["url"] in seen:
                continue
            seen.add(r["url"])
            deduped.append(r)
        if deduped:
            return deduped, "", False
        # zero parsed cards is not proof that no results exist: classify the
        # page honestly (recognizable no-results vs challenge vs unknown)
        low = re.sub(r"\s+", " ", resp.body.lower())
        if "no results" in low or "did not match" in low:
            return [], "recognizable no-results page", False
        if any(m in low for m in ("just a moment", "captcha", "verify you are human", "anomaly")):
            return None, "bot-challenge page (HTTP " + str(resp.status) + ")", True
        return None, "zero result cards parsed; layout changed or JS required", True

    def _rate_limited(self, name, q, retry_after):
        # honor Retry-After once if it is short; a long cooldown is reported,
        # never slept through, and never evaded by hopping to another provider
        ra_s = None
        if retry_after:
            try:
                ra_s = float(retry_after)
            except (TypeError, ValueError):
                pass
        if ra_s is not None and 0 < ra_s <= self.s.retry_after_max_sleep:
            time.sleep(ra_s)
            result = self._search_one(name, q)
            if result[0] is not None or not result[2]:
                return result
        msg = "HTTP 429 rate limit" + (" (Retry-After: " + str(int(ra_s)) + "s)" if ra_s else "")
        return None, msg, True

    def _format_block(self, q, engine_used, outcome, why=""):
        results, note, failed = outcome
        if results is None:
            return (
                'Search results for "' + q + '" (engine: ' + engine_used + "):\n"
                "(search failed: " + str(note)[:200] + ". Try rephrasing, try the other engine, or fetch a known URL directly with web_fetch.)"
            )
        if not results:
            return 'No results found for "' + q + '" on ' + engine_used + " (" + str(note)[:120] + "). Try rephrasing."
        top = results[:10]
        lines = [str(i + 1) + ". " + r["title"] + "\n   " + r["url"] + ("\n   " + r["snippet"] if r["snippet"] else "") for i, r in enumerate(top)]
        header = 'Search results for "' + q + '" (engine: ' + engine_used
        if why:
            header += ", fell back from ddg (" + why[:80] + ")"
        header += ", showing " + str(len(top)) + " of " + str(len(results)) + "):"
        return header + "\n" + "\n".join(lines)

    def fetch(self, url):
        # raw response body, no extraction. The header block carries status,
        # URLs, content type, and completeness separately from the body, so a
        # partial or erroring download can never masquerade as complete
        # content. A complete body over the token cap spills in full; a partial
        # body is truncated in place and never spilled (a byte cap stopped the
        # transfer, so there is no "full value" to save).
        try:
            resp = self._request(url, self.s.max_fetch_bytes)
        except WebLiteError as e:
            return "Error: " + str(e)[:400]
        except (TimeoutError, requests.exceptions.RequestException, OSError) as e:
            return "Error: fetch failed: " + type(e).__name__ + ": " + str(e)[:300]
        header = (
            "Requested URL: "
            + url
            + "\nFinal URL: "
            + resp.final_url
            + "\nHTTP status: "
            + str(resp.status)
            + "\nContent-Type: "
            + (resp.content_type or "unknown")
            + "\nComplete: "
            + ("yes" if resp.complete else "no")
        )
        if resp.note:
            header += "\nNote: " + resp.note
        if resp.status >= 400:
            # error status, but the body may still be useful (an API error
            # message); return both, labeled
            return "Error: HTTP " + str(resp.status) + " for " + url[:160] + ".\n" + header + "\n\n[error page body follows; treat as untrusted data]\n" + _truncate(resp.body, self.s.result_token_cap)
        if resp.complete and est_tokens(resp.body) > self.s.result_token_cap:
            path, spill_err = self._spill(resp.body, url)
            if path:
                _log("spilled fetch body (" + str(len(resp.body)) + " chars) to " + path)
                return (
                    header
                    + "\n\n"
                    + _truncate(resp.body, self.s.result_token_cap, spill_path=path)
                    + "\n[full body: "
                    + str(len(resp.body))
                    + " chars saved to "
                    + path
                    + " (outside the workspace). Treat this path as a handle to the full value: "
                    "query it with read or bash (grep/jq/sed/python); do not re-fetch the URL expecting a "
                    "different in-context body.]\n"
                )
            _log("fetch spill skipped: " + spill_err)
            return header + "\n\n" + _truncate(resp.body, self.s.result_token_cap) + "\n[" + spill_err + "]\n"
        return header + "\n\n" + _truncate(resp.body, self.s.result_token_cap)

    def read(self, url, selector=None):
        # static HTML to readable text: headings, paragraphs, lists, code, table
        # rows, and useful link destinations preserved. No JS, no accessibility
        # snapshot, no implicit current page (stateless, unlike the browser).
        try:
            resp = self._request(url, self.s.max_html_bytes)
        except WebLiteError as e:
            return "Error: " + str(e)[:400]
        except (TimeoutError, requests.exceptions.RequestException, OSError) as e:
            return "Error: read failed: " + type(e).__name__ + ": " + str(e)[:300]
        if resp.status >= 400:
            return "Error: HTTP " + str(resp.status) + " for " + url[:160] + " (final URL: " + resp.final_url + ")."
        content_type = (resp.content_type or "").split(";")[0].strip().lower()
        if content_type and content_type not in ("text/html", "application/xhtml+xml"):
            return "Error: web_read extracts HTML pages; this URL served '" + content_type + "'. Use web_fetch for the raw body."
        try:
            soup = BeautifulSoup(resp.body, "html.parser")
            title = soup.title.get_text(strip=True) if soup.title else ""
            if selector:
                matches = soup.select(selector)
                if not matches:
                    return "No elements match selector '" + selector + "' on " + url[:160] + " (final URL: " + resp.final_url + ")."
                if len(matches) == 1:
                    scope = "selector '" + selector + "' (1 match)"
                    body_text = _render(matches[0])
                else:
                    shown = min(len(matches), 5)
                    scope = "selector '" + selector + "' (" + str(len(matches)) + " matches, first " + str(shown) + " shown)"
                    parts = []
                    for i in range(shown):
                        parts.append("=== " + selector + " match " + str(i + 1) + " of " + str(len(matches)) + " ===\n" + _render(matches[i]))
                    body_text = "\n\n".join(parts)
            else:
                # prefer main, then article, then a cleaned body; label the
                # chosen scope so the model knows what it got
                scope = None
                for candidate in ("main", "article"):
                    found = soup.select(candidate)
                    if found:
                        scope = candidate
                        body_text = _render(found[0])
                        break
                if scope is None:
                    scope = "body (no main/article element found)"
                    # fragment responses can lack <body>; render the whole soup then
                    body_text = _render(soup.body if soup.body else soup)
        except (SelectorSyntaxError, ValueError) as e:
            return "Error: invalid CSS selector '" + selector + "': " + str(e)[:200]
        header = "Requested URL: " + url + "\nFinal URL: " + resp.final_url + "\nHTTP status: " + str(resp.status) + "\nScope: " + scope
        if title:
            header += "\nTitle: " + title
        if not resp.complete:
            header += "\nNote: " + resp.note
        if not body_text or not body_text.strip():
            low = resp.body.lower()
            if "enable javascript" in low or "noscript" in low:
                header += "\nNote: page appears to require JavaScript; static extraction produced no text. " "Try the site's API with web_fetch."
            return header + "\n\n(no extractable text)"
        return header + "\n\n" + _truncate(body_text, self.s.result_token_cap)


# ---------- HTML rendering ----------

# excluded from extraction entirely
_SKIP_TAGS = {"script", "style", "noscript", "template", "svg", "iframe", "form", "button", "select"}
_BLOCK_TAGS = {
    "address",
    "article",
    "aside",
    "blockquote",
    "dd",
    "div",
    "dl",
    "dt",
    "fieldset",
    "figcaption",
    "figure",
    "footer",
    "header",
    "hr",
    "legend",
    "main",
    "nav",
    "ol",
    "p",
    "section",
    "table",
    "ul",
}
_HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}


def _collapse_ws(s):
    # collapse whitespace to single spaces but preserve word boundaries across
    # inline elements (a bare split would glue "foo<b>bar</b>" into "foobar")
    if not s:
        return ""
    lead = s[:1].isspace()
    trail = s[-1:].isspace()
    mid = " ".join(s.split())
    return (" " if lead and mid else "") + mid + (" " if trail and mid else "")


def _inline(node):
    # inline text of a node's descendants; block children inside are flattened.
    # Word boundaries survive two markup shapes that naive joining glues:
    # whitespace-only text nodes between inline tags (<kbd>get</kbd> <kbd>install</kbd>
    # carries its only space there) and block children next to other content
    # (<td><p>a</p><p>b</p></td>). Found live on the bs4 docs, 2026-09-07.
    parts = []
    for child in node.children:
        if isinstance(child, NavigableString):
            s = str(child)
            if s.strip():
                parts.append(_collapse_ws(s))
            elif s:
                parts.append(" ")
            continue
        if not isinstance(child, Tag):
            continue
        if child.name in _SKIP_TAGS:
            continue
        if child.name == "br":
            parts.append("\n")
            continue
        inner = _inline(child)
        if child.name == "a":
            href = (child.get("href") or "").strip()
            if href.startswith(("http://", "https://")):
                parts.append((inner + " (" + href + ")") if inner.strip() else href)
                continue
        if child.name in _BLOCK_TAGS and inner.strip():
            parts.append(" " + inner.strip() + " ")
            continue
        parts.append(inner)
    return re.sub(r"[ \t]{2,}", " ", "".join(parts))


def _render(node):
    # block-structured text: headings as markdown, li as bullets, pre fenced,
    # table rows as pipe rows, link destinations preserved inline
    if node is None:
        return ""
    out = []

    def walk(el):
        for child in el.children:
            if isinstance(child, NavigableString):
                t = _collapse_ws(str(child))
                if t.strip():
                    out.append(t)
                continue
            if not isinstance(child, Tag):
                continue
            name = child.name
            if name in _SKIP_TAGS:
                continue
            if name in _HEADINGS:
                out.append("\n" + "#" * int(name[1]) + " " + _inline(child).strip() + "\n")
            elif name == "p":
                out.append("\n" + _inline(child).strip() + "\n")
            elif name == "li":
                out.append("\n- " + _inline(child).strip())
            elif name == "pre":
                out.append("\n```\n" + child.get_text().rstrip() + "\n```\n")
            elif name == "table":
                out.append("\n")
                for tr in child.find_all("tr"):
                    cells = [_inline(c).strip() for c in tr.find_all(["td", "th"])]
                    if any(cells):
                        out.append("| " + " | ".join(cells) + " |\n")
            elif name == "hr":
                out.append("\n---\n")
            elif name in _BLOCK_TAGS or name in ("html", "body"):
                out.append("\n")
                walk(child)
            else:
                # inline or unknown tag: recurse structurally so unknown
                # semantic wrappers still yield block children
                walk(child)

    walk(node)
    text = "".join(out)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------- search engine adapters ----------
# Mirrors mcp_server.js's ENGINES so both modes speak the same result format.
# DDG html is the default (blocker-free with a browser UA, stable lean DOM);
# Brave is the explicit override and the single automatic fallback. No third
# parser: Bing looked lenient in one probe, but one query is not evidence for
# maintaining another fragile scraper (001 report, section 5).


def _parse_ddg(soup):
    out = []
    for r in soup.select(".result"):
        a = r.select_one(".result__a")
        if not a or not a.get("href"):
            continue
        url = a["href"]
        # ddg wraps outbound links in a redirect carrying the target as uddg=
        try:
            uddg = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query).get("uddg")
            if uddg:
                url = uddg[0]
        except ValueError:
            pass
        # ad cards link through duckduckgo.com/y.js with no uddg param (paid
        # placements, not results; mcp_server.js v2.4.1 lesson)
        if url.startswith("https://duckduckgo.com/y.js"):
            continue
        s = r.select_one(".result__snippet")
        snippet = re.sub(r"\s+", " ", s.get_text() if s else "").strip()
        title = re.sub(r"\s+", " ", a.get_text()).strip()
        if title:
            out.append({"title": title, "url": url, "snippet": snippet[:300]})
    return out


def _parse_brave(soup):
    out = []
    for c in soup.select('div.snippet[data-type="web"]'):
        a = c.select_one('a[href^="http"]')
        if not a:
            continue
        t = c.select_one(".title")
        title = re.sub(r"\s+", " ", (t.get_text() if t else a.get_text()) or "").strip()
        snippet = ""
        for sel in (".generic-snippet .content", ".inline-qa-question", ".snippet-content"):
            el = c.select_one(sel)
            if el and el.get_text().strip():
                snippet = re.sub(r"\s+", " ", el.get_text()).strip()
                break
        if title:
            out.append({"title": title, "url": a["href"], "snippet": snippet[:300]})
    return out


ENGINES = {
    "ddg": {
        "url": lambda q: "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote_plus(q),
        "parse": _parse_ddg,
    },
    "brave": {
        "url": lambda q: "https://search.brave.com/search?q=" + urllib.parse.quote_plus(q),
        "parse": _parse_brave,
    },
}


# ---------- module-level facade for agent.py ----------

_LITE = None


def init(settings=None):
    global _LITE
    if _LITE is None:
        _LITE = WebLite(settings)
    return _LITE


def close():
    global _LITE
    if _LITE is not None:
        _LITE.close()
        _LITE = None


def _require():
    if _LITE is None:
        raise WebLiteError("web_lite is not initialized")
    return _LITE


def search(queries, engine=None):
    return _require().search(queries, engine)


def fetch(url):
    return _require().fetch(url)


def read(url, selector=None):
    return _require().read(url, selector)


if __name__ == "__main__":
    # tiny manual smoke: python3 web_lite.py <url> [url ...] reads each URL
    init()
    try:
        for target in sys.argv[1:]:
            print("=== read " + target + " ===")
            print(read(target)[:2000])
    finally:
        close()
