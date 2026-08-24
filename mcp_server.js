#!/usr/bin/env node

// MCP server for HOME use: connects to an existing headed Chrome instance
// via CDP at localhost:9222. Requires playwright 1.57+ for ariaSnapshot.
//
// Dependencies: @modelcontextprotocol/sdk, playwright, zod
// (turndown is no longer needed - ariaSnapshot replaces HTML-to-markdown)
//
// v2.1: added web_search (structured top-10) and fetch_url (raw response body,
// no aria conversion). Both run in a
// throwaway tab so they never clobber the shared page the model may be
// mid-reading with playwright_extract_content. One shared CDP connection is
// kept alive for the whole process: repeatedly connecting/disconnecting
// Playwright clients wedged the Chrome debugger in soak tests, while tab
// create/close on a single connection stayed stable.
//
// v2.2: web_search default is DuckDuckGo html; Brave is an explicit override
// only, with its result-card selector fixed for the current layout.
// v2.3: web_search accepts the DSH 'queries' array (1-4 strings) and fans it
// out into concurrent searches, one result set per query. ensureBrowser is
// promise-guarded so a concurrent fan-out shares one CDP connection.

const { McpServer } = require('@modelcontextprotocol/sdk/server/mcp.js');
const { StdioServerTransport } = require('@modelcontextprotocol/sdk/server/stdio.js');
const { chromium } = require('playwright');
const { z } = require('zod');

let browser = null;
let context = null;
let page = null;
// in-flight connect promise: a concurrent first use (a web_search fan-out)
// must share one CDP connection instead of racing to open several
let browserPromise = null;

const server = new McpServer({
  name: 'playwright-chrome',
  version: '2.3.0',
});

function ensureBrowser() {
  if (!browserPromise) {
    browserPromise = (async () => {
      try {
        browser = await chromium.connectOverCDP('http://localhost:9222');
      } catch (err) {
        browserPromise = null;
        throw new Error(
          'Could not connect to Chrome at localhost:9222. '
          + 'Is Chrome running with --remote-debugging-port=9222? '
          + '(' + err.message + ')'
        );
      }
      context = browser.contexts()[0];
      const pages = context.pages();
      page = pages.length > 0 ? pages[0] : await context.newPage();
      console.error('Connected to existing Chrome instance');
      return page;
    })();
  }
  return browserPromise;
}

// shared helper: extract page or element content via ariaSnapshot with
// innerText fallback. keeps tool handlers thin.
async function extractContent(target, label) {
  try {
    const snapshot = await target.ariaSnapshot({ mode: 'ai' });
    return snapshot || 'No content found';
  } catch (err) {
    console.error('ariaSnapshot failed (' + label + '), falling back to innerText: ' + err.message);
    // target is either a Page or a Locator; Page has .evaluate, Locator does not
    if (typeof target.evaluate === 'function') {
      const text = await target.evaluate(() => document.body.innerText);
      return text || 'No content found';
    }
    // locator fallback
    const text = await target.innerText().catch(() => 'No content found');
    return text;
  }
}

// shared helper: navigate with domcontentloaded then load fallback.
// returns the page; logs failures to stderr.
async function navigatePage(p, url) {
  let navigationError = null;
  try {
    await p.goto(url, { waitUntil: 'domcontentloaded', timeout: 15000 });
  } catch (err) {
    console.error('domcontentloaded timed out for ' + url + ', falling back to load');
    navigationError = err;
    try {
      await p.goto(url, { waitUntil: 'load', timeout: 30000 });
      navigationError = null;
    } catch (err2) {
      navigationError = err2;
    }
  }
  if (navigationError) {
    console.error('Navigation failed for ' + url + ': ' + navigationError.message);
  }
  // brief settle for JS-rendered content
  await p.waitForTimeout(2000);
  return p;
}

// run fn on a fresh throwaway tab. search/fetch get their own tab so they do
// not navigate the shared page away from whatever the model was reading.
async function withTab(fn) {
  await ensureBrowser();
  const tab = await context.newPage();
  try {
    return await fn(tab);
  } finally {
    await tab.close().catch(() => {});
  }
}

server.tool(
  'playwright_navigate',
  'Navigate to a URL in the existing Chrome browser and return page content as an accessibility tree snapshot',
  { url: z.string().describe('URL to navigate to') },
  async ({ url }) => {
    const p = await ensureBrowser();
    await navigatePage(p, url);
    const text = await extractContent(p, url);
    return { content: [{ type: 'text', text }] };
  }
);

server.tool(
  'playwright_extract_content',
  'Extract the current page content as an accessibility tree snapshot',
  {
    selector: z.string().optional().describe(
      'Optional CSS selector to scope extraction (e.g. "article", "main"). Defaults to entire body.'
    ),
  },
  async ({ selector }) => {
    const p = await ensureBrowser();
    if (!selector || selector === 'body' || selector === '*') {
      const text = await extractContent(p, 'full page');
      return { content: [{ type: 'text', text }] };
    }
    const element = p.locator(selector);
    await element.waitFor({ state: 'visible', timeout: 5000 }).catch(() => {});
    const text = await extractContent(element, selector);
    return { content: [{ type: 'text', text }] };
  }
);

// in-page extractors for search result cards. DDG html is the default: it is
// blocker-free from this IP and has a stable, lean DOM. Brave is an explicit
// override only (engine=brave); it currently 429s from this IP, and its card
// selector was updated in v2.2 (the #results wrapper is gone).
const BRAVE_EXTRACT = `(() => {
  const out = [];
  for (const c of document.querySelectorAll('div.snippet[data-type="web"]')) {
    const a = c.querySelector('a[href^="http"]');
    if (!a) continue;
    const t = c.querySelector('.title');
    const title = ((t ? t.textContent : a.textContent) || '').replace(/\\s+/g, ' ').trim();
    let snippet = '';
    for (const sel of ['.generic-snippet .content', '.inline-qa-question', '.snippet-content']) {
      const el = c.querySelector(sel);
      if (el && el.textContent.trim()) { snippet = el.textContent.replace(/\\s+/g, ' ').trim(); break; }
    }
    if (title) out.push({ title, url: a.href, snippet: snippet.slice(0, 300) });
  }
  return out;
})()`;

const DDG_EXTRACT = `(() => {
  const out = [];
  for (const r of document.querySelectorAll('.result')) {
    const a = r.querySelector('.result__a');
    if (!a) continue;
    let url = a.href;
    // ddg wraps outbound links in a redirect carrying the target as uddg=
    try { const q = new URL(url).searchParams.get('uddg'); if (q) url = q; } catch (e) {}
    const s = r.querySelector('.result__snippet');
    const snippet = (s ? s.textContent : '').replace(/\\s+/g, ' ').trim();
    out.push({ title: a.textContent.replace(/\\s+/g, ' ').trim(), url, snippet: snippet.slice(0, 300) });
  }
  return out;
})()`;

const ENGINES = {
  ddg: {
    url: (q) => 'https://html.duckduckgo.com/html/?q=' + encodeURIComponent(q),
    wait: '.result',
    extract: DDG_EXTRACT,
  },
  brave: {
    url: (q) => 'https://search.brave.com/search?q=' + encodeURIComponent(q),
    wait: 'div.snippet[data-type="web"]',
    extract: BRAVE_EXTRACT,
  },
};

async function searchOn(engineName, query) {
  const eng = ENGINES[engineName];
  return withTab(async (tab) => {
    await navigatePage(tab, eng.url(query));
    await tab.waitForSelector(eng.wait, { timeout: 6000 }).catch(() => {});
    return await tab.evaluate(eng.extract);
  });
}

server.tool(
  'web_search',
  'Search the web and return a structured top-10 list of {title, url, snippet} per query. DuckDuckGo by default, Brave optional override.',
  {
    query: z.string().optional().describe('Plain text search query (single search).'),
    queries: z.array(z.string()).max(4).optional().describe('DSH form: 1-4 search query strings. Each is searched separately and returned as its own result set.'),
    engine: z.string().optional().describe('Optional engine override: "ddg" (default) or "brave". Omit unless you have a reason.'),
  },
  async ({ query, queries, engine }) => {
    // ddg is the default; brave is an explicit override only and is not in the
    // fallback chain (it 429s from this IP, and a wasted Brave load would
    // otherwise precede every DDG search).
    const name = engine && ENGINES[engine] ? engine : 'ddg';
    const list = [];
    if (Array.isArray(queries)) {
      for (const q of queries.slice(0, 4)) list.push(String(q));
    }
    if (typeof query === 'string' && query.trim()) list.push(query);
    if (!list.length) {
      return {
        content: [{
          type: 'text',
          text: 'web_search requires a plain text query or a queries array of 1-4 strings.',
        }],
      };
    }
    // fan out: one search per query, each in its own throwaway tab. run them
    // concurrently so 4 searches cost about one search of wall clock. one bad
    // query must not lose the others, so each is isolated.
    const settled = await Promise.all(list.map(async (q) => {
      try {
        return { q, results: await searchOn(name, q) };
      } catch (err) {
        console.error('search on ' + name + ' for "' + q + '" failed: ' + err.message);
        return { q, results: null };
      }
    }));
    const blocks = [];
    for (const { q, results } of settled) {
      if (results === null) {
        blocks.push('Search results for "' + q + '" (engine: ' + name + '):\n(search failed on this engine; try rephrasing or a different query.)');
      } else if (!results.length) {
        blocks.push('No results found for "' + q + '" on ' + name
          + ' (possible bot wall). Try rephrasing, or playwright_navigate to a known URL directly.');
      } else {
        const top = results.slice(0, 10);
        const lines = top.map((r, i) =>
          (i + 1) + '. ' + r.title + '\n   ' + r.url + (r.snippet ? '\n   ' + r.snippet : '')
        );
        const header = 'Search results for "' + q + '" (engine: ' + name + ', showing ' + top.length + ' of ' + results.length + '):';
        blocks.push(header + '\n' + lines.join('\n'));
      }
    }
    return { content: [{ type: 'text', text: blocks.join('\n\n') }] };
  }
);

server.tool(
  'fetch_url',
  'Fetch a URL and return the raw response body (JSON, HTML, text) with no accessibility-tree conversion. Runs in a throwaway tab.',
  { url: z.string().describe('URL to fetch') },
  async ({ url }) => {
    const body = await withTab(async (tab) => {
      let resp = null;
      try {
        resp = await tab.goto(url, { waitUntil: 'domcontentloaded', timeout: 30000 });
      } catch (err) {
        // goto throws on downloads and some redirects; the page may still hold content
        console.error('fetch_url goto error for ' + url + ': ' + err.message);
      }
      if (resp) {
        try {
          return await resp.text();
        } catch (err) {
          console.error('fetch_url response.text() failed for ' + url + ': ' + err.message);
        }
      }
      // fallback: whatever ended up rendered (Chrome wraps JSON in a <pre>)
      const text = await tab.evaluate(() => {
        const pre = document.querySelector('pre');
        return pre ? pre.textContent : document.body ? document.body.innerText : '';
      }).catch(() => '');
      return text || 'Error: no response body for ' + url;
    });
    return { content: [{ type: 'text', text: body }] };
  }
);

async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error('Playwright MCP server started, will connect to Chrome at localhost:9222 on first use');
}

main().catch((error) => {
  console.error('Fatal error:', error);
  process.exit(1);
});
