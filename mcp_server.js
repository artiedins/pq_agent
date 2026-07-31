#!/usr/bin/env node

// MCP server for HOME use: connects to an existing headed Chrome instance
// via CDP at localhost:9222. Requires playwright 1.57+ for ariaSnapshot.
//
// Dependencies: @modelcontextprotocol/sdk, playwright, zod
// (turndown is no longer needed - ariaSnapshot replaces HTML-to-markdown)
//
// v2.1: added web_search (structured top-10, Brave default with DDG fallback)
// and fetch_url (raw response body, no aria conversion). Both run in a
// throwaway tab so they never clobber the shared page the model may be
// mid-reading with playwright_extract_content. One shared CDP connection is
// kept alive for the whole process: repeatedly connecting/disconnecting
// Playwright clients wedged the Chrome debugger in soak tests, while tab
// create/close on a single connection stayed stable.

const { McpServer } = require('@modelcontextprotocol/sdk/server/mcp.js');
const { StdioServerTransport } = require('@modelcontextprotocol/sdk/server/stdio.js');
const { chromium } = require('playwright');
const { z } = require('zod');

let browser = null;
let context = null;
let page = null;

const server = new McpServer({
  name: 'playwright-chrome',
  version: '2.1.0',
});

async function ensureBrowser() {
  if (!browser) {
    try {
      browser = await chromium.connectOverCDP('http://localhost:9222');
    } catch (err) {
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
  }
  return page;
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

// in-page extractors for search result cards. measured in the round-2 engine
// eval: Brave surfaced practitioner threads (Reddit/HN) where DDG's open-query
// top hits were affiliate listicles, and the old ampersand DDG URL silently
// broke site: queries, so Brave is the default and DDG proper (html/?q=) is
// the fallback.
const BRAVE_EXTRACT = `(() => {
  const out = [];
  for (const c of document.querySelectorAll('#results div.snippet[data-type="web"]')) {
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
  brave: {
    url: (q) => 'https://search.brave.com/search?q=' + encodeURIComponent(q),
    wait: '#results div.snippet[data-type="web"]',
    extract: BRAVE_EXTRACT,
  },
  ddg: {
    url: (q) => 'https://html.duckduckgo.com/html/?q=' + encodeURIComponent(q),
    wait: '.result',
    extract: DDG_EXTRACT,
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
  'Search the web and return a structured top-10 list of {title, url, snippet}. Brave by default, DuckDuckGo fallback.',
  {
    query: z.string().describe('Plain text search query'),
    engine: z.string().optional().describe('Optional engine override: "brave" or "ddg". Default tries brave then ddg.'),
  },
  async ({ query, engine }) => {
    const preferred = engine && ENGINES[engine] ? [engine] : [];
    const order = preferred.concat(Object.keys(ENGINES).filter((e) => e !== preferred[0]));
    let used = order[0];
    let results = [];
    for (const name of order) {
      try {
        results = await searchOn(name, query);
        used = name;
      } catch (err) {
        console.error('search on ' + name + ' failed: ' + err.message);
        results = [];
      }
      // zero results usually means a bot wall or a dead layout, not an
      // honest empty SERP, so fall through to the next engine
      if (results.length) break;
    }
    if (!results.length) {
      return {
        content: [{
          type: 'text',
          text: 'No results found for "' + query + '" on ' + order.join(' or ')
            + ' (possible bot wall). Try rephrasing, or playwright_navigate to a known URL directly.',
        }],
      };
    }
    const top = results.slice(0, 10);
    const lines = top.map((r, i) =>
      (i + 1) + '. ' + r.title + '\n   ' + r.url + (r.snippet ? '\n   ' + r.snippet : '')
    );
    const header = 'Search results for "' + query + '" (engine: ' + used + ', showing ' + top.length + ' of ' + results.length + '):';
    return { content: [{ type: 'text', text: header + '\n' + lines.join('\n') }] };
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
