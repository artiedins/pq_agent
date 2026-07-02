#!/usr/bin/env node

// MCP server for HOME use: connects to an existing headed Chrome instance
// via CDP at localhost:9222. Requires playwright 1.57+ for ariaSnapshot.
//
// Dependencies: @modelcontextprotocol/sdk, playwright, zod
// (turndown is no longer needed - ariaSnapshot replaces HTML-to-markdown)

const { McpServer } = require('@modelcontextprotocol/sdk/server/mcp.js');
const { StdioServerTransport } = require('@modelcontextprotocol/sdk/server/stdio.js');
const { chromium } = require('playwright');
const { z } = require('zod');

let browser = null;
let page = null;

const server = new McpServer({
  name: 'playwright-chrome',
  version: '2.0.0',
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
    const context = browser.contexts()[0];
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

async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error('Playwright MCP server started, will connect to Chrome at localhost:9222 on first use');
}

main().catch((error) => {
  console.error('Fatal error:', error);
  process.exit(1);
});
