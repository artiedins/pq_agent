#!/usr/bin/env node

const { McpServer } = require('@modelcontextprotocol/sdk/server/mcp.js');
const { StdioServerTransport } = require('@modelcontextprotocol/sdk/server/stdio.js');
const { chromium } = require('playwright');
const { z } = require('zod');
const TurndownService = require('turndown');
const turndown = new TurndownService({ headingStyle: 'atx', codeBlockStyle: 'fenced' });

let browser = null;
let page = null;

const server = new McpServer({
  name: 'playwright-chrome',
  version: '1.0.0',
});

async function ensureBrowser() {
  if (!browser) {
    browser = await chromium.connectOverCDP('http://localhost:9222');
    const context = browser.contexts()[0];
    const pages = context.pages();
    page = pages.length > 0 ? pages[0] : await context.newPage();
    console.error('Connected to existing Chrome instance');
  }
  return page;
}

server.tool(
  'playwright_navigate',
  'Navigate to a URL in the existing Chrome browser',
  { url: z.string().describe('URL to navigate to') },
  async ({ url }) => {
    const p = await ensureBrowser();
    await p.goto(url, { waitUntil: 'networkidle' });
    const title = await p.title();
    return { content: [{ type: 'text', text: `Navigated to: ${url}\nPage title: ${title}` }] };
  }
);

server.tool(
  'playwright_screenshot',
  'Take a screenshot of the current page',
  { name: z.string().describe('Filename for screenshot') },
  async ({ name }) => {
    const p = await ensureBrowser();
    await p.screenshot({ path: name, fullPage: true });
    return { content: [{ type: 'text', text: `Screenshot saved to: ${name}` }] };
  }
);

server.tool(
  'playwright_extract_content',
  'Extract text content from the current page as clean markdown',
  { selector: z.string().optional().describe('Optional CSS selector (e.g. "article", "main"). Defaults to entire body.') },
  async ({ selector }) => {
    const p = await ensureBrowser();
    let html;
    if (selector) {
      html = await p.locator(selector).innerHTML();
    } else {
      html = await p.evaluate(() => document.body.innerHTML);
    }
    const markdown = turndown.turndown(html);
    return { content: [{ type: 'text', text: markdown || 'No content found' }] };
  }
);

server.tool(
  'playwright_click',
  'Click an element on the page',
  { selector: z.string().describe('CSS selector for element to click') },
  async ({ selector }) => {
    const p = await ensureBrowser();
    await p.click(selector);
    return { content: [{ type: 'text', text: `Clicked: ${selector}` }] };
  }
);

server.tool(
  'playwright_fill',
  'Fill a form field',
  {
    selector: z.string().describe('CSS selector for input field'),
    value: z.string().describe('Value to fill'),
  },
  async ({ selector, value }) => {
    const p = await ensureBrowser();
    await p.fill(selector, value);
    return { content: [{ type: 'text', text: `Filled ${selector} with: ${value}` }] };
  }
);

async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error('Playwright MCP server started, connecting to Chrome at localhost:9222');
}

main().catch((error) => {
  console.error('Fatal error:', error);
  process.exit(1);
});
