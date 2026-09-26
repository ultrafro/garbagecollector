const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright');

const durationMs = 4 * 60 * 1000;
const stamp = new Date().toISOString().replace(/[:.]/g, '-');
const outDir = path.resolve('recordings');
fs.mkdirSync(outDir, { recursive: true });

(async () => {
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    viewport: { width: 1440, height: 1000 },
    recordVideo: { dir: outDir, size: { width: 1440, height: 1000 } },
  });
  const page = await context.newPage();
  const events = [];
  page.on('pageerror', error => events.push({ type: 'pageerror', message: error.message, t: Date.now() }));
  await page.goto('http://127.0.0.1:8080', { waitUntil: 'domcontentloaded' });
  await page.waitForSelector('#auto');
  await page.waitForTimeout(2500);
  await page.evaluate(() => {
    window.__autoRecording = [];
    window.addEventListener('robot-message', event => {
      const message = event.detail;
      if (message && (message.type === 'autonomy' || message.type === 'state' || message.type === 'notice' || message.type === 'error')) {
        window.__autoRecording.push({ t: Date.now(), message });
      }
    });
  });
  const started = new Date().toISOString();
  await page.locator('#auto').click();
  await page.waitForTimeout(durationMs);
  const ended = new Date().toISOString();
  await page.locator('#stop').click();
  await page.waitForTimeout(1200);
  const captured = await page.evaluate(() => window.__autoRecording || []);
  captured.push(...events);
  fs.writeFileSync(path.join(outDir, `auto-test-${stamp}.json`), JSON.stringify({ started, ended, durationMs, events: captured }, null, 2));
  await context.close();
  await browser.close();
  console.log(JSON.stringify({ started, ended, durationMs, eventCount: captured.length, outDir }));
})().catch(error => { console.error(error); process.exitCode = 1; });
