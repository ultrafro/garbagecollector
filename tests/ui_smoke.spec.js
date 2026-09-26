const { test, expect } = require('playwright/test');

test('console connects, streams, and renders responsively', async ({ page }) => {
  const errors = [];
  page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
  await page.goto('http://127.0.0.1:8080');
  await expect(page.locator('#status')).toHaveText('robot connected');
  await expect(page.locator('#cameraStatus')).toHaveText('YOLO live', {timeout: 15000});
  await expect(page.locator('#telemetry')).toContainText('connected');
  await page.locator('#setHome').click();
  await expect(page.locator('#armNotice')).toHaveText('Mock home saved.');
  await page.locator('#auto').click();
  await expect(page.locator('#autoStatus')).toContainText('AUTO');
  await expect(page.locator('#decision')).toContainText('command:', {timeout: 5000});
  await page.screenshot({path: 'screenshots/control-desktop.png', fullPage: true});
  await page.setViewportSize({width: 390, height: 844});
  await page.screenshot({path: 'screenshots/control-mobile.png', fullPage: true});
  await page.locator('#stop').click();
  await expect(page.locator('#autoStatus')).toHaveText('manual');
  expect(errors).toEqual([]);
});
