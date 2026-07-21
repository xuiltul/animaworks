/**
 * Unit tests for activity page tab routing helpers.
 *
 * Run with: node --test tests/unit/frontend/test_activity_tabs.mjs
 *
 * activity.js imports i18n via an absolute path ("/shared/..."), which Node
 * cannot resolve, so we load the module source with that import stubbed out.
 */

import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ACTIVITY_PATH = resolve(
  __dirname,
  "../../../server/static/pages/activity.js",
);

const source = readFileSync(ACTIVITY_PATH, "utf8")
  .replace(
    /^import\s+\{\s*createPageTabs\s*\}\s+from\s+["'][^"']+["'];?\s*$/m,
    `export function createPageTabs() { return { el: null, setActive() {}, getActive() { return ""; }, destroy() {} }; }`,
  )
  .replace(
    /^import\s+\{\s*t\s*\}\s+from\s+["'][^"']+["'];?\s*$/m,
    `export function t(k) { return k; }`,
  );

const moduleUrl =
  "data:text/javascript;base64," + Buffer.from(source, "utf8").toString("base64");

const {
  resolveActivityTab,
  buildActivityTabHash,
  activityTabLoader,
} = await import(moduleUrl);

describe("resolveActivityTab", () => {
  it("defaults to timeline for empty / missing subPath", () => {
    assert.equal(resolveActivityTab(""), "timeline");
    assert.equal(resolveActivityTab(undefined), "timeline");
    assert.equal(resolveActivityTab(null), "timeline");
  });

  it("maps removed report subPath to timeline tab", () => {
    assert.equal(resolveActivityTab("report"), "timeline");
  });

  it("maps logs subPath to logs tab", () => {
    assert.equal(resolveActivityTab("logs"), "logs");
    assert.equal(resolveActivityTab("logs/foo"), "logs");
  });

  it("falls back to timeline for unknown subPath", () => {
    assert.equal(resolveActivityTab("unknown"), "timeline");
    assert.equal(resolveActivityTab("timeline"), "timeline");
  });
});

describe("buildActivityTabHash", () => {
  it("builds hashes for each tab", () => {
    assert.equal(buildActivityTabHash("timeline"), "#/activity");
    assert.equal(buildActivityTabHash("logs"), "#/activity/logs");
  });
});

describe("activityTabLoader (subPath → module path)", () => {
  it("loads activity-timeline for timeline tab", () => {
    const fn = activityTabLoader("timeline");
    assert.equal(typeof fn, "function");
    // Function body references the relative module path
    assert.match(String(fn), /activity-timeline\.js/);
  });


  it("loads logs for logs tab", () => {
    assert.match(String(activityTabLoader("logs")), /logs\.js/);
  });
});
