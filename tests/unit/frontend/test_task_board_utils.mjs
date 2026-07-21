/**
 * Unit tests for task-board-utils.js pure helpers (splitTaskDescription).
 *
 * Run with: node --test tests/unit/frontend/test_task_board_utils.mjs
 *
 * task-board-utils.js imports i18n via an absolute path ("/shared/..."), which
 * Node cannot resolve, so we load the module source with that import stubbed.
 */

import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const __dirname = dirname(fileURLToPath(import.meta.url));
const UTILS_PATH = resolve(
  __dirname,
  "../../../server/static/pages/task-board-utils.js",
);

const source = readFileSync(UTILS_PATH, "utf8").replace(
  /^import\s+\{\s*t\s*\}\s+from\s+["'][^"']+["'];?\s*$/m,
  "const t = (k) => k;",
);
const moduleUrl =
  "data:text/javascript;base64," + Buffer.from(source, "utf8").toString("base64");
const { splitTaskDescription } = await import(moduleUrl);

describe("splitTaskDescription", () => {
  it("returns empty title/body for nullish or blank input", () => {
    for (const input of [null, undefined, "", "   ", "\n\t"]) {
      assert.deepEqual(splitTaskDescription(input), {
        title: "",
        body: "",
        titleTruncated: false,
      });
    }
  });

  it("uses the first line as title and the rest as body", () => {
    const result = splitTaskDescription("Short title\nLong body line one.\nLine two.");
    assert.equal(result.title, "Short title");
    assert.equal(result.body, "Long body line one.\nLine two.");
    assert.equal(result.titleTruncated, false);
  });

  it("splits on the first sentence boundary when there is no newline", () => {
    const result = splitTaskDescription(
      "First sentence ends here. Second sentence continues with more detail.",
    );
    assert.equal(result.title, "First sentence ends here.");
    assert.equal(result.body, "Second sentence continues with more detail.");
    assert.equal(result.titleTruncated, false);
  });

  it("supports CJK sentence punctuation", () => {
    const result = splitTaskDescription("これはタイトルです。本文はこちらに続きます。");
    assert.equal(result.title, "これはタイトルです。");
    assert.equal(result.body, "本文はこちらに続きます。");
  });

  it("keeps short single-sentence text as title only", () => {
    const result = splitTaskDescription("Just a short task title.");
    assert.equal(result.title, "Just a short task title.");
    assert.equal(result.body, "");
    assert.equal(result.titleTruncated, false);
  });

  it("caps title at 80 chars and moves overflow into body", () => {
    const long = "A".repeat(100);
    const result = splitTaskDescription(long);
    assert.equal(result.title, `${"A".repeat(80)}…`);
    assert.equal(result.body, "A".repeat(20));
    assert.equal(result.titleTruncated, true);
  });

  it("when first line exceeds 80 chars, truncates title and prefixes body", () => {
    const first = "B".repeat(90);
    const result = splitTaskDescription(`${first}\nremainder body`);
    assert.equal(result.title, `${"B".repeat(80)}…`);
    assert.equal(result.body, `${"B".repeat(10)}\nremainder body`);
    assert.equal(result.titleTruncated, true);
  });

  it("respects a custom maxTitleLength", () => {
    const result = splitTaskDescription("abcdefghij", 5);
    assert.equal(result.title, "abcde…");
    assert.equal(result.body, "fghij");
    assert.equal(result.titleTruncated, true);
  });

  it("trims surrounding whitespace", () => {
    const result = splitTaskDescription("  Title line  \n  Body text  ");
    assert.equal(result.title, "Title line");
    assert.equal(result.body, "Body text");
  });

  it("handles multi-paragraph audit-log style descriptions", () => {
    const description = [
      "Security audit: review failed authentication spikes",
      "",
      "Observed 240 failed logins from 12 subnets within 15 minutes.",
      "Suggested follow-up: rotate credentials and enable rate limits.",
    ].join("\n");
    const result = splitTaskDescription(description);
    assert.equal(result.title, "Security audit: review failed authentication spikes");
    assert.ok(result.body.includes("Observed 240 failed logins"));
    assert.ok(result.body.includes("Suggested follow-up"));
    assert.equal(result.titleTruncated, false);
  });
});
