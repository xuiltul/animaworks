/**
 * Unit tests for inline tool-call text collapsing in render-utils.js.
 *
 * Run with: node --test tests/unit/frontend/test_tool_call_display_text.mjs
 *
 * Regression guard for the bug where an assistant message containing a full
 * prose answer PLUS a trailing inline `send_message` / `post_channel` tool-call
 * JSON was collapsed to just the tool-call payload, silently discarding the
 * surrounding answer text.
 *
 * render-utils.js imports i18n via an absolute path ("/shared/..."), which Node
 * cannot resolve, so we load the module source with that import stubbed out and
 * evaluate it from a data: URL to exercise the real functions.
 */

import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const __dirname = dirname(fileURLToPath(import.meta.url));
const RENDER_UTILS = resolve(
  __dirname,
  "../../../server/static/shared/chat/render-utils.js",
);

const source = readFileSync(RENDER_UTILS, "utf8").replace(
  /^import\s+\{\s*t\s*\}\s+from\s+["'][^"']+["'];?\s*$/m,
  "const t = (k) => k;",
);
const moduleUrl =
  "data:text/javascript;base64," + Buffer.from(source, "utf8").toString("base64");
const { renderHistoryMessage } = await import(moduleUrl);

const opts = {
  escapeHtml: (s) => String(s),
  renderMarkdown: (s) => s, // identity: makes the visible body easy to assert on
  smartTimestamp: () => "",
};

function renderAssistant(content, extra = {}) {
  return renderHistoryMessage(
    { role: "assistant", content, ...extra },
    opts,
  );
}

describe("inline tool-call text collapsing", () => {
  it("keeps prose that precedes an inline send_message tool call", () => {
    const prose =
      "これはサンドボックスエスケープの詳しい解説です。マウント条件が重要です。";
    const toolCall =
      '{"name":"send_message","arguments":{"content":"まとめは以上です。ふふ。"}}';
    const html = renderAssistant(`${prose}\n\n${toolCall}`);
    assert.ok(html.includes(prose), "prose body must be preserved");
  });

  it("keeps prose surrounding a fenced send_message tool call", () => {
    const prose = "本文の解説です。";
    const fenced =
      '```json\n{"name":"send_message","arguments":{"content":"締めの一言"}}\n```';
    const html = renderAssistant(`${prose}\n\n${fenced}`);
    assert.ok(html.includes(prose), "prose body must be preserved");
  });

  it("still collapses a message that is ONLY a send_message tool call", () => {
    const content =
      '{"name":"send_message","arguments":{"content":"配信されるべき本文"}}';
    const html = renderAssistant(content);
    assert.ok(html.includes("配信されるべき本文"), "payload must be shown");
    assert.ok(!html.includes('"send_message"'), "raw JSON must not be shown");
  });

  it("collapses a fence-only send_message tool call (no prose)", () => {
    const content =
      '```json\n{"name":"send_message","arguments":{"content":"配信本文"}}\n```';
    const html = renderAssistant(content);
    assert.ok(html.includes("配信本文"), "payload must be shown");
    assert.ok(!html.includes('"send_message"'), "raw JSON must not be shown");
  });

  it("passes plain prose through unchanged", () => {
    const content = "いい選択です！週末の自由食＋ビール1日ルールにもちょうど収まります。";
    const html = renderAssistant(content);
    assert.ok(html.includes(content), "plain prose must be shown verbatim");
  });

  it("keeps prose that precedes an inline post_channel tool call", () => {
    const prose = "チャンネル向けの詳しい報告本文です。";
    const toolCall =
      '{"name":"post_channel","arguments":{"text":"短い告知"}}';
    const html = renderAssistant(`${prose}\n\n${toolCall}`);
    assert.ok(html.includes(prose), "prose body must be preserved");
  });
});
