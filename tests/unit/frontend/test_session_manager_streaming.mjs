/**
 * Unit tests for ChatSessionManager streaming timestamps
 * (server/static/shared/chat/session-manager.js).
 *
 * Run with: node --test tests/unit/frontend/test_session_manager_streaming.mjs
 *
 * Covers `getStreamingSince()`, which the chat page polling fail-safe uses to
 * bypass the streaming guard when a stream has been stuck for too long.
 */

import { describe, it, beforeEach } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const STATIC = resolve(__dirname, "../../../server/static");

// ── Import Module Under Test ──────────────────────────

// session-manager.js imports "./history-loader.js" and "/shared/base-path.js"
// (browser-absolute). Load as data: modules with stubs so Node can resolve them.
let ChatSessionManager;

{
  const toDataUrl = (body, tag) =>
    "data:text/javascript;base64," +
    Buffer.from(body, "utf8").toString("base64") +
    "#" +
    tag;

  const stripImports = (src) =>
    src.replace(/(?:^|\n)\s*import\b[\s\S]*?;/g, "\n");

  const historyUrl = toDataUrl(
    readFileSync(resolve(STATIC, "shared/chat/history-loader.js"), "utf8"),
    "history-loader",
  );

  const managerBody =
    `
    import {
      createHistoryState, applyHistoryData, mergePolledHistory, hasHistoryContent, hasFeedConversation,
    } from "${historyUrl}";
    const basePath = "";
  ` + stripImports(readFileSync(resolve(STATIC, "shared/chat/session-manager.js"), "utf8"));

  const mod = await import(toDataUrl(managerBody, "session-manager"));
  ChatSessionManager = mod.ChatSessionManager;
}

// ── Helpers ──────────────────────────────

function configure(mgr, streamChat) {
  mgr.configure({
    streamChat,
    fetchActiveStream: async () => null,
    fetchStreamProgress: async () => null,
    getUser: () => "human",
    fetchHistory: async () => ({ sessions: [] }),
  });
}

// ── Tests ──────────────────────────────

describe("ChatSessionManager.getStreamingSince", () => {
  let mgr;

  beforeEach(() => {
    ChatSessionManager.resetInstance();
    mgr = ChatSessionManager.getInstance();
  });

  it("returns null when no session exists or nothing is streaming", () => {
    assert.strictEqual(mgr.getStreamingSince("nico", "default"), null);
    mgr.getSession("nico", "default");
    assert.strictEqual(mgr.getStreamingSince("nico", "default"), null);
  });

  it("is set while a stream is in progress and cleared when it completes", async () => {
    let resolveStream;
    configure(mgr, () => new Promise((r) => { resolveStream = r; }));

    const before = Date.now();
    const pending = mgr.sendChat("nico", "default", "hello");

    assert.strictEqual(mgr.isStreamingFor("nico", "default"), true);
    const since = mgr.getStreamingSince("nico", "default");
    assert.strictEqual(typeof since, "number");
    assert.ok(since >= before && since <= Date.now(), "timestamp should be captured at stream start");

    resolveStream();
    const result = await pending;

    assert.strictEqual(result.success, true);
    assert.strictEqual(mgr.isStreamingFor("nico", "default"), false);
    assert.strictEqual(mgr.getStreamingSince("nico", "default"), null);
  });

  it("is cleared when the stream rejects (e.g. SSE idle timeout)", async () => {
    configure(mgr, () => {
      const err = new Error("SSE idle timeout: no data received for 90000ms");
      err.name = "SseIdleTimeoutError";
      return Promise.reject(err);
    });

    const errors = [];
    const result = await mgr.sendChat("nico", "default", "hello", {
      callbacks: { onError: (e) => errors.push(e.message) },
    });

    assert.strictEqual(result.success, false);
    assert.strictEqual(result.error.name, "SseIdleTimeoutError");
    assert.deepStrictEqual(errors, ["SSE idle timeout: no data received for 90000ms"]);
    assert.strictEqual(mgr.isStreamingFor("nico", "default"), false);
    assert.strictEqual(mgr.getStreamingSince("nico", "default"), null);
  });

  it("is scoped per anima:thread", async () => {
    let resolveStream;
    configure(mgr, () => new Promise((r) => { resolveStream = r; }));

    const pending = mgr.sendChat("nico", "t1", "hello");
    assert.strictEqual(typeof mgr.getStreamingSince("nico", "t1"), "number");
    assert.strictEqual(mgr.getStreamingSince("nico", "default"), null);
    assert.strictEqual(mgr.getStreamingSince("other", "t1"), null);

    resolveStream();
    await pending;
    assert.strictEqual(mgr.getStreamingSince("nico", "t1"), null);
  });
});
