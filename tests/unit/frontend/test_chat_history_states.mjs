/**
 * Behaviour tests for the chat history states: a failed fetch must not look
 * like an empty history, an archived thread must not stay selected, and a busy
 * Anima's background-only feed must not leave the user looking at nothing.
 *
 * Run with: node --test tests/unit/frontend/test_chat_history_states.mjs
 *
 * The real controllers run against a small fake DOM. Browser-only imports are
 * replaced by stubs; history-loader, session-manager, thread-logic and
 * render-utils are the real modules.
 */

import { beforeEach, describe, it } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const STATIC = resolve(__dirname, "../../../server/static");

const read = (path) => readFileSync(resolve(STATIC, path), "utf8");
const toModuleUrl = (body, tag) =>
  "data:text/javascript;base64," + Buffer.from(body, "utf8").toString("base64") + "#" + tag;
const stripImports = (source) => source.replace(/(?:^|\n)\s*import\b[\s\S]*?;/g, "\n");

const escapeHtml = (value) => String(value ?? "").replace(/[&<>"']/g, (ch) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
})[ch]);

// Keys come back untranslated so tests assert which message was chosen.
globalThis.__t = (key) => key;
globalThis.__escapeHtml = escapeHtml;
globalThis.__api = async () => { throw new Error("unexpected api call"); };

// ── Real modules ──────────────────────

const historyUrl = toModuleUrl(read("shared/chat/history-loader.js"), "history-loader");
const historyLoader = await import(historyUrl);

const threadLogicUrl = toModuleUrl(
  `const t = (...args) => globalThis.__t(...args);\n${stripImports(read("shared/chat/thread-logic.js"))}`,
  "thread-logic",
);
const threadLogic = await import(threadLogicUrl);

const renderUtilsUrl = toModuleUrl(
  `const t = (...args) => globalThis.__t(...args);
   const _escapeAttr = (...args) => globalThis.__escapeHtml(...args);
   const api = (...args) => globalThis.__api(...args);
   const bustUrl = () => "";
   const faceUrl = () => "";
   ${stripImports(read("shared/chat/render-utils.js"))}`,
  "render-utils",
);
const renderUtils = await import(renderUtilsUrl);

const managerUrl = toModuleUrl(
  `import {
     createHistoryState, applyHistoryData, mergePolledHistory, hasHistoryContent, hasFeedConversation,
   } from "${historyUrl}";
   const api = (...args) => globalThis.__api(...args);
   ${stripImports(read("shared/chat/session-manager.js"))}`,
  "session-manager",
);
const { ChatSessionManager } = await import(managerUrl);

// ── Fake DOM ──────────────────────

class FakeButton {
  constructor(attribute) { this.attribute = attribute; this.listeners = []; this.disabled = false; }
  addEventListener(name, callback) { if (name === "click") this.listeners.push(callback); }
  click() { for (const callback of this.listeners) callback({ stopPropagation() {} }); }
}

class FakeElement {
  #html = "";
  #buttons = new Map();
  scrollHeight = 0;
  scrollTop = 0;
  clientHeight = 0;
  dataset = {};
  value = "";
  style = {};
  disabled = false;
  addEventListener() {}
  set innerHTML(html) { this.#html = html; this.#buttons = new Map(); }
  get innerHTML() { return this.#html; }
  #button(attribute) {
    // Only the buttons these tests press; other queries find nothing to bind.
    if (!["data-history-retry", "data-feed-load-older"].includes(attribute)) return null;
    if (!this.#html.includes(attribute)) return null;
    if (!this.#buttons.has(attribute)) this.#buttons.set(attribute, new FakeButton(attribute));
    return this.#buttons.get(attribute);
  }
  querySelector(selector) {
    const match = /^\[([a-z-]+)\]$/.exec(selector);
    return match ? this.#button(match[1]) : null;
  }
  querySelectorAll(selector) {
    const found = this.querySelector(selector);
    return found ? [found] : [];
  }
  /** The button a person would click, by its data attribute. */
  press(attribute) {
    const button = this.#button(attribute);
    assert.ok(button, `no ${attribute} button in: ${this.#html}`);
    button.click();
  }
}

globalThis.window = { matchMedia: () => ({ matches: false }), setTimeout, requestIdleCallback: undefined };
globalThis.document = {
  getElementById: () => null,
  createElement: () => new FakeElement(),
  body: { classList: { contains: () => false } },
};
globalThis.localStorage = { getItem: () => null, setItem() {} };

const settle = async (rounds = 25) => {
  for (let index = 0; index < rounds; index += 1) await new Promise((done) => setImmediate(done));
};

/** Records calls but answers nothing: every controller the code under test merely pokes. */
const noopController = () => new Proxy({}, { get: () => () => undefined });

function deferred() {
  let resolvePromise;
  let rejectPromise;
  const promise = new Promise((done, fail) => { resolvePromise = done; rejectPromise = fail; });
  return { promise, resolve: resolvePromise, reject: rejectPromise };
}

// ── Chat page renderer under test ──────────────────────

const rendererUrl = toModuleUrl(
  `import { mergePolledHistory, hasFeedConversation } from "${historyUrl}";
   import {
     renderHistoryMessage as _sharedRenderHistoryMessage,
     renderSessionDivider as _sharedRenderSessionDivider,
     bindToolCallHandlers as _sharedBindToolCallHandlers,
     bindBubbleActionHandlers as _sharedBindBubbleActionHandlers,
     bindChatLineHandlers as _sharedBindChatLineHandlers,
     bindHumanNotifyHandlers as _sharedBindHumanNotifyHandlers,
     renderLiveBubble,
     renderStreamingBubbleInner,
     updateStreamingZone,
     renderCollapsibleActivityBundle as _sharedRenderCollapsibleActivityBundle,
     getSessionActivityType as _sharedGetSessionActivityType,
     getMessageActivityType as _sharedGetMessageActivityType,
     bindCollapsibleSessionHandlers as _sharedBindCollapsibleSessionHandlers,
     renderFeedBackgroundGroup as _sharedRenderFeedBackgroundGroup,
     feedBackgroundUrl as _sharedFeedBackgroundUrl,
     bindLazyBackgroundGroups as _sharedBindLazyBackgroundGroups,
     renderFeedHiddenCount as _sharedRenderFeedHiddenCount,
     renderHistoryError as _sharedRenderHistoryError,
     bindHistoryRetry as _sharedBindHistoryRetry,
   } from "${renderUtilsUrl}";
   const setThreadUnread = () => {};
   const refreshAnimaUnread = () => {};
   const scheduleSaveChatUiState = () => {};
   const CONSTANTS = { HISTORY_PAGE_SIZE: 50, TOOL_RESULT_TRUNCATE: 500, STREAMING_GUARD_MAX_MS: 120000 };
   const createScrollObserver = () => ({ observe() {}, disconnect() {}, refresh() {} });
   const thinkRenderHooks = null;
   const initTextArtifactHandlers = () => {};
   const companyColor = () => "";
   const createLogger = () => ({ warn() {}, error() {}, debug() {}, info() {} });
   ${stripImports(read("pages/chat/chat-renderer.js"))}`,
  "chat-renderer",
);
const { createChatRenderer } = await import(rendererUrl);

function feedPage(items, { hasMore = false, nextBefore = null } = {}) {
  return { items, has_more: hasMore, next_before: nextBefore };
}
const messageItem = (ts, content, role = "human") => ({
  kind: "message",
  message: { ts, role, content, from_person: role === "human" ? "human" : "", tool_calls: [] },
});
const backgroundItem = (fromTs, toTs = fromTs) => ({
  kind: "bg_group", from_ts: fromTs, to_ts: toTs, count: 5, kinds: { cron: 5 }, participants: [], label_hint: "cron",
});

function chatPage({ animaFlags = {} } = {}) {
  ChatSessionManager.resetInstance();
  const manager = ChatSessionManager.getInstance();
  const messagesEl = new FakeElement();
  const elements = { chatPageMessages: messagesEl };
  const api = (...args) => globalThis.__api(...args);
  const state = {
    manager,
    selectedAnima: "mei",
    selectedThreadId: "default",
    animas: [{ name: "mei", status: "idle", ...animaFlags }],
    users: [],
    threads: { mei: [{ id: "default" }, { id: "research" }] },
    activeThreadByAnima: {},
    animaTabAvatarUrls: {},
    firstMeetingRequested: new Set(),
    firstMeetingResponses: new Map(),
    firstMeetingFailed: new Set(),
    chatPollingInFlight: false,
    get demoMode() { return false; },
    fetchChatFeedHistory: null,
  };
  const ctx = {
    state,
    $: (id) => elements[id] ?? null,
    controllers: { thread: noopController(), anima: noopController(), events: noopController() },
    deps: {
      t: (key) => key,
      api,
      escapeHtml,
      renderMarkdown: (text) => escapeHtml(text),
      smartTimestamp: (ts) => ts,
      timeStr: (ts) => ts,
      renderChatFiles: () => "",
      renderChatImages: () => "",
    },
  };
  const renderer = createChatRenderer(ctx);
  ctx.controllers.renderer = renderer;
  return { ctx, state, manager, renderer, messagesEl };
}

describe("failed history fetch on the chat page", () => {
  beforeEach(() => { globalThis.__api = async () => { throw new Error("unexpected api call"); }; });

  it("shows a retryable error instead of the empty state", async () => {
    const { state, manager, renderer, messagesEl } = chatPage();
    state.fetchChatFeedHistory = async () => { throw new Error("502"); };

    await manager.loadHistory("mei", "default", 50, state.fetchChatFeedHistory);
    renderer.renderChat();

    assert.match(messagesEl.innerHTML, /chat\.history_load_failed/);
    assert.doesNotMatch(messagesEl.innerHTML, /chat\.messages_empty/);
    assert.match(messagesEl.innerHTML, /role="alert"/);

    state.fetchChatFeedHistory = async () => feedPage([messageItem("2026-11-01T10:00:00+09:00", "hello")]);
    messagesEl.press("data-history-retry");
    await settle();

    assert.match(messagesEl.innerHTML, /hello/);
    assert.doesNotMatch(messagesEl.innerHTML, /chat\.history_load_failed/);
  });

  it("still shows the empty state for a history that really is empty", async () => {
    const { state, manager, renderer, messagesEl } = chatPage();
    state.fetchChatFeedHistory = async () => feedPage([]);

    await manager.loadHistory("mei", "default", 50, state.fetchChatFeedHistory);
    renderer.renderChat();

    assert.match(messagesEl.innerHTML, /chat\.messages_empty/);
    assert.doesNotMatch(messagesEl.innerHTML, /data-history-retry/);
  });

  it("does not start the first-meeting greeting on a failed fetch, only on a confirmed empty one", async () => {
    const { state, manager, renderer, messagesEl } = chatPage({ animaFlags: { needs_user_input: true } });
    const greetings = [];
    globalThis.__api = async (url) => {
      greetings.push(url);
      return { response: "nice to meet you" };
    };
    state.fetchChatFeedHistory = async () => { throw new Error("502"); };

    await manager.loadHistory("mei", "default", 50, state.fetchChatFeedHistory);
    renderer.renderChat();
    await settle();

    assert.deepEqual(greetings, []);
    assert.match(messagesEl.innerHTML, /chat\.history_load_failed/);

    state.fetchChatFeedHistory = async () => feedPage([]);
    messagesEl.press("data-history-retry");
    await settle();

    assert.deepEqual(greetings, ["/api/animas/mei/greet"]);
  });

  it("does not start the first-meeting greeting while the fetch is still running", async () => {
    const { state, manager, renderer } = chatPage({ animaFlags: { needs_user_input: true } });
    const greetings = [];
    globalThis.__api = async (url) => { greetings.push(url); return { response: "hi" }; };
    const pending = deferred();
    state.fetchChatFeedHistory = () => pending.promise;

    const loading = manager.loadHistory("mei", "default", 50, state.fetchChatFeedHistory);
    renderer.renderChat();
    await settle();
    assert.deepEqual(greetings, []);

    pending.resolve(feedPage([messageItem("2026-11-01T10:00:00+09:00", "hello")]));
    await loading;
    renderer.renderChat();
    await settle();
    assert.deepEqual(greetings, [], "an Anima with history gets no first meeting");
  });

  it("keeps the history on screen when a poll fails", async () => {
    const { state, manager, renderer, messagesEl } = chatPage();
    state.fetchChatFeedHistory = async () => feedPage([messageItem("2026-11-01T10:00:00+09:00", "hello")]);
    await manager.loadHistory("mei", "default", 50, state.fetchChatFeedHistory);
    renderer.renderChat();

    state.fetchChatFeedHistory = async () => { throw new Error("502"); };
    await renderer.pollSelectedChat();
    renderer.renderChat();

    assert.match(messagesEl.innerHTML, /hello/);
    assert.equal(manager.getHistoryState("mei", "default").loadFailed, false);
  });

  it("recovers from a failed load when a later poll succeeds", async () => {
    const { state, manager, renderer, messagesEl } = chatPage();
    state.fetchChatFeedHistory = async () => { throw new Error("502"); };
    await manager.loadHistory("mei", "default", 50, state.fetchChatFeedHistory);
    renderer.renderChat();
    assert.match(messagesEl.innerHTML, /chat\.history_load_failed/);

    state.fetchChatFeedHistory = async () => feedPage([messageItem("2026-11-01T10:00:00+09:00", "hello")]);
    await renderer.pollSelectedChat();

    assert.match(messagesEl.innerHTML, /hello/);
    const hs = manager.getHistoryState("mei", "default");
    assert.equal(hs.loadFailed, false);
    assert.equal(hs.loaded, true);
  });

  it("keeps a failed load visible while polls keep failing", async () => {
    const { state, manager, renderer, messagesEl } = chatPage();
    state.fetchChatFeedHistory = async () => { throw new Error("502"); };
    await manager.loadHistory("mei", "default", 50, state.fetchChatFeedHistory);
    renderer.renderChat();

    await renderer.pollSelectedChat();
    renderer.renderChat();

    assert.match(messagesEl.innerHTML, /chat\.history_load_failed/);
  });

  it("keeps history that was on screen when a full reload fails", async () => {
    const { state, manager, renderer, messagesEl } = chatPage();
    state.fetchChatFeedHistory = async () => feedPage([messageItem("2026-11-01T10:00:00+09:00", "hello")]);
    await manager.loadHistory("mei", "default", 50, state.fetchChatFeedHistory);

    state.fetchChatFeedHistory = async () => { throw new Error("502"); };
    await manager.loadHistory("mei", "default", 50, state.fetchChatFeedHistory);
    renderer.renderChat();

    assert.match(messagesEl.innerHTML, /hello/);
    assert.equal(manager.getHistoryState("mei", "default").loaded, true);
  });

  it("offers a retry when an older page fails and keeps the rest on screen", async () => {
    const { state, manager, renderer, messagesEl } = chatPage();
    state.fetchChatFeedHistory = async () => feedPage(
      [messageItem("2026-11-01T10:00:00+09:00", "recent")],
      { hasMore: true, nextBefore: "2026-11-01T10:00:00+09:00" },
    );
    await manager.loadHistory("mei", "default", 50, state.fetchChatFeedHistory);

    state.fetchChatFeedHistory = async () => { throw new Error("502"); };
    await renderer.loadOlderMessages();

    assert.match(messagesEl.innerHTML, /recent/);
    assert.match(messagesEl.innerHTML, /chat\.history_older_failed/);
    assert.doesNotMatch(messagesEl.innerHTML, /chat-load-sentinel/, "no sentinel, or the failure would retrigger itself");
    assert.equal(manager.getHistoryState("mei", "default").hasMore, true);

    state.fetchChatFeedHistory = async () => feedPage([messageItem("2026-10-01T10:00:00+09:00", "older")]);
    messagesEl.press("data-history-retry");
    await settle();

    assert.match(messagesEl.innerHTML, /older/);
    assert.doesNotMatch(messagesEl.innerHTML, /chat\.history_older_failed/);
  });
});

describe("busy Anima whose first feed page has no conversation", () => {
  beforeEach(() => { globalThis.__api = async () => { throw new Error("unexpected api call"); }; });

  /** A fetcher answering from a page list by call order, recording each cursor. */
  function pagedFetcher(pages) {
    const cursors = [];
    const fetcher = async (_anima, _limit, before) => {
      cursors.push(before);
      return pages[Math.min(cursors.length - 1, pages.length - 1)];
    };
    fetcher.cursors = cursors;
    return fetcher;
  }

  it("loads older pages until a conversation appears, then stops", async () => {
    const { state, manager, renderer, messagesEl } = chatPage();
    state.fetchChatFeedHistory = pagedFetcher([
      feedPage([backgroundItem("2026-11-03T00:00:00+09:00")], { hasMore: true, nextBefore: "2026-11-03T00:00:00+09:00" }),
      feedPage([messageItem("2026-11-01T10:00:00+09:00", "found it")], { hasMore: true, nextBefore: "2026-11-01T10:00:00+09:00" }),
      feedPage([messageItem("2026-10-01T10:00:00+09:00", "too far")]),
    ]);

    await manager.loadHistory("mei", "default", 50, state.fetchChatFeedHistory);
    renderer.renderChat();
    await settle();

    assert.deepEqual(state.fetchChatFeedHistory.cursors, [null, "2026-11-03T00:00:00+09:00"]);
    assert.match(messagesEl.innerHTML, /found it/);
    assert.doesNotMatch(messagesEl.innerHTML, /chat\.feed\.no_/);
  });

  it("stops at the page bound, says so, and resumes only when asked", async () => {
    const { state, manager, renderer, messagesEl } = chatPage();
    let page = 0;
    const fetcher = async () => {
      page += 1;
      const ts = `2026-10-${String(30 - page).padStart(2, "0")}T00:00:00+09:00`;
      return feedPage([backgroundItem(ts)], { hasMore: true, nextBefore: ts });
    };
    state.fetchChatFeedHistory = fetcher;

    await manager.loadHistory("mei", "default", 50, fetcher);
    renderer.renderChat();
    assert.match(messagesEl.innerHTML, /chat\.past_loading/, "searching is visible");
    await settle();

    assert.equal(page, 4, "the first page plus three more");
    assert.match(messagesEl.innerHTML, /chat\.feed\.no_recent_conversation/);
    assert.doesNotMatch(messagesEl.innerHTML, /chat\.past_loading/);

    renderer.renderChat();
    await settle();
    assert.equal(page, 4, "rendering again does not start another search");

    messagesEl.press("data-feed-load-older");
    await settle();
    assert.equal(page, 7, "one click looks three pages further");
  });

  it("stops at the time bound", async () => {
    const { state, manager } = chatPage();
    let calls = 0;
    state.fetchChatFeedHistory = async () => {
      calls += 1;
      return feedPage([backgroundItem("2026-10-20T00:00:00+09:00")], { hasMore: true, nextBefore: `2026-10-${30 - calls}` });
    };
    await manager.loadHistory("mei", "default", 50, state.fetchChatFeedHistory);

    const pages = await manager.loadUntilConversation("mei", "default", {
      fetchHistory: state.fetchChatFeedHistory, maxPages: 10, maxMs: 0,
    });

    assert.equal(pages, 0);
    assert.equal(calls, 1);
  });

  it("says there is no conversation when the log is exhausted, without fetching more", async () => {
    const { state, manager, renderer, messagesEl } = chatPage();
    state.fetchChatFeedHistory = pagedFetcher([feedPage([backgroundItem("2026-11-03T00:00:00+09:00")])]);

    await manager.loadHistory("mei", "default", 50, state.fetchChatFeedHistory);
    renderer.renderChat();
    await settle();

    assert.equal(state.fetchChatFeedHistory.cursors.length, 1);
    assert.match(messagesEl.innerHTML, /chat\.feed\.no_conversation/);
    assert.doesNotMatch(messagesEl.innerHTML, /data-feed-load-older/);
    assert.match(messagesEl.innerHTML, /bg-feed-group/, "the background line stays");
  });

  it("does not call an empty first page with older pages left \"no messages\"", async () => {
    const { state, manager, renderer, messagesEl } = chatPage();
    state.fetchChatFeedHistory = pagedFetcher([
      feedPage([], { hasMore: true, nextBefore: "2026-10-22" }),
      feedPage([messageItem("2026-10-01T10:00:00+09:00", "old message")]),
    ]);

    await manager.loadHistory("mei", "research", 50, state.fetchChatFeedHistory);
    state.selectedThreadId = "research";
    renderer.renderChat();
    assert.doesNotMatch(messagesEl.innerHTML, /chat\.messages_empty/);
    await settle();

    assert.match(messagesEl.innerHTML, /old message/);
  });

  it("leaves paging alone when the first page has a conversation", async () => {
    const { state, manager, renderer, messagesEl } = chatPage();
    state.fetchChatFeedHistory = pagedFetcher([
      feedPage([messageItem("2026-11-03T00:00:00+09:00", "recent")], { hasMore: true, nextBefore: "2026-11-03T00:00:00+09:00" }),
    ]);

    await manager.loadHistory("mei", "default", 50, state.fetchChatFeedHistory);
    renderer.renderChat();
    await settle();

    assert.equal(state.fetchChatFeedHistory.cursors.length, 1);
    assert.match(messagesEl.innerHTML, /chat-load-sentinel/);
    assert.doesNotMatch(messagesEl.innerHTML, /chat\.feed\.no_/);
  });

  it("stops searching when the user leaves the thread and resumes on return", async () => {
    const { state, manager, renderer } = chatPage();
    let calls = 0;
    const gate = deferred();
    state.fetchChatFeedHistory = async () => {
      calls += 1;
      if (calls === 2) await gate.promise;
      return feedPage([backgroundItem(`2026-10-${30 - calls}T00:00:00+09:00`)], {
        hasMore: true, nextBefore: `2026-10-${30 - calls}T00:00:00+09:00`,
      });
    };

    await manager.loadHistory("mei", "default", 50, state.fetchChatFeedHistory);
    renderer.renderChat();
    await settle();
    assert.equal(calls, 2);

    state.selectedThreadId = "research";
    gate.resolve();
    await settle();
    assert.equal(calls, 2, "no further pages for a thread nobody is looking at");

    state.selectedThreadId = "default";
    renderer.renderChat();
    await settle();
    assert.equal(calls, 5, "the search resumes with a fresh bound");
  });

  it("stops at a failed page and offers the retry", async () => {
    const { state, manager, renderer, messagesEl } = chatPage();
    let calls = 0;
    state.fetchChatFeedHistory = async () => {
      calls += 1;
      if (calls === 2) throw new Error("502");
      return feedPage([backgroundItem("2026-11-03T00:00:00+09:00")], { hasMore: true, nextBefore: "2026-11-03T00:00:00+09:00" });
    };

    await manager.loadHistory("mei", "default", 50, state.fetchChatFeedHistory);
    renderer.renderChat();
    await settle();

    assert.equal(calls, 2);
    assert.match(messagesEl.innerHTML, /chat\.history_older_failed/);
  });
});

describe("feed polling", () => {
  it("does not re-render on an empty poll of an empty history", () => {
    const hs = historyLoader.createHistoryState();
    historyLoader.applyHistoryData(hs, feedPage([]));
    hs.loaded = true;

    const { changed } = historyLoader.mergePolledHistory(hs, feedPage([]));

    assert.equal(changed, false);
  });

  it("does not rewind a cursor that older empty pages already moved past", () => {
    const hs = historyLoader.createHistoryState();
    historyLoader.applyHistoryData(hs, feedPage([], { hasMore: true, nextBefore: "2026-10-22" }));
    hs.loaded = true;
    historyLoader.applyHistoryData(hs, feedPage([], { hasMore: true, nextBefore: "2026-10-08" }), { prepend: true });

    assert.equal(historyLoader.mergePolledHistory(hs, feedPage([], { hasMore: true, nextBefore: "2026-10-22" })).changed, false);

    const { merged } = historyLoader.mergePolledHistory(
      hs,
      feedPage([messageItem("2026-11-01T10:00:00+09:00", "new")], { hasMore: true, nextBefore: "2026-10-22" }),
    );
    assert.equal(merged.nextBefore, "2026-10-08");
  });

  it("keeps the cursor when items were on screen and the newest page changes", () => {
    const hs = historyLoader.createHistoryState();
    const first = messageItem("2026-11-01T10:00:00+09:00", "one");
    historyLoader.applyHistoryData(hs, feedPage([first], { hasMore: true, nextBefore: first.message.ts }));
    historyLoader.applyHistoryData(hs, feedPage([], { hasMore: true, nextBefore: "2026-10-08" }), { prepend: true });

    const { merged } = historyLoader.mergePolledHistory(
      hs,
      feedPage([first, messageItem("2026-11-01T11:00:00+09:00", "two")], { hasMore: true, nextBefore: first.message.ts }),
    );

    assert.equal(merged.feedItems.length, 2);
    assert.equal(merged.nextBefore, "2026-10-08");
  });
});

describe("feed summary item", () => {
  it("renders the compressed summary as a fold, not a chat bubble, and escapes it", () => {
    const html = renderUtils.renderHistoryMessage(
      { type: "thread_summary", role: "system", content: "<b>agreed</b>\nnext", compressed_turn_count: 8, ts: "x" },
      { escapeHtml, renderMarkdown: (text) => text, smartTimestamp: (ts) => ts, lineChat: true },
    );

    assert.match(html, /<details class="chat-thread-summary">/);
    assert.match(html, /chat\.feed\.compressed_summary/);
    assert.match(html, /&lt;b&gt;agreed&lt;\/b&gt;<br>next/);
  });

  it("does not count the summary as a conversation", () => {
    assert.equal(historyLoader.hasFeedConversation({
      feedItems: [{ kind: "message", message: { type: "thread_summary" } }, backgroundItem("x")],
    }), false);
    assert.equal(historyLoader.hasFeedConversation({ feedItems: [messageItem("x", "hello")] }), true);
  });
});

// ── Workspace view under test ──────────────────────

const workspaceUrl = toModuleUrl(
  `import {
     renderHistoryMessage, renderSessionDivider, bindToolCallHandlers,
     bindBubbleActionHandlers, bindHumanNotifyHandlers,
     renderLiveBubble, renderStreamingBubbleInner,
     renderHistoryError, bindHistoryRetry,
   } from "${renderUtilsUrl}";
   import { ChatSessionManager } from "${managerUrl}";
   const getState = () => globalThis.__wsState;
   const setState = () => {};
   const t = (key) => key;
   const escapeHtml = (...args) => globalThis.__escapeHtml(...args);
   const renderSimpleMarkdown = (text) => escapeHtml(text);
   const smartTimestamp = (ts) => ts;
   const renderChatImages = () => "";
   const createScrollObserver = () => ({ observe() {}, disconnect() {}, refresh() {} });
   const createLogger = () => ({ warn() {}, error() {}, debug() {}, info() {} });
   ${stripImports(read("workspace/modules/chat-history.js"))}`,
  "ws-chat-history",
);
const workspace = await import(workspaceUrl);
const { ChatSessionManager: WorkspaceManager } = await import(managerUrl);

function workspaceView() {
  WorkspaceManager.resetInstance();
  const manager = WorkspaceManager.getInstance();
  const convMessages = new FakeElement();
  convMessages.insertBefore = () => {};
  Object.defineProperty(convMessages, "firstChild", { get: () => null });
  globalThis.__wsState = { conversationAnima: "mei", activeThreadId: "default" };
  workspace.initHistory({ getDom: () => ({ convMessages }) });
  return { manager, convMessages };
}

const sessionsPage = (content, { hasMore = false, nextBefore = null } = {}) => ({
  sessions: [{
    session_start: "2026-11-01T10:00:00+09:00",
    session_end: "2026-11-01T10:00:00+09:00",
    trigger: "chat",
    messages: [{ ts: "2026-11-01T10:00:00+09:00", role: "human", content, from_person: "human", tool_calls: [] }],
  }],
  has_more: hasMore,
  next_before: nextBefore,
});

describe("failed history fetch in the workspace view", () => {
  it("shows a retryable error instead of the empty state", async () => {
    const { manager, convMessages } = workspaceView();
    let fetcher = async () => { throw new Error("502"); };
    manager.configure({ fetchHistory: (...args) => fetcher(...args) });

    await workspace.loadAndRenderConvMessages("mei", { resumeStream() {} });

    assert.match(convMessages.innerHTML, /chat\.history_load_failed/);
    assert.doesNotMatch(convMessages.innerHTML, /chat\.messages_empty/);

    fetcher = async () => sessionsPage("hello");
    convMessages.press("data-history-retry");
    await settle();

    assert.match(convMessages.innerHTML, /hello/);
    assert.doesNotMatch(convMessages.innerHTML, /chat\.history_load_failed/);
  });

  it("shows a spinner, not the empty state, while the history loads", async () => {
    const { manager, convMessages } = workspaceView();
    const pending = deferred();
    manager.configure({ fetchHistory: () => pending.promise });

    const opening = workspace.loadAndRenderConvMessages("mei", { resumeStream() {} });

    assert.match(convMessages.innerHTML, /common\.loading/);
    assert.doesNotMatch(convMessages.innerHTML, /chat\.messages_empty/);

    pending.resolve(sessionsPage("hello"));
    await opening;
    assert.match(convMessages.innerHTML, /hello/);
  });

  it("still shows the empty state for a history that really is empty", async () => {
    const { manager, convMessages } = workspaceView();
    manager.configure({ fetchHistory: async () => ({ sessions: [], has_more: false, next_before: null }) });

    await workspace.loadAndRenderConvMessages("mei", { resumeStream() {} });

    assert.match(convMessages.innerHTML, /chat\.messages_empty/);
  });

  it("keeps messages and offers a retry when an older page fails", async () => {
    const { manager, convMessages } = workspaceView();
    let fetcher = async () => sessionsPage("recent", { hasMore: true, nextBefore: "2026-11-01T10:00:00+09:00" });
    manager.configure({ fetchHistory: (...args) => fetcher(...args) });
    await workspace.loadAndRenderConvMessages("mei", { resumeStream() {} });

    fetcher = async () => { throw new Error("502"); };
    await workspace.loadMoreHistory();

    assert.match(convMessages.innerHTML, /recent/);
    assert.match(convMessages.innerHTML, /chat\.history_older_failed/);
    assert.doesNotMatch(convMessages.innerHTML, /chat-load-sentinel/);

    fetcher = async () => sessionsPage("older");
    convMessages.press("data-history-retry");
    await settle();

    assert.match(convMessages.innerHTML, /older/);
    assert.match(convMessages.innerHTML, /recent/);
  });
});

// ── Thread selection under test ──────────────────────

const animaControllerUrl = toModuleUrl(
  `import { hasHistoryContent } from "${historyUrl}";
   import { resolveActiveThreadId } from "${threadLogicUrl}";
   const stubs = globalThis.__animaStubs;
   const isTabOpen = (ctx, name) => ctx.state.animaTabs.some((tab) => tab.name === name);
   const refreshAnimaUnread = () => {};
   const clearUnreadForActiveThread = () => {};
   const loadDraft = () => "";
   const saveDraft = () => {};
   const chatInputMaxHeight = () => 200;
   const fetchChatUiState = async () => stubs.uiState;
   const scheduleSaveChatUiState = () => { stubs.saves += 1; };
   const mergeThreadsFromSessions = () => {};
   const setThreadUnread = () => {};
   const threadTimeValue = () => 0;
   const syncModelSelect = () => {};
   const CONSTANTS = { HISTORY_PAGE_SIZE: 50 };
   const animaStatusLabel = (status) => status;
   const bustupCandidates = () => [];
   const resolveCachedAvatar = async () => null;
   const companyColor = () => "";
   const openCallScreen = () => {};
   const presence = { takePendingOpen: () => null, setWorking() {}, get: () => null };
   const displayStatus = () => "idle";
   ${stripImports(read("pages/chat/anima-controller.js"))}`,
  "anima-controller",
);
globalThis.__animaStubs = { uiState: null, saves: 0 };
const { createAnimaController } = await import(animaControllerUrl);

const threadControllerUrl = toModuleUrl(
  `import { resolveActiveThreadId, renderThreadTabsHtml, createThread as sharedCreateThread,
     closeThread as sharedCloseThread, restoreThread as sharedRestoreThread } from "${threadLogicUrl}";
   const isTabOpen = () => true;
   const refreshAnimaUnread = () => {};
   const clearUnreadForActiveThread = () => {};
   const setThreadUnread = () => {};
   const threadTimeValue = () => 0;
   const scheduleSaveChatUiState = () => {};
   const saveDraft = () => {};
   const loadDraft = () => "";
   const chatInputMaxHeight = () => 200;
   const CONSTANTS = { HISTORY_PAGE_SIZE: 50, THREAD_VISIBLE_NON_DEFAULT: 5 };
   const syncModelSelect = () => {};
   ${stripImports(read("pages/chat/thread-controller.js"))}`,
  "thread-controller",
);
const { createThreadController } = await import(threadControllerUrl);

function chatControllers({ uiState = null, paneState = null, fetchFeed = async () => feedPage([]) } = {}) {
  ChatSessionManager.resetInstance();
  const manager = ChatSessionManager.getInstance();
  globalThis.__animaStubs.uiState = uiState;
  globalThis.__animaStubs.saves = 0;
  const events = [];
  const paneStates = [];
  const state = {
    manager,
    container: null,
    animas: [],
    users: [],
    selectedAnima: null,
    selectedThreadId: "default",
    threads: {},
    activeThreadByAnima: {},
    animaLastAccess: {},
    animaTabs: [],
    animaTabAvatarUrls: {},
    animaTabAvatarLoading: {},
    modelByThread: {},
    fetchChatFeedHistory: fetchFeed,
    paneIdx: 0,
    paneHost: {
      getPaneState: () => paneState,
      savePaneStates: () => paneStates.push({ anima: state.selectedAnima, threadId: state.selectedThreadId }),
      renderRail() {},
    },
  };
  const ctx = {
    state,
    $: () => null,
    controllers: {
      renderer: {
        renderChat() {
          events.push({
            selected: state.selectedThreadId,
            loading: manager.getHistoryState(state.selectedAnima, state.selectedThreadId).loading,
          });
        },
        fetchConversationHistory: fetchFeed,
      },
      imageVoice: noopController(),
      workIndicator: noopController(),
      streaming: noopController(),
      thread: noopController(),
      events: noopController(),
      memoryPopover: noopController(),
    },
    deps: {
      t: (key) => key,
      escapeHtml,
      logger: { error(...args) { throw new Error(args.join(" ")); } },
      api: async (url) => {
        if (url === "/api/animas") return [{ name: "sakura", status: "idle" }, { name: "rin", status: "idle" }];
        if (url === "/api/users") return [];
        return {};
      },
    },
  };
  ctx.controllers.anima = createAnimaController(ctx);
  return { ctx, state, manager, events, paneStates };
}

const savedState = (activeThreadId, archivedId = "e2628517") => ({
  version: 1,
  active_anima: "sakura",
  anima_tabs: [{ name: "sakura", unread_star: false }],
  anima_last_access: {},
  thread_state: {
    sakura: {
      active_thread_id: activeThreadId,
      threads: [
        { id: "default", label: "main", unread: false },
        { id: archivedId, label: "old", unread: false, archived: true },
        { id: "fresh", label: "fresh", unread: false },
      ],
    },
  },
});

describe("archived thread remembered as the active one", () => {
  it("opens the default thread for state already saved that way, and saves the repair", async () => {
    const { ctx, state, manager } = chatControllers({ uiState: savedState("e2628517") });

    await ctx.controllers.anima.loadAnimas();
    await settle();

    assert.equal(state.selectedAnima, "sakura");
    assert.equal(state.selectedThreadId, "default");
    assert.equal(state.activeThreadByAnima.sakura, "default");
    assert.ok(globalThis.__animaStubs.saves > 0, "the repaired selection is written back");
    const archived = state.threads.sakura.find((thread) => thread.id === "e2628517");
    assert.equal(archived.archived, true, "the thread stays in the archive, restorable");
    assert.equal(manager.hasSession("sakura", "e2628517"), false, "the archived thread is not even loaded");
  });

  it("opens the default thread when pane state still points at an archived thread", async () => {
    const { ctx, state } = chatControllers({
      uiState: savedState("default"),
      paneState: { anima: "sakura", threadId: "e2628517" },
    });

    await ctx.controllers.anima.loadAnimas();
    await settle();

    assert.equal(state.selectedThreadId, "default");
    assert.equal(state.activeThreadByAnima.sakura, "default");
  });

  it("repairs animas in other tabs too", () => {
    const { ctx, state } = chatControllers();
    state.animas = [{ name: "sakura" }, { name: "rin" }];
    const saved = savedState("default");
    saved.anima_tabs.push({ name: "rin", unread_star: false });
    saved.thread_state.rin = {
      active_thread_id: "stale",
      threads: [{ id: "default", label: "main" }, { id: "stale", label: "old", archived: true }],
    };

    ctx.controllers.anima.restoreChatUiState(saved);

    assert.equal(state.activeThreadByAnima.rin, "default");
  });

  it("keeps a live thread selected across a reload", async () => {
    const { ctx, state } = chatControllers({
      uiState: savedState("fresh"),
      paneState: { anima: "sakura", threadId: "fresh" },
    });

    await ctx.controllers.anima.loadAnimas();
    await settle();

    assert.equal(state.selectedThreadId, "fresh");
  });

  it("closing the open thread updates pane state, so a reload does not reopen it", async () => {
    const { ctx, state, paneStates } = chatControllers({ uiState: savedState("fresh") });
    await ctx.controllers.anima.loadAnimas();
    await settle();
    assert.equal(state.selectedThreadId, "fresh");
    const thread = createThreadController(ctx);
    ctx.controllers.thread = thread;
    ctx.controllers.anima.refreshSessions = async () => null;
    paneStates.length = 0;

    thread.closeThread("fresh");

    assert.equal(state.selectedThreadId, "default");
    assert.equal(state.threads.sakura.find((entry) => entry.id === "fresh").archived, true);
    assert.deepEqual(paneStates.at(-1), { anima: "sakura", threadId: "default" });
  });
});

describe("selecting an Anima loads its feed", () => {
  it("shows the fetched items right away and keeps a spinner up meanwhile", async () => {
    const pending = deferred();
    const { ctx, state, manager, events } = chatControllers({
      uiState: savedState("default"),
      fetchFeed: () => pending.promise,
    });
    state.animas = [{ name: "sakura", status: "idle" }];
    ctx.controllers.anima.restoreChatUiState(savedState("default"));

    const selecting = ctx.controllers.anima.selectAnima("sakura");

    assert.deepEqual(events[0], { selected: "default", loading: true }, "no empty state during the fetch");

    pending.resolve(feedPage([messageItem("2026-11-01T10:00:00+09:00", "hello")]));
    await selecting;

    const hs = manager.getHistoryState("sakura", "default");
    assert.equal(hs.isFeed, true);
    assert.equal(hs.loaded, true);
    assert.equal(hs.feedItems.length, 1, "feed data is applied, not discarded until the next poll");
  });

  it("reports a failed fetch instead of a confirmed empty history", async () => {
    const { ctx, state, manager } = chatControllers({
      uiState: savedState("default"),
      fetchFeed: async () => { throw new Error("502"); },
    });
    state.animas = [{ name: "sakura", status: "idle" }];
    ctx.controllers.anima.restoreChatUiState(savedState("default"));

    await ctx.controllers.anima.selectAnima("sakura");

    const hs = manager.getHistoryState("sakura", "default");
    assert.equal(hs.loaded, false);
    assert.equal(hs.loadFailed, true);
  });

  it("does not refetch a thread whose background-only feed is already loaded", async () => {
    let fetches = 0;
    const { ctx, state, manager } = chatControllers({
      uiState: savedState("default"),
      fetchFeed: async () => { fetches += 1; return feedPage([backgroundItem("2026-11-01T10:00:00+09:00")]); },
    });
    state.animas = [{ name: "sakura", status: "idle" }];
    ctx.controllers.anima.restoreChatUiState(savedState("default"));
    await manager.loadHistory("sakura", "default", 50, state.fetchChatFeedHistory);

    await ctx.controllers.anima.selectAnima("sakura");

    assert.equal(fetches, 1);
  });
});

describe("selecting a thread loads its feed", () => {
  it("keeps a spinner up during the fetch rather than the empty state", async () => {
    const pending = deferred();
    const { ctx, state, manager, events } = chatControllers({
      uiState: savedState("default"),
      fetchFeed: () => pending.promise,
    });
    state.animas = [{ name: "sakura", status: "idle" }];
    ctx.controllers.anima.restoreChatUiState(savedState("default"));
    state.selectedAnima = "sakura";
    ctx.controllers.anima.refreshSessions = async () => null;
    const thread = createThreadController(ctx);
    ctx.controllers.thread = thread;
    ctx.$ = () => null;

    const selecting = thread.selectThread("fresh");

    assert.deepEqual(events[0], { selected: "fresh", loading: true });

    pending.resolve(feedPage([messageItem("2026-11-01T10:00:00+09:00", "hello")]));
    await selecting;
    assert.equal(manager.getHistoryState("sakura", "fresh").feedItems.length, 1);
  });
});

describe("thread selection logic", () => {
  const { resolveActiveThreadId, autoArchiveStaleThreads } = threadLogic;
  const list = [{ id: "default" }, { id: "live" }, { id: "gone", archived: true }];

  it("falls back to the default thread for archived and unknown threads", () => {
    assert.equal(resolveActiveThreadId(list, "live"), "live");
    assert.equal(resolveActiveThreadId(list, "gone"), "default");
    assert.equal(resolveActiveThreadId(list, "missing"), "default");
    assert.equal(resolveActiveThreadId(list, ""), "default");
    assert.equal(resolveActiveThreadId(list, undefined), "default");
    assert.equal(resolveActiveThreadId(undefined, "live"), "default");
    assert.equal(resolveActiveThreadId(list, "default"), "default");
  });

  it("does not auto-archive a thread open in another pane", () => {
    const stale = new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString();
    const threads = [{ id: "default" }, { id: "open-elsewhere", lastTs: stale }, { id: "idle", lastTs: stale }];

    const { list: out } = autoArchiveStaleThreads(threads, { activeThreadId: "default", openThreadIds: ["open-elsewhere"] });

    assert.ok(!out.find((thread) => thread.id === "open-elsewhere").archived);
    assert.equal(out.find((thread) => thread.id === "idle").archived, true);
  });
});

describe("history state strings", () => {
  const keys = [
    "chat.history_load_failed",
    "chat.history_older_failed",
    "chat.history_retry",
    "chat.feed.compressed_summary",
    "chat.feed.load_older",
    "chat.feed.no_conversation",
    "chat.feed.no_recent_conversation",
  ];

  it("exist in every language file, translated", () => {
    const locales = ["ja", "en", "ko"].map((locale) => JSON.parse(read(`i18n/${locale}.json`)));
    for (const key of keys) {
      const values = locales.map((data) => data[key]);
      assert.ok(values.every(Boolean), `${key} is missing from a language file`);
      assert.equal(new Set(values).size, values.length, `${key} is not translated in every language`);
    }
  });
});
