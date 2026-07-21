/** Behavioral DOM tests for the activity session replay modules. */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { afterEach, describe, it } from "node:test";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(HERE, "../..");
const REPLAY_PATH = resolve(
  REPO_ROOT,
  "server/static/pages/activity/session-replay.js",
);
const GROUP_DETAIL_PATH = resolve(
  REPO_ROOT,
  "server/static/pages/activity/group-detail.js",
);
const ACTIVITY_TYPES_PATH = resolve(
  REPO_ROOT,
  "server/static/shared/activity-types.js",
);

class FakeClassList {
  constructor(element) {
    this.element = element;
  }

  _tokens() {
    return new Set(String(this.element.className || "").split(/\s+/).filter(Boolean));
  }

  _save(tokens) {
    this.element.className = [...tokens].join(" ");
  }

  add(...names) {
    const tokens = this._tokens();
    names.forEach((name) => tokens.add(name));
    this._save(tokens);
  }

  remove(...names) {
    const tokens = this._tokens();
    names.forEach((name) => tokens.delete(name));
    this._save(tokens);
  }

  contains(name) {
    return this._tokens().has(name);
  }

  toggle(name, force) {
    const tokens = this._tokens();
    const enabled = force === undefined ? !tokens.has(name) : Boolean(force);
    if (enabled) tokens.add(name);
    else tokens.delete(name);
    this._save(tokens);
    return enabled;
  }
}

class FakeElement {
  constructor(tagName = "div") {
    this.tagName = tagName.toUpperCase();
    this.children = [];
    this.parentNode = null;
    this.className = "";
    this.classList = new FakeClassList(this);
    this.dataset = {};
    this.attributes = new Map();
    this.listeners = new Map();
    this.hidden = false;
    this.isConnected = true;
    this.scrollTop = 0;
    this.clientHeight = 100;
    this._scrollHeight = null;
    this._innerHTML = "";
    this.textContent = "";
  }

  set innerHTML(value) {
    this._innerHTML = String(value);
    this.children = [];
    if (this._innerHTML.includes('class="replay-status"')) {
      const status = new FakeElement("div");
      status.className = "replay-status";
      const scroll = new FakeElement("div");
      scroll.className = "replay-scroll";
      if (this._innerHTML.includes('class="replay-events"')) {
        const events = new FakeElement("div");
        events.className = "replay-events";
        scroll.appendChild(events);
      }
      this.append(status, scroll);
    }
  }

  get innerHTML() {
    return this._innerHTML;
  }

  get scrollHeight() {
    if (this._scrollHeight !== null) return this._scrollHeight;
    return Math.max(100, this.children.length * 100);
  }

  set scrollHeight(value) {
    this._scrollHeight = value;
  }

  appendChild(child) {
    child.parentNode = this;
    this.children.push(child);
    return child;
  }

  append(...children) {
    children.forEach((child) => this.appendChild(child));
  }

  remove() {
    if (!this.parentNode) return;
    this.parentNode.children = this.parentNode.children.filter((child) => child !== this);
    this.parentNode = null;
  }

  setAttribute(name, value) {
    this.attributes.set(name, String(value));
  }

  getAttribute(name) {
    return this.attributes.get(name) ?? null;
  }

  addEventListener(name, callback) {
    if (!this.listeners.has(name)) this.listeners.set(name, []);
    this.listeners.get(name).push(callback);
  }

  dispatchEvent(event) {
    const value = typeof event === "string" ? { type: event } : event;
    for (const callback of this.listeners.get(value.type) || []) callback(value);
  }

  click() {
    this.dispatchEvent({ type: "click", stopPropagation() {} });
  }

  querySelector(selector) {
    const className = selector.startsWith(".") ? selector.slice(1) : null;
    if (!className) return null;
    for (const child of this.children) {
      if (child.classList.contains(className)) return child;
      const nested = child.querySelector(selector);
      if (nested) return nested;
    }
    return null;
  }

  before() {}

  after() {}
}

function stripImports(source) {
  return source.replace(/^import[\s\S]*?;\s*$/gm, "");
}

async function importWithPrelude(path, prelude) {
  const source = `${prelude}\n${stripImports(readFileSync(path, "utf8"))}`;
  const encoded = Buffer.from(source, "utf8").toString("base64");
  return import(`data:text/javascript;base64,${encoded}`);
}

function findByClass(root, className) {
  if (root.classList.contains(className)) return root;
  for (const child of root.children) {
    const found = findByClass(child, className);
    if (found) return found;
  }
  return null;
}

function installDom() {
  const body = new FakeElement("body");
  globalThis.document = {
    body,
    createElement: (tagName) => new FakeElement(tagName),
  };
  globalThis.window = { lucide: null };
  return body;
}

function makeGroup({ events, isOpen = true } = {}) {
  return {
    id: "grp-alice:2026-07-21T12:00:00Z:chat",
    type: "chat",
    anima: "alice",
    start_ts: "2026-07-21T12:00:00Z",
    end_ts: "2026-07-21T12:00:02Z",
    event_count: events.length,
    is_open: isOpen,
    events,
  };
}

function replayPrelude() {
  return `
const api = (...args) => globalThis.__replayDeps.api(...args);
const escapeHtml = (value) => String(value ?? "");
const renderSafeMarkdown = (value) => "<p>" + String(value ?? "") + "</p>";
const smartTimestamp = (value) => String(value ?? "").slice(11, 19);
const getDisplaySummary = (event) => event.summary || event.content || event.type || "";
const getIcon = (type) => "icon:" + type;
const t = (key) => key;
const isGroupInProgress = (group) => Boolean(group && group.is_open);
`;
}

let activeReplayModule = null;

afterEach(() => {
  activeReplayModule?.destroySessionReplay();
  activeReplayModule = null;
  delete globalThis.__replayDeps;
  delete globalThis.__groupDeps;
  delete globalThis.document;
  delete globalThis.window;
});

describe("session replay behavior", () => {
  it("renders event roles, schedules 5s polling, appends by id, and preserves upward scroll", async () => {
    installDom();
    const timers = [];
    let timerId = 0;
    globalThis.setTimeout = (callback, delay) => {
      const timer = { id: ++timerId, callback, delay, cleared: false };
      timers.push(timer);
      return timer.id;
    };
    globalThis.clearTimeout = (id) => {
      const timer = timers.find((item) => item.id === id);
      if (timer) timer.cleared = true;
    };

    const initialEvents = [
      { id: "m1", ts: "2026-07-21T12:00:00Z", type: "message_received", content: "hello" },
      {
        id: "t1",
        ts: "2026-07-21T12:00:01Z",
        type: "tool_use",
        tool: "read_file",
        meta: { args: { path: "notes.md" } },
        tool_result: {
          ts: "2026-07-21T12:00:02Z",
          content: "done",
          is_error: false,
        },
      },
      {
        id: "r1",
        ts: "2026-07-21T12:00:03Z",
        type: "response_sent",
        content: "finished",
        meta: { thinking_text: "reasoning" },
      },
      { id: "s1", ts: "2026-07-21T12:00:04Z", type: "heartbeat_end", summary: "system" },
    ];
    const appended = {
      id: "s2",
      ts: "2026-07-21T12:00:05Z",
      type: "memory_write",
      summary: "saved",
    };
    const responses = [
      { group: makeGroup({ events: initialEvents }) },
      { group: makeGroup({ events: [...initialEvents, appended] }) },
      { group: makeGroup({ events: [...initialEvents, appended], isOpen: false }) },
    ];
    globalThis.__replayDeps = { api: async () => responses.shift() };
    const replay = await importWithPrelude(REPLAY_PATH, replayPrelude());
    activeReplayModule = replay;

    const container = new FakeElement("section");
    await replay.renderSessionReplay(container, {
      anima: "alice",
      groupId: "grp-alice:2026-07-21T12:00:00Z:chat",
    });

    const eventList = findByClass(container, "replay-events");
    assert.ok(eventList);
    assert.deepEqual(
      eventList.children.map((row) => row.className),
      [
        "replay-message replay-message--incoming",
        "replay-tool-row",
        "replay-message replay-message--outgoing",
        "replay-system-row",
      ],
    );
    assert.ok(findByClass(eventList.children[2], "replay-thinking"));
    assert.equal(timers.length, 1);
    assert.equal(timers[0].delay, 5000);

    const scroll = findByClass(container, "replay-scroll");
    scroll.scrollHeight = 1000;
    scroll.clientHeight = 200;
    scroll.scrollTop = 100;
    scroll.dispatchEvent("scroll");
    await timers.shift().callback();

    assert.equal(eventList.children.length, 5, "only the unseen event is appended");
    assert.equal(eventList.children[4].dataset.eventId, "s2");
    assert.equal(scroll.scrollTop, 100, "upward user scroll is preserved");
    assert.equal(timers.length, 1, "open group schedules the next poll");

    await timers.shift().callback();
    assert.equal(timers.length, 0, "finished group stops polling");
    assert.equal(findByClass(container, "replay-status").classList.contains("is-live"), false);
  });

  it("renders the localized not-found state for a 404", async () => {
    installDom();
    globalThis.__replayDeps = {
      api: async () => {
        throw new Error("API 404: Not Found");
      },
    };
    const replay = await importWithPrelude(REPLAY_PATH, replayPrelude());
    activeReplayModule = replay;
    const container = new FakeElement("section");

    await replay.renderSessionReplay(container, { anima: "alice", groupId: "missing" });

    const scroll = findByClass(container, "replay-scroll");
    assert.match(
      scroll.innerHTML,
      /activity\.replay_not_found/,
      JSON.stringify(
        container.children.map((child) => ({
          className: child.className,
          hidden: child.hidden,
          innerHTML: child.innerHTML,
        })),
      ),
    );
    assert.equal(findByClass(container, "replay-status").hidden, true);
  });
});

describe("group detail behavior", () => {
  it("defaults to conversation and toggles raw events in both directions", async () => {
    installDom();
    const calls = [];
    globalThis.__groupDeps = {
      destroySessionReplay: () => calls.push("destroy"),
      renderSessionReplay: (container, options) => calls.push({ container, options }),
    };
    const prelude = `
const escapeHtml = (value) => String(value ?? "");
const smartTimestamp = (value) => String(value ?? "");
const getIcon = (type) => "icon:" + type;
const getDisplaySummary = (event) => event.summary || event.content || event.type || "";
const createContextBadge = () => null;
const decorateContextElement = () => {};
const getParallelTaskCounts = () => new Map();
const t = (key, vars) => vars && vars.count !== undefined ? key + ":" + vars.count : key;
const isGroupInProgress = () => false;
const destroySessionReplay = globalThis.__groupDeps.destroySessionReplay;
const renderSessionReplay = globalThis.__groupDeps.renderSessionReplay;
`;
    const detail = await importWithPrelude(GROUP_DETAIL_PATH, prelude);
    const container = new FakeElement("section");
    const group = makeGroup({
      events: [
        { id: "m1", type: "message_received", ts: "2026-07-21T12:00:00Z", content: "hi" },
        { id: "r1", type: "response_sent", ts: "2026-07-21T12:00:01Z", content: "ok" },
      ],
      isOpen: false,
    });

    detail.renderGroupDetail(container, group);

    const tabs = findByClass(container, "replay-view-tabs");
    const [conversationButton, rawButton] = tabs.children;
    const replayPanel = findByClass(container, "replay-panel");
    const rawPanel = findByClass(container, "replay-raw-panel");
    assert.equal(conversationButton.getAttribute("aria-selected"), "true");
    assert.equal(replayPanel.hidden, false);
    assert.equal(rawPanel.hidden, true);
    assert.equal(calls[1].options.groupId, group.id);

    rawButton.click();
    assert.equal(rawButton.getAttribute("aria-selected"), "true");
    assert.equal(replayPanel.hidden, true);
    assert.equal(rawPanel.hidden, false);

    conversationButton.click();
    assert.equal(conversationButton.getAttribute("aria-selected"), "true");
    assert.equal(replayPanel.hidden, false);
    assert.equal(rawPanel.hidden, true);
  });
});

describe("activity icon modes", () => {
  it("returns emoji in default mode and lucide markup in realistic mode", async () => {
    const body = installDom();
    const icons = await importWithPrelude(ACTIVITY_TYPES_PATH, "const t = (key) => key;");

    assert.equal(icons.getIcon("chat"), "💬");
    body.classList.add("mode-realistic");
    assert.match(icons.getIcon("chat"), /data-lucide="message-circle"/);
  });
});
