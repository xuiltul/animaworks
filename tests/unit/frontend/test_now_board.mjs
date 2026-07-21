import { describe, it } from "node:test";
import assert from "node:assert/strict";

import {
  deriveNowStatus,
  renderNowBoard,
} from "../../../server/static/pages/activity/now-board.js";

class MockClassList {
  constructor(element) {
    this.element = element;
    this.values = new Set();
  }

  add(...names) {
    for (const name of names) this.values.add(name);
  }

  contains(name) {
    return this.values.has(name) || this.element.className.split(/\s+/).includes(name);
  }
}

class MockElement {
  constructor(tagName) {
    this.tagName = tagName.toUpperCase();
    this.children = [];
    this.className = "";
    this.classList = new MockClassList(this);
    this.dataset = {};
    this.textContent = "";
    this.title = "";
    this.attributes = {};
  }

  appendChild(child) {
    child.parentElement = this;
    this.children.push(child);
    return child;
  }

  replaceChildren(...children) {
    this.children = [...children];
    for (const child of children) child.parentElement = this;
  }

  setAttribute(name, value) {
    this.attributes[name] = String(value);
  }
}

class MockDocument {
  constructor() {
    this.listeners = new Map();
  }

  createElement(tagName) {
    return new MockElement(tagName);
  }

  addEventListener(type, handler) {
    if (!this.listeners.has(type)) this.listeners.set(type, new Set());
    this.listeners.get(type).add(handler);
  }

  removeEventListener(type, handler) {
    this.listeners.get(type)?.delete(handler);
  }

  dispatchEvent(event) {
    for (const handler of this.listeners.get(event.type) || []) handler(event);
  }
}

function findAll(root, className) {
  const matches = [];
  const visit = (element) => {
    if (element.classList?.contains(className)) matches.push(element);
    for (const child of element.children || []) visit(child);
  };
  visit(root);
  return matches;
}

function translate(key, params = {}) {
  const values = {
    "activity.now_chat": "In conversation",
    "activity.now_cron": "Running cron",
    "activity.now_dropped": `+${params.count} omitted`,
    "activity.now_error": "Error",
    "activity.now_idle": "Idle",
    "activity.now_last_activity": `Last active ${params.time}`,
    "activity.now_live": "Live",
    "activity.now_no_activity": "No activity yet",
    "activity.now_no_tools": "Waiting for tool events",
    "activity.now_task": "Running task",
    "activity.now_title": "Now",
  };
  return values[key] || key;
}

describe("Now board status composition", () => {
  it("expires a leaked thinking state after 120 seconds", () => {
    const card = {
      runtimeStatus: "thinking",
      statusUpdatedAt: 1_000,
      lastActivityAt: 1_000,
      transientStatus: "",
      runningTasks: [],
    };
    assert.equal(deriveNowStatus(card, 120_999), "chat");
    assert.equal(deriveNowStatus(card, 121_001), "idle");
  });

  it("prioritizes errors and running tasks", () => {
    const base = { statusUpdatedAt: 0, lastActivityAt: 0, transientStatus: "", runningTasks: [{}] };
    assert.equal(deriveNowStatus({ ...base, runtimeStatus: "idle" }, 10), "task");
    assert.equal(deriveNowStatus({ ...base, runtimeStatus: "error" }, 10), "error");
  });
});

describe("Now board live event flow", () => {
  it("updates a card from status and tool CustomEvents", () => {
    const documentRef = new MockDocument();
    const container = new MockElement("section");
    let currentTime = Date.parse("2026-07-21T10:00:00Z");
    let statusCallback = null;
    let unsubscribed = false;
    const board = renderNowBoard(container, {
      documentRef,
      now: () => currentTime,
      onEvent(type, callback) {
        assert.equal(type, "anima.status");
        statusCallback = callback;
        return () => { unsubscribed = true; };
      },
      smartTimestamp: (value) => value.slice(11, 16),
      t: translate,
    });

    board.applySnapshot(
      [{ name: "alice", status: "idle" }, { name: "bob", status: "idle" }],
      { animas: [], total: 0 },
      { groups: [] },
    );
    assert.equal(findAll(container, "now-card").length, 2);
    assert.equal(findAll(container, "now-card--compact").length, 2);

    statusCallback({ name: "alice", status: "thinking" });
    const alice = findAll(container, "now-card").find((card) => card.dataset.anima === "alice");
    assert.equal(alice.dataset.status, "chat");

    documentRef.dispatchEvent({
      type: "anima-tool-activity",
      detail: {
        name: "alice",
        kind: "tool_result",
        tool: "Bash",
        summary: "command failed",
        is_error: true,
        dropped: 4,
        ctx: "task:build",
        ts: "2026-07-21T10:00:01Z",
      },
    });
    assert.equal(findAll(container, "now-ticker-row").length, 2);
    assert.equal(findAll(container, "now-ticker-result")[0].textContent, "✗");
    assert.equal(findAll(container, "now-ticker-summary")[1].textContent, "+4 omitted");

    currentTime += 121_000;
    statusCallback({ name: "alice", status: "idle" });
    const idleAlice = findAll(container, "now-card").find((card) => card.dataset.anima === "alice");
    assert.equal(idleAlice.dataset.status, "idle");
    assert.equal(idleAlice.classList.contains("now-card--compact"), true);

    board.destroy();
    assert.equal(unsubscribed, true);
    assert.equal(documentRef.listeners.get("anima-tool-activity").size, 0);
  });

  it("restores task and recent ticker state from polling", () => {
    const documentRef = new MockDocument();
    const container = new MockElement("section");
    const board = renderNowBoard(container, { documentRef, t: translate });
    board.applySnapshot(
      [{ name: "worker", status: "idle" }],
      { animas: [{ name: "worker", tasks: [{ task_id: "t1", title: "Build report" }] }] },
      {
        groups: [{
          anima: "worker",
          type: "task_exec",
          is_open: true,
          end_ts: "2026-07-21T10:00:00Z",
          events: [{
            id: "event-1",
            type: "tool_use",
            tool: "Read",
            summary: "input.md",
            ts: "2026-07-21T10:00:00Z",
          }],
        }],
      },
    );

    const card = findAll(container, "now-card")[0];
    assert.equal(card.dataset.status, "task");
    assert.equal(findAll(container, "now-ticker-tool")[0].textContent, "Read");
    board.destroy();
  });

  it("keeps five tool rows when an omitted-events row is shown", () => {
    const documentRef = new MockDocument();
    const container = new MockElement("section");
    const board = renderNowBoard(container, { documentRef, t: translate });

    for (let index = 0; index < 4; index += 1) {
      board.handleToolActivity({
        id: `event-${index}`,
        name: "worker",
        kind: "tool_use",
        tool: `Tool${index}`,
        ts: `2026-07-21T10:00:0${index}Z`,
      });
    }
    board.handleToolActivity({
      id: "event-latest",
      name: "worker",
      kind: "tool_result",
      tool: "LatestTool",
      dropped: 3,
      ts: "2026-07-21T10:00:05Z",
    });

    assert.equal(findAll(container, "now-ticker-row").length, 6);
    assert.equal(findAll(container, "now-ticker-row--dropped").length, 1);
    assert.equal(findAll(container, "now-ticker-summary")[1].textContent, "+3 omitted");
    board.destroy();
  });
});
