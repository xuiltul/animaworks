/** Unit tests for shared activity execution-context DOM helpers.
 * Run with: node --test tests/unit/frontend/test_activity_context.mjs
 */

import { describe, it } from "node:test";
import assert from "node:assert/strict";

import {
  buildParallelGroups,
  createContextBadge,
  decorateContextElement,
  getContextPresentation,
  getParallelTaskCounts,
  renderRunningTasksStrip,
  updateLiveParallelIndicators,
} from "../../../server/static/shared/activity-context.js";

class MockClassList {
  constructor() {
    this.values = new Set();
  }

  add(...names) {
    for (const name of names) this.values.add(name);
  }

  contains(name) {
    return this.values.has(name);
  }
}

class MockElement {
  constructor(tagName) {
    this.tagName = tagName.toUpperCase();
    this.children = [];
    this.className = "";
    this.classList = new MockClassList();
    this.dataset = {};
    this.hidden = false;
    this.textContent = "";
    this.title = "";
    this.style = {
      values: new Map(),
      setProperty: (key, value) => this.style.values.set(key, value),
    };
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

  querySelector(selector) {
    if (!selector.startsWith(".")) return null;
    const className = selector.slice(1);
    return this.children.find((child) => child.className.split(/\s+/).includes(className)) || null;
  }

  before(sibling) {
    if (!this.parentElement) return;
    const index = this.parentElement.children.indexOf(this);
    sibling.parentElement = this.parentElement;
    this.parentElement.children.splice(index, 0, sibling);
  }

  remove() {
    if (!this.parentElement) return;
    const index = this.parentElement.children.indexOf(this);
    if (index >= 0) this.parentElement.children.splice(index, 1);
    this.parentElement = null;
  }
}

const mockDocument = {
  createElement(tagName) {
    return new MockElement(tagName);
  },
};

function translate(key, params) {
  if (key === "activity.running_tasks") return `Running tasks (${params.count})`;
  if (key === "activity.running_task_slot") return `Slot ${params.slot}`;
  return key;
}

describe("activity context presentation", () => {
  it("assigns a stable colour and short task badge", () => {
    const first = getContextPresentation("task:abcdef1234567890");
    const second = getContextPresentation("task:abcdef1234567890");
    assert.deepEqual(first, second);
    assert.equal(first.isTask, true);
    assert.equal(first.label, "⚙ abcdef1234…");
  });

  it("leaves legacy ctx-less events undecorated", () => {
    const row = new MockElement("div");
    assert.equal(decorateContextElement(row, undefined), null);
    assert.equal(row.classList.contains("has-activity-context"), false);
    assert.equal(createContextBadge("", mockDocument), null);
  });

  it("creates a safe badge using textContent", () => {
    const badge = createContextBadge("task:<unsafe>", mockDocument);
    assert.equal(badge.tagName, "SPAN");
    assert.equal(badge.dataset.activityCtx, "task:<unsafe>");
    assert.equal(badge.textContent, "⚙ <unsafe>");
    assert.ok(badge.style.values.get("--activity-ctx-color"));
  });
});

describe("parallel task detection", () => {
  it("uses task execution boundaries for interleaved activity", () => {
    const events = [
      { type: "task_exec_start", ctx: "task:one", anima: "alpha", ts: "2026-07-13T10:00:00" },
      { type: "task_exec_start", ctx: "task:two", anima: "alpha", ts: "2026-07-13T10:00:10" },
      { type: "tool_use", ctx: "task:one", anima: "alpha", ts: "2026-07-13T10:02:00" },
      { type: "task_exec_end", ctx: "task:two", anima: "alpha", ts: "2026-07-13T10:03:00" },
      { type: "tool_use", ctx: "task:one", anima: "alpha", ts: "2026-07-13T10:05:00" },
    ];
    const counts = getParallelTaskCounts(events);
    assert.equal(counts.get(events[2]), 2);
    assert.equal(counts.get(events[3]), 2);
    assert.equal(counts.get(events[4]), 1);
  });

  it("falls back to a local mixing window without lifecycle events", () => {
    const events = [
      { type: "tool_use", ctx: "task:one", anima: "alpha", ts: "2026-07-13T10:00:00" },
      { type: "tool_use", ctx: "task:two", anima: "alpha", ts: "2026-07-13T10:00:30" },
      { type: "tool_use", ctx: "task:other", anima: "beta", ts: "2026-07-13T10:00:20" },
    ];
    const counts = getParallelTaskCounts(events);
    assert.equal(counts.get(events[0]), 2);
    assert.equal(counts.get(events[1]), 2);
    assert.equal(counts.get(events[2]), 1);
  });

  it("adds indicators to mixed compact-feed rows only", () => {
    const container = new MockElement("div");
    for (const [ctx, time, anima] of [
      ["task:one", 1_000, "alpha"],
      ["task:two", 20_000, "alpha"],
      ["heartbeat", 25_000, "alpha"],
      ["task:three", 22_000, "beta"],
    ]) {
      const row = new MockElement("div");
      row.dataset.activityCtx = ctx;
      row.dataset.activityTs = String(time);
      row.dataset.activityAnima = anima;
      const summary = new MockElement("span");
      summary.className = "activity-summary";
      row.appendChild(summary);
      container.appendChild(row);
    }
    updateLiveParallelIndicators(container, (key, params) => `Parallel ${params.count}`, mockDocument);
    assert.equal(container.children[0].querySelector(".activity-parallel-badge").textContent, "Parallel 2");
    assert.equal(container.children[1].querySelector(".activity-parallel-badge").textContent, "Parallel 2");
    assert.equal(container.children[2].querySelector(".activity-parallel-badge"), null);
    assert.equal(container.children[3].querySelector(".activity-parallel-badge"), null);
  });
});

describe("parallel task lane grouping", () => {
  it("groups only the overlap window and keeps non-task activity flat", () => {
    const events = [
      { type: "task_exec_end", ctx: "task:one", anima: "alpha", ts: "2026-07-13T10:06:00" },
      { type: "task_exec_end", ctx: "task:two", anima: "alpha", ts: "2026-07-13T10:05:00" },
      { type: "tool_use", ctx: "task:one", anima: "alpha", ts: "2026-07-13T10:04:00" },
      { type: "heartbeat", ctx: "heartbeat", anima: "alpha", ts: "2026-07-13T10:03:00" },
      {
        type: "task_exec_start",
        ctx: "task:two",
        anima: "alpha",
        ts: "2026-07-13T10:02:00",
        summary: "Second task summary",
      },
      {
        type: "task_exec_start",
        ctx: "task:one",
        anima: "alpha",
        ts: "2026-07-13T10:00:00",
        meta: { title: "First task" },
      },
      { type: "task_exec_start", ctx: "task:other", anima: "beta", ts: "2026-07-13T10:01:00" },
      { type: "task_exec_end", ctx: "task:other", anima: "beta", ts: "2026-07-13T10:04:00" },
    ];

    const groups = buildParallelGroups(events);
    assert.equal(groups.length, 1);
    assert.equal(groups[0].anima, "alpha");
    assert.deepEqual(groups[0].lanes.map((lane) => lane.taskId), ["one", "two"]);
    assert.deepEqual(groups[0].lanes.map((lane) => lane.title), ["First task", "Second task summary"]);
    assert.deepEqual(groups[0].lanes[0].events, [events[2]]);
    assert.deepEqual(groups[0].lanes[1].events, [events[1], events[4]]);
    assert.deepEqual(groups[0].flatEvents, [events[3]]);
  });

  it("does not group sequential task intervals", () => {
    const events = [
      { type: "task_exec_start", ctx: "task:one", anima: "alpha", ts: "2026-07-13T10:00:00" },
      { type: "task_exec_end", ctx: "task:one", anima: "alpha", ts: "2026-07-13T10:01:00" },
      { type: "task_exec_start", ctx: "task:two", anima: "alpha", ts: "2026-07-13T10:01:00" },
      { type: "task_exec_end", ctx: "task:two", anima: "alpha", ts: "2026-07-13T10:02:00" },
    ];
    assert.deepEqual(buildParallelGroups(events), []);
  });

  it("uses the last task event as the end of an unfinished interval", () => {
    const events = [
      {
        type: "task_exec_start",
        ctx: "task:1234567890",
        anima: "alpha",
        ts: "2026-07-13T10:00:00",
      },
      { type: "task_exec_start", ctx: "task:two", anima: "alpha", ts: "2026-07-13T10:02:00" },
      { type: "task_exec_end", ctx: "task:two", anima: "alpha", ts: "2026-07-13T10:03:00" },
      { type: "tool_use", ctx: "task:1234567890", anima: "alpha", ts: "2026-07-13T10:04:00" },
    ];

    const groups = buildParallelGroups(events);
    assert.equal(groups.length, 1);
    assert.equal(groups[0].lanes[0].title, "task:12345678");
    assert.equal(groups[0].lanes.length, 2);
  });

  it("groups overlapping tasks across a date boundary", () => {
    const events = [
      { type: "task_exec_start", ctx: "task:one", anima: "alpha", ts: "2026-07-13T23:58:00+09:00" },
      { type: "task_exec_start", ctx: "task:two", anima: "alpha", ts: "2026-07-13T23:59:30+09:00" },
      { type: "task_exec_end", ctx: "task:one", anima: "alpha", ts: "2026-07-14T00:01:00+09:00" },
      { type: "task_exec_end", ctx: "task:two", anima: "alpha", ts: "2026-07-14T00:02:00+09:00" },
    ];

    const groups = buildParallelGroups(events);
    assert.equal(groups.length, 1);
    assert.deepEqual(groups[0].lanes.map((lane) => lane.taskId), ["one", "two"]);
  });

  it("keeps a non-parallel gap out of separate overlap windows", () => {
    const events = [
      { type: "task_exec_start", ctx: "task:one", anima: "alpha", ts: "2026-07-13T10:00:00" },
      { type: "task_exec_start", ctx: "task:two", anima: "alpha", ts: "2026-07-13T10:02:00" },
      { type: "task_exec_end", ctx: "task:two", anima: "alpha", ts: "2026-07-13T10:04:00" },
      { type: "tool_use", ctx: "task:one", anima: "alpha", ts: "2026-07-13T10:05:00" },
      { type: "task_exec_start", ctx: "task:three", anima: "alpha", ts: "2026-07-13T10:06:00" },
      { type: "task_exec_end", ctx: "task:three", anima: "alpha", ts: "2026-07-13T10:08:00" },
      { type: "task_exec_end", ctx: "task:one", anima: "alpha", ts: "2026-07-13T10:10:00" },
    ];

    const groups = buildParallelGroups(events);
    assert.equal(groups.length, 2);
    assert.deepEqual(groups[0].lanes.map((lane) => lane.taskId), ["one", "two"]);
    assert.deepEqual(groups[1].lanes.map((lane) => lane.taskId), ["one", "three"]);
    const groupedEvents = groups.flatMap((group) => group.lanes.flatMap((lane) => lane.events));
    assert.equal(groupedEvents.includes(events[3]), false);
  });
});

describe("running task strip DOM generation", () => {
  it("renders slot and title chips then hides cleanly for an empty payload", () => {
    const container = new MockElement("div");
    const payload = {
      animas: [{
        name: "alpha",
        tasks: [
          { slot_id: 0, task_id: "task-a", title: "Compile report" },
          { slot_id: 2, task_id: "task-b", title: "Check tests" },
        ],
      }],
      total: 2,
    };

    const tasks = renderRunningTasksStrip(container, payload, translate, mockDocument);
    assert.equal(tasks.length, 2);
    assert.equal(container.hidden, false);
    assert.equal(container.children[0].textContent, "Running tasks (2)");
    assert.equal(container.children[1].dataset.slotId, "0");
    assert.equal(container.children[1].children[0].textContent, "Slot 0");
    assert.equal(container.children[1].children[1].textContent, "alpha · Compile report");
    assert.equal(container.children[1].classList.contains("has-activity-context"), true);

    renderRunningTasksStrip(container, { animas: [], total: 0 }, translate, mockDocument);
    assert.equal(container.hidden, true);
    assert.equal(container.children.length, 0);
  });
});
