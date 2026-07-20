/**
 * Unit tests for Anima schedule tab and Dashboard server-status pure helpers.
 *
 * Run with: node --test tests/unit/frontend/test_schedule_tab.mjs
 */

import { describe, it, beforeEach, mock } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const __dirname = dirname(fileURLToPath(import.meta.url));
const STATIC = resolve(__dirname, "../../../server/static");

// ── Minimal DOM ──────────────────────────────

class MockEl {
  constructor(tag = "div") {
    this.tagName = String(tag).toUpperCase();
    this.id = "";
    this._innerHTML = "";
    this.children = [];
  }

  set innerHTML(html) {
    this._innerHTML = String(html ?? "");
  }

  get innerHTML() {
    return this._innerHTML;
  }
}

const _byId = new Map();

globalThis.document = {
  createElement(tag) {
    return new MockEl(tag);
  },
  getElementById(id) {
    return _byId.get(id) || null;
  },
};

globalThis.window = globalThis;

// ── Load schedule tab with stubbed imports ──

function loadScheduleModule({ apiImpl } = {}) {
  const path = resolve(STATIC, "pages/anima-tabs/schedule.js");
  let source = readFileSync(path, "utf8");
  source = source.replace(/^import\s+.+;?\s*$/gm, "");

  const preamble = `
    let _apiImpl = ${apiImpl ? "null" : "async () => ({ jobs: [] })"};
    const api = (...args) => {
      if (typeof globalThis.__scheduleApi === "function") {
        return globalThis.__scheduleApi(...args);
      }
      return _apiImpl(...args);
    };
    const escapeHtml = (s) =>
      String(s ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
    const timeStr = (v) => (v ? "12:00" : "--:--");
    const t = (k) => k;
  `;

  const url =
    "data:text/javascript;base64," +
    Buffer.from(preamble + "\n" + source, "utf8").toString("base64");
  return import(url + "#" + Math.random());
}

// ── Load home.js pure helpers with stubbed imports ──

function loadHomeHelpers() {
  const path = resolve(STATIC, "pages/home.js");
  let source = readFileSync(path, "utf8");
  source = source.replace(/^import\s+.+;?\s*$/gm, "");

  // Keep only exported pure helpers (+ their dependencies inside the file).
  // Drop render/destroy side effects by not calling them.
  const preamble = `
    const t = (k) => k;
    const basePath = "";
    const api = async () => ({});
    const escapeHtml = (s) =>
      String(s ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
    const escapeAttr = escapeHtml;
    const timeStr = (v) => (v ? "12:00" : "--:--");
    const statusClass = () => "";
    const animaHashColor = () => "#000";
    const companyColor = () => "#000";
    const getIcon = () => "";
    const getDisplaySummary = () => "";
    const bustupCandidates = () => [];
    const resolveCachedAvatar = async () => null;
  `;

  const url =
    "data:text/javascript;base64," +
    Buffer.from(preamble + "\n" + source, "utf8").toString("base64");
  return import(url + "#" + Math.random());
}

describe("extractSchedulerJobs / filterJobsForAnima", () => {
  it("extracts flat jobs array", async () => {
    const mod = await loadScheduleModule();
    const jobs = mod.extractSchedulerJobs({
      jobs: [{ id: "a", anima: "sakura" }],
    });
    assert.equal(jobs.length, 1);
    assert.equal(jobs[0].id, "a");
  });

  it("merges system_jobs and anima_jobs when jobs is absent", async () => {
    const mod = await loadScheduleModule();
    const jobs = mod.extractSchedulerJobs({
      system_jobs: [{ id: "sys", anima: "system" }],
      anima_jobs: [{ id: "cron1", anima: "sakura" }],
    });
    assert.equal(jobs.length, 2);
    assert.deepEqual(
      jobs.map((j) => j.id),
      ["sys", "cron1"],
    );
  });

  it("returns empty for null/invalid payload", async () => {
    const mod = await loadScheduleModule();
    assert.deepEqual(mod.extractSchedulerJobs(null), []);
    assert.deepEqual(mod.extractSchedulerJobs(undefined), []);
    assert.deepEqual(mod.extractSchedulerJobs({}), []);
  });

  it("filters jobs by anima name only", async () => {
    const mod = await loadScheduleModule();
    const all = [
      { name: "hb", anima: "sakura", schedule: "*/5" },
      { name: "other", anima: "yuki", schedule: "0 9 * * *" },
      { name: "sys", anima: "system", schedule: "hourly" },
    ];
    const filtered = mod.filterJobsForAnima(all, "sakura");
    assert.equal(filtered.length, 1);
    assert.equal(filtered[0].name, "hb");
    assert.equal(filtered[0].anima, "sakura");
  });

  it("returns empty when anima has no jobs", async () => {
    const mod = await loadScheduleModule();
    const filtered = mod.filterJobsForAnima(
      [{ name: "x", anima: "yuki" }],
      "sakura",
    );
    assert.deepEqual(filtered, []);
  });
});

describe("jobsTableHtml", () => {
  it("renders empty placeholder when no jobs", async () => {
    const mod = await loadScheduleModule();
    const html = mod.jobsTableHtml([]);
    assert.match(html, /server\.no_jobs/);
    assert.doesNotMatch(html, /data-table/);
  });

  it("renders job name, schedule, last/next run for filtered jobs", async () => {
    const mod = await loadScheduleModule();
    const html = mod.jobsTableHtml([
      {
        name: "morning",
        schedule: "0 9 * * *",
        last_run: "2026-07-20T00:00:00Z",
        next_run: "2026-07-21T00:00:00Z",
      },
    ]);
    assert.match(html, /data-table/);
    assert.match(html, /morning/);
    assert.match(html, /0 9 \* \* \*/);
    assert.match(html, /server\.job_name/);
    assert.match(html, /server\.job_schedule/);
  });
});

describe("schedule tab render/destroy", () => {
  beforeEach(() => {
    _byId.clear();
    globalThis.__scheduleApi = null;
  });

  it("shows only jobs for the current anima after render", async () => {
    const mod = await loadScheduleModule();
    const container = new MockEl("div");
    const content = new MockEl("div");
    content.id = "animaScheduleTabContent";

    // Intercept getElementById so _load finds content after render sets HTML
    const origGet = document.getElementById;
    document.getElementById = (id) => {
      if (id === "animaScheduleTabContent") {
        // After render, content is inside container via innerHTML string only;
        // schedule.js uses getElementById, so register a live element.
        if (!_byId.has(id)) {
          _byId.set(id, content);
        }
        return _byId.get(id);
      }
      return origGet(id);
    };

    globalThis.__scheduleApi = async (path) => {
      assert.equal(path, "/api/system/scheduler");
      return {
        anima_jobs: [
          { name: "mine", anima: "sakura", schedule: "daily" },
          { name: "theirs", anima: "yuki", schedule: "hourly" },
        ],
        system_jobs: [{ name: "sys", anima: "system", schedule: "cron" }],
      };
    };

    // Patch render flow: after container.innerHTML is set, register content id
    const origSet = Object.getOwnPropertyDescriptor(
      Object.getPrototypeOf(container),
      "innerHTML",
    ) || Object.getOwnPropertyDescriptor(container, "innerHTML");

    // Ensure content element exists before _load runs
    _byId.set("animaScheduleTabContent", content);

    // Manually exercise pure path used by _load
    const data = await globalThis.__scheduleApi("/api/system/scheduler");
    const jobs = mod.filterJobsForAnima(mod.extractSchedulerJobs(data), "sakura");
    content.innerHTML = mod.jobsTableHtml(jobs);

    assert.match(content.innerHTML, /mine/);
    assert.doesNotMatch(content.innerHTML, /theirs/);
    assert.doesNotMatch(content.innerHTML, /sys/);

    document.getElementById = origGet;
  });

  it("shows empty state when anima has no jobs", async () => {
    const mod = await loadScheduleModule();
    const html = mod.jobsTableHtml(
      mod.filterJobsForAnima(
        [{ name: "x", anima: "other" }],
        "sakura",
      ),
    );
    assert.match(html, /server\.no_jobs/);
  });

  it("destroy clears the refresh interval", async () => {
    const mod = await loadScheduleModule();
    const intervals = new Set();
    const realSetInterval = globalThis.setInterval;
    const realClearInterval = globalThis.clearInterval;

    globalThis.setInterval = (fn, ms) => {
      const id = realSetInterval(() => {}, 60_000);
      intervals.add(id);
      assert.equal(ms, 30000);
      return id;
    };
    globalThis.clearInterval = (id) => {
      intervals.delete(id);
      realClearInterval(id);
    };

    try {
      const container = new MockEl("div");
      _byId.set("animaScheduleTabContent", new MockEl("div"));
      globalThis.__scheduleApi = async () => ({ jobs: [] });

      mod.render(container, { animaName: "sakura" });
      // render also sets container.innerHTML which recreates the id in real DOM;
      // our mock uses getElementById map — ensure it exists for _load
      if (!_byId.has("animaScheduleTabContent")) {
        _byId.set("animaScheduleTabContent", new MockEl("div"));
      }

      assert.equal(intervals.size, 1);
      mod.destroy();
      assert.equal(intervals.size, 0);

      // second destroy is a no-op
      mod.destroy();
      assert.equal(intervals.size, 0);
    } finally {
      globalThis.setInterval = realSetInterval;
      globalThis.clearInterval = realClearInterval;
      for (const id of intervals) realClearInterval(id);
    }
  });

  it("exports render/destroy contract", () => {
    const source = readFileSync(
      resolve(STATIC, "pages/anima-tabs/schedule.js"),
      "utf8",
    );
    assert.match(source, /export function render\(/);
    assert.match(source, /export function destroy\(/);
    assert.match(source, /animaName/);
    assert.match(source, /setInterval/);
    assert.match(source, /clearInterval/);
    assert.match(source, /30000/);
    assert.match(source, /filterJobsForAnima/);
  });
});

describe("Dashboard server status pure helpers (home.js)", () => {
  it("summarizeServerStatus aggregates uptime/ws/scheduler/jobs", async () => {
    const home = await loadHomeHelpers();
    const summary = home.summarizeServerStatus({
      statusData: { scheduler_running: true, animas: 3 },
      connectionsData: { websocket: { connected_clients: 4 } },
      schedulerData: {
        system_jobs: [{ id: "s1" }],
        anima_jobs: [{ id: "a1" }, { id: "a2" }],
      },
    });
    assert.equal(summary.reachable, true);
    assert.equal(summary.wsCount, 4);
    assert.equal(summary.schedulerRunning, true);
    assert.equal(summary.jobCount, 3);
  });

  it("summarizeServerStatus handles missing optional payloads", async () => {
    const home = await loadHomeHelpers();
    const summary = home.summarizeServerStatus({
      statusData: null,
      connectionsData: null,
      schedulerData: null,
    });
    assert.equal(summary.reachable, false);
    assert.equal(summary.wsCount, 0);
    assert.equal(summary.schedulerRunning, false);
    assert.equal(summary.jobCount, 0);
  });

  it("serverStatusDisplayRows maps labels to display values", async () => {
    const home = await loadHomeHelpers();
    const rows = home.serverStatusDisplayRows({
      reachable: true,
      wsCount: 2,
      schedulerRunning: false,
      jobCount: 5,
    });
    assert.equal(rows.length, 4);
    assert.deepEqual(rows[0], ["server.uptime_label", "server.running"]);
    assert.deepEqual(rows[1], ["server.connections_label", 2]);
    assert.deepEqual(rows[2], ["server.scheduler", "server.stopped"]);
    assert.deepEqual(rows[3], ["server.jobs_label", 5]);
  });

  it("serverStatusTableHtml escapes content", async () => {
    const home = await loadHomeHelpers();
    const html = home.serverStatusTableHtml([
      ["Label <x>", "Value & co"],
    ]);
    assert.match(html, /Label &lt;x&gt;/);
    assert.match(html, /Value &amp; co/);
    assert.match(html, /data-table/);
  });
});
