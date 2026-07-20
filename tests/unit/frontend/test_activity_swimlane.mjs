/** Unit tests for activity swimlane pure layout helpers.
 * Run with: node --test tests/unit/frontend/test_activity_swimlane.mjs
 */

import { describe, it } from "node:test";
import assert from "node:assert/strict";

import {
  parseTs,
  isGroupInProgress,
  groupHasError,
  isAmbientType,
  barHeightForType,
  barOpacityForType,
  assignOverlapRows,
  buildLanes,
  computeBarGeometry,
  createTimeScale,
} from "../../../server/static/pages/activity/swimlane-layout.js";

describe("isGroupInProgress", () => {
  const now = Date.parse("2026-07-21T12:00:00Z");

  it("returns false when is_open is false", () => {
    assert.equal(
      isGroupInProgress({ is_open: false, end_ts: "2026-07-21T11:59:00Z" }, now),
      false,
    );
  });

  it("returns true when open and end_ts within 5 minutes", () => {
    assert.equal(
      isGroupInProgress({ is_open: true, end_ts: "2026-07-21T11:56:00Z" }, now),
      true,
    );
  });

  it("returns false for cron-like open residue older than 5 minutes", () => {
    // cron groups stay is_open forever — must not appear as in-progress
    assert.equal(
      isGroupInProgress(
        {
          type: "cron",
          is_open: true,
          start_ts: "2026-07-21T10:00:00Z",
          end_ts: "2026-07-21T10:00:30Z",
        },
        now,
      ),
      false,
    );
  });

  it("returns false at exactly beyond 5 minute window", () => {
    assert.equal(
      isGroupInProgress({ is_open: true, end_ts: "2026-07-21T11:54:59Z" }, now),
      false,
    );
  });
});

describe("groupHasError", () => {
  it("detects type=error events", () => {
    assert.equal(
      groupHasError({ events: [{ type: "tool_use" }, { type: "error" }] }),
      true,
    );
  });

  it("detects tool_result.is_error", () => {
    assert.equal(
      groupHasError({
        events: [{ type: "tool_use", tool_result: { is_error: true, content: "fail" } }],
      }),
      true,
    );
  });

  it("returns false without errors", () => {
    assert.equal(
      groupHasError({
        events: [{ type: "tool_use", tool_result: { is_error: false } }, { type: "response_sent" }],
      }),
      false,
    );
  });
});

describe("ambient vs signal visual tokens", () => {
  it("cron/heartbeat are ambient (short + dim)", () => {
    assert.equal(isAmbientType("cron"), true);
    assert.equal(isAmbientType("heartbeat"), true);
    assert.equal(barHeightForType("cron"), 8);
    assert.equal(barOpacityForType("heartbeat"), 0.35);
  });

  it("chat/task_exec are signal (tall + opaque)", () => {
    assert.equal(isAmbientType("chat"), false);
    assert.equal(barHeightForType("chat"), 18);
    assert.equal(barOpacityForType("task_exec"), 1.0);
  });
});

describe("assignOverlapRows", () => {
  it("places non-overlapping groups on row 0", () => {
    const groups = [
      { id: "a", start_ts: "2026-07-21T10:00:00Z", end_ts: "2026-07-21T10:05:00Z" },
      { id: "b", start_ts: "2026-07-21T10:10:00Z", end_ts: "2026-07-21T10:15:00Z" },
    ];
    const map = assignOverlapRows(groups);
    assert.equal(map.get("a").row, 0);
    assert.equal(map.get("b").row, 0);
    assert.equal(map.get("a").overflowCount, 0);
  });

  it("splits two overlapping groups onto rows 0 and 1", () => {
    const groups = [
      { id: "a", start_ts: "2026-07-21T10:00:00Z", end_ts: "2026-07-21T10:20:00Z" },
      { id: "b", start_ts: "2026-07-21T10:05:00Z", end_ts: "2026-07-21T10:15:00Z" },
    ];
    const map = assignOverlapRows(groups);
    assert.equal(map.get("a").row, 0);
    assert.equal(map.get("b").row, 1);
  });

  it("marks overflow when more than 2 concurrent groups", () => {
    const groups = [
      { id: "a", start_ts: "2026-07-21T10:00:00Z", end_ts: "2026-07-21T10:30:00Z" },
      { id: "b", start_ts: "2026-07-21T10:05:00Z", end_ts: "2026-07-21T10:25:00Z" },
      { id: "c", start_ts: "2026-07-21T10:10:00Z", end_ts: "2026-07-21T10:20:00Z" },
    ];
    const map = assignOverlapRows(groups);
    // c overlaps both a and b → concurrent size 2 → overflowCount = 1
    assert.ok(map.get("c").overflowCount >= 1);
    assert.ok([0, 1].includes(map.get("c").row));
  });
});

describe("buildLanes", () => {
  const now = Date.parse("2026-07-21T12:00:00Z");

  it("orders by recent-hour event count descending", () => {
    const groups = [
      {
        id: "1",
        anima: "beta",
        start_ts: "2026-07-21T11:50:00Z",
        end_ts: "2026-07-21T11:51:00Z",
        event_count: 1,
        events: [{}],
      },
      {
        id: "2",
        anima: "alpha",
        start_ts: "2026-07-21T11:40:00Z",
        end_ts: "2026-07-21T11:55:00Z",
        event_count: 10,
        events: new Array(10).fill({}),
      },
    ];
    const lanes = buildLanes(groups, { nowMs: now });
    assert.equal(lanes[0].anima, "alpha");
    assert.equal(lanes[1].anima, "beta");
  });

  it("puts empty animas at the bottom with half height", () => {
    const groups = [
      {
        id: "1",
        anima: "active",
        start_ts: "2026-07-21T11:50:00Z",
        end_ts: "2026-07-21T11:51:00Z",
        event_count: 1,
      },
    ];
    const lanes = buildLanes(groups, { nowMs: now, animaOrder: ["active", "idle"] });
    assert.equal(lanes[0].anima, "active");
    assert.equal(lanes[0].height, 28);
    assert.equal(lanes[1].anima, "idle");
    assert.equal(lanes[1].isEmpty, true);
    assert.equal(lanes[1].height, 14);
  });
});

describe("computeBarGeometry", () => {
  const now = Date.parse("2026-07-21T12:00:00Z");
  const windowStart = Date.parse("2026-07-21T06:00:00Z");
  const windowEnd = now;
  const scale = createTimeScale(windowStart, windowEnd, 100, 700);

  it("enforces minimum 4px width for instantaneous events", () => {
    const geom = computeBarGeometry(
      {
        start_ts: "2026-07-21T10:00:00Z",
        end_ts: "2026-07-21T10:00:00Z",
        is_open: false,
      },
      scale,
      windowStart,
      windowEnd,
      now,
    );
    assert.ok(geom.width >= 4);
  });

  it("extends in-progress groups to now", () => {
    const geom = computeBarGeometry(
      {
        start_ts: "2026-07-21T11:58:00Z",
        end_ts: "2026-07-21T11:59:00Z",
        is_open: true,
      },
      scale,
      windowStart,
      windowEnd,
      now,
    );
    assert.equal(geom.extendsToNow, true);
  });

  it("clips groups that start before the window", () => {
    const geom = computeBarGeometry(
      {
        start_ts: "2026-07-21T04:00:00Z",
        end_ts: "2026-07-21T08:00:00Z",
        is_open: false,
      },
      scale,
      windowStart,
      windowEnd,
      now,
    );
    assert.equal(geom.clipped, true);
    assert.ok(geom.width > 0);
  });
});

describe("parseTs", () => {
  it("parses ISO timestamps", () => {
    assert.ok(Number.isFinite(parseTs("2026-07-21T12:00:00Z")));
  });

  it("returns NaN for empty input", () => {
    assert.equal(Number.isNaN(parseTs("")), true);
    assert.equal(Number.isNaN(parseTs(null)), true);
  });
});
