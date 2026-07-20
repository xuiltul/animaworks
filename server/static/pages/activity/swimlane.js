// ── Activity swimlane SVG renderer ──────────────────────────
// Layout pure helpers live in swimlane-layout.js (unit-tested).
// This module owns SVG DOM generation via renderSwimlane().

import {
  GROUP_TYPE_COLORS,
  getDisplaySummary,
  TYPE_ICONS,
} from "../../shared/activity-types.js";
import { t } from "/shared/i18n.js";
import {
  LANE_HEIGHT,
  isGroupInProgress,
  groupHasError,
  barHeightForType,
  barOpacityForType,
  assignOverlapRows,
  buildLanes,
  computeBarGeometry,
  createTimeScale,
} from "./swimlane-layout.js";

// Re-export pure helpers so callers can import from either module
export {
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
} from "./swimlane-layout.js";

const SVG_NS = "http://www.w3.org/2000/svg";
const LABEL_WIDTH = 88;
const AXIS_HEIGHT = 28;
const LABEL_MIN_BAR_PX = 40;

function _emojiForType(type) {
  const entry = TYPE_ICONS[type];
  return entry?.emoji || "📌";
}

function _groupLabel(grp) {
  if (grp.type === "single" && grp.events?.[0]) {
    return getDisplaySummary(grp.events[0]) || grp.summary || grp.type;
  }
  if (grp.type === "heartbeat") return "Heartbeat";
  if (grp.type === "chat") return t("activity.label_user_chat");
  if (grp.type === "dm") return `DM ${grp.summary || ""}`.trim();
  if (grp.type === "cron") return `Cron ${grp.summary || ""}`.trim();
  if (grp.type === "task") return `Task ${grp.summary || ""}`.trim();
  if (grp.type === "inbox") return t("activity.label_inbox");
  if (grp.type === "task_exec") {
    return `${t("activity.label_task_exec")} ${grp.summary || ""}`.trim();
  }
  return grp.summary || grp.type || "";
}

function _formatRange(startTs, endTs) {
  const s = startTs ? String(startTs).slice(11, 16) : "";
  const e = endTs ? String(endTs).slice(11, 16) : "";
  if (!s) return "";
  return s === e ? s : `${s}–${e}`;
}

function _el(tag, attrs = {}, text) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "className") node.setAttribute("class", v);
    else if (k === "textContent") node.textContent = v;
    else node.setAttribute(k, String(v));
  }
  if (text != null) node.textContent = text;
  return node;
}

function _buildScale(windowStartMs, windowEndMs, x0, x1) {
  if (typeof window !== "undefined" && window.d3?.scaleTime) {
    return window.d3.scaleTime()
      .domain([new Date(windowStartMs), new Date(windowEndMs)])
      .range([x0, x1]);
  }
  return createTimeScale(windowStartMs, windowEndMs, x0, x1);
}

function _tickValues(windowStartMs, windowEndMs, hours) {
  let stepMs;
  if (hours <= 6) stepMs = 30 * 60 * 1000;
  else if (hours <= 24) stepMs = 2 * 60 * 60 * 1000;
  else stepMs = 4 * 60 * 60 * 1000;

  const ticks = [];
  const startAligned = Math.ceil(windowStartMs / stepMs) * stepMs;
  for (let t = startAligned; t <= windowEndMs; t += stepMs) {
    ticks.push(t);
  }
  if (ticks.length === 0) ticks.push(windowStartMs, windowEndMs);
  return ticks;
}

function _formatTick(ms, hours) {
  const d = new Date(ms);
  const hh = String(d.getHours()).padStart(2, "0");
  const mm = String(d.getMinutes()).padStart(2, "0");
  if (hours > 24) {
    const mo = String(d.getMonth() + 1).padStart(2, "0");
    const day = String(d.getDate()).padStart(2, "0");
    return `${mo}/${day} ${hh}:${mm}`;
  }
  return `${hh}:${mm}`;
}

/**
 * Render swimlane into an SVG element.
 *
 * @param {SVGElement} svgEl
 * @param {object[]} groups
 * @param {{
 *   timeRange: { startMs: number, endMs: number, hours: number },
 *   nowMs?: number,
 *   selectedGroupId?: string|null,
 *   onSelectGroup?: (group: object|null) => void,
 *   animaOrder?: string[],
 *   width?: number,
 * }} options
 */
export function renderSwimlane(svgEl, groups, options = {}) {
  if (!svgEl) return { lanes: [], height: 0 };

  const nowMs = options.nowMs ?? Date.now();
  const timeRange = options.timeRange || {
    startMs: nowMs - 6 * 3600_000,
    endMs: nowMs,
    hours: 6,
  };
  const windowStartMs = timeRange.startMs;
  const windowEndMs = timeRange.endMs;
  const hours = timeRange.hours || 6;
  const selectedId = options.selectedGroupId || null;
  const onSelect = options.onSelectGroup || (() => {});

  const containerWidth = options.width
    || svgEl.clientWidth
    || svgEl.parentElement?.clientWidth
    || 800;

  const chartLeft = LABEL_WIDTH;
  const chartRight = Math.max(chartLeft + 100, containerWidth - 8);

  const lanes = buildLanes(groups, { nowMs, animaOrder: options.animaOrder });
  let y = AXIS_HEIGHT;
  const laneLayouts = [];
  for (const lane of lanes) {
    laneLayouts.push({ ...lane, y });
    y += lane.height;
  }
  const totalHeight = Math.max(y + 4, AXIS_HEIGHT + LANE_HEIGHT);

  while (svgEl.firstChild) svgEl.removeChild(svgEl.firstChild);
  svgEl.setAttribute("width", String(containerWidth));
  svgEl.setAttribute("height", String(totalHeight));
  svgEl.setAttribute("viewBox", `0 0 ${containerWidth} ${totalHeight}`);
  svgEl.setAttribute("class", "swimlane-svg");
  svgEl.setAttribute("role", "img");
  svgEl.setAttribute("aria-label", t("activity.page_title"));

  const scaleX = _buildScale(windowStartMs, windowEndMs, chartLeft, chartRight);
  const sx = (ms) => {
    const v = scaleX(ms instanceof Date ? ms : new Date(ms));
    return typeof v === "number" ? v : +v;
  };

  const bg = _el("rect", {
    class: "swimlane-bg",
    x: 0,
    y: 0,
    width: containerWidth,
    height: totalHeight,
  });
  svgEl.appendChild(bg);

  svgEl.appendChild(_el("rect", {
    class: "swimlane-axis-bg",
    x: 0,
    y: 0,
    width: containerWidth,
    height: AXIS_HEIGHT,
  }));

  const ticks = _tickValues(windowStartMs, windowEndMs, hours);
  for (const tick of ticks) {
    const x = sx(tick);
    if (x < chartLeft || x > chartRight) continue;
    svgEl.appendChild(_el("line", {
      class: "swimlane-grid-line",
      x1: x,
      y1: AXIS_HEIGHT,
      x2: x,
      y2: totalHeight,
    }));
    svgEl.appendChild(_el("text", {
      class: "swimlane-tick-label",
      x,
      y: AXIS_HEIGHT - 8,
      "text-anchor": "middle",
    }, _formatTick(tick, hours)));
  }

  const nowX = sx(Math.min(Math.max(nowMs, windowStartMs), windowEndMs));
  svgEl.appendChild(_el("line", {
    class: "swimlane-now-line",
    x1: nowX,
    y1: AXIS_HEIGHT - 4,
    x2: nowX,
    y2: totalHeight,
  }));
  svgEl.appendChild(_el("text", {
    class: "swimlane-now-label",
    x: nowX,
    y: 12,
    "text-anchor": "middle",
  }, t("activity.swimlane_now")));

  const isRealistic = typeof document !== "undefined"
    && document.body?.classList?.contains("mode-realistic");

  for (const lane of laneLayouts) {
    svgEl.appendChild(_el("rect", {
      class: `swimlane-lane-bg${lane.isEmpty ? " swimlane-lane-bg--empty" : ""}`,
      x: 0,
      y: lane.y,
      width: containerWidth,
      height: lane.height,
    }));

    svgEl.appendChild(_el("line", {
      class: "swimlane-lane-sep",
      x1: 0,
      y1: lane.y + lane.height,
      x2: containerWidth,
      y2: lane.y + lane.height,
    }));

    svgEl.appendChild(_el("text", {
      class: `swimlane-lane-label${lane.isEmpty ? " swimlane-lane-label--empty" : ""}`,
      x: 6,
      y: lane.y + lane.height / 2,
      "dominant-baseline": "middle",
    }, lane.anima || "—"));

    if (lane.isEmpty) continue;

    const overlap = assignOverlapRows(lane.groups);
    const dualRow = [...overlap.values()].some((v) => v.row === 1);

    for (const grp of lane.groups) {
      const geom = computeBarGeometry(grp, sx, windowStartMs, windowEndMs, nowMs);
      if (geom.width <= 0) continue;

      const rowInfo = overlap.get(grp.id) || { row: 0, overflowCount: 0 };
      const barH = barHeightForType(grp.type);
      const opacity = barOpacityForType(grp.type);
      const hasError = groupHasError(grp);
      const inProgress = isGroupInProgress(grp, nowMs);
      const color = GROUP_TYPE_COLORS[grp.type] || GROUP_TYPE_COLORS.single;

      let barY;
      if (dualRow) {
        const half = lane.height / 2;
        barY = lane.y + rowInfo.row * half + (half - barH) / 2;
      } else {
        barY = lane.y + (lane.height - barH) / 2;
      }

      const g = _el("g", {
        class: "swimlane-bar-group"
          + (selectedId === grp.id ? " is-selected" : "")
          + (inProgress ? " is-in-progress" : "")
          + (hasError ? " has-error" : ""),
        "data-group-id": grp.id,
        role: "button",
        tabindex: "0",
      });

      const rect = _el("rect", {
        class: "swimlane-bar",
        x: geom.x,
        y: barY,
        width: geom.width,
        height: barH,
        rx: 3,
        ry: 3,
        fill: color,
        opacity: String(opacity),
      });
      if (hasError) {
        rect.setAttribute("stroke", "var(--aw-color-error-dark, #dc2626)");
        rect.setAttribute("stroke-width", "2");
      }
      g.appendChild(rect);

      if (inProgress) {
        g.appendChild(_el("circle", {
          class: "swimlane-bar-pulse",
          cx: geom.x + geom.width,
          cy: barY + barH / 2,
          r: 3,
        }));
      }

      if (geom.width >= LABEL_MIN_BAR_PX && !isRealistic) {
        const emoji = _emojiForType(grp.type);
        const label = _groupLabel(grp);
        const text = `${emoji} ${label}`.slice(0, 40);
        g.appendChild(_el("text", {
          class: "swimlane-bar-label",
          x: geom.x + 4,
          y: barY + barH / 2,
          "dominant-baseline": "middle",
        }, text));
      }

      if (rowInfo.overflowCount > 0) {
        g.appendChild(_el("text", {
          class: "swimlane-bar-overflow",
          x: geom.x + geom.width - 2,
          y: barY + 2,
          "text-anchor": "end",
          "dominant-baseline": "hanging",
        }, `+${rowInfo.overflowCount + 2}`));
      }

      const count = grp.event_count || grp.events?.length || 0;
      const tip = [
        _emojiForType(grp.type),
        _groupLabel(grp),
        _formatRange(grp.start_ts, grp.end_ts),
        t("activity.swimlane_tooltip_events", { count }),
      ].filter(Boolean).join(" · ");
      g.appendChild(_el("title", {}, tip));

      g.addEventListener("click", (e) => {
        e.stopPropagation();
        onSelect(grp);
      });
      g.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          onSelect(grp);
        }
      });

      svgEl.appendChild(g);
    }
  }

  bg.addEventListener("click", () => onSelect(null));

  return { lanes: laneLayouts, height: totalHeight, width: containerWidth };
}
