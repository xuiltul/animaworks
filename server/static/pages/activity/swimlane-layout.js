// ── Swimlane layout pure helpers (dependency-free for Node tests) ──

export const MIN_BAR_PX = 4;
export const LANE_HEIGHT = 28;
export const EMPTY_LANE_HEIGHT = 14;
export const AMBIENT_BAR_H = 8;
export const SIGNAL_BAR_H = 18;
export const OPEN_WINDOW_MS = 5 * 60 * 1000;
export const RECENT_WINDOW_MS = 60 * 60 * 1000;

export const AMBIENT_GROUP_TYPES = new Set(["cron", "heartbeat"]);

/** Parse activity timestamps (ISO-like, with or without timezone). */
export function parseTs(ts) {
  if (!ts) return NaN;
  const ms = Date.parse(ts);
  if (Number.isFinite(ms)) return ms;
  const m = String(ts).match(
    /^(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2}):(\d{2})/,
  );
  if (!m) return NaN;
  return new Date(+m[1], +m[2] - 1, +m[3], +m[4], +m[5], +m[6]).getTime();
}

/**
 * Ongoing only when is_open AND end_ts is within the last 5 minutes.
 * Cron/task groups remain is_open forever but must NOT show as live.
 */
export function isGroupInProgress(group, nowMs = Date.now()) {
  if (!group || !group.is_open) return false;
  const endMs = parseTs(group.end_ts || group.start_ts);
  if (!Number.isFinite(endMs)) return false;
  return nowMs - endMs <= OPEN_WINDOW_MS;
}

/** True when any event is an error or a tool_result with is_error. */
export function groupHasError(group) {
  for (const evt of group?.events || []) {
    if (evt.type === "error") return true;
    if (evt.tool_result && evt.tool_result.is_error === true) return true;
  }
  return false;
}

export function isAmbientType(type) {
  return AMBIENT_GROUP_TYPES.has(type);
}

export function barHeightForType(type) {
  return isAmbientType(type) ? AMBIENT_BAR_H : SIGNAL_BAR_H;
}

export function barOpacityForType(type) {
  return isAmbientType(type) ? 0.35 : 1.0;
}

/**
 * Detect pairwise time-range overlaps within a list of groups
 * (same Anima lane). Assigns each group a row index 0 or 1.
 *
 * @returns {Map<string, {row: number, overflowCount: number}>}
 */
export function assignOverlapRows(groups) {
  const result = new Map();
  if (!groups || groups.length === 0) return result;

  const items = groups.map((g) => ({
    id: g.id,
    start: parseTs(g.start_ts),
    end: parseTs(g.end_ts || g.start_ts),
    group: g,
  })).filter((it) => Number.isFinite(it.start));

  for (const it of items) {
    if (!Number.isFinite(it.end) || it.end < it.start) it.end = it.start;
  }

  items.sort((a, b) => a.start - b.start || a.end - b.end);

  const rowEnds = [0, 0];
  const concurrent = new Map();
  for (const it of items) concurrent.set(it.id, new Set());

  for (let i = 0; i < items.length; i++) {
    for (let j = i + 1; j < items.length; j++) {
      const a = items[i];
      const b = items[j];
      if (b.start >= a.end) break;
      concurrent.get(a.id).add(b.id);
      concurrent.get(b.id).add(a.id);
    }
  }

  for (const it of items) {
    let row = 0;
    if (it.start < rowEnds[0]) row = 1;
    rowEnds[row] = Math.max(rowEnds[row], it.end);
    const overflowCount = Math.max(0, concurrent.get(it.id).size + 1 - 2);
    result.set(it.id, { row, overflowCount });
  }

  return result;
}

/**
 * Build ordered lane descriptors from groups.
 * Sort: recent-hour event count desc; zero-count animas at bottom with half height.
 */
export function buildLanes(groups, opts = {}) {
  const nowMs = opts.nowMs ?? Date.now();
  const recentCutoff = nowMs - RECENT_WINDOW_MS;
  const byAnima = new Map();

  for (const g of groups || []) {
    const anima = g.anima || "";
    if (!byAnima.has(anima)) byAnima.set(anima, []);
    byAnima.get(anima).push(g);
  }

  for (const name of opts.animaOrder || []) {
    if (!byAnima.has(name)) byAnima.set(name, []);
  }

  const lanes = [];
  for (const [anima, gs] of byAnima) {
    let recentCount = 0;
    for (const g of gs) {
      const s = parseTs(g.start_ts);
      const e = parseTs(g.end_ts || g.start_ts);
      if ((Number.isFinite(e) && e >= recentCutoff) ||
          (Number.isFinite(s) && s >= recentCutoff)) {
        recentCount += g.event_count || (g.events ? g.events.length : 1);
      }
    }
    const isEmpty = gs.length === 0;
    lanes.push({
      anima,
      height: isEmpty ? EMPTY_LANE_HEIGHT : LANE_HEIGHT,
      isEmpty,
      recentCount,
      groups: gs,
    });
  }

  lanes.sort((a, b) => {
    if (a.isEmpty !== b.isEmpty) return a.isEmpty ? 1 : -1;
    if (b.recentCount !== a.recentCount) return b.recentCount - a.recentCount;
    return a.anima.localeCompare(b.anima);
  });

  return lanes;
}

/** Simple linear time scale (fallback when d3 is unavailable). */
export function createTimeScale(domainStart, domainEnd, rangeStart, rangeEnd) {
  const d0 = +domainStart;
  const d1 = +domainEnd;
  const r0 = rangeStart;
  const r1 = rangeEnd;
  const span = d1 - d0 || 1;
  const scale = (t) => r0 + ((+t - d0) / span) * (r1 - r0);
  scale.domain = () => [d0, d1];
  scale.range = () => [r0, r1];
  scale.invert = (x) => d0 + ((x - r0) / (r1 - r0 || 1)) * span;
  return scale;
}

/**
 * Compute bar geometry in pixel space.
 * @returns {{ x: number, width: number, clipped: boolean, extendsToNow: boolean }}
 */
export function computeBarGeometry(group, scaleX, windowStartMs, windowEndMs, nowMs) {
  const startMs = parseTs(group.start_ts);
  let endMs = parseTs(group.end_ts || group.start_ts);
  if (!Number.isFinite(startMs)) {
    return { x: 0, width: 0, clipped: true, extendsToNow: false };
  }
  if (!Number.isFinite(endMs) || endMs < startMs) endMs = startMs;

  const inProgress = isGroupInProgress(group, nowMs);
  let extendsToNow = false;
  if (inProgress) {
    endMs = Math.max(endMs, nowMs);
    extendsToNow = true;
  }

  const clippedStart = Math.max(startMs, windowStartMs);
  const clippedEnd = Math.min(endMs, windowEndMs);
  if (clippedEnd < windowStartMs || clippedStart > windowEndMs) {
    return { x: 0, width: 0, clipped: true, extendsToNow: false };
  }

  const x0 = scaleX(clippedStart);
  const x1 = scaleX(Math.max(clippedEnd, clippedStart));
  let width = Math.max(MIN_BAR_PX, x1 - x0);
  const maxW = scaleX(windowEndMs) - x0;
  if (width > maxW && maxW > 0) width = Math.max(MIN_BAR_PX, maxW);

  return {
    x: x0,
    width,
    clipped: startMs < windowStartMs,
    extendsToNow,
  };
}
