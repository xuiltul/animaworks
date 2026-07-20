// ── Activity execution-context visual helpers ─────────────────
// Kept dependency-free so every activity surface can share the same stable
// task colours and the DOM helpers can be tested with Node's built-in runner.

const CONTEXT_COLORS = [
  "#2563eb",
  "#7c3aed",
  "#db2777",
  "#dc2626",
  "#d97706",
  "#059669",
  "#0891b2",
  "#4f46e5",
];

const PARALLEL_WINDOW_MS = 60_000;

function _hash(value) {
  let hash = 2166136261;
  for (let i = 0; i < value.length; i++) {
    hash ^= value.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

/** Return a stable colour and short label for an activity context. */
export function getContextPresentation(ctx) {
  if (typeof ctx !== "string" || !ctx.trim()) return null;
  const value = ctx.trim();
  let icon = "◇";
  let label = value;

  if (value.startsWith("task:")) {
    icon = "⚙";
    label = value.slice(5) || value;
  } else if (value.startsWith("cron:")) {
    icon = "⏰";
    label = value.slice(5) || "cron";
  } else if (value === "heartbeat") {
    icon = "💓";
  } else if (value === "chat") {
    icon = "💬";
  } else if (value.startsWith("inbox")) {
    icon = "📬";
  }

  if (label.length > 12) label = `${label.slice(0, 10)}…`;
  return {
    ctx: value,
    color: CONTEXT_COLORS[_hash(value) % CONTEXT_COLORS.length],
    label: `${icon} ${label}`,
    isTask: value.startsWith("task:"),
  };
}

/** Add the shared context colour treatment to an event row. */
export function decorateContextElement(element, ctx) {
  const presentation = getContextPresentation(ctx);
  if (!element || !presentation) return presentation;
  element.classList.add("has-activity-context");
  element.dataset.activityCtx = presentation.ctx;
  element.style.setProperty("--activity-ctx-color", presentation.color);
  return presentation;
}

/** Create a compact, accessible context badge. */
export function createContextBadge(ctx, documentRef = document) {
  const presentation = getContextPresentation(ctx);
  if (!presentation) return null;
  const badge = documentRef.createElement("span");
  badge.className = "activity-context-badge";
  badge.dataset.activityCtx = presentation.ctx;
  badge.style.setProperty("--activity-ctx-color", presentation.color);
  badge.textContent = presentation.label;
  badge.title = presentation.ctx;
  return badge;
}

function _eventTime(evt) {
  const value = new Date(evt?.ts || evt?.timestamp || 0).getTime();
  return Number.isFinite(value) ? value : 0;
}

function _animaKey(evt) {
  if (typeof evt?.anima === "string") return evt.anima;
  if (Array.isArray(evt?.animas)) return evt.animas.join(",");
  return "";
}

function _taskContext(evt) {
  if (typeof evt?.ctx !== "string" || !evt.ctx.startsWith("task:") || evt.ctx.length <= 5) {
    return null;
  }
  return evt.ctx;
}

function _validEventTime(evt) {
  if (!evt?.ts && !evt?.timestamp) return null;
  const value = new Date(evt.ts || evt.timestamp).getTime();
  return Number.isFinite(value) ? value : null;
}

function _laneTitle(startEvent, taskId) {
  const meta = startEvent?.meta;
  if (meta && typeof meta === "object") {
    for (const value of [meta.title, meta.task_title, meta.task_name]) {
      if (typeof value === "string" && value.trim()) return value.trim();
    }
  }
  if (typeof startEvent?.summary === "string" && startEvent.summary.trim()) {
    return startEvent.summary.trim();
  }
  return `task:${taskId.slice(0, 8)}`;
}

function _buildTaskIntervals(indexedEvents) {
  const bucketsByAnima = new Map();
  for (const item of indexedEvents) {
    const ctx = _taskContext(item.evt);
    if (!ctx || item.time === null) continue;
    const anima = _animaKey(item.evt);
    let byContext = bucketsByAnima.get(anima);
    if (!byContext) {
      byContext = new Map();
      bucketsByAnima.set(anima, byContext);
    }
    let bucket = byContext.get(ctx);
    if (!bucket) {
      bucket = [];
      byContext.set(ctx, bucket);
    }
    bucket.push(item);
  }

  const intervalsByAnima = new Map();
  for (const [anima, byContext] of bucketsByAnima) {
    const intervals = [];
    for (const [ctx, bucket] of byContext) {
      bucket.sort((a, b) => a.time - b.time || a.index - b.index);
      let open = null;
      const closeOpen = (end) => {
        if (!open) return;
        intervals.push({
          anima,
          ctx,
          taskId: ctx.slice(5),
          start: open.start,
          end: Math.max(open.start, end),
          startEvent: open.startEvent,
          items: open.items,
        });
        open = null;
      };

      for (const item of bucket) {
        if (item.evt.type === "task_exec_start") {
          if (open) closeOpen(open.items[open.items.length - 1].time);
          open = {
            start: item.time,
            startEvent: item.evt,
            items: [item],
          };
          continue;
        }
        if (!open) continue;
        open.items.push(item);
        if (item.evt.type === "task_exec_end") closeOpen(item.time);
      }
      if (open) closeOpen(open.items[open.items.length - 1].time);
    }
    intervalsByAnima.set(anima, intervals);
  }
  return intervalsByAnima;
}

function _parallelWindows(intervals) {
  const boundaries = [];
  for (const interval of intervals) {
    if (interval.end <= interval.start) continue;
    boundaries.push({ time: interval.start, kind: "start", interval });
    boundaries.push({ time: interval.end, kind: "end", interval });
  }
  boundaries.sort((a, b) => a.time - b.time);
  if (boundaries.length === 0) return [];

  const activeByTask = new Map();
  const windows = [];
  let previousTime = boundaries[0].time;
  let cursor = 0;

  while (cursor < boundaries.length) {
    const time = boundaries[cursor].time;
    if (time > previousTime) {
      if (activeByTask.size >= 2) {
        const last = windows[windows.length - 1];
        if (last && last.end === previousTime) {
          last.end = time;
        } else {
          windows.push({
            start: previousTime,
            end: time,
            intervals: new Set(),
          });
        }
      }
    }

    const atTime = [];
    while (cursor < boundaries.length && boundaries[cursor].time === time) {
      atTime.push(boundaries[cursor]);
      cursor += 1;
    }
    for (const boundary of atTime) {
      if (boundary.kind !== "end") continue;
      const count = activeByTask.get(boundary.interval.taskId) || 0;
      if (count <= 1) activeByTask.delete(boundary.interval.taskId);
      else activeByTask.set(boundary.interval.taskId, count - 1);
    }
    for (const boundary of atTime) {
      if (boundary.kind !== "start") continue;
      activeByTask.set(
        boundary.interval.taskId,
        (activeByTask.get(boundary.interval.taskId) || 0) + 1,
      );
    }
    previousTime = time;
  }

  for (const interval of intervals) {
    let low = 0;
    let high = windows.length;
    while (low < high) {
      const middle = Math.floor((low + high) / 2);
      if (windows[middle].end <= interval.start) low = middle + 1;
      else high = middle;
    }
    for (let index = low; index < windows.length; index++) {
      const window = windows[index];
      if (window.start >= interval.end) break;
      window.intervals.add(interval);
    }
  }

  return windows;
}

/**
 * Build nested lane data for time windows with two or more concurrent tasks.
 *
 * Only task events inside an actual overlap window enter a lane. Task events
 * before or after that window remain available to the caller for flat display.
 * Non-task activity from the same Anima is returned in flatEvents so it can be
 * rendered directly under the Anima group instead of inside a task lane.
 */
export function buildParallelGroups(events) {
  const source = Array.isArray(events) ? events : [];
  const indexed = source.map((evt, index) => ({
    evt,
    index,
    time: _validEventTime(evt),
  }));
  const intervalsByAnima = _buildTaskIntervals(indexed);
  const indexedByAnima = new Map();
  for (const item of indexed) {
    if (item.time === null) continue;
    const anima = _animaKey(item.evt);
    let animaItems = indexedByAnima.get(anima);
    if (!animaItems) {
      animaItems = [];
      indexedByAnima.set(anima, animaItems);
    }
    animaItems.push(item);
  }
  for (const animaItems of indexedByAnima.values()) {
    animaItems.sort((a, b) => a.time - b.time || a.index - b.index);
  }
  const groups = [];

  for (const [anima, intervals] of intervalsByAnima) {
    const intervalByEventIndex = new Map();
    for (const interval of intervals) {
      for (const item of interval.items) intervalByEventIndex.set(item.index, interval);
    }
    const windows = _parallelWindows(intervals).map((window) => {
      const lanesByTask = new Map();
      for (const interval of window.intervals) {
        let lane = lanesByTask.get(interval.taskId);
        if (!lane) {
          lane = {
            taskId: interval.taskId,
            title: _laneTitle(interval.startEvent, interval.taskId),
            eventsByIndex: new Map(),
            firstStart: interval.start,
          };
          lanesByTask.set(interval.taskId, lane);
        }
        lane.firstStart = Math.min(lane.firstStart, interval.start);
      }
      return { ...window, lanesByTask, flatItems: [], anchor: source.length };
    });

    let windowIndex = 0;
    for (const item of indexedByAnima.get(anima) || []) {
      while (windowIndex < windows.length && item.time > windows[windowIndex].end) {
        windowIndex += 1;
      }
      const window = windows[windowIndex];
      if (!window || item.time < window.start) continue;

      const ctx = _taskContext(item.evt);
      if (!ctx) {
        window.flatItems.push(item);
        window.anchor = Math.min(window.anchor, item.index);
        continue;
      }
      const interval = intervalByEventIndex.get(item.index);
      if (!interval || !window.intervals.has(interval)) continue;
      const lane = window.lanesByTask.get(interval.taskId);
      if (!lane) continue;
      lane.eventsByIndex.set(item.index, item.evt);
      window.anchor = Math.min(window.anchor, item.index);
    }

    for (const window of windows) {
      const lanes = Array.from(window.lanesByTask.values())
        .sort((a, b) => a.firstStart - b.firstStart || a.taskId.localeCompare(b.taskId))
        .map((lane) => ({
          taskId: lane.taskId,
          title: lane.title,
          events: Array.from(lane.eventsByIndex.entries())
            .sort((a, b) => a[0] - b[0])
            .map((entry) => entry[1]),
        }));
      const flatEvents = window.flatItems
        .sort((a, b) => a.index - b.index)
        .map((item) => item.evt);
      const group = { anima, lanes, flatEvents };
      groups.push({ group, anchor: window.anchor });
    }
  }

  return groups
    .sort((a, b) => a.anchor - b.anchor)
    .map((entry) => entry.group);
}

/**
 * Compute concurrent task counts for rendered events.
 *
 * task_exec_start/end boundaries are used when present. A one-minute local
 * mixing window is the fallback for a page that begins midway through tasks
 * or for older producers that only attach ctx to tool events.
 */
export function getParallelTaskCounts(events) {
  const source = Array.isArray(events) ? events : [];
  const counts = new Map();
  const ordered = source
    .map((evt, index) => ({ evt, index, time: _eventTime(evt) }))
    .sort((a, b) => a.time - b.time || a.index - b.index);
  const activeByAnima = new Map();

  for (const item of ordered) {
    const { evt } = item;
    if (typeof evt?.ctx !== "string" || !evt.ctx.startsWith("task:")) continue;
    const anima = _animaKey(evt);
    let active = activeByAnima.get(anima);
    if (!active) {
      active = new Set();
      activeByAnima.set(anima, active);
    }
    if (evt.type === "task_exec_start") active.add(evt.ctx);

    let count = active.has(evt.ctx) ? active.size : 0;
    if (count < 2) {
      const nearby = new Set();
      for (const other of source) {
        if (_animaKey(other) !== anima) continue;
        if (typeof other?.ctx !== "string" || !other.ctx.startsWith("task:")) continue;
        if (Math.abs(_eventTime(other) - item.time) <= PARALLEL_WINDOW_MS) nearby.add(other.ctx);
      }
      count = Math.max(count, nearby.size);
    }
    counts.set(evt, count);

    if (evt.type === "task_exec_end") active.delete(evt.ctx);
  }

  return counts;
}

/** Refresh parallel badges in compact feeds whose rows carry ctx/time data. */
export function updateLiveParallelIndicators(container, translate, documentRef = document) {
  if (!container) return;
  const entries = Array.from(container.children || []);
  for (const entry of entries) entry.querySelector?.(".activity-parallel-badge")?.remove();

  for (const entry of entries) {
    const ctx = entry.dataset?.activityCtx || "";
    const anima = entry.dataset?.activityAnima || "";
    const time = Number(entry.dataset?.activityTs || 0);
    if (!ctx.startsWith("task:") || !time) continue;
    const contexts = new Set();
    for (const other of entries) {
      const otherCtx = other.dataset?.activityCtx || "";
      const otherAnima = other.dataset?.activityAnima || "";
      const otherTime = Number(other.dataset?.activityTs || 0);
      if (
        otherAnima === anima
        && otherCtx.startsWith("task:")
        && Math.abs(otherTime - time) <= PARALLEL_WINDOW_MS
      ) {
        contexts.add(otherCtx);
      }
    }
    if (contexts.size < 2) continue;
    const badge = documentRef.createElement("span");
    badge.className = "activity-parallel-badge";
    badge.textContent = translate("activity.parallel_count", { count: contexts.size });
    const summary = entry.querySelector?.(".activity-summary");
    if (summary) summary.before(badge);
    else entry.appendChild(badge);
  }
}

/** Flatten the running-task API response while retaining the Anima name. */
export function normalizeRunningTasks(payload) {
  const result = [];
  for (const anima of payload?.animas || []) {
    for (const task of anima?.tasks || []) {
      result.push({ ...task, anima: anima.name || "" });
    }
  }
  return result;
}

/** Render the shared running-task strip into an existing container. */
export function renderRunningTasksStrip(container, payload, translate, documentRef = document) {
  if (!container) return [];
  const tasks = normalizeRunningTasks(payload);
  container.replaceChildren();
  container.hidden = tasks.length === 0;
  if (tasks.length === 0) return tasks;

  const heading = documentRef.createElement("span");
  heading.className = "running-tasks-heading";
  heading.textContent = translate("activity.running_tasks", { count: tasks.length });
  container.appendChild(heading);

  for (const task of tasks) {
    const chip = documentRef.createElement("span");
    chip.className = "running-task-chip";
    chip.dataset.slotId = String(task.slot_id ?? "");
    chip.dataset.taskId = String(task.task_id || "");
    decorateContextElement(chip, `task:${task.task_id || ""}`);

    const slot = documentRef.createElement("span");
    slot.className = "running-task-slot";
    slot.textContent = translate("activity.running_task_slot", { slot: task.slot_id ?? "?" });

    const title = documentRef.createElement("span");
    title.className = "running-task-title";
    const taskTitle = task.title || task.task_id || "";
    title.textContent = task.anima ? `${task.anima} · ${taskTitle}` : taskTitle;
    title.title = taskTitle;

    chip.appendChild(slot);
    chip.appendChild(title);
    container.appendChild(chip);
  }

  return tasks;
}
