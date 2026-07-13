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
