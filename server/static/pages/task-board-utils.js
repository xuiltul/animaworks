import { t } from "/shared/i18n.js";

export const COLUMNS = ["todo", "running", "blocked", "waiting", "review", "done", "suppressed"];
export const SUPPRESSED_VISIBILITIES = new Set(["expired", "archived", "tombstoned"]);

export function taskKey(task) {
  return `${task.anima_name}:${task.task_id}`;
}

export function shortId(taskId) {
  return (taskId || "").slice(0, 8);
}

export function visibilityLabel(visibility) {
  return t(`taskboard.visibility_${visibility || "active"}`);
}

export function visibilityPayload(action) {
  if (action === "expire") return "expired";
  if (action === "archive") return "archived";
  if (action === "tombstone") return "tombstoned";
  return action;
}

export function statusClassSuffix(status) {
  return String(status || "missing").replace(/[^a-zA-Z0-9_-]/g, "-");
}

export function deadlineText(deadline) {
  if (!deadline) return t("taskboard.no_deadline");
  const date = new Date(deadline);
  if (Number.isNaN(date.getTime())) return t("taskboard.no_deadline");
  return date.toLocaleString([], { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" });
}

export function ageText(updatedAt) {
  if (!updatedAt) return t("taskboard.updated_unknown");
  const then = new Date(updatedAt).getTime();
  if (Number.isNaN(then)) return t("taskboard.updated_unknown");
  const minutes = Math.max(0, Math.floor((Date.now() - then) / 60000));
  if (minutes < 60) return t("taskboard.age_minutes", { count: minutes });
  const hours = Math.floor(minutes / 60);
  if (hours < 48) return t("taskboard.age_hours", { count: hours });
  return t("taskboard.age_days", { count: Math.floor(hours / 24) });
}

export function isOverdue(deadline) {
  if (!deadline) return false;
  const ts = new Date(deadline).getTime();
  return !Number.isNaN(ts) && Date.now() > ts;
}

export function defaultLocalDateTime() {
  const date = new Date(Date.now() + 60 * 60 * 1000);
  date.setMinutes(Math.ceil(date.getMinutes() / 5) * 5, 0, 0);
  const pad = (value) => String(value).padStart(2, "0");
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}T${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

/**
 * Sentence-ending punctuation (Latin + CJK).
 * Latin .!? require trailing whitespace or end-of-string (avoid "3.14" / abbreviations).
 * CJK 。！？ split even without following whitespace.
 */
const SENTENCE_BOUNDARY = /[.!?](?=\s|$)|[。！？]/;

/**
 * Split a task description into a short title and remaining body for card display.
 *
 * Title = first line, or first sentence if single-line, capped at maxTitleLength
 * (with ellipsis). Remainder becomes body. Empty / null-safe.
 *
 * @param {string|null|undefined} description
 * @param {number} [maxTitleLength=80]
 * @returns {{ title: string, body: string, titleTruncated: boolean }}
 */
export function splitTaskDescription(description, maxTitleLength = 80) {
  const limit = Number.isFinite(maxTitleLength) && maxTitleLength > 0 ? maxTitleLength : 80;
  const text = String(description ?? "")
    .replace(/\r\n/g, "\n")
    .replace(/\r/g, "\n")
    .trim();

  if (!text) {
    return { title: "", body: "", titleTruncated: false };
  }

  let title = text;
  let body = "";

  const nl = text.indexOf("\n");
  if (nl >= 0) {
    title = text.slice(0, nl).trim();
    body = text.slice(nl + 1).trim();
  } else {
    const match = SENTENCE_BOUNDARY.exec(text);
    if (match && match.index + match[0].length < text.length) {
      const end = match.index + match[0].length;
      title = text.slice(0, end).trim();
      body = text.slice(end).trim();
    }
  }

  let titleTruncated = false;
  if (title.length > limit) {
    const overflow = title.slice(limit).trim();
    title = `${title.slice(0, limit).trimEnd()}…`;
    body = body ? `${overflow}\n${body}`.trim() : overflow;
    titleTruncated = true;
  }

  return { title, body, titleTruncated };
}
