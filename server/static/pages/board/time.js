// ── Board time helpers ───────────────────────
import { getLocale, t } from "/shared/i18n.js";

const DAY_MS = 24 * 60 * 60 * 1000;

function _locale() {
  return getLocale() || "ja";
}

function _startOfDay(date) {
  return new Date(date.getFullYear(), date.getMonth(), date.getDate());
}

/** Stable local-date key ("2026-11-07") used to group posts by day. */
export function localDayKey(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${date.getFullYear()}-${month}-${day}`;
}

/** Parse a local-date key back into a Date (midnight local time). */
function _dateFromDayKey(key) {
  const [year, month, day] = String(key).split("-").map(Number);
  return new Date(year, (month || 1) - 1, day || 1);
}

function _weekday(date) {
  return new Intl.DateTimeFormat(_locale(), { weekday: "short" }).format(date);
}

function _monthDay(date) {
  const now = new Date();
  const options = date.getFullYear() === now.getFullYear()
    ? { month: "long", day: "numeric" }
    : { year: "numeric", month: "long", day: "numeric" };
  return new Intl.DateTimeFormat(_locale(), options).format(date);
}

/** "今日 · 11月7日（土）" style label for a local-date key. */
export function dayLabel(dayKey) {
  if (!dayKey) return "";
  const date = _dateFromDayKey(dayKey);
  const diffDays = Math.round((_startOfDay(new Date()) - _startOfDay(date)) / DAY_MS);
  const prefix = diffDays === 0
    ? t("board.day_today")
    : diffDays === 1
      ? t("board.day_yesterday")
      : "";
  const body = `${_monthDay(date)}${_locale() === "ja" ? `（${_weekday(date)}）` : ` (${_weekday(date)})`}`;
  return prefix ? `${prefix} · ${body}` : body;
}

/** Clock time ("09:05") in the active locale. */
export function clockTime(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  return new Intl.DateTimeFormat(_locale(), { hour: "2-digit", minute: "2-digit" }).format(date);
}

/** Relative label like "12分前" for the rail. */
export function relativeTime(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  const diffMs = Date.now() - date.getTime();
  if (diffMs < 60_000) return t("board.time_now");
  const locale = _locale();
  const rtf = new Intl.RelativeTimeFormat(locale, { numeric: "auto" });
  const minutes = Math.round(diffMs / 60_000);
  let label;
  if (minutes < 60) label = rtf.format(-minutes, "minute");
  else if (minutes < 60 * 24) label = rtf.format(-Math.round(minutes / 60), "hour");
  else if (minutes < 60 * 24 * 7) label = rtf.format(-Math.round(minutes / (60 * 24)), "day");
  else label = _monthDay(date);
  return locale === "en" ? label : label.replace(/\s+/g, "");
}

/** "2,814" formatted for the active locale. */
export function numberText(value) {
  return new Intl.NumberFormat(_locale()).format(Number(value) || 0);
}
