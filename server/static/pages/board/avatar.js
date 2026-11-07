// ── Board avatars & kind helpers ─────────────
import { escapeHtml } from "../../modules/state.js";
import { t } from "/shared/i18n.js";

const PERSON_ICON = `<svg class="board-avatar-icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><circle cx="12" cy="7.5" r="3.2"></circle><path d="M5.5 20c.6-4 3-6 6.5-6s5.9 2 6.5 6"></path></svg>`;
const CLOCK_ICON = `<svg class="board-avatar-icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><circle cx="12" cy="12" r="7.5"></circle><path d="M12 7.5V12l3 2"></path></svg>`;

/** Collapse a raw API kind into one of the four visual families. */
export function kindGroup(kind) {
  const value = String(kind || "").toLowerCase();
  if (value === "human") return "human";
  if (value === "external_user" || value === "external_bot" || value.startsWith("external")) return "external";
  if (value === "operator") return "operator";
  return "anima";
}

/** i18n label for the kind badge (empty for anima). */
export function kindLabel(kind) {
  const value = String(kind || "").toLowerCase();
  if (value === "human") return t("board.kind_human");
  if (value === "external_bot") return t("board.source_slack_bot");
  if (value === "external_user" || value.startsWith("external")) return t("board.source_slack_user");
  if (value === "operator") return t("board.kind_operator");
  return "";
}

/** i18n label for a board presence status. */
export function statusLabel(status) {
  const key = `board.mention_status_${status || "idle"}`;
  return t(key);
}

function _sizeClass(size) {
  if (size === "xs") return " board-avatar-xs";
  if (size === "sm") return " board-avatar-sm";
  return "";
}

/**
 * Shape-coded avatar markup.
 * anima: circle (image or initial) / human: rounded square person icon /
 * external: hexagon (Slack family) / operator: clock square.
 */
export function avatarMarkup({ name, kind, url = null, size = "md" } = {}) {
  const safeName = escapeHtml(name || "?");
  const initial = escapeHtml((name || "?").charAt(0).toUpperCase() || "?");
  const group = kindGroup(kind);
  const sizeClass = _sizeClass(size);

  if (group === "anima" && url) {
    return `<span class="board-avatar board-avatar-anima${sizeClass}" title="${safeName}"><img class="board-avatar-img" src="${escapeHtml(url)}" alt="${safeName}" loading="lazy"></span>`;
  }
  if (group === "human") {
    return `<span class="board-avatar board-avatar-human${sizeClass}" title="${safeName}">${PERSON_ICON}</span>`;
  }
  if (group === "external") {
    return `<span class="board-avatar board-avatar-external${sizeClass}" title="${safeName}">${initial}</span>`;
  }
  if (group === "operator") {
    return `<span class="board-avatar board-avatar-operator${sizeClass}" title="${safeName}">${CLOCK_ICON}</span>`;
  }
  return `<span class="board-avatar board-avatar-anima board-avatar-initial${sizeClass}" title="${safeName}">${initial}</span>`;
}
