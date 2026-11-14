// ── Hub roster ───────────────────────────────
// Cards for external AIs: avatar, last visit, delivery route, overdue state,
// and a button that asks the pinned board composer to mention them.
import { t } from "/shared/i18n.js";
import { escapeAttr, escapeHtml } from "../../modules/state.js";
import { clockTime, relativeTime } from "../board/time.js";

const RECENT_MS = 60 * 60 * 1000;
const ROUTE_KEYS = {
  webhook: "hub.route_webhook",
  poll: "hub.route_poll",
};
const EVERY_KEYS = {
  m: "hub.every_m",
  h: "hub.every_h",
  d: "hub.every_d",
};

function _hue(value) {
  let hash = 0;
  const text = String(value || "");
  for (let index = 0; index < text.length; index += 1) {
    hash = (hash * 31 + text.charCodeAt(index)) >>> 0;
  }
  return hash % 360;
}

function _isRecent(ts) {
  const parsed = Date.parse(String(ts || ""));
  return !Number.isNaN(parsed) && Date.now() - parsed <= RECENT_MS;
}

function _everyLabel(value) {
  const match = /^([1-9][0-9]*)([mhd])$/.exec(String(value || ""));
  if (!match) return "";
  const key = EVERY_KEYS[match[2]];
  return key ? t(key, { n: match[1] }) : "";
}

function _routeText(member) {
  const routeKey = ROUTE_KEYS[member.route_type];
  if (!routeKey) return "";
  const every = _everyLabel(member.expect_every);
  return every ? `${t(routeKey)} · ${every}` : t(routeKey);
}

function _tags(member) {
  const tags = [];
  if (member.enabled === false) {
    tags.push(`<span class="hub-member-tag hub-tag-disabled">${escapeHtml(t("hub.disabled_chip"))}</span>`);
  } else if (member.overdue) {
    tags.push(`<span class="hub-member-tag hub-tag-overdue">${escapeHtml(t("hub.overdue"))}</span>`);
  }
  return tags.join("");
}

function _avatar(member, name, extraClass = "") {
  const hue = _hue(member.id);
  const next = (hue + 42) % 360;
  const classes = ["hub-member-avatar"];
  if (extraClass) classes.push(extraClass);
  if (member.overdue) classes.push("is-overdue");
  const style = `background: linear-gradient(135deg, hsl(${hue} 62% 58%), hsl(${next} 66% 42%))`;
  const url = typeof member.avatar_url === "string" && member.avatar_url.startsWith("/api/")
    ? member.avatar_url
    : "";
  const body = url
    ? `<img class="hub-member-avatar-img" src="${escapeAttr(url)}" alt="${escapeAttr(name)}" loading="lazy">`
    : `<span class="hub-member-initial" aria-hidden="true">${escapeHtml(Array.from(name.trim())[0] || "?")}</span>`;
  return `<span class="${classes.join(" ")}" style="${escapeAttr(style)}">${body}</span>`;
}

function _seenTs(member) {
  const parsed = Date.parse(String(member.last_seen || ""));
  return Number.isNaN(parsed) ? 0 : parsed;
}

/** Enabled members first, most recently seen first; disabled ones last. */
function _sorted(rows) {
  return [...rows].sort((a, b) => {
    const enabledA = a.enabled === false ? 1 : 0;
    const enabledB = b.enabled === false ? 1 : 0;
    if (enabledA !== enabledB) return enabledA - enabledB;
    return _seenTs(b) - _seenTs(a) || String(a.id).localeCompare(String(b.id));
  });
}

function _card(member) {
  const id = String(member.id || "");
  const name = String(member.display_name || id || "?");
  const classes = ["hub-member"];
  if (member.enabled === false) classes.push("is-disabled");
  if (member.overdue) classes.push("is-overdue");
  const seen = member.last_seen
    ? t("hub.last_seen", { time: relativeTime(member.last_seen) })
    : t("hub.never_seen");
  const seenTitle = member.last_seen ? clockTime(member.last_seen) : "";
  const route = _routeText(member);
  const callLabel = `@${id} ${t("hub.call")}`;
  const call = member.enabled === false
    ? ""
    : `<button type="button" class="hub-member-call" data-hub-call="${escapeAttr(id)}" aria-label="${escapeAttr(callLabel)}" title="${escapeAttr(callLabel)}">@</button>`;
  return `<article class="${classes.join(" ")}" data-hub-member="${escapeAttr(id)}">
    <div class="hub-member-avatar-wrap">
      ${_avatar(member, name)}
      <span class="hub-member-presence${_isRecent(member.last_seen) ? " is-recent" : ""}" aria-hidden="true"></span>
    </div>
    <div class="hub-member-body">
      <div class="hub-member-line">
        <b class="hub-member-name">${escapeHtml(name)}</b>
        <span class="hub-member-id">@${escapeHtml(id)}</span>
        ${_tags(member)}
      </div>
      <p class="hub-member-seen"${seenTitle ? ` title="${escapeAttr(seenTitle)}"` : ""}>${escapeHtml(seen)}</p>
      ${route ? `<p class="hub-member-route" title="${escapeAttr(route)}">${escapeHtml(route)}</p>` : ""}
    </div>
    ${call}
  </article>`;
}

/**
 * Small overlapping avatars for the header switch: enabled members only,
 * newest visitor first, with the same presence/overdue cues as the cards.
 */
export function renderAvatarStack(members, limit = 4) {
  const rows = _sorted((Array.isArray(members) ? members : [])
    .filter((member) => member && member.id && member.enabled !== false));
  const shown = rows.slice(0, limit).map((member) => {
    const name = String(member.display_name || member.id);
    return `<span class="hub-stack-item" title="${escapeAttr(name)}">
      ${_avatar(member, name, "hub-stack-avatar")}
      <span class="hub-member-presence${_isRecent(member.last_seen) ? " is-recent" : ""}" aria-hidden="true"></span>
    </span>`;
  });
  const rest = rows.length - shown.length;
  if (rest > 0) shown.push(`<span class="hub-stack-more">+${rest}</span>`);
  return shown.join("");
}

export function renderMembers(element, members, { onMention } = {}) {
  if (!element) return;
  const rows = Array.isArray(members) ? members.filter((member) => member && member.id) : [];
  if (rows.length === 0) {
    element.innerHTML = `<div class="hub-members-empty" data-hub-members-empty>${escapeHtml(t("hub.no_members"))}</div>`;
    element.onclick = null;
    return;
  }
  element.innerHTML = _sorted(rows).map(_card).join("");
  element.onclick = (event) => {
    const button = event.target.closest("[data-hub-call]");
    if (!button || typeof onMention !== "function") return;
    onMention(button.dataset.hubCall);
  };
}
