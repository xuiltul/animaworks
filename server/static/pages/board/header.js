// ── Board channel header ─────────────────────
import { t } from "/shared/i18n.js";
import { escapeHtml } from "../../modules/state.js";
import { avatarMarkup } from "./avatar.js";

function _accessBadge(access, memberCount) {
  if (access === "closed") return `<span class="board-badge board-badge-access board-badge-access-closed">${escapeHtml(t("board.access_closed"))}</span>`;
  if (access === "members") {
    return `<span class="board-badge board-badge-access board-badge-access-members">${escapeHtml(t("board.access_members", { count: memberCount }))}</span>`;
  }
  return `<span class="board-badge board-badge-access board-badge-access-open">${escapeHtml(t("board.access_open"))}</span>`;
}

export function renderChannelHeader(element, channel, store) {
  if (!channel) {
    element.innerHTML = `<div class="board-header-title">${escapeHtml(t("board.loading"))}</div>`;
    return;
  }

  const members = Array.isArray(channel.members) ? channel.members : [];
  const visibleMembers = members.slice(0, 6);
  const memberAvatars = visibleMembers.map((name) => avatarMarkup({
    name,
    kind: "anima",
    url: store.avatarUrl(name),
    size: "sm",
  })).join("");
  const moreMembers = members.length > visibleMembers.length
    ? `<span class="board-members-more" title="${escapeHtml(members.slice(6).join(", "))}">${escapeHtml(t("board.members_more", { count: members.length - visibleMembers.length }))}</span>`
    : "";

  const slack = channel.slack || {};
  const slackBadges = [
    slack.inbound ? `<span class="board-badge board-badge-slack">${escapeHtml(t("board.slack_inbound"))}</span>` : "",
    slack.outbound ? `<span class="board-badge board-badge-slack">${escapeHtml(t("board.slack_outbound"))}</span>` : "",
  ].filter(Boolean).join("");

  element.innerHTML = `
    <div class="board-header-title">
      <span class="board-header-hash" aria-hidden="true">#</span>
      <span class="board-header-name">${escapeHtml(channel.name)}</span>
    </div>
    <div class="board-header-meta">
      ${_accessBadge(channel.access, members.length)}
      ${channel.company ? `<span class="board-badge">${escapeHtml(channel.company)}</span>` : ""}
      ${slackBadges}
      ${members.length ? `<span class="board-members">${memberAvatars}${moreMembers}</span>` : ""}
    </div>
    ${channel.description ? `<div class="board-header-description">${escapeHtml(channel.description)}</div>` : ""}`;
}
