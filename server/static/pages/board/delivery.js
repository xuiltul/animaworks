// ── Board delivery footer ────────────────────
// Renders who a post actually reached, and what happened afterwards:
// delivered → read → replied (or still waiting / thinking).
import { t } from "/shared/i18n.js";
import { escapeHtml } from "../../modules/state.js";
import { avatarMarkup } from "./avatar.js";

function _slackLabel(status) {
  if (status === "sent") return t("board.delivery_slack_sent");
  if (status === "failed") return t("board.delivery_slack_failed");
  if (status === "skipped") return t("board.delivery_slack_skipped");
  return "";
}

function _replyMinutes(message, reply) {
  const start = Date.parse(String(message?.ts || ""));
  const end = Date.parse(String(reply?.ts || ""));
  if (Number.isNaN(start) || Number.isNaN(end)) return null;
  return Math.max(1, Math.round(Math.max(0, end - start) / 60_000));
}

function _replyTargetId(reply) {
  if (!reply) return "";
  return reply.id ?? reply.message_id ?? "";
}

function _callMarkup(name, message, store) {
  const status = store ? store.statusOf(name) : "idle";
  const kind = store?.anima(name)?.kind || "anima";
  const url = store ? store.avatarUrl(name) : null;
  const read = store ? store.hasReadByName(name, message) : false;
  const reply = store ? store.replyFor(message, name) : null;

  let stage;
  if (reply) {
    const minutes = _replyMinutes(message, reply);
    const label = minutes === null
      ? t("board.delivery_replied")
      : t("board.reply_in_minutes", { count: minutes });
    const target = _replyTargetId(reply);
    stage = `<button type="button" class="board-delivery-stage board-delivery-replied"${target ? ` data-board-jump="${escapeHtml(target)}"` : ""}>${escapeHtml(label)}</button>`;
  } else if (read) {
    stage = `<span class="board-delivery-stage board-delivery-read">${escapeHtml(t("board.delivery_read"))}</span><span class="board-delivery-stage board-delivery-waiting">${escapeHtml(t("board.delivery_reply_waiting"))}</span>`;
  } else if (status === "busy") {
    stage = `<span class="board-delivery-stage board-delivery-thinking">${escapeHtml(t("board.delivery_thinking"))}</span>`;
  } else {
    stage = `<span class="board-delivery-stage board-delivery-unread">${escapeHtml(t("board.delivery_unread"))}</span>`;
  }

  return `<span class="board-delivery-call board-delivery-ok" data-board-call="${escapeHtml(name)}">
    ${avatarMarkup({ name, kind, url, size: "xs" })}
    <span class="board-delivery-name">${escapeHtml(name)}</span>
    <span class="board-delivery-stage board-delivery-delivered">${escapeHtml(t("board.delivery_delivered"))}</span>
    ${stage}
  </span>`;
}

/**
 * Build the delivery HTML for one message, or "" when there is nothing to
 * show. Shared by the post renderer so refetched and optimistically inserted
 * posts look identical.
 */
export function deliveryMarkup(message, ctx = {}) {
  const store = ctx.store || null;
  const delivery = message?.delivery || {};
  const notified = Array.isArray(delivery.notified) ? delivery.notified : [];
  const notNotified = delivery.not_notified && typeof delivery.not_notified === "object"
    ? Object.entries(delivery.not_notified)
    : [];
  const hooks = message?.hooks || {};
  const slack = _slackLabel(hooks.slack_outbound);

  if (notified.length === 0 && notNotified.length === 0 && !slack) return "";

  const items = [];
  if (notified.length > 0) {
    items.push(`<span class="board-delivery-title">${escapeHtml(t("board.delivery_notified"))}</span>`);
    for (const name of notified) items.push(_callMarkup(String(name), message, store));
  }
  if (notNotified.length > 0) {
    items.push(`<span class="board-delivery-title">${escapeHtml(t("board.delivery_not_notified"))}</span>`);
    for (const [name, reason] of notNotified) {
      items.push(`<span class="board-delivery-item board-delivery-fail">${escapeHtml(name)} · ${escapeHtml(reason || "")}</span>`);
    }
  }
  if (slack) {
    items.push(`<span class="board-delivery-item board-delivery-slack">${escapeHtml(slack)}</span>`);
  }

  return `<div class="board-delivery" role="status">${items.join("")}</div>`;
}
