// ── Board delivery footer ────────────────────
// Renders who a post actually reached. Rows fetched from the API carry
// `delivery`; the POST response additionally carries `hooks` (Slack etc.).
import { t } from "/shared/i18n.js";
import { escapeHtml } from "../../modules/state.js";

function _slackLabel(status) {
  if (status === "sent") return t("board.delivery_slack_sent");
  if (status === "failed") return t("board.delivery_slack_failed");
  if (status === "skipped") return t("board.delivery_slack_skipped");
  return "";
}

/**
 * Build the delivery HTML for one message, or "" when there is nothing to
 * show. Shared by the post renderer so refetched and optimistically inserted
 * posts look identical.
 */
export function deliveryMarkup(message) {
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
    for (const name of notified) {
      items.push(`<span class="board-delivery-item board-delivery-ok">✓ ${escapeHtml(name)}</span>`);
    }
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
