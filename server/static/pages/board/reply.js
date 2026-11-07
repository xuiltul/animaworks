// ── Reply-to line ────────────────────────────
// "↩ alice の 09:05 の投稿への返事" above the text, clicking jumps to the
// original row (loaded with the `around` query when necessary).
import { t } from "/shared/i18n.js";
import { escapeHtml } from "../../modules/state.js";
import { clockTime, dayLabel, localDayKey } from "./time.js";

function _replyTime(target, message) {
  const time = clockTime(target.ts);
  if (!time) return "";
  const targetDay = localDayKey(target.ts);
  const currentDay = localDayKey(message?.ts);
  if (targetDay && currentDay && targetDay !== currentDay) {
    const day = dayLabel(targetDay);
    return day ? `${day} ${time}` : time;
  }
  return time;
}

export function replyMarkup(message, ctx = {}) {
  if (!message || !message.reply_to) return "";
  const store = ctx.store || null;
  const target = store ? store.replyTarget(message) : null;
  const isDm = store?.state?.selectedType === "dm";
  const jumpId = target ? (target.id ?? target.message_id ?? message.reply_to) : message.reply_to;

  let label;
  if (target) {
    const time = _replyTime(target, message);
    label = isDm
      ? t("board.reply_to_dm", { name: target.from, time })
      : t("board.reply_to_post", { name: target.from, time });
  } else {
    label = isDm ? t("board.reply_to_dm_unknown") : t("board.reply_to_post_unknown");
  }

  return `<div class="board-post-reply"><button type="button" class="board-reply-link" data-board-jump="${escapeHtml(String(jumpId))}" title="${escapeHtml(t("board.reply_jump"))}">${escapeHtml(label)}</button></div>`;
}
