// ── "Who has read this" footer ───────────────
// Small overlapping anima faces at the bottom right of every post, with a
// count ("n人が読んだ") or "全員が読んだ" when the whole reader pool is done.
import { t } from "/shared/i18n.js";
import { escapeHtml } from "../../modules/state.js";
import { avatarMarkup } from "./avatar.js";

export function seenMarkup(message, ctx = {}) {
  const store = ctx.store;
  if (!store || !message) return "";
  const readers = store.readersForMessage(message);
  if (readers.length === 0) return "";
  const pool = store.readerPoolForMessage(message);
  const all = pool.length > 0 && readers.length >= pool.length;
  const label = all
    ? t("board.seen_all")
    : t("board.seen_count", { count: readers.length });
  const separator = t("board.seen_separator") || "、";
  const names = readers.join(separator);
  const title = t("board.seen_names", { names });
  const newFaces = ctx.newFaces instanceof Set ? ctx.newFaces : new Set();

  const faces = readers.map((name) => {
    const kind = store.anima(name)?.kind || "anima";
    const url = store.avatarUrl(name);
    const fresh = newFaces.has(name) ? " board-is-new" : "";
    return `<span class="board-seen-face${fresh}">${avatarMarkup({ name, kind, url, size: "xs" })}</span>`;
  }).join("");

  return `<button type="button" class="board-seen" title="${escapeHtml(title)}" aria-label="${escapeHtml(title)}" data-board-seen="${escapeHtml(names)}">
    <span class="board-seen-faces">${faces}</span>
    <span class="board-seen-label">${escapeHtml(label)}</span>
    <span class="board-seen-names">${escapeHtml(title)}</span>
  </button>`;
}
