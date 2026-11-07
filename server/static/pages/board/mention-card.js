// ── Mention profile card ─────────────────────
import { t } from "/shared/i18n.js";
import { escapeHtml } from "../../modules/state.js";
import { avatarMarkup, statusLabel } from "./avatar.js";

export function createMentionCard() {
  let card = null;
  let currentName = "";
  let filterHandler = null;

  function close() {
    if (card) card.remove();
    card = null;
    currentName = "";
    filterHandler = null;
  }

  function _onPointerDown(event) {
    if (card && !card.contains(event.target)) close();
  }

  function _onKeyDown(event) {
    if (event.key === "Escape") close();
  }

  document.addEventListener("pointerdown", _onPointerDown);
  document.addEventListener("keydown", _onKeyDown);

  function open(name, anchorRect, store, onFilter) {
    close();
    currentName = name;
    filterHandler = onFilter;
    const anima = store.anima(name) || {};
    const status = store.statusOf(name);

    card = document.createElement("div");
    card.className = "board-mention-card";
    card.setAttribute("role", "dialog");
    card.setAttribute("aria-label", name);
    card.innerHTML = `
      <button type="button" class="board-mention-card-close" data-board-card-close aria-label="${escapeHtml(t("board.mention_card_close"))}">×</button>
      <div class="board-mention-card-head">
        ${avatarMarkup({ name, kind: "anima", url: store.avatarUrl(name), size: "md" })}
        <div class="board-mention-card-identity">
          <div class="board-mention-card-name">${escapeHtml(name)}</div>
          <div class="board-mention-card-role">${escapeHtml(anima.role || anima.title || anima.department || "—")}</div>
        </div>
      </div>
      <div class="board-mention-card-status">
        <span class="board-status-dot board-status-${escapeHtml(status)}" data-board-card-status></span>
        <span class="board-mention-card-status-label" data-board-card-status-label>${escapeHtml(statusLabel(status))}</span>
      </div>
      <button type="button" class="board-mention-card-filter" data-board-card-filter>${escapeHtml(t("board.mention_only"))}</button>`;

    document.body.append(card);

    const width = 260;
    const margin = 8;
    const left = Math.max(margin, Math.min(anchorRect.left, window.innerWidth - width - margin));
    const top = Math.min(anchorRect.bottom + 8, window.innerHeight - 160);
    card.style.left = `${left}px`;
    card.style.top = `${Math.max(margin, top)}px`;
    card.style.width = `${width}px`;

    card.addEventListener("click", (event) => {
      if (event.target.closest("[data-board-card-close]")) {
        close();
        return;
      }
      if (event.target.closest("[data-board-card-filter]")) {
        const handler = filterHandler;
        const target = currentName;
        close();
        handler?.(target);
      }
    });
  }

  function update(store) {
    if (!card || !currentName) return;
    const status = store.statusOf(currentName);
    const dot = card.querySelector("[data-board-card-status]");
    if (dot) dot.className = `board-status-dot board-status-${status}`;
    const label = card.querySelector("[data-board-card-status-label]");
    if (label) label.textContent = statusLabel(status);
  }

  function destroy() {
    close();
    document.removeEventListener("pointerdown", _onPointerDown);
    document.removeEventListener("keydown", _onKeyDown);
  }

  return { open, close, update, destroy };
}
