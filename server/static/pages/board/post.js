// ── Board post rendering ─────────────────────
import { t } from "/shared/i18n.js";
import { escapeAttr, escapeHtml, renderMarkdown } from "../../modules/state.js";
import { avatarMarkup, kindGroup, kindLabel } from "./avatar.js";
import { deliveryMarkup } from "./delivery.js";
import { decorateMarkdown } from "./markdown.js";
import { replyMarkup } from "./reply.js";
import { seenMarkup } from "./seen.js";
import { clockTime } from "./time.js";

function _isTitled(text) {
  const first = String(text || "").split("\n").find((line) => line.trim() !== "");
  return Boolean(first && /^\s*#{1,3}\s+/.test(first));
}

function _actionsMarkup(message, ctx) {
  if (!ctx.actions) return "";
  const hasId = message.id !== undefined && message.id !== null && message.id !== "";
  const quote = ctx.canQuote
    ? `<button type="button" class="board-post-action" data-board-action="quote">${escapeHtml(t("board.action_quote"))}</button>`
    : "";
  const link = hasId
    ? `<button type="button" class="board-post-action" data-board-action="link">${escapeHtml(t("board.action_link"))}</button>`
    : "";
  if (!quote && !link) return "";
  return `<div class="board-post-actions" role="toolbar" aria-label="${escapeHtml(t("board.post_actions"))}">${quote}${link}</div>`;
}

/**
 * Build one post element. The caller owns insertion/reuse; this function only
 * creates a fresh node (used when a key is new or its content changed).
 */
export function buildPostElement(message, ctx) {
  const kind = message.kind || "anima";
  const group = kindGroup(kind);
  const compact = Boolean(ctx.compact);
  const mine = group === "human" && Boolean(ctx.isMine?.(message.from));
  const text = String(message.text || "");
  const titled = _isTitled(text);
  const badge = kindLabel(kind);
  const delivery = deliveryMarkup(message, ctx);
  const seen = seenMarkup(message, ctx);
  const reply = replyMarkup(message, ctx);
  const actions = _actionsMarkup(message, ctx);

  const element = document.createElement("article");
  element.className = [
    "board-post",
    `board-post-kind-${group}`,
    mine ? "board-post-mine" : "",
    compact ? "board-post-compact" : "",
  ].filter(Boolean).join(" ");
  if (group === "anima") {
    element.dataset.boardFrom = String(message.from || "");
  }
  element.dataset.boardKey = ctx.expansionKey;
  element.dataset.boardFoldKey = ctx.expansionKey;
  element.dataset.boardTextLength = String(text.length);
  if (message.id !== undefined && message.id !== null) {
    element.dataset.boardId = String(message.id);
  }

  const rendered = renderMarkdown(text, group === "anima" ? message.from : null);
  const decorated = decorateMarkdown(rendered, ctx);
  const time = `<time class="board-post-time" datetime="${escapeHtml(message.ts || "")}">${escapeHtml(clockTime(message.ts))}</time>`;
  const bubble = `<div class="board-post-bubble">
      ${reply}
      <div class="board-post-text board-markdown${titled ? " board-markdown-titled" : ""}">${decorated}</div>
      ${delivery}
    </div>`;

  if (compact) {
    if (mine) {
      element.innerHTML = `
        <div class="board-post-main">
          <div class="board-post-head board-post-head-mine">${time}</div>
          ${bubble}
          ${seen}
        </div>
        ${actions}`;
      return element;
    }
    element.innerHTML = `
      <div class="board-post-gutter"><time class="board-post-compact-time" datetime="${escapeHtml(message.ts || "")}">${escapeHtml(clockTime(message.ts))}</time></div>
      <div class="board-post-main">
        ${bubble}
        ${seen}
      </div>
      ${actions}`;
    return element;
  }

  if (mine) {
    element.innerHTML = `
      <div class="board-post-main">
        <div class="board-post-head board-post-head-mine">${time}</div>
        ${bubble}
        ${seen}
      </div>
      ${actions}`;
    return element;
  }

  // Hub members may choose a display name; their id stays in the tooltip.
  const from = String(message.from || "");
  const shown = kind === "external_agent" ? (ctx.displayName?.(from) || from) : from;
  const avatar = avatarMarkup({
    name: shown,
    kind,
    // Slack users can share a name with an anima; only animas and hub members own avatars.
    url: group === "anima" || kind === "external_agent" ? ctx.avatarUrl(message.from) : null,
    size: "md",
  });

  element.innerHTML = `
    <div class="board-post-gutter">${avatar}</div>
    <div class="board-post-main">
      <div class="board-post-head">
        <span class="board-post-author"${shown !== from ? ` title="@${escapeAttr(from)}"` : ""}>${escapeHtml(shown || "?")}</span>
        ${badge ? `<span class="board-post-kind-badge board-kind-badge-${group}">${escapeHtml(badge)}</span>` : ""}
        ${time}
      </div>
      ${bubble}
      ${seen}
    </div>
    ${actions}`;
  return element;
}
