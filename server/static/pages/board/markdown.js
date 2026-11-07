// ── Board markdown decoration ───────────────
// renderMarkdown() produces the raw HTML. This module adds the board-specific
// affordances: mention/channel chips, horizontally scrollable tables, numeric
// column alignment and copyable code blocks.
import { escapeHtml } from "../../modules/state.js";
import { avatarMarkup } from "./avatar.js";

const MENTION_RE = /@([a-z][a-z0-9_-]{0,30})|#([a-z][a-z0-9_-]{0,30})/g;
const NUMERIC_RE = /^[-+¥$€£]?[\d][\d.,:%+\-−\s]*$/;

function _statusDot(name, status, label) {
  return `<span class="board-status-dot board-status-${escapeHtml(status)}" data-board-status-anima="${escapeHtml(name)}" title="${escapeHtml(label)}" aria-label="${escapeHtml(label)}"></span>`;
}

function _mentionChip(name, ctx) {
  const status = ctx.statusOf(name);
  const label = ctx.statusLabel(name);
  return `<span class="board-mention-avatar">${avatarMarkup({ name, kind: "anima", url: ctx.avatarUrl(name), size: "xs" })}</span><span class="board-mention-name">@${escapeHtml(name)}</span>${_statusDot(name, status, label)}`;
}

function _decorateMentions(root, ctx) {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode(node) {
      if (!node.nodeValue || !/[@#]/.test(node.nodeValue)) return NodeFilter.FILTER_REJECT;
      const parent = node.parentElement;
      if (!parent || parent.closest("pre, code, a, button")) return NodeFilter.FILTER_REJECT;
      return NodeFilter.FILTER_ACCEPT;
    },
  });

  const textNodes = [];
  while (walker.nextNode()) textNodes.push(walker.currentNode);

  for (const node of textNodes) {
    const text = node.nodeValue;
    const parts = [];
    let lastIndex = 0;
    let replaced = false;
    MENTION_RE.lastIndex = 0;
    let match = MENTION_RE.exec(text);

    while (match) {
      const mentionName = match[1];
      const channelName = match[2];
      const name = mentionName || channelName;
      const before = match.index > 0 ? text[match.index - 1] : "";
      const validBoundary = !before || !/[a-z0-9_-]/i.test(before);
      const known = mentionName ? ctx.isAnima(name) : ctx.hasChannel(name);

      if (!validBoundary || !known) {
        match = MENTION_RE.exec(text);
        continue;
      }

      if (match.index > lastIndex) parts.push(text.slice(lastIndex, match.index));

      if (mentionName) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "board-mention";
        button.dataset.boardMention = name;
        button.title = `@${name}`;
        button.innerHTML = _mentionChip(name, ctx);
        parts.push(button);
      } else {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "board-channel-ref";
        button.dataset.boardChannel = name;
        button.textContent = `#${name}`;
        parts.push(button);
      }

      replaced = true;
      lastIndex = match.index + match[0].length;
      match = MENTION_RE.exec(text);
    }

    if (!replaced) continue;
    if (lastIndex < text.length) parts.push(text.slice(lastIndex));

    const fragment = document.createDocumentFragment();
    for (const part of parts) {
      fragment.append(typeof part === "string" ? document.createTextNode(part) : part);
    }
    node.replaceWith(fragment);
  }
}

function _decorateTables(root) {
  for (const table of [...root.querySelectorAll("table")]) {
    if (table.parentElement?.classList.contains("board-table-wrap")) continue;
    const wrapper = document.createElement("div");
    wrapper.className = "board-table-wrap";
    table.replaceWith(wrapper);
    wrapper.append(table);

    for (const cell of table.querySelectorAll("th, td")) {
      const text = (cell.textContent || "").trim();
      if (text && text.length <= 24 && NUMERIC_RE.test(text) && /\d/.test(text)) {
        cell.classList.add("board-num");
      }
    }
  }
}

function _decorateCode(root, ctx) {
  for (const pre of [...root.querySelectorAll("pre")]) {
    if (pre.parentElement?.classList.contains("board-code")) continue;
    const wrapper = document.createElement("div");
    wrapper.className = "board-code";
    pre.replaceWith(wrapper);
    wrapper.append(pre);

    const button = document.createElement("button");
    button.type = "button";
    button.className = "board-code-copy";
    button.dataset.boardCopy = "";
    button.textContent = ctx.t("board.copy");
    button.setAttribute("aria-label", ctx.t("board.copy_code"));
    wrapper.append(button);
  }
}

/** Decorate rendered markdown HTML with board affordances. */
export function decorateMarkdown(html, ctx) {
  const root = document.createElement("div");
  root.innerHTML = html || "";
  _decorateTables(root);
  _decorateCode(root, ctx);
  _decorateMentions(root, ctx);
  return root.innerHTML;
}
