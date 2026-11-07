// ── Board message stream ─────────────────────
// Keyed reconciliation: existing post nodes are reused so expanded long posts,
// scroll position and in-flight selection survive new posts and status events.
import { t } from "/shared/i18n.js";
import { escapeHtml } from "../../modules/state.js";
import { buildPostElement } from "./post.js";
import { isCompactWithPrevious } from "./store.js";
import { dayLabel, localDayKey, numberText } from "./time.js";

const FOLD_THRESHOLD = 290;

function _nearBottom(element) {
  return element.scrollHeight - element.scrollTop - element.clientHeight < 80;
}

export function createStream({ element, store }) {
  const nodes = new Map();
  const signatures = new Map();

  function avatarSignature(message) {
    const names = [message.from];
    const text = String(message.text || "");
    const mentionRe = /@([a-z][a-z0-9_-]{0,30})/g;
    let match = mentionRe.exec(text);
    while (match) {
      names.push(match[1]);
      match = mentionRe.exec(text);
    }
    return names.map((name) => store.avatarUrl(name) || "").join(",");
  }

  const markdownCtx = {
    t,
    isAnima: (name) => store.hasAnima(name),
    hasChannel: (name) => store.hasChannel(name),
    avatarUrl: (name) => store.avatarUrl(name),
    statusOf: (name) => store.statusOf(name),
    statusLabel: (name) => t(`board.mention_status_${store.statusOf(name)}`),
    // Post actions belong to channel boards. Quoting needs a writable board.
    get actions() {
      return store.state.selectedType === "channel";
    },
    get canQuote() {
      if (store.state.selectedType !== "channel") return false;
      const channel = store.channel(store.state.selectedName);
      return !channel || channel.writable !== false;
    },
  };

  function _reset(html) {
    element.innerHTML = html;
    nodes.clear();
    signatures.clear();
  }

  function _makeDay(dayKey) {
    const day = document.createElement("div");
    day.className = "board-day";
    day.dataset.boardDay = dayKey;
    day.innerHTML = `<span class="board-day-label">${escapeHtml(dayLabel(dayKey))}</span>`;
    return day;
  }

  function _filterBar() {
    const bar = document.createElement("div");
    bar.className = "board-filter-bar";
    bar.innerHTML = `<button type="button" class="board-filter-chip board-filter-person board-is-active" data-board-clear-filter title="${escapeHtml(t("board.filter_clear"))}">${escapeHtml(t("board.filter_person_chip", { name: store.state.personFilter }))}</button>`;
    return bar;
  }

  function _reconcile(items) {
    const keep = new Set();
    let cursor = element.firstChild;

    for (const item of items) {
      let node = nodes.get(item.key);
      if (!node || signatures.get(item.key) !== item.signature) {
        const fresh = item.create();
        if (node) node.replaceWith(fresh);
        else element.insertBefore(fresh, cursor);
        node = fresh;
        nodes.set(item.key, node);
        signatures.set(item.key, item.signature);
      }
      keep.add(item.key);
      if (node !== cursor) element.insertBefore(node, cursor);
      cursor = node.nextSibling;
    }

    while (cursor) {
      const next = cursor.nextSibling;
      cursor.remove();
      cursor = next;
    }
    for (const key of [...nodes.keys()]) {
      if (!keep.has(key)) {
        nodes.delete(key);
        signatures.delete(key);
      }
    }
  }

  function _applyFold(postElement, message) {
    if (postElement.dataset.boardFoldChecked === "1") return;
    const textElement = postElement.querySelector(".board-post-text");
    if (!textElement) return;
    postElement.dataset.boardFoldChecked = "1";

    if (textElement.scrollHeight <= FOLD_THRESHOLD) return;

    const key = store.expansionKey(message);
    const expanded = store.isExpanded(key);
    postElement.dataset.boardFoldable = "1";
    if (!expanded) textElement.classList.add("board-md-folded");

    const button = document.createElement("button");
    button.type = "button";
    button.className = "board-more";
    button.dataset.boardFold = key;
    button.setAttribute("aria-expanded", expanded ? "true" : "false");
    button.innerHTML = `<span class="board-more-label">${escapeHtml(expanded ? t("board.fold") : t("board.read_more"))}</span><span class="board-more-count">· ${escapeHtml(t("board.char_count", { count: numberText(message.text?.length || 0) }))}</span>`;
    textElement.after(button);
  }

  function render({ reason = "update" } = {}) {
    const atBottom = _nearBottom(element);
    const previousHeight = element.scrollHeight;
    const previousTop = element.scrollTop;

    if (store.state.loading && store.state.messages.length === 0) {
      _reset(`<div class="board-loading">${escapeHtml(t("board.loading"))}</div>`);
      return;
    }
    if (store.state.error && store.state.messages.length === 0) {
      _reset(`<div class="board-messages-empty">${escapeHtml(t("board.load_failed"))}: ${escapeHtml(store.state.error)}</div>`);
      return;
    }

    const visible = store.visibleMessages();
    if (visible.length === 0) {
      const html = store.state.messages.length === 0
        ? `<div class="board-messages-empty">${escapeHtml(t("board.messages_empty"))}</div>`
        : `<div class="board-messages-empty">${escapeHtml(t("board.no_matches"))}</div>`;
      _reset(html);
      return;
    }

    const items = [];
    if (store.state.personFilter) {
      items.push({
        key: "filter",
        signature: `filter:${store.state.personFilter}`,
        create: _filterBar,
      });
    }

    let previous = null;
    let previousDay = "";
    for (const message of visible) {
      const day = localDayKey(message.ts);
      const dayChanged = day !== previousDay;
      if (dayChanged) {
        items.push({
          key: `day:${day}`,
          signature: `day:${day}`,
          create: () => _makeDay(day),
        });
      }

      const compact = isCompactWithPrevious(previous, message, dayChanged);
      const key = store.messageKey(message);
      const expansionKey = store.expansionKey(message);
      const signature = [
        compact ? "c" : "f",
        expansionKey,
        message.from,
        message.kind || "anima",
        message.ts,
        message.id ?? "",
        message.text || "",
        avatarSignature(message),
        JSON.stringify(message.delivery || {}),
        JSON.stringify(message.hooks || {}),
      ].join("|");

      items.push({
        key,
        signature,
        message,
        create: () => buildPostElement(message, {
          ...markdownCtx,
          compact,
          expansionKey,
        }),
      });

      previous = message;
      previousDay = day;
    }

    _reconcile(items);

    for (const item of items) {
      if (!item.message) continue;
      const node = nodes.get(item.key);
      if (node) _applyFold(node, item.message);
    }

    if (reason === "older") {
      element.scrollTop = previousTop + Math.max(0, element.scrollHeight - previousHeight);
    } else if (reason === "initial" || atBottom) {
      element.scrollTop = element.scrollHeight;
    } else {
      element.scrollTop = previousTop;
    }
  }

  function applyExpanded(key, open) {
    for (const node of nodes.values()) {
      if (node.dataset.boardFoldKey !== key) continue;
      const textElement = node.querySelector(".board-post-text");
      if (!textElement) return;
      textElement.classList.toggle("board-md-folded", !open);
      const button = node.querySelector(".board-more");
      if (button) {
        const length = Number(node.dataset.boardTextLength || 0);
        button.setAttribute("aria-expanded", open ? "true" : "false");
        button.innerHTML = `<span class="board-more-label">${escapeHtml(open ? t("board.fold") : t("board.read_more"))}</span><span class="board-more-count">· ${escapeHtml(t("board.char_count", { count: numberText(length) }))}</span>`;
      }
      return;
    }
  }

  function updateStatusDots() {
    for (const dot of element.querySelectorAll("[data-board-status-anima]")) {
      const name = dot.dataset.boardStatusAnima;
      const status = store.statusOf(name);
      dot.className = `board-status-dot board-status-${status}`;
      const label = t(`board.mention_status_${status}`);
      dot.title = label;
      dot.setAttribute("aria-label", label);
    }
  }

  function destroy() {
    nodes.clear();
    signatures.clear();
  }

  return { render, applyExpanded, updateStatusDots, destroy };
}
