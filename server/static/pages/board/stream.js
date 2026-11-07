// ── Board message stream ─────────────────────
// Keyed reconciliation: existing post nodes are reused so expanded long posts,
// scroll position and in-flight selection survive new posts and status events.
import { t } from "/shared/i18n.js";
import { escapeHtml } from "../../modules/state.js";
import { buildPostElement } from "./post.js";
import { isCompactWithPrevious } from "./store.js";
import { clockTime, dayLabel, localDayKey, numberText } from "./time.js";

const FOLD_THRESHOLD = 290;

function _nearBottom(element) {
  return element.scrollHeight - element.scrollTop - element.clientHeight < 80;
}

function _reducedMotion() {
  return typeof matchMedia === "function" && matchMedia("(prefers-reduced-motion: reduce)").matches;
}

/** Set scrollTop immediately, bypassing the stylesheet's smooth scrolling. */
function _scrollToEnd(element) {
  const previous = element.style.scrollBehavior;
  element.style.scrollBehavior = "auto";
  element.scrollTop = element.scrollHeight;
  element.style.scrollBehavior = previous;
}

export function createStream({ element, store }) {
  const nodes = new Map();
  const signatures = new Map();
  // Readers seen at the previous render, per post key. Faces that appear after
  // that are the ones that get the drop animation.
  const seenBefore = new Map();
  let pendingNew = 0;
  let pill = null;

  function avatarSignature(message) {
    const names = [message.from];
    const text = String(message.text || "");
    const mentionRe = /@([a-z][a-z0-9_-]{0,30})/gi;
    let match = mentionRe.exec(text);
    while (match) {
      names.push(match[1]);
      match = mentionRe.exec(text);
    }
    // Delivery chips and read faces render avatars too. Include them so a post
    // is rebuilt once those avatars resolve (the reads can arrive first).
    const notified = message?.delivery?.notified;
    if (Array.isArray(notified)) names.push(...notified);
    names.push(...store.readersForMessage(message));
    return names.map((name) => store.avatarUrl(name) || "").join(",");
  }

  const markdownCtx = {
    t,
    store,
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
    pill = null;
  }

  function _makeDay(dayKey) {
    const day = document.createElement("div");
    day.className = "board-day";
    day.dataset.boardDay = dayKey;
    day.innerHTML = `<span class="board-day-label">${escapeHtml(dayLabel(dayKey))}</span>`;
    return day;
  }

  function _makeNewline(line) {
    const wrap = document.createElement("div");
    wrap.className = "board-newline";
    wrap.dataset.boardNewline = line.afterId;
    const label = t("board.newline");
    const note = line.name
      ? t("board.newline_previous", { name: line.name, time: clockTime(line.ts) })
      : t("board.newline_previous_unknown", { time: clockTime(line.ts) });
    wrap.innerHTML = `<span class="board-newline-label">${escapeHtml(label)}</span><span class="board-newline-note">${escapeHtml(note)}</span>`;
    wrap.title = `${label} · ${note}`;
    return wrap;
  }

  function _filterBar() {
    const bar = document.createElement("div");
    bar.className = "board-filter-bar";
    bar.innerHTML = `<button type="button" class="board-filter-chip board-filter-person board-is-active" data-board-clear-filter title="${escapeHtml(t("board.filter_clear"))}">${escapeHtml(t("board.filter_person_chip", { name: store.state.personFilter }))}</button>`;
    return bar;
  }

  function _newFacesFor(message) {
    const key = store.messageKey(message);
    const current = store.readersForMessage(message);
    const previous = seenBefore.get(key);
    const fresh = new Set(previous ? current.filter((name) => !previous.has(name)) : []);
    seenBefore.set(key, new Set(current));
    return fresh;
  }

  function _reconcile(items) {
    const keep = new Set();
    let cursor = element.firstChild;

    for (const item of items) {
      let node = nodes.get(item.key);
      if (!node) {
        node = item.create();
        nodes.set(item.key, node);
        signatures.set(item.key, item.signature);
      } else if (signatures.get(item.key) !== item.signature) {
        const fresh = item.create();
        // If the node being replaced is the reconciliation cursor, advance the
        // cursor before detaching it; otherwise insertBefore() would receive a
        // node that is no longer a child of the stream.
        if (node === cursor) cursor = node.nextSibling;
        node.replaceWith(fresh);
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

  function _markArrival(node) {
    if (!node || node.dataset.boardArrived === "1") return;
    node.dataset.boardArrived = "1";
    node.classList.add("board-post-arrive");
    setTimeout(() => {
      if (!node.isConnected) return;
      node.classList.remove("board-post-arrive");
      delete node.dataset.boardArrived;
    }, 1200);
  }

  function _focusNode(postId) {
    const message = store.messageById(postId);
    if (!message) return false;
    const node = nodes.get(store.messageKey(message));
    if (!node) return false;
    node.classList.remove("board-post-focus");
    // Force a reflow so re-focusing the same post replays the highlight.
    void node.offsetWidth;
    node.classList.add("board-post-focus");
    node.scrollIntoView({
      block: "center",
      behavior: _reducedMotion() ? "auto" : "smooth",
    });
    setTimeout(() => {
      if (node.isConnected) node.classList.remove("board-post-focus");
    }, 2000);
    store.clearFocus();
    return true;
  }

  function _ensurePill() {
    if (!pill) {
      pill = document.createElement("button");
      pill.type = "button";
      pill.className = "board-new-pill";
      pill.dataset.boardNewPill = "1";
      pill.setAttribute("aria-live", "polite");
    }
    element.append(pill);
    pill.hidden = pendingNew <= 0;
    if (pendingNew > 0) {
      pill.textContent = t("board.new_posts", { count: numberText(pendingNew) });
    }
    return pill;
  }

  function render({ reason = "update" } = {}) {
    const atBottom = _nearBottom(element);
    const previousHeight = element.scrollHeight;
    const previousTop = element.scrollTop;
    const liveKeys = new Set(store.state.liveKeys || []);
    store.state.liveKeys = new Set();

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

    if (reason === "initial" || atBottom) pendingNew = 0;
    if (liveKeys.size > 0 && !atBottom && reason !== "initial") {
      pendingNew += liveKeys.size;
    }

    const items = [];
    if (store.state.personFilter) {
      items.push({
        key: "filter",
        signature: `filter:${store.state.personFilter}`,
        create: _filterBar,
      });
    }

    // The "new from here" line describes the whole board, so hide it while filtering.
    const filtering = store.state.kindFilter !== "all" || Boolean(store.state.query);
    const line = filtering ? null : store.state.readLine;
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
        store.stateSignature(message),
      ].join("|");

      items.push({
        key,
        signature,
        message,
        create: () => buildPostElement(message, {
          ...markdownCtx,
          compact,
          expansionKey,
          newFaces: _newFacesFor(message),
        }),
      });

      if (line && line.afterId === String(message.id)) {
        items.push({
          key: `newline:${line.afterId}`,
          signature: `newline:${line.afterId}:${line.name}:${line.ts}`,
          create: () => _makeNewline(line),
        });
      }

      previous = message;
      previousDay = day;
    }

    _reconcile(items);

    for (const item of items) {
      if (!item.message) continue;
      const node = nodes.get(item.key);
      if (!node) continue;
      _applyFold(node, item.message);
      if (liveKeys.has(item.key)) _markArrival(node);
    }

    _ensurePill();

    const focusId = store.state.focusId;
    const focused = focusId ? _focusNode(focusId) : false;
    if (!focused) {
      if (reason === "older") {
        element.scrollTop = previousTop + Math.max(0, element.scrollHeight - previousHeight);
      } else if (reason === "initial" || atBottom) {
        _scrollToEnd(element);
      } else {
        element.scrollTop = previousTop;
      }
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

  function isNearBottom() {
    return _nearBottom(element);
  }

  function clearNewPill() {
    pendingNew = 0;
    if (pill) pill.hidden = true;
  }

  function scrollToBottom() {
    clearNewPill();
    _scrollToEnd(element);
    store.scheduleMarkRead();
  }

  function destroy() {
    nodes.clear();
    signatures.clear();
    seenBefore.clear();
    pill = null;
  }

  return {
    render,
    applyExpanded,
    updateStatusDots,
    isNearBottom,
    clearNewPill,
    scrollToBottom,
    destroy,
  };
}
