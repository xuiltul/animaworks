// ── Board page assembly ──────────────────────
import { t } from "/shared/i18n.js";
import { escapeHtml } from "../../modules/state.js";
import { onEvent } from "../../modules/websocket.js";
import { createBoardStore } from "./store.js";
import { renderRail } from "./rail.js";
import { renderChannelHeader } from "./header.js";
import { renderDmHeader } from "./dm.js";
import { createStream } from "./stream.js";
import { createComposer } from "./composer.js";
import { createDrawer } from "./drawer.js";
import { createMentionCard } from "./mention-card.js";
import { createHomeView } from "./home.js";
import { createFlowsView } from "./flows.js";
import { createPalette } from "./palette.js";

export function createBoardPage(container, options = {}) {
  const store = createBoardStore();
  const mentionCard = createMentionCard();
  const abort = new AbortController();
  let stream = null;
  let composer = null;
  let unsubscribeStore = null;
  let unsubscribeStatus = null;
  let unsubscribeRead = null;
  let composingSearch = false;

  container.innerHTML = `
    <div class="board-layout" id="boardLayout">
      <aside class="board-rail" id="boardRail" aria-label="${escapeHtml(t("board.channels"))}"></aside>
      <section class="board-main">
        <section class="board-home" id="boardHome" hidden></section>
        <section class="board-flows" id="boardFlows" hidden></section>
        <div class="board-placeholder" id="boardPlaceholder">
          <div class="board-placeholder-icon" aria-hidden="true">#</div>
          <div class="board-placeholder-text">${escapeHtml(t("board.channel_select"))}</div>
        </div>
        <div class="board-view" id="boardView" hidden>
          <header class="board-header" id="boardHeader"></header>
          <div class="board-messages" id="boardMessages"></div>
          <div class="board-readonly" id="boardReadonly" hidden></div>
        </div>
      </section>
      <div class="board-palette-backdrop" id="boardPalette" hidden></div>
    </div>`;

  const layout = container.querySelector("#boardLayout");
  const railElement = container.querySelector("#boardRail");
  const homeElement = container.querySelector("#boardHome");
  const flowsElement = container.querySelector("#boardFlows");
  const placeholder = container.querySelector("#boardPlaceholder");
  const view = container.querySelector("#boardView");
  const headerElement = container.querySelector("#boardHeader");
  const messagesElement = container.querySelector("#boardMessages");
  const readonlyElement = container.querySelector("#boardReadonly");
  const mainElement = container.querySelector(".board-main");
  const paletteRoot = container.querySelector("#boardPalette");
  const signal = abort.signal;

  composer = createComposer({
    store,
    onSubmit: (text, options) => store.submit(text, options?.replyTo),
  });
  view.append(composer.element);
  stream = createStream({ element: messagesElement, store });
  const home = createHomeView({ element: homeElement, store });
  const flows = createFlowsView({ element: flowsElement, store });
  const palette = createPalette({
    root: paletteRoot,
    store,
    onSelect: (type, name) => {
      store.select(type, name);
      closeRailOnMobile();
    },
    onView: (next) => {
      store.setView(next);
      closeRailOnMobile();
    },
  });

  function updateDrawerToggle() {
    const button = headerElement.querySelector("[data-board-info]");
    if (button) button.setAttribute("aria-expanded", String(drawer.isOpen()));
  }

  const drawer = createDrawer({
    store,
    onToggle: () => updateDrawerToggle(),
  });
  mainElement.append(drawer.element);

  function syncRailToggles() {
    const open = layout.classList.contains("board-is-rail-open");
    const buttons = [...container.querySelectorAll("[data-board-rail-toggle]")];
    for (const button of buttons) {
      button.setAttribute("aria-expanded", String(open));
    }
    // Keep the legacy mobile-toggle id on the top bar that is actually shown.
    const visible = buttons.find((button) => button.offsetParent !== null);
    if (visible) {
      for (const button of buttons) {
        if (button !== visible && button.id === "boardMobileToggle") {
          button.removeAttribute("id");
        }
      }
      visible.id = "boardMobileToggle";
    }
  }

  function setRailOpen(open) {
    layout.classList.toggle("board-is-rail-open", open);
    syncRailToggles();
  }

  function closeRailOnMobile() {
    if (window.innerWidth <= 760) setRailOpen(false);
  }

  // Touch devices reveal a post's quote/link buttons only after a tap.
  let openActionsPost = null;

  function _isTouchLike() {
    return typeof matchMedia === "function" && matchMedia("(hover: none)").matches;
  }

  function closePostActions() {
    if (openActionsPost) openActionsPost.classList.remove("board-actions-open");
    openActionsPost = null;
  }

  function toggleTouchPostActions(event) {
    if (!_isTouchLike()) return;
    const post = event.target.closest(".board-post");
    const interactive = event.target.closest("a, button, [data-board-mention], .board-seen-face");
    if (interactive) return;
    if (!post) {
      closePostActions();
      return;
    }
    const selection = typeof window.getSelection === "function" ? window.getSelection() : null;
    if (selection && String(selection).trim()) return;
    if (openActionsPost && openActionsPost !== post) closePostActions();
    if (!openActionsPost) {
      post.classList.add("board-actions-open");
      openActionsPost = post;
    }
  }

  function refresh(reason) {
    renderRail(railElement, store);
    syncRailToggles();

    const hasSelection = Boolean(store.state.selectedType && store.state.selectedName);
    const activeView = hasSelection ? null : (store.state.view || "home");
    placeholder.hidden = hasSelection || activeView === "home" || activeView === "flows";
    view.hidden = !hasSelection;
    homeElement.hidden = hasSelection || activeView !== "home";
    flowsElement.hidden = hasSelection || activeView !== "flows";
    if (!hasSelection) {
      drawer.close();
      if (activeView === "flows") flows.show();
      else home.render();
      syncRailToggles();
      return;
    }

    if (store.state.selectedType === "channel") {
      const channel = store.channel(store.state.selectedName);
      renderChannelHeader(headerElement, channel, store);
      const writable = !channel || channel.writable !== false;
      composer.setContext({ type: "channel", name: store.state.selectedName, writable, channel });
      readonlyElement.hidden = writable;
      readonlyElement.textContent = t("board.closed_notice");
      updateDrawerToggle();
      if (drawer.isOpen()) drawer.render(channel);
    } else {
      renderDmHeader(headerElement, store.pair(store.state.selectedName), store);
      composer.setContext({ type: "dm", name: null, writable: false });
      readonlyElement.hidden = false;
      readonlyElement.textContent = t("board.dm_readonly_notice");
      drawer.close();
    }
    syncRailToggles();

    const streamReason = reason === "older"
      ? "older"
      : (reason === "selection" || reason === "loading" ? "initial" : "update");
    stream.render({ reason: streamReason });

    // Opening a board (or returning to the bottom while a live post arrives)
    // advances the read position after a short pause.
    if (
      reason === "messages" &&
      store.state.selectedType === "channel" &&
      store.state.messages.length > 0 &&
      stream.isNearBottom()
    ) {
      store.scheduleMarkRead();
    }
  }

  function onRailClick(event) {
    const viewButton = event.target.closest("[data-board-view]");
    if (viewButton) {
      store.setView(viewButton.dataset.boardView);
      closeRailOnMobile();
      return;
    }
    const searchButton = event.target.closest("[data-board-search]");
    if (searchButton) {
      palette.open();
      return;
    }
    const modeButton = event.target.closest("[data-board-mode]");
    if (modeButton) {
      store.setMode(modeButton.dataset.boardMode);
      return;
    }
    const item = event.target.closest("[data-board-type][data-board-name]");
    if (!item) return;
    store.select(item.dataset.boardType, item.dataset.boardName);
    closeRailOnMobile();
  }

  function onHomeClick(event) {
    const card = event.target.closest("[data-board-home-card]");
    if (!card) return;
    store.select("channel", card.dataset.boardHomeCard);
    closeRailOnMobile();
  }

  function onRailChange(event) {
    const select = event.target.closest("[data-board-dm-filter]");
    if (select) store.setDmFilter(select.value);
  }

  async function copyText(text) {
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(text);
      } else {
        const textarea = document.createElement("textarea");
        textarea.value = text;
        document.body.append(textarea);
        textarea.select();
        document.execCommand("copy");
        textarea.remove();
      }
      return true;
    } catch {
      return false;
    }
  }

  async function copyCode(button) {
    const wrapper = button.closest(".board-code");
    const code = wrapper?.querySelector("pre")?.textContent || "";
    const original = button.textContent;
    if (!(await copyText(code))) {
      button.textContent = original;
      return;
    }
    button.textContent = t("board.copied");
    setTimeout(() => {
      if (button.isConnected) button.textContent = original;
    }, 1500);
  }

  async function onStreamClick(event) {
    toggleTouchPostActions(event);

    const pill = event.target.closest("[data-board-new-pill]");
    if (pill) {
      stream.scrollToBottom();
      return;
    }

    const jump = event.target.closest("[data-board-jump]");
    if (jump) {
      void store.jumpTo(jump.dataset.boardJump);
      return;
    }

    const seen = event.target.closest("[data-board-seen]");
    if (seen) {
      seen.classList.toggle("board-is-open");
      return;
    }

    const mention = event.target.closest("[data-board-mention]");
    if (mention) {
      const name = mention.dataset.boardMention;
      mentionCard.open(name, mention.getBoundingClientRect(), store, (filterName) => {
        store.setPersonFilter(filterName);
      });
      return;
    }

    const channelLink = event.target.closest("[data-board-channel]");
    if (channelLink) {
      store.select("channel", channelLink.dataset.boardChannel);
      closeRailOnMobile();
      return;
    }

    const fold = event.target.closest("[data-board-fold]");
    if (fold) {
      const key = fold.dataset.boardFold;
      const open = store.toggleExpanded(key);
      stream.applyExpanded(key, open);
      return;
    }

    const copyButton = event.target.closest("[data-board-copy]");
    if (copyButton) {
      void copyCode(copyButton);
      return;
    }

    const actionButton = event.target.closest("[data-board-action]");
    if (actionButton) {
      const postElement = actionButton.closest(".board-post");
      const id = postElement?.dataset.boardId;
      const message = store.messageById(id);
      if (!message) return;
      if (actionButton.dataset.boardAction === "quote") {
        composer.quote(message);
        return;
      }
      if (actionButton.dataset.boardAction === "link") {
        const original = actionButton.textContent;
        const copied = await copyText(`#/board/${store.state.selectedName}/${message.id}`);
        if (copied) {
          actionButton.textContent = t("board.link_copied");
          setTimeout(() => {
            if (actionButton.isConnected) actionButton.textContent = original;
          }, 1500);
        }
        return;
      }
    }

    if (event.target.closest("[data-board-clear-filter]")) {
      store.setPersonFilter("");
    }
  }

  function onScroll() {
    if (messagesElement.scrollTop < 48) void store.loadOlder();
    if (stream.isNearBottom()) {
      stream.clearNewPill();
      store.scheduleMarkRead();
    }
  }

  function onHeaderClick(event) {
    const searchToggle = event.target.closest("[data-board-search-toggle]");
    if (searchToggle) {
      const open = !store.state.filtersOpen;
      store.setFiltersOpen(open);
      if (open) {
        const input = headerElement.querySelector("[data-board-search-input]");
        if (input) input.focus();
      }
      return;
    }
    if (!event.target.closest("[data-board-info]")) return;
    if (store.state.selectedType !== "channel") return;
    const channel = store.channel(store.state.selectedName);
    if (!channel) return;
    if (drawer.isOpen()) drawer.close();
    else drawer.open(channel);
    updateDrawerToggle();
  }

  function onHeaderKeydown(event) {
    if (event.key !== "Escape" || !store.state.filtersOpen) return;
    store.setFiltersOpen(false);
  }

  function onHeaderFilterClick(event) {
    const kindChip = event.target.closest("[data-board-kind]");
    if (kindChip) {
      store.setKindFilter(kindChip.dataset.boardKind);
      return;
    }
    if (event.target.closest("[data-board-clear-filter]")) {
      store.setPersonFilter("");
    }
  }

  function onHeaderFilterInput(event) {
    const input = event.target.closest("[data-board-search-input]");
    if (!input || composingSearch || event.isComposing) return;
    store.setQuery(input.value);
  }

  function onHeaderCompositionEnd(event) {
    composingSearch = false;
    const input = event.target.closest("[data-board-search-input]");
    if (input) store.setQuery(input.value);
  }

  railElement.addEventListener("click", onRailClick, { signal });
  railElement.addEventListener("change", onRailChange, { signal });
  homeElement.addEventListener("click", onHomeClick, { signal });
  headerElement.addEventListener("click", onHeaderClick, { signal });
  document.addEventListener("keydown", onHeaderKeydown, { signal });
  headerElement.addEventListener("click", onHeaderFilterClick, { signal });
  headerElement.addEventListener("input", onHeaderFilterInput, { signal });
  headerElement.addEventListener("compositionstart", () => {
    composingSearch = true;
  }, { signal });
  headerElement.addEventListener("compositionend", onHeaderCompositionEnd, { signal });
  messagesElement.addEventListener("click", onStreamClick, { signal });
  messagesElement.addEventListener("scroll", onScroll, { signal });
  container.addEventListener("click", (event) => {
    if (event.target.closest("[data-board-nav]")) {
      document.body.classList.toggle("mobile-nav-open");
      return;
    }
    if (event.target.closest("[data-board-rail-toggle]")) {
      setRailOpen(!layout.classList.contains("board-is-rail-open"));
      return;
    }
    const viewLink = event.target.closest("[data-board-view-link]");
    if (viewLink) {
      store.setView(viewLink.dataset.boardViewLink);
      closeRailOnMobile();
      return;
    }
    if (_isTouchLike() && !event.target.closest(".board-post")) closePostActions();
  }, { signal });

  unsubscribeStore = store.subscribe((reason) => {
    if (reason === "status") {
      stream.updateStatusDots();
      // Delivery chips depend on the anima's current work status ("考え中…"),
      // so a status change must reconcile the affected posts as well.
      stream.render({ reason: "update" });
      mentionCard.update(store);
      composer.refresh();
      return;
    }
    if (reason === "mode" || reason === "dm-filter") {
      renderRail(railElement, store);
      return;
    }
    if (reason === "filter" || reason === "filters" || reason === "filters-open") {
      refresh(reason);
      return;
    }
    refresh(reason);
  });

  unsubscribeStatus = onEvent("anima.status", (data) => store.handleAnimaStatus(data));
  window.__boardWsHandler = (data) => {
    const timelineItem =
      data && data.id !== undefined && ["notified", "done", "decided"].includes(data.kind);
    if (timelineItem) {
      home.handleTimelineItem(data);
      return;
    }
    if (
      data &&
      data.text === undefined &&
      data.status !== undefined &&
      (data.name !== undefined || data.anima !== undefined)
    ) {
      store.handleAnimaStatus(data);
    } else if (data && data.text === undefined && (data.reader !== undefined || data.upto_id !== undefined)) {
      void store.handleBoardRead(data);
    } else {
      void store.handleBoardPost(data);
    }
  };
  window.__boardReadWsHandler = (data) => {
    void store.handleBoardRead(data);
  };
  window.__boardTimelineWsHandler = (data) => home.handleTimelineItem(data);

  function parseSubPath(subPath) {
    const text = String(subPath || "").replace(/^\/+|\/+$/g, "");
    if (!text) return null;
    const slash = text.indexOf("/");
    const rawName = slash >= 0 ? text.slice(0, slash) : text;
    const rawId = slash >= 0 ? text.slice(slash + 1) : "";
    if (!rawName) return null;
    try {
      return {
        channel: decodeURIComponent(rawName),
        id: rawId ? decodeURIComponent(rawId) : "",
      };
    } catch {
      return { channel: rawName, id: rawId };
    }
  }

  function render() {
    const target = parseSubPath(options.subPath);
    if (target) {
      store.select("channel", target.channel);
      if (target.id) void store.jumpTo(target.id);
    } else {
      refresh("selection");
    }
    store.start();
  }

  function destroy() {
    abort.abort();
    store.stop();
    if (unsubscribeStore) unsubscribeStore();
    if (unsubscribeStatus) unsubscribeStatus();
    if (unsubscribeRead) unsubscribeRead();
    if (window.__boardWsHandler) window.__boardWsHandler = null;
    if (window.__boardReadWsHandler) window.__boardReadWsHandler = null;
    if (window.__boardTimelineWsHandler) window.__boardTimelineWsHandler = null;
    mentionCard.destroy();
    stream.destroy();
    composer.destroy();
    home.destroy();
    flows.destroy();
    palette.destroy();
    drawer.destroy();
    container.innerHTML = "";
  }

  return { render, destroy };
}
