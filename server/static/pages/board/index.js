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
import { createMentionCard } from "./mention-card.js";
import { createHomeView } from "./home.js";
import { createFlowsView } from "./flows.js";
import { createPalette } from "./palette.js";

export function createBoardPage(container) {
  const store = createBoardStore();
  const mentionCard = createMentionCard();
  const abort = new AbortController();
  let stream = null;
  let composer = null;
  let unsubscribeStore = null;
  let unsubscribeStatus = null;

  container.innerHTML = `
    <div class="board-layout" id="boardLayout">
      <button type="button" class="board-mobile-toggle" id="boardMobileToggle" aria-expanded="false">${escapeHtml(t("board.rail_toggle"))}</button>
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
  const mobileToggle = container.querySelector("#boardMobileToggle");
  const paletteRoot = container.querySelector("#boardPalette");
  const signal = abort.signal;

  composer = createComposer({ onSubmit: (text) => store.submit(text) });
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

  function setRailOpen(open) {
    layout.classList.toggle("board-is-rail-open", open);
    mobileToggle.setAttribute("aria-expanded", String(open));
  }

  function closeRailOnMobile() {
    if (window.innerWidth <= 760) setRailOpen(false);
  }

  function refresh(reason) {
    renderRail(railElement, store);

    const hasSelection = Boolean(store.state.selectedType && store.state.selectedName);
    const activeView = hasSelection ? null : (store.state.view || "home");
    placeholder.hidden = hasSelection || activeView === "home" || activeView === "flows";
    view.hidden = !hasSelection;
    homeElement.hidden = hasSelection || activeView !== "home";
    flowsElement.hidden = hasSelection || activeView !== "flows";
    if (!hasSelection) {
      if (activeView === "flows") flows.show();
      else home.render();
      return;
    }

    if (store.state.selectedType === "channel") {
      const channel = store.channel(store.state.selectedName);
      renderChannelHeader(headerElement, channel, store);
      const writable = !channel || channel.writable !== false;
      composer.setContext({ type: "channel", name: store.state.selectedName, writable });
      readonlyElement.hidden = writable;
      readonlyElement.textContent = t("board.closed_notice");
    } else {
      renderDmHeader(headerElement, store.pair(store.state.selectedName), store);
      composer.setContext({ type: "dm", name: null, writable: false });
      readonlyElement.hidden = false;
      readonlyElement.textContent = t("board.dm_readonly_notice");
    }

    const streamReason = reason === "older"
      ? "older"
      : (reason === "selection" || reason === "loading" ? "initial" : "update");
    stream.render({ reason: streamReason });
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

  async function copyCode(button) {
    const wrapper = button.closest(".board-code");
    const code = wrapper?.querySelector("pre")?.textContent || "";
    const original = button.textContent;
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(code);
      } else {
        const textarea = document.createElement("textarea");
        textarea.value = code;
        document.body.append(textarea);
        textarea.select();
        document.execCommand("copy");
        textarea.remove();
      }
      button.textContent = t("board.copied");
      setTimeout(() => {
        if (button.isConnected) button.textContent = original;
      }, 1500);
    } catch {
      button.textContent = original;
    }
  }

  function onStreamClick(event) {
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

    if (event.target.closest("[data-board-clear-filter]")) {
      store.setPersonFilter("");
    }
  }

  function onScroll() {
    if (messagesElement.scrollTop < 48) void store.loadOlder();
  }

  railElement.addEventListener("click", onRailClick, { signal });
  railElement.addEventListener("change", onRailChange, { signal });
  homeElement.addEventListener("click", onHomeClick, { signal });
  messagesElement.addEventListener("click", onStreamClick, { signal });
  messagesElement.addEventListener("scroll", onScroll, { signal });
  mobileToggle.addEventListener("click", () => {
    setRailOpen(!layout.classList.contains("board-is-rail-open"));
  }, { signal });

  unsubscribeStore = store.subscribe((reason) => {
    if (reason === "status") {
      stream.updateStatusDots();
      mentionCard.update(store);
      return;
    }
    if (reason === "mode" || reason === "dm-filter") {
      renderRail(railElement, store);
      return;
    }
    if (reason === "filter") {
      stream.render({ reason: "filter" });
      return;
    }
    refresh(reason);
  });

  unsubscribeStatus = onEvent("anima.status", (data) => store.handleAnimaStatus(data));
  window.__boardWsHandler = (data) => {
    void store.handleBoardPost(data);
  };

  function render() {
    refresh("selection");
    store.start();
  }

  function destroy() {
    abort.abort();
    store.stop();
    if (unsubscribeStore) unsubscribeStore();
    if (unsubscribeStatus) unsubscribeStatus();
    if (window.__boardWsHandler) window.__boardWsHandler = null;
    mentionCard.destroy();
    stream.destroy();
    composer.destroy();
    home.destroy();
    flows.destroy();
    palette.destroy();
    container.innerHTML = "";
  }

  return { render, destroy };
}
