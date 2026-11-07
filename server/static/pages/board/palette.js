// ── Board palette (⌘K / Ctrl+K) ─────────────
// Tiny command window for jumping to a board, the home or the flow view.
import { escapeAttr, escapeHtml } from "../../modules/state.js";
import { t } from "/shared/i18n.js";

const MAX_ITEMS = 20;

function _itemLabel(item) {
  return item.kind === "view" ? item.label : `#${item.label}`;
}

export function createPalette({ root, store, onSelect, onView }) {
  root.innerHTML = `<div class="board-palette" role="dialog" aria-modal="true" aria-label="${escapeAttr(t("board.search_aria"))}">
    <form class="board-palette-form" role="search">
      <input type="search" class="board-palette-input" id="boardPaletteInput" autocomplete="off" spellcheck="false" placeholder="${escapeAttr(t("board.search_placeholder"))}" aria-label="${escapeAttr(t("board.search_placeholder"))}">
    </form>
    <div class="board-palette-list" id="boardPaletteList" role="listbox"></div>
  </div>`;

  const input = root.querySelector("#boardPaletteInput");
  const list = root.querySelector("#boardPaletteList");
  let items = [];
  let active = 0;
  let isOpen = false;

  function _candidates() {
    const home = {
      kind: "view",
      id: "home",
      label: t("board.home_rail"),
      sub: t("board.search_home_sub"),
      search: `${t("board.home_rail")} ${t("board.search_home_sub")}`.toLowerCase(),
    };
    const flows = {
      kind: "view",
      id: "flows",
      label: t("board.flows_title"),
      sub: t("board.search_flows_sub"),
      search: `${t("board.flows_title")} ${t("board.search_flows_sub")}`.toLowerCase(),
    };
    const boards = store.state.channels.map((channel) => ({
      kind: "channel",
      id: channel.name,
      label: channel.name,
      sub: [channel.company, channel.description].filter(Boolean).join(" · "),
      search: `${channel.name} ${channel.company || ""} ${channel.description || ""}`.toLowerCase(),
    }));
    return [home, flows, ...boards];
  }

  function _filter(query) {
    const needle = String(query || "").trim().toLowerCase();
    const source = _candidates();
    const matched = needle
      ? source.filter((item) => item.search.includes(needle))
      : source;
    return matched.slice(0, MAX_ITEMS);
  }

  function _render() {
    if (items.length === 0) {
      list.innerHTML = `<div class="board-palette-empty">${escapeHtml(t("board.search_empty"))}</div>`;
      return;
    }
    list.innerHTML = items.map((item, index) => `<button type="button" class="board-palette-item${index === active ? " is-active" : ""}" role="option" aria-selected="${index === active}" data-board-palette-item="${escapeAttr(item.id)}" data-board-palette-kind="${item.kind}" data-board-palette-index="${index}">
      <span class="board-palette-label">${escapeHtml(_itemLabel(item))}</span>
      ${item.sub ? `<span class="board-palette-sub">${escapeHtml(item.sub)}</span>` : ""}
    </button>`).join("");
  }

  function _refresh(query) {
    items = _filter(query);
    active = Math.min(active, Math.max(0, items.length - 1));
    _render();
  }

  function open() {
    isOpen = true;
    root.hidden = false;
    active = 0;
    input.value = "";
    _refresh("");
    input.focus();
  }

  function close() {
    isOpen = false;
    root.hidden = true;
    input.value = "";
  }

  function choose(item) {
    if (!item) return;
    if (item.kind === "view") onView(item.id);
    else onSelect(item.kind, item.id);
    close();
  }

  function _move(delta) {
    if (items.length === 0) return;
    active = (active + delta + items.length) % items.length;
    _render();
    const current = list.querySelector(`[data-board-palette-index="${active}"]`);
    current?.scrollIntoView({ block: "nearest" });
  }

  function onInputKeyDown(event) {
    if (event.isComposing || event.keyCode === 229) return;
    if (event.key === "ArrowDown") {
      event.preventDefault();
      _move(1);
      return;
    }
    if (event.key === "ArrowUp") {
      event.preventDefault();
      _move(-1);
      return;
    }
    if (event.key === "Enter") {
      event.preventDefault();
      choose(items[active]);
      return;
    }
    if (event.key === "Escape") {
      event.preventDefault();
      close();
    }
  }

  function onRootClick(event) {
    const item = event.target.closest("[data-board-palette-item]");
    if (item) {
      const index = Number(item.dataset.boardPaletteIndex);
      choose(items[index]);
      return;
    }
    if (event.target === root) close();
  }

  function onGlobalKeyDown(event) {
    const key = String(event.key || "").toLowerCase();
    if ((event.metaKey || event.ctrlKey) && key === "k") {
      event.preventDefault();
      if (isOpen) close();
      else open();
      return;
    }
    if (event.key === "Escape" && isOpen) {
      event.preventDefault();
      close();
    }
  }

  input.addEventListener("input", () => {
    active = 0;
    _refresh(input.value);
    const current = list.querySelector(`[data-board-palette-index="${active}"]`);
    if (current) current.scrollIntoView({ block: "nearest" });
  });
  input.addEventListener("keydown", onInputKeyDown);
  root.addEventListener("click", onRootClick);
  window.addEventListener("keydown", onGlobalKeyDown);

  function destroy() {
    window.removeEventListener("keydown", onGlobalKeyDown);
    root.removeEventListener("click", onRootClick);
    input.removeEventListener("keydown", onInputKeyDown);
    root.innerHTML = "";
  }

  return { open, close, destroy };
}
