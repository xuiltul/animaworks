// ── Left rail: every Anima with live status, plus split / meeting tools ──
import { presence, displayStatus } from "../../modules/anima-presence.js";
import { faceUrl } from "../../shared/chat/emotion-face.js";
import { attachFaceFallback } from "./pane-head.js";
import { hiddenRailAnimas, orderRailAnimas, railBadge } from "./rail-logic.js";
import { openWaitingList } from "./waiting-list.js";

export { orderRailAnimas, railBadge };

const HIDDEN_KEY = "aw-chat-rail-hidden";

function loadHidden() {
  try {
    const list = JSON.parse(localStorage.getItem(HIDDEN_KEY) || "[]");
    return new Set(Array.isArray(list) ? list.filter((n) => typeof n === "string") : []);
  } catch { return new Set(); }
}

function saveHidden(hidden) {
  try { localStorage.setItem(HIDDEN_KEY, JSON.stringify([...hidden])); } catch { /* storage off */ }
}


export function createAnimaRail({ el, getPanes, getFocused, t, escapeHtml, onToolsChanged, onPick }) {
  let _last = "";

  attachFaceFallback(el, (name) => getFocused()?.ctx.state.animaTabAvatarUrls?.[name] || "");

  let _lastActive = null;
  const hidden = loadHidden();
  let addMenu = null;

  function render() {
    const focused = getFocused();
    if (!focused) return;
    const fctx = focused.ctx;
    const panes = getPanes();
    const tabNames = fctx.state.animaTabs.map((tab) => tab.name);
    const active = fctx.state.selectedAnima;
    // A hidden anima still shows up while it waits for a reply or is open in a pane.
    const keep = new Set(panes.map((p) => p.ctx.state.selectedAnima).filter(Boolean));
    for (const name of hidden) if (presence.get(name).awaitingHuman) keep.add(name);
    const list = orderRailAnimas(fctx.state.animas, tabNames, hidden, keep);
    const others = new Set(
      panes.filter((p) => p !== focused).map((p) => p.ctx.state.selectedAnima).filter(Boolean),
    );
    // Folded rail on phones: the header shows a dot when another anima needs attention.
    el.classList.toggle("has-alert", list.some((a) => a.name !== active && railBadge(presence.get(a.name))));

    const items = list.map((a) => {
      const name = a.name;
      const entry = presence.get(name);
      const kind = displayStatus(entry);
      const badge = railBadge(entry);
      const cls = ["chat-rail-item"];
      if (name === active) cls.push("active");
      if (others.has(name)) cls.push("pane2");
      if (kind === "working" || kind === "calling") cls.push("working");
      if (kind === "waiting") cls.push("waiting");
      const stateText = kind === "waiting" ? t("chat.presence.status_waiting")
        : kind === "calling" ? t("chat.presence.status_calling")
          : kind === "working" ? t("chat.presence.status_working") : "";
      const src = faceUrl(name, presence.emotionFor(name), "S");
      const n = escapeHtml(name);
      const fb = fctx.state.animaTabAvatarUrls?.[name];
      if (!fb) fctx.controllers.anima?.ensureAnimaTabAvatar?.(name);
      return `<button type="button" class="${cls.join(" ")}" data-anima="${n}" title="${n}" aria-label="${escapeHtml(t("chat.rail.open_chat", { name }))}"${name === active ? ' aria-current="true"' : ""}>`
        + `<span class="chat-rail-face"><img src="${escapeHtml(src)}" data-face="${n}" alt=""><span class="chat-rail-ring"></span>`
        + (badge ? `<span class="chat-rail-badge ${badge.kind}">${escapeHtml(badge.text)}</span>` : "")
        + `<span class="chat-rail-x" role="button" tabindex="-1" data-hide="${n}" title="${escapeHtml(t("chat.rail.hide"))}" aria-label="${escapeHtml(t("chat.rail.hide"))}">×</span>`
        + `</span><span class="chat-rail-name">${n}</span><span class="chat-rail-state${kind === "waiting" ? " wait" : ""}">${escapeHtml(stateText)}</span></button>`;
    }).join("") + `<button type="button" class="chat-rail-add" title="${escapeHtml(t("chat.rail.add"))}" aria-label="${escapeHtml(t("chat.rail.add"))}">+</button>`;

    const multi = panes.length > 1;
    const full = panes.length >= 4;
    const meetingOn = Boolean(fctx.state.meetingMode);
    const tools = `<span class="chat-rail-sep"></span>`
      + (full ? "" : `<button type="button" class="chat-rail-tool${multi ? " on" : ""}" data-tool="split" title="${escapeHtml(t("pane.split"))}">${escapeHtml(t("chat.presence.rail_split"))}</button>`)
      + `<button type="button" class="chat-rail-tool${meetingOn ? " on" : ""}" data-tool="meeting" title="${escapeHtml(t("meeting.toggle"))}">${escapeHtml(t("chat.presence.rail_meeting"))}</button>`;

    const html = items + tools;
    if (html === _last) return;
    const prevActive = _lastActive;
    _last = html;
    _lastActive = active;
    el.innerHTML = html;

    // Keep the open Anima visible when the rail is longer than the screen.
    if (active && active !== prevActive) {
      el.querySelector(".chat-rail-item.active")?.scrollIntoView?.({ block: "nearest", inline: "nearest" });
    }
  }

  function hideAnima(name, focused) {
    const fctx = focused.ctx;
    const visible = orderRailAnimas(fctx.state.animas, [], hidden).filter((a) => a.name !== name);
    if (visible.length === 0) return; // keep at least one anima on the rail
    hidden.add(name);
    saveHidden(hidden);
    const anima = fctx.controllers.anima;
    if (fctx.state.selectedAnima === name) {
      const next = fctx.state.animaTabs.find((tab) => tab.name !== name && !hidden.has(tab.name))?.name
        || visible[0].name;
      anima.openOrSelectAnima(next);
    }
    if (fctx.state.animaTabs.some((tab) => tab.name === name)) anima.closeAnimaTab?.(name);
    _last = "";
    render();
  }

  function closeAddMenu() {
    addMenu?.remove();
    addMenu = null;
    document.removeEventListener("pointerdown", onOutside, true);
  }

  function onOutside(e) {
    if (addMenu && !addMenu.contains(e.target) && !e.target.closest?.(".chat-rail-add")) closeAddMenu();
  }

  function openAddMenu(button, focused) {
    if (addMenu) { closeAddMenu(); return; }
    const fctx = focused.ctx;
    const options = hiddenRailAnimas(fctx.state.animas, hidden);
    addMenu = document.createElement("div");
    addMenu.className = "chat-rail-add-menu";
    addMenu.setAttribute("role", "menu");
    addMenu.innerHTML = options.length
      ? options.map((a) => {
        const n = escapeHtml(a.name);
        return `<button type="button" role="menuitem" data-show="${n}"><img src="${escapeHtml(faceUrl(a.name, "neutral", "S"))}" data-face="${n}" alt="">${n}</button>`;
      }).join("")
      : `<p>${escapeHtml(t("chat.rail.add_empty"))}</p>`;
    const r = button.getBoundingClientRect();
    addMenu.style.left = `${r.right + 8}px`;
    addMenu.style.top = `${Math.max(8, Math.min(r.top, window.innerHeight - 320))}px`;
    document.body.appendChild(addMenu);
    attachFaceFallback(addMenu, (name) => fctx.state.animaTabAvatarUrls?.[name] || "");
    addMenu.addEventListener("click", (ev) => {
      const pick = ev.target.closest?.("[data-show]");
      if (!pick) return;
      hidden.delete(pick.dataset.show);
      saveHidden(hidden);
      closeAddMenu();
      _last = "";
      getFocused()?.ctx.controllers.anima.openOrSelectAnima(pick.dataset.show);
      onPick?.();
      render();
    });
    document.addEventListener("pointerdown", onOutside, true);
  }

  el.addEventListener("click", (e) => {
    const focused = getFocused();
    if (!focused) return;
    const hide = e.target.closest?.(".chat-rail-x");
    if (hide) {
      e.preventDefault();
      e.stopPropagation();
      hideAnima(hide.dataset.hide, focused);
      return;
    }
    const add = e.target.closest?.(".chat-rail-add");
    if (add) {
      openAddMenu(add, focused);
      return;
    }
    const item = e.target.closest?.(".chat-rail-item");
    const badge = e.target.closest?.(".chat-rail-badge.wait");
    if (item && badge) {
      e.preventDefault();
      openWaitingList({ anchor: badge, anima: item.dataset.anima, getFocused, t, escapeHtml });
      return;
    }
    if (item) {
      if (item._longPressFired) { item._longPressFired = false; return; }
      const name = item.dataset.anima;
      if (!name) return;
      const anima = focused.ctx.controllers.anima;
      if (name === focused.ctx.state.selectedAnima) focused.ctx.controllers.memoryPopover.open(item);
      else anima.openOrSelectAnima(name);
      onPick?.();
      return;
    }
    const tool = e.target.closest?.(".chat-rail-tool");
    if (tool) {
      const id = tool.dataset.tool === "split" ? "chatSplitPaneBtn" : "meetingModeToggle";
      focused.el.querySelector(`[data-chat-id="${id}"]`)?.click();
      setTimeout(() => { _last = ""; onToolsChanged?.(); }, 50);
    }
  });

  return { render, invalidate() { _last = ""; } };
}
