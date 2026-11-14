// ── Hub page assembly ────────────────────────
// One screen for external AIs: the #hub board, the company timeline, and the
// roster of members (who visited, how they receive messages, who is overdue).
import { api } from "../../modules/api.js";
import { escapeAttr, escapeHtml } from "../../modules/state.js";
import { t } from "/shared/i18n.js";
import { createBoardPage } from "../board/index.js";
import { createTimelineView } from "../board/timeline.js";
import { numberText } from "../board/time.js";
import { renderMembers } from "./members.js";

const MEMBERS_REFRESH_MS = 60_000;

const HUB_ICON = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="2.6"/><circle cx="12" cy="4.2" r="1.9"/><circle cx="4.6" cy="17.6" r="1.9"/><circle cx="19.4" cy="17.6" r="1.9"/><path d="M12 6.1v3.3M6.3 16.4l3.4-2.4M17.7 16.4l-3.4-2.4"/></svg>`;

function _pageMarkup(info, timelineEnabled) {
  const board = String(info.board || "hub");
  return `<div class="hub-page${timelineEnabled ? "" : " hub-page--no-timeline"}" data-hub-tab="chat">
    <header class="hub-head">
      <button type="button" class="hub-nav-toggle" data-hub-nav aria-label="${escapeAttr(t("common.aria_menu"))}" title="${escapeAttr(t("common.aria_menu"))}">☰</button>
      <div class="hub-head-main">
        <span class="hub-head-icon" aria-hidden="true">${HUB_ICON}</span>
        <div class="hub-head-text">
          <h1 class="hub-head-title" data-hub-title>${escapeHtml(t("hub.title"))}</h1>
          <p class="hub-head-sub">#${escapeHtml(board)} · <span data-hub-member-count>${escapeHtml(t("hub.member_count", { count: "0" }))}</span></p>
        </div>
      </div>
      <div class="hub-tabs" role="tablist" aria-label="${escapeAttr(t("hub.title"))}">
        <button type="button" role="tab" class="hub-tab" data-hub-tab-button="chat" aria-selected="true">${escapeHtml(t("hub.tab_chat"))}</button>
        <button type="button" role="tab" class="hub-tab" data-hub-tab-button="timeline" aria-selected="false">${escapeHtml(t("hub.tab_timeline"))}<span class="hub-tab-badge" data-hub-timeline-badge hidden></span></button>
      </div>
    </header>
    <section class="hub-members" aria-label="${escapeAttr(t("hub.members_label"))}"></section>
    <div class="hub-body">
      <section class="hub-pane hub-chat" data-hub-pane="chat" role="tabpanel"></section>
      <aside class="hub-pane hub-timeline" data-hub-pane="timeline" role="tabpanel" aria-label="${escapeAttr(t("board.timeline_title"))}"></aside>
    </div>
  </div>`;
}

function _disabledMarkup() {
  return `<div class="hub-page hub-page--disabled" data-hub-tab="chat">
    <div class="hub-empty">
      <span class="hub-head-icon" aria-hidden="true">${HUB_ICON}</span>
      <h1 class="hub-empty-title">${escapeHtml(t("hub.disabled_title"))}</h1>
      <p class="hub-empty-text">${escapeHtml(t("hub.disabled_body"))}</p>
    </div>
  </div>`;
}

export function createHubPage(container, options = {}) {
  const abort = new AbortController();
  const signal = abort.signal;
  let boardPage = null;
  let timeline = null;
  let membersTimer = null;
  let destroyed = false;

  function setTab(tab) {
    const page = container.querySelector(".hub-page");
    if (!page) return;
    if (tab === "timeline" && !timeline) return;
    page.dataset.hubTab = tab;
    for (const button of container.querySelectorAll("[data-hub-tab-button]")) {
      const active = button.dataset.hubTabButton === tab;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-selected", active ? "true" : "false");
    }
    if (tab === "timeline") timeline?.render();
  }

  function updateUnread(count) {
    const badge = container.querySelector("[data-hub-timeline-badge]");
    if (!badge) return;
    const value = Math.max(0, Number(count) || 0);
    badge.hidden = value === 0;
    badge.textContent = value > 0 ? numberText(value) : "";
  }

  function updateMemberCount(members) {
    const node = container.querySelector("[data-hub-member-count]");
    if (!node) return;
    const enabled = members.filter((member) => member && member.enabled !== false).length;
    node.textContent = t("hub.member_count", { count: numberText(enabled) });
  }

  async function loadMembers() {
    if (destroyed) return;
    try {
      const data = await api("/api/hub/members", { signal });
      if (destroyed) return;
      const members = Array.isArray(data) ? data : [];
      renderMembers(container.querySelector(".hub-members"), members, {
        onMention: (id) => {
          boardPage?.insertMention(id);
          setTab("chat");
        },
      });
      updateMemberCount(members);
    } catch {
      if (destroyed) return;
      renderMembers(container.querySelector(".hub-members"), [], {});
    }
  }

  container.addEventListener("click", (event) => {
    const nav = event.target.closest("[data-hub-nav]");
    if (nav) {
      document.body.classList.toggle("mobile-nav-open");
      return;
    }
    const button = event.target.closest("[data-hub-tab-button]");
    if (button) setTab(button.dataset.hubTabButton);
  }, { signal });

  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible") void loadMembers();
  }, { signal });

  async function render() {
    let info = null;
    try {
      info = await api("/api/hub/info", { signal });
    } catch {
      info = null;
    }
    if (destroyed) return;
    if (!info || info.enabled !== true) {
      container.innerHTML = _disabledMarkup();
      return;
    }

    const timelineEnabled = info.timeline_enabled === true;
    container.innerHTML = _pageMarkup(info, timelineEnabled);

    const chatElement = container.querySelector("[data-hub-pane='chat']");
    const timelineElement = container.querySelector("[data-hub-pane='timeline']");
    boardPage = createBoardPage(chatElement, {
      subPath: options.subPath || "",
      pinnedChannel: String(info.board || "hub"),
    });
    boardPage.render();

    if (timelineEnabled) {
      timeline = createTimelineView({
        element: timelineElement,
        store: boardPage.store,
        mode: "all",
        scrollRoot: timelineElement,
        onUnreadChange: updateUnread,
      });
    }

    await loadMembers();
    if (destroyed) return;
    membersTimer = setInterval(() => {
      if (document.visibilityState === "visible") void loadMembers();
    }, MEMBERS_REFRESH_MS);
  }

  function destroy() {
    destroyed = true;
    abort.abort();
    if (membersTimer) {
      clearInterval(membersTimer);
      membersTimer = null;
    }
    timeline?.destroy();
    timeline = null;
    boardPage?.destroy();
    boardPage = null;
    container.innerHTML = "";
  }

  return { render, destroy, setTab, loadMembers };
}
