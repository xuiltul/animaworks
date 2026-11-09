// ── Workspace Chat Dock — embeds the shared Chat pane beside the stage ──
import { createPaneHost } from "../../pages/chat/pane-host.js";
import { ChatSessionManager } from "../../shared/chat/session-manager.js";
import { presence } from "../../modules/anima-presence.js";
import { setReplyWaiting } from "./office-renderer.js";
import { initChatDockResizer } from "./chat-dock-resizer.js";
import { onEvent } from "./websocket.js";
import { t } from "/shared/i18n.js";

let _dockEl = null;
let _host = null;
let _onClosed = null;
let _resizer = null;
let _presenceUnsub = null;
let _timelineObserver = null;

/** Keep the portrait clear of the Activity Timeline, expanded or not. */
function _syncPresenceSlotOffset() {
  const slot = document.getElementById("wsPresenceSlot");
  const timeline = document.querySelector("#wsStage .ws-timeline");
  if (!slot || !timeline) return;
  if (!_timelineObserver) {
    _timelineObserver = new ResizeObserver(_syncPresenceSlotOffset);
    _timelineObserver.observe(timeline);
  }
  const height = timeline.getBoundingClientRect().height;
  if (height > 0) {
    slot.style.setProperty("--ws-timeline-bar-height", `${Math.round(height)}px`);
  }
}

/** Drive the pixel-office "reply waiting" bubbles from the presence store. */
function _syncReplyWaiting() {
  const names = presence.names().filter((name) => presence.get(name).unread > 0);
  setReplyWaiting(names);
}

function _bindPresence() {
  if (_presenceUnsub) return;
  _presenceUnsub = presence.subscribe(_syncReplyWaiting);
  _syncReplyWaiting();
}

/** Mark an incoming message unread unless its conversation is on screen. */
function _markIncomingUnread(anima, threadId) {
  const pane = _host?.getFocused?.();
  const visible = isConversationOpen()
    && pane?.ctx.state.selectedAnima === anima
    && (pane.ctx.state.selectedThreadId || "default") === threadId;
  if (!visible) presence.addUnread(anima, 1);
}

function _toggleDockFull() {
  const layoutEl = _dockEl?.parentElement;
  if (!layoutEl) return;
  const full = layoutEl.classList.toggle("is-dock-full");
  const handle = _dockEl.querySelector('[data-chat-id="wsDockHandle"]');
  handle?.setAttribute("aria-pressed", String(full));
  _resizer?.refresh();
}

function _render(container) {
  container.innerHTML = `
    <button type="button" class="ws-dock-handle" data-chat-id="wsDockHandle"
      aria-label="${t("ws.dock_expand")}" aria-pressed="false">
      <span class="ws-dock-handle-grip" aria-hidden="true"></span>
    </button>
    <div class="chat-page-layout" data-chat-id="wsChatDockLayout">
      <div class="chat-pane-host" data-chat-id="chatPaneHost"></div>
    </div>
  `;
  container.querySelector('[data-chat-id="wsDockHandle"]')?.addEventListener("click", _toggleDockFull);
  const layoutEl = container.querySelector('[data-chat-id="wsChatDockLayout"]');
  _host = createPaneHost(layoutEl, {
    embedded: true,
    onEvent,
    onClose: closeConversation,
    presenceMount: () => document.getElementById("wsPresenceSlot"),
  });
}

/** Render an incoming message in the pane currently showing that conversation. */
function _renderIncoming(anima, threadId) {
  const pane = _host?.panes.find((p) => p.ctx.state.selectedAnima === anima
    && (p.ctx.state.selectedThreadId || "default") === threadId);
  pane?.ctx.controllers.renderer.renderChat();
}

/** Mirror the Chat screen's WS session-store writes for incoming messages. */
function _bindChatEvents() {
  onEvent("chat.response", (data) => {
    const anima = data.anima || data.name;
    const response = data.response || data.message;
    if (!anima || !response) return;

    const threadId = data.thread_id || "default";
    const mgr = ChatSessionManager.getInstance();
    if (mgr.isStreamingFor(anima, threadId)) return;

    mgr.addMessage(anima, threadId, {
      role: "assistant",
      text: response,
      timestamp: new Date().toISOString(),
    });
    _renderIncoming(anima, threadId);
    _markIncomingUnread(anima, threadId);
  });

  onEvent("anima.proactive_message", (data) => {
    const anima = data.anima || data.name;
    const subject = data.subject || "";
    const body = data.body || data.message || data.summary || "";
    if (!anima || (!subject && !body)) return;

    const text = subject ? `**${subject}**\n${body}` : body;
    const threadId = data.thread_id || "default";
    ChatSessionManager.getInstance().addMessage(anima, threadId, {
      role: "assistant",
      text,
      proactive: true,
      priority: data.priority || "normal",
      timestamp: new Date().toISOString(),
    });
    _renderIncoming(anima, threadId);
    _markIncomingUnread(anima, threadId);
  });
}

function _onWindowResize() {
  _syncPresenceSlotOffset();
  if (window.innerWidth > 768) _dockEl?.parentElement?.classList.remove("is-dock-full");
  _resizer?.refresh();
}

export function initChatDock(container, { onClosed } = {}) {
  if (_host) return;
  _dockEl = container;
  _onClosed = typeof onClosed === "function" ? onClosed : null;
  _dockEl.hidden = true;
  _bindPresence();
  _resizer = initChatDockResizer({
    dock: _dockEl,
    resizer: document.getElementById("wsDockResizer"),
    layout: _dockEl.parentElement,
  });
  _syncPresenceSlotOffset();
  window.addEventListener("resize", _onWindowResize);
}

export function isConversationOpen() {
  return Boolean(_dockEl && !_dockEl.hidden);
}

export function isChatPopoverOpen() {
  return Boolean(document.querySelector(
    ".aw-memory-overlay:not([hidden]), .bustup-overlay.visible, .aw-call-screen:not([hidden]), "
    + ".chat-waiting-list, .chat-add-conversation.open, .chat-unified-user-menu.open, "
    + ".chat-thread-dropdown.open, .thread-archive-wrap.open, .chat-slash-palette:not([hidden]), "
    + ".image-lightbox, .chat-file-viewer-overlay, .text-artifact-modal-overlay"
  ));
}

export async function openConversation(name) {
  if (!name || !_dockEl) return;
  if (!_host) {
    _render(_dockEl);
    _bindChatEvents();
  }
  if (_host.panes.length === 0) _host.addPane();
  _host.bindSharedEvents();
  _dockEl.hidden = false;
  _host.setVisible(true);
  _resizer?.refresh();
  _syncPresenceSlotOffset();
  requestAnimationFrame(_syncPresenceSlotOffset);
  await _host.openAnima(name);
}

export function closeConversation() {
  if (!_host || !_dockEl) return;
  _host.setVisible(false);
  _dockEl.hidden = true;
  _dockEl.parentElement?.classList.remove("is-dock-full");
  const handle = _dockEl.querySelector('[data-chat-id="wsDockHandle"]');
  handle?.setAttribute("aria-pressed", "false");
  _resizer?.refresh();
  _onClosed?.();
}
