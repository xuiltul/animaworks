// ── Workspace Chat Dock — embeds the shared Chat pane beside the stage ──
import { createPaneHost } from "../../pages/chat/pane-host.js";
import { ChatSessionManager } from "../../shared/chat/session-manager.js";
import { onEvent } from "./websocket.js";

let _dockEl = null;
let _host = null;
let _onClosed = null;

function _render(container) {
  container.innerHTML = `
    <div class="chat-page-layout" data-chat-id="wsChatDockLayout">
      <div class="chat-pane-host" data-chat-id="chatPaneHost"></div>
    </div>
  `;
  const layoutEl = container.querySelector('[data-chat-id="wsChatDockLayout"]');
  _host = createPaneHost(layoutEl, {
    embedded: true,
    onEvent,
    onClose: closeConversation,
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
  });
}

export function initChatDock(container, { onClosed } = {}) {
  if (_host) return;
  _dockEl = container;
  _onClosed = typeof onClosed === "function" ? onClosed : null;
  _dockEl.hidden = true;
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
  await _host.openAnima(name);
}

export function closeConversation() {
  if (!_host || !_dockEl) return;
  _host.setVisible(false);
  _dockEl.hidden = true;
  _onClosed?.();
}
