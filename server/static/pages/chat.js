// ── Chat Page (Orchestrator) ──────────────────
import { t } from "/shared/i18n.js";
import { createChatContext, CONSTANTS } from "./chat/ctx.js";
import { createAnimaController } from "./chat/anima-controller.js";
import { createThreadController } from "./chat/thread-controller.js";
import { createChatRenderer } from "./chat/chat-renderer.js";
import { createAvatarController } from "./chat/avatar-controller.js";
import { createActivityController } from "./chat/activity-controller.js";
import { createStreamingController } from "./chat/streaming-controller.js";
import { createHistoryController } from "./chat/history-controller.js";
import { createMemoryController } from "./chat/memory-controller.js";
import { createSidebarController } from "./chat/sidebar-controller.js";
import { createEventsController } from "./chat/events-controller.js";
import { createImageVoiceController } from "./chat/image-voice-controller.js";

let _ctx = null;

export function render(container) {
  const ctx = createChatContext();
  _ctx = ctx;
  ctx.state.container = container;

  // ── Create controllers (order doesn't matter — cross-refs are lazy via ctx.controllers) ──
  ctx.controllers.anima = createAnimaController(ctx);
  ctx.controllers.thread = createThreadController(ctx);
  ctx.controllers.renderer = createChatRenderer(ctx);
  ctx.controllers.avatar = createAvatarController(ctx);
  ctx.controllers.activity = createActivityController(ctx);
  ctx.controllers.streaming = createStreamingController(ctx);
  ctx.controllers.history = createHistoryController(ctx);
  ctx.controllers.memory = createMemoryController(ctx);
  ctx.controllers.sidebar = createSidebarController(ctx);
  ctx.controllers.events = createEventsController(ctx);
  ctx.controllers.imageVoice = createImageVoiceController(ctx);

  // ── DOM ──
  container.innerHTML = `
    <!-- Mobile Tab Bar -->
    <nav class="chat-mobile-tabs" id="chatMobileTabs">
      <button class="chat-mobile-tab active" data-panel="chat" id="chatMobileTabChat">${t("nav.chat")}</button>
      <button class="chat-mobile-tab" data-panel="info" id="chatMobileTabInfo">${t("chat.character_summary")}</button>
    </nav>

    <div class="chat-page-layout" id="chatPageLayout">
      <!-- Left: Chat Panel -->
      <div class="chat-page-main">
        <div class="chat-anima-tabs-header">
          <div class="anima-tabs" id="chatAnimaTabs"></div>
          <div class="chat-header-actions">
            <div class="chat-add-conversation" id="chatAddConversationArea">
              <button type="button" id="chatAddConversationBtn" class="chat-add-conversation-btn">${t("chat.anima_select")}</button>
              <div id="chatAddConversationMenu" class="chat-add-conversation-menu" role="listbox" aria-label="${t("chat.anima_select")}"></div>
            </div>
          </div>
        </div>

        <div class="thread-tabs" id="chatThreadTabs">
          <button class="thread-tab active" data-thread="default">メイン</button>
          <button class="thread-tab-new" id="chatNewThreadBtn" title="新しいスレッド">＋</button>
        </div>

        <!-- Chat Messages -->
        <div id="chatPageMessages" class="chat-messages" style="flex:1; overflow-y:auto; padding:1rem;">
          <div class="chat-empty">${t("chat.anima_select_first")}</div>
        </div>

        <!-- Chat Input -->
        <form id="chatPageForm" class="chat-input-form">
          <div class="image-preview-bar" id="chatPagePreviewBar" style="display:none"></div>
          <div class="pending-queue-bar" id="chatPagePending" style="display:none">
            <div class="pending-queue-header">
              <span class="pending-queue-label" id="chatPagePendingLabel">${t("chat.queue_label")}</span>
              <button class="pending-queue-clear" id="chatPagePendingCancel" type="button" title="${t("chat.queue_clear_all")}">✕ all</button>
            </div>
            <div id="chatPagePendingList"></div>
          </div>
          <div class="chat-input-wrap">
            <textarea
              id="chatPageInput"
              class="chat-input"
              placeholder="${t("chat.placeholder")}"
              autocomplete="off"
              rows="1"
              disabled
            ></textarea>
            <div class="chat-input-actions">
              <button type="button" class="chat-attach-btn" id="chatPageAttachBtn" title="${t("chat.attach_image")}">+</button>
              <button type="button" class="chat-queue-btn" id="chatPageQueueBtn" disabled title="${t("chat.queue_add")}">
                <svg class="chat-queue-icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false">
                  <path d="M12 5v14M5 12l7 7 7-7" />
                </svg>
              </button>
              <button type="submit" class="chat-send-btn" id="chatPageSendBtn" disabled>
                <svg class="chat-send-icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false">
                  <path d="M12 19V5M5 12l7-7 7 7" />
                </svg>
              </button>
            </div>
          </div>
          <input type="file" id="chatPageFileInput" accept="image/jpeg,image/png,image/gif,image/webp" multiple style="display:none" />
        </form>
      </div>

      <div class="chat-right-pane-handle">
        <button
          type="button"
          id="chatRightPaneToggleBtn"
          class="chat-right-pane-toggle-btn"
          aria-label="右ペインを表示・非表示"
          title="右ペインを表示・非表示"
        >
          <svg class="chat-right-pane-toggle-icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false">
            <path d="M15 6l-6 6 6 6" />
          </svg>
        </button>
      </div>

      <!-- Right: Sidebar -->
      <div class="chat-page-sidebar mobile-hidden">
        <nav class="right-tabs" style="display:flex; border-bottom:1px solid var(--border-color, #eee);">
          <button class="right-tab active" data-tab="state" id="chatTabState">${t("chat.state_current")}</button>
          <button class="right-tab" data-tab="activity" id="chatTabActivity">${t("nav.activity")}</button>
          <button class="right-tab" data-tab="history" id="chatTabHistory">${t("chat.history_conversation")}</button>
        </nav>

        <div id="chatRightTabContent" style="padding:0.75rem;">
          <div id="chatPaneState">
            <pre class="state-content" id="chatAnimaState" style="white-space:pre-wrap; word-break:break-word; margin:0;">${t("chat.anima_select_first")}</pre>
          </div>
          <div id="chatPaneActivity" style="display:none;">
            <div id="chatActivityFeed" class="activity-feed">
              <div class="activity-empty">${t("activity.feed_empty")}</div>
            </div>
          </div>
          <div id="chatPaneHistory" style="display:none;">
            <div id="chatHistorySessionList">
              <div class="loading-placeholder">${t("chat.anima_select_first")}</div>
            </div>
            <div id="chatHistoryDetail" style="display:none;">
              <button class="memory-back-btn" id="chatHistoryBackBtn">&larr; ${t("chat.back_list")}</button>
              <h3 id="chatHistoryDetailTitle" style="margin:0.5rem 0;"></h3>
              <div id="chatHistoryConversation" style="max-height:400px; overflow-y:auto;"></div>
            </div>
          </div>

          <div class="chat-memory-section">
            <nav class="memory-tabs" style="display:flex; border-bottom:1px solid var(--border-color, #eee);">
              <button class="memory-tab active" data-tab="episodes">${t("chat.memory_episodes")}</button>
              <button class="memory-tab" data-tab="knowledge">${t("chat.memory_knowledge")}</button>
              <button class="memory-tab" data-tab="procedures">${t("chat.memory_procedures")}</button>
            </nav>
            <div id="chatMemoryFileList" class="memory-file-list" style="padding:0.5rem;">
              <div class="loading-placeholder">${t("chat.anima_select")}</div>
            </div>
            <div id="chatMemoryContentArea" style="display:none; padding:0.5rem;">
              <button class="memory-back-btn" id="chatMemoryBackBtn">&larr; ${t("chat.memory_back")}</button>
              <h3 id="chatMemoryContentTitle" style="margin:0.5rem 0;"></h3>
              <pre id="chatMemoryContentBody" class="memory-content-body" style="white-space:pre-wrap; word-break:break-word;"></pre>
            </div>
          </div>
        </div>
      </div>
    </div>
  `;

  // ── Wire up ──
  ctx.controllers.sidebar.initRightPaneVisibility();
  ctx.controllers.events.bindEvents();
  ctx.controllers.anima.loadAnimas();

  // Auto-refresh intervals
  const actInterval = setInterval(() => ctx.controllers.activity.loadActivity(), 30000);
  ctx.state.intervals.push(actInterval);
  const chatInterval = setInterval(() => ctx.controllers.renderer.pollSelectedChat(), CONSTANTS.CHAT_POLL_INTERVAL_MS);
  ctx.state.intervals.push(chatInterval);
}

export function destroy() {
  if (!_ctx) return;
  const { state } = _ctx;

  for (const id of state.intervals) clearInterval(id);
  state.intervals = [];

  if (state.chatUiStateSaveTimer) {
    clearTimeout(state.chatUiStateSaveTimer);
    state.chatUiStateSaveTimer = null;
  }
  for (const { el, event, handler } of state.boundListeners) {
    el.removeEventListener(event, handler);
  }
  state.boundListeners = [];

  if (state.chatObserver) { state.chatObserver.disconnect(); state.chatObserver = null; }
  for (const [, s] of Object.entries(state.activeStreams)) {
    if (s.abortController) s.abortController.abort();
  }
  state.activeStreams = {};
  state.pendingQueue = [];
  _ctx.controllers.avatar.removeBustupOverlay();

  // Reset all state
  state.bustupUrl = null;
  state.container = null;
  state.animas = [];
  state.selectedAnima = null;
  state.chatHistories = {};
  state.historyState = {};
  state.animaTabs = [];
  state.selectedThreadId = "default";
  state.threads = {};
  state.activeThreadByAnima = {};
  state.animaLastAccess = {};
  state.animaDetail = null;
  state.imageInputManager = null;
  state.animaTabAvatarUrls = {};
  state.animaTabAvatarLoading = {};
  state.rightPaneVisible = true;

  _ctx = null;
}
