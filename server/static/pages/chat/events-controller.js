// ── Event Binding Controller ──────────────────
import { $, saveDraft, chatInputMaxHeight } from "./ctx.js";

export function createEventsController(ctx) {
  const { state, deps } = ctx;
  const { t } = deps;

  function addListener(id, event, handler) {
    const el = $(id);
    if (el) {
      el.addEventListener(event, handler);
      state.boundListeners.push({ el, event, handler });
    }
  }

  function bindEvents() {
    // Mobile tab switching
    for (const tabId of ["chatMobileTabChat", "chatMobileTabInfo"]) {
      addListener(tabId, "click", e => ctx.controllers.sidebar.switchMobileTab(e.target.dataset.panel));
    }

    // Escape to dismiss bustup overlay
    document.addEventListener("keydown", ctx.controllers.avatar.onBustupEscape);
    state.boundListeners.push({ el: document, event: "keydown", handler: ctx.controllers.avatar.onBustupEscape });

    // Add conversation picker
    addListener("chatAddConversationBtn", "click", e => {
      e.stopPropagation();
      const area = $("chatAddConversationArea");
      if (!area) return;
      const nextOpen = !area.classList.contains("open");
      area.classList.toggle("open", nextOpen);
      if (nextOpen) ctx.controllers.anima.renderAddConversationMenu();
    });
    const closeMenu = e => {
      const area = $("chatAddConversationArea");
      if (!area || !area.classList.contains("open")) return;
      if (e.target instanceof Element && area.contains(e.target)) return;
      area.classList.remove("open");
    };
    document.addEventListener("pointerdown", closeMenu);
    state.boundListeners.push({ el: document, event: "pointerdown", handler: closeMenu });

    // New thread
    addListener("chatNewThreadBtn", "click", () => ctx.controllers.thread.createNewThread());

    // Chat form submit
    addListener("chatPageForm", "submit", e => { e.preventDefault(); ctx.controllers.streaming.submitChat(); });

    // Focus textarea on wrap click
    const inputWrap = state.container.querySelector(".chat-input-wrap");
    const focusInput = e => {
      if (e.target instanceof Element && e.target.closest("button, input, select, textarea, a")) return;
      $("chatPageInput")?.focus();
    };
    if (inputWrap) {
      inputWrap.addEventListener("pointerdown", focusInput);
      state.boundListeners.push({ el: inputWrap, event: "pointerdown", handler: focusInput });
      inputWrap.addEventListener("click", focusInput);
      state.boundListeners.push({ el: inputWrap, event: "click", handler: focusInput });
    }

    // Textarea shortcuts
    addListener("chatPageInput", "keydown", e => {
      if (e.key === "Enter" && e.altKey) { e.preventDefault(); ctx.controllers.streaming.addToQueue(); }
      else if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); ctx.controllers.streaming.submitChat(); }
    });

    // Queue / pending
    addListener("chatPageQueueBtn", "click", () => ctx.controllers.streaming.addToQueue());
    addListener("chatPagePendingCancel", "click", () => {
      state.pendingQueue = [];
      ctx.controllers.streaming.hidePendingIndicator();
      ctx.controllers.streaming.updateSendButton();
    });

    // Auto-resize + draft save
    addListener("chatPageInput", "input", () => {
      const el = $("chatPageInput");
      if (el) {
        el.style.height = "auto";
        el.style.height = Math.min(el.scrollHeight, chatInputMaxHeight()) + "px";
        saveDraft(state.selectedAnima, el.value || "");
      }
      ctx.controllers.streaming.updateSendButton();
    });

    // Right tab switching
    for (const tabId of ["chatTabState", "chatTabActivity", "chatTabHistory"]) {
      addListener(tabId, "click", e => ctx.controllers.sidebar.switchRightTab(e.target.dataset.tab));
    }
    addListener("chatRightPaneToggleBtn", "click", () => ctx.controllers.sidebar.toggleRightPane());

    // Memory tabs
    state.container.querySelectorAll(".memory-tab").forEach(btn => {
      const handler = () => {
        state.activeMemoryTab = btn.dataset.tab;
        state.container.querySelectorAll(".memory-tab").forEach(b => b.classList.toggle("active", b.dataset.tab === state.activeMemoryTab));
        const contentArea = $("chatMemoryContentArea");
        const fileList = $("chatMemoryFileList");
        if (contentArea) contentArea.style.display = "none";
        if (fileList) fileList.style.display = "";
        ctx.controllers.memory.loadMemoryTab();
      };
      btn.addEventListener("click", handler);
      state.boundListeners.push({ el: btn, event: "click", handler });
    });

    // Attach / file input
    addListener("chatPageAttachBtn", "click", () => { $("chatPageFileInput")?.click(); });
    addListener("chatPageFileInput", "change", () => {
      const fi = $("chatPageFileInput");
      if (fi?.files.length > 0) { state.imageInputManager?.addFiles(fi.files); fi.value = ""; }
    });

    // Image input + voice init
    ctx.controllers.imageVoice.initImageInput();

    // Memory back
    addListener("chatMemoryBackBtn", "click", () => {
      const ca = $("chatMemoryContentArea");
      const fl = $("chatMemoryFileList");
      if (ca) ca.style.display = "none";
      if (fl) fl.style.display = "";
    });

    // History back
    addListener("chatHistoryBackBtn", "click", () => {
      const detail = $("chatHistoryDetail");
      const list = $("chatHistorySessionList");
      if (detail) detail.style.display = "none";
      if (list) list.style.display = "";
    });

    // Infinite scroll observer
    ctx.controllers.renderer.setupChatObserver();
  }

  return { bindEvents };
}
