// ── Streaming / Send / Queue Controller ────────
import {
  saveDraft, clearDraft, chatInputMaxHeight,
  scheduleSaveChatUiState, CONSTANTS,
} from "./ctx.js";
import { getDescendants } from "../../shared/chat/org-utils.js";

const SEND_BTN_ICONS = {
  send: `<svg class="chat-send-icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M12 19V5M5 12l7-7 7 7" /></svg>`,
  stop: `<svg class="chat-send-icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><rect x="5" y="5" width="14" height="14" rx="2.5" /></svg>`,
  interrupt: `<span class="chat-send-icon-group" aria-hidden="true"><svg class="chat-send-icon chat-send-icon-square" viewBox="0 0 24 24" focusable="false"><rect x="5" y="5" width="14" height="14" rx="2.5" /></svg><svg class="chat-send-icon" viewBox="0 0 24 24" focusable="false"><path d="M12 19V5M5 12l7-7 7 7" /></svg></span>`,
};

export function createStreamingController(ctx) {
  const $ = ctx.$;
  const { state, deps } = ctx;
  const { t, escapeHtml, logger, fetchActiveStream, fetchStreamProgress } = deps;
  const mgr = state.manager;

  mgr.addEventListener("stream-state-changed", () => {
    updateSendButton();
    ctx.controllers.anima?.renderAnimaTabs();
  });

  function isAnimaStreaming(name) { return mgr.isStreamingForAnima(name); }

  function setSendButtonIcon(sendBtn, mode) {
    sendBtn.innerHTML = SEND_BTN_ICONS[mode] || SEND_BTN_ICONS.send;
  }

  function updateSendButton() {
    const sendBtn = $("chatPageSendBtn");
    const queueBtn = $("chatPageQueueBtn");
    const inputVal = $("chatPageInput")?.value?.trim() || "";
    const hasInput = inputVal.length > 0;
    const name = state.selectedAnima;
    const tid = state.selectedThreadId;
    const streamCtx = name ? mgr.getStreamingContext(name) : null;
    const isChatStreaming = streamCtx && streamCtx.thread === tid;
    const pendingQueue = name ? mgr.getPendingQueue(name, tid) : [];

    if (queueBtn) queueBtn.disabled = !hasInput || !name;
    if (!sendBtn) return;

    sendBtn.classList.remove("stop", "interrupt");
    if (!isChatStreaming) {
      setSendButtonIcon(sendBtn, "send");
      sendBtn.disabled = !name || (!hasInput && pendingQueue.length === 0);
    } else if (hasInput) {
      setSendButtonIcon(sendBtn, "send");
      sendBtn.disabled = false;
    } else if (pendingQueue.length > 0) {
      setSendButtonIcon(sendBtn, "interrupt");
      sendBtn.classList.add("interrupt");
      sendBtn.disabled = false;
    } else {
      setSendButtonIcon(sendBtn, "stop");
      sendBtn.classList.add("stop");
      sendBtn.disabled = false;
    }
  }

  function showPendingIndicator() {
    const bar = $("chatPagePending");
    const list = $("chatPagePendingList");
    const label = $("chatPagePendingLabel");
    const name = state.selectedAnima;
    const tid = state.selectedThreadId;
    const pendingQueue = name ? mgr.getPendingQueue(name, tid) : [];
    if (!bar || !list) return;
    if (pendingQueue.length === 0) { bar.style.display = "none"; return; }
    if (label) label.textContent = `${t("chat.queue_label")} (${pendingQueue.length})`;
    list.innerHTML = pendingQueue.map((p, i) => {
      const txt = escapeHtml(p.text.length > 60 ? p.text.slice(0, 60) + "\u2026" : p.text);
      const img = p.images?.length ? ` <span style="opacity:0.6">(+${p.images.length} images)</span>` : "";
      return `<div class="pending-queue-item" data-idx="${i}"><span class="pending-queue-item-num">${i + 1}.</span><span class="pending-queue-item-text">${txt || "(images only)"}${img}</span><button class="pending-queue-item-del" data-idx="${i}" type="button">\u2715</button></div>`;
    }).join("");
    bar.style.display = "";

    list.onclick = e => {
      const delBtn = e.target.closest(".pending-queue-item-del");
      if (delBtn) {
        e.stopPropagation();
        mgr.removeFromQueue(name, tid, parseInt(delBtn.dataset.idx, 10));
        showPendingIndicator();
        updateSendButton();
        return;
      }
      const item = e.target.closest(".pending-queue-item");
      if (item) {
        const removed = mgr.removeFromQueue(name, tid, parseInt(item.dataset.idx, 10));
        if (removed) {
          const input = $("chatPageInput");
          if (input) {
            input.value = removed.text;
            input.style.height = "auto";
            input.style.height = Math.min(input.scrollHeight, chatInputMaxHeight()) + "px";
            input.focus();
          }
        }
        showPendingIndicator();
        updateSendButton();
      }
    };
  }

  function hidePendingIndicator() {
    const bar = $("chatPagePending");
    if (bar) bar.style.display = "none";
  }

  function addToQueue() {
    const input = $("chatPageInput");
    if (!input) return;
    const msg = input.value.trim();
    const hasImages = state.imageInputManager && state.imageInputManager.getImageCount() > 0;
    if (!msg && !hasImages) return;
    const name = state.selectedAnima;
    const tid = state.selectedThreadId;
    mgr.enqueue(name, tid, {
      text: msg,
      images: state.imageInputManager?.getPendingImages() || [],
      displayImages: state.imageInputManager?.getDisplayImages() || [],
    });
    input.value = "";
    input.style.height = "auto";
    saveDraft(name, "", tid);
    state.imageInputManager?.clearImages();
    showPendingIndicator();
    updateSendButton();
  }

  function stopStreaming() {
    if (!state.selectedAnima) return;
    mgr.stopStreaming(state.selectedAnima);
  }

  function interruptAndSendPending() {
    stopStreaming();
  }

  function submitChat() {
    const input = $("chatPageInput");
    if (!input) return;
    const msg = input.value.trim();
    const hasImages = state.imageInputManager && state.imageInputManager.getImageCount() > 0;
    const name = state.selectedAnima;
    const tid = state.selectedThreadId;
    const streamCtx = name ? mgr.getStreamingContext(name) : null;
    const isChatStreaming = streamCtx && streamCtx.thread === tid;
    const pendingQueue = name ? mgr.getPendingQueue(name, tid) : [];

    if (!isChatStreaming) {
      if (msg || hasImages) {
        mgr.enqueue(name, tid, {
          text: msg,
          images: state.imageInputManager?.getPendingImages() || [],
          displayImages: state.imageInputManager?.getDisplayImages() || [],
        });
        input.value = "";
        input.style.height = "auto";
        saveDraft(name, "", tid);
        state.imageInputManager?.clearImages();
      }
      if (mgr.getPendingQueue(name, tid).length === 0) return;
      const next = mgr.dequeue(name, tid);
      showPendingIndicator();
      if (mgr.getPendingQueue(name, tid).length === 0) hidePendingIndicator();
      sendChat(next.text, { images: next.images, displayImages: next.displayImages });
      return;
    }

    if (msg || hasImages) {
      mgr.enqueue(name, tid, {
        text: msg,
        images: state.imageInputManager?.getPendingImages() || [],
        displayImages: state.imageInputManager?.getDisplayImages() || [],
      });
      input.value = "";
      input.style.height = "auto";
      saveDraft(name, "", tid);
      state.imageInputManager?.clearImages();
      showPendingIndicator();
      updateSendButton();
      return;
    }

    if (pendingQueue.length > 0) { interruptAndSendPending(); return; }
    stopStreaming();
  }

  async function sendChat(message, overrideImages = null) {
    const name = overrideImages?.targetAnima || state.selectedAnima;
    const images = overrideImages?.images || state.imageInputManager?.getPendingImages() || [];
    const displayImages = overrideImages?.displayImages || state.imageInputManager?.getDisplayImages() || [];
    const tid = overrideImages?.targetThread || state.selectedThreadId;
    if (!name || (!message.trim() && images.length === 0)) return;
    if (mgr.isStreamingFor(name, tid)) {
      logger.warn("Blocked: this thread is already streaming", { anima: name, thread: tid });
      return;
    }

    const currentAnima = state.animas.find(p => p.name === name);
    if (currentAnima?.status === "bootstrapping" || currentAnima?.bootstrapping) {
      const msgs = $("chatPageMessages");
      if (msgs) {
        const el = document.createElement("div");
        el.className = "chat-bubble assistant";
        el.textContent = t("chat.bootstrapping");
        msgs.appendChild(el);
        msgs.scrollTop = msgs.scrollHeight;
      }
      return;
    }
    const threadList = state.threads[name] || [];
    const threadEntry = threadList.find(th => th.id === tid);
    if (threadEntry && threadEntry.label === "新しいスレッド" && message.trim()) {
      threadEntry.label = message.trim().slice(0, 20) + (message.trim().length > 20 ? "..." : "");
      ctx.controllers.thread.renderThreadTabs();
      scheduleSaveChatUiState(ctx);
    }

    const input = $("chatPageInput");
    updateSendButton();
    ctx.controllers.anima.renderAnimaTabs();
    ctx.controllers.thread.renderThreadTabs();
    if (input) input.placeholder = t("chat.message_to", { name });
    if (!overrideImages) state.imageInputManager?.clearImages();

    // User actively sent a message → re-attach scroll to bottom
    ctx.controllers.renderer.reattach();

    ctx.controllers.activity.addLocalActivity("chat", name, `${t("chat.user_prefix")} ${message}`);

    const isVisible = () => state.selectedAnima === name;

    const finalizeStreamError = (streamingMsg, errorMsg, recoveredText = "") => {
      const recovered = recoveredText || "";
      if (recovered && recovered.length > (streamingMsg.text || "").length) {
        streamingMsg.text = recovered;
      }
      const errLine = `${t("chat.error_prefix")} ${errorMsg}`;
      if (!streamingMsg.text) {
        streamingMsg.text = errLine;
      } else if (!streamingMsg.text.includes(errorMsg)) {
        streamingMsg.text += `\n${errLine}`;
      }
      streamingMsg.streaming = false;
      streamingMsg.activeTool = null;
      if (isVisible()) ctx.controllers.renderer.renderChat();
    };

    const renderBubble = (streamingMsg, zone = "all") => { if (isVisible()) ctx.controllers.renderer.renderStreamingBubble(streamingMsg, zone); };
    const renderFull = () => { if (isVisible()) ctx.controllers.renderer.renderChat(!ctx.controllers.renderer.isUserDetached()); };

    let _subThrottleTimer = null;
    let _subThrottlePending = false;
    const _throttledSubRender = () => {
      if (_subThrottleTimer) { _subThrottlePending = true; return; }
      renderBubble(streamingMsg, "subordinate");
      _subThrottleTimer = setTimeout(() => {
        _subThrottleTimer = null;
        if (_subThrottlePending) { _subThrottlePending = false; renderBubble(streamingMsg, "subordinate"); }
      }, 150);
    };

    const _toolDetailTimers = new Map();
    const _throttledToolDetail = (toolId) => {
      if (_toolDetailTimers.has(toolId)) return;
      renderBubble(streamingMsg, "tools");
      _toolDetailTimers.set(toolId, setTimeout(() => { _toolDetailTimers.delete(toolId); renderBubble(streamingMsg, "tools"); }, 200));
    };

    logger.debug(`_sendChat: starting stream for ${name} msg_len=${message.length}`);

    // Use let + onStreamCreated to avoid TDZ: the const destructuring from
    // await would not be initialized when SSE callbacks fire during streaming.
    let streamingMsg = null;

    const _SUB_ACTIVITY_TYPES = new Set([
      "tool_start", "tool_detail", "tool_end",
      "inbox_processing_start", "inbox_processing_end",
    ]);
    const _descendants = getDescendants(name, state.animas || []);
    const _onSubordinateActivity = (e) => {
      const { name: subName, event: evtType, tool_name: toolName, detail: toolDetail } = e.detail || {};
      if (!streamingMsg?.streaming || subName === name) return;
      if (!_descendants.has(subName)) return;
      if (!_SUB_ACTIVITY_TYPES.has(evtType)) return;
      if (!streamingMsg.subordinateActivity) streamingMsg.subordinateActivity = {};
      if (evtType === "tool_start") {
        streamingMsg.subordinateActivity[subName] = { type: evtType, tool: toolName, summary: `${toolName} 実行中...` };
      } else if (evtType === "tool_detail") {
        streamingMsg.subordinateActivity[subName] = { type: evtType, tool: toolName, summary: `${toolName}: ${toolDetail || ""}` };
      } else if (evtType === "tool_end") {
        streamingMsg.subordinateActivity[subName] = { type: evtType, tool: toolName, summary: `${toolName} 完了` };
      } else {
        streamingMsg.subordinateActivity[subName] = { type: evtType, tool: toolName || "", summary: evtType };
      }
      _throttledSubRender();
    };
    document.addEventListener("anima-tool-activity", _onSubordinateActivity);

    const { success, error } = await mgr.sendChat(name, tid, message, {
      images,
      displayImages,
      callbacks: {
        onStreamCreated: msg => { streamingMsg = msg; renderFull(); updateSendButton(); },
        onTextDelta: text => {
          if (!streamingMsg?.streaming) return;
          streamingMsg.afterHeartbeatRelay = false;
          streamingMsg.text += text;
          renderBubble(streamingMsg, "text");
        },
        onToolStart: (toolName, detail) => {
          if (!streamingMsg?.streaming) return;
          streamingMsg.activeTool = toolName;
          if (!streamingMsg.toolHistory) streamingMsg.toolHistory = [];
          streamingMsg.toolHistory.push({
            tool_name: toolName,
            tool_id: detail?.tool_id || "",
            started_at: Date.now(),
          });
          renderBubble(streamingMsg, "tools");
        },
        onToolDetail: (_toolName, detailText, info) => {
          if (!streamingMsg?.streaming) return;
          if (streamingMsg.toolHistory && info?.tool_id) {
            for (let i = streamingMsg.toolHistory.length - 1; i >= 0; i--) {
              const entry = streamingMsg.toolHistory[i];
              if (entry.tool_id === info.tool_id && !entry.completed) {
                entry.detail = detailText;
                break;
              }
            }
          }
          _throttledToolDetail(info?.tool_id || "_");
        },
        onToolEnd: (detail) => {
          if (!streamingMsg?.streaming) return;
          streamingMsg.activeTool = null;
          if (streamingMsg.toolHistory && detail?.tool_id) {
            for (let i = streamingMsg.toolHistory.length - 1; i >= 0; i--) {
              const entry = streamingMsg.toolHistory[i];
              if (entry.tool_id === detail.tool_id && !entry.completed) {
                entry.completed = true;
                entry.duration_ms = Date.now() - entry.started_at;
                entry.result_summary = detail.result_summary || "";
                entry.input_summary = detail.input_summary || "";
                entry.is_error = !!detail.is_error;
                break;
              }
            }
          }
          renderBubble(streamingMsg, "tools");
        },
        onChainStart: () => {},
        onCompressionStart: () => { if (!streamingMsg?.streaming) return; streamingMsg.compressing = true; renderBubble(streamingMsg, "text"); },
        onCompressionEnd: () => { if (!streamingMsg?.streaming) return; streamingMsg.compressing = false; renderBubble(streamingMsg, "text"); },
        onHeartbeatRelayStart: ({ message: msg }) => {
          if (!streamingMsg?.streaming) return;
          streamingMsg.heartbeatRelay = true; streamingMsg.heartbeatText = "";
          renderBubble(streamingMsg, "text");
          ctx.controllers.activity.addLocalActivity("system", name, `${t("chat.heartbeat_relay")}: ${msg}`);
        },
        onHeartbeatRelay: ({ text }) => {
          if (!streamingMsg?.streaming) return;
          streamingMsg.heartbeatText = (streamingMsg.heartbeatText || "") + text;
          renderBubble(streamingMsg, "text");
        },
        onHeartbeatRelayDone: () => {
          if (!streamingMsg?.streaming) return;
          streamingMsg.heartbeatRelay = false; streamingMsg.heartbeatText = ""; streamingMsg.afterHeartbeatRelay = true;
          renderBubble(streamingMsg, "text");
        },
        onThinkingStart: () => { if (!streamingMsg?.streaming) return; streamingMsg.thinkingText = ""; streamingMsg.thinking = true; renderBubble(streamingMsg, "thinking"); },
        onThinkingDelta: text => { if (!streamingMsg?.streaming) return; streamingMsg.thinkingText = (streamingMsg.thinkingText || "") + text; renderBubble(streamingMsg, "thinking"); },
        onThinkingEnd: () => { if (!streamingMsg?.streaming) return; streamingMsg.thinking = false; renderBubble(streamingMsg, "thinking"); },
        onError: ({ message: errorMsg }) => {
          logger.debug(`onError: ${errorMsg}`);
          if (!streamingMsg?.text) {
            void (async () => {
              let recoveredText = "";
              try {
                const active = await fetchActiveStream(name, tid);
                if (active?.response_id) {
                  const progress = await fetchStreamProgress(name, active.response_id);
                  recoveredText = progress?.full_text || "";
                }
              } catch (progressErr) {
                logger.debug("Failed to load stream progress on error", { anima: name, error: progressErr?.message || String(progressErr) });
              }
              if (streamingMsg) finalizeStreamError(streamingMsg, errorMsg, recoveredText);
            })();
            return;
          }
          finalizeStreamError(streamingMsg, errorMsg);
        },
        onDone: ({ summary, images: doneImages }) => {
          if (!streamingMsg) return;
          const text = summary || streamingMsg.text;
          streamingMsg.text = text || t("chat.empty_response");
          streamingMsg.images = doneImages || [];
          streamingMsg.streaming = false;
          streamingMsg.activeTool = null;
          streamingMsg.heartbeatRelay = false; streamingMsg.heartbeatText = ""; streamingMsg.afterHeartbeatRelay = false;
          renderFull();
          ctx.controllers.activity.addLocalActivity("chat", name, `${t("chat.response_prefix")} ${streamingMsg.text.slice(0, 100)}`);
          ctx.controllers.renderer.markResponseComplete(name, tid);

          const paneEl = ctx.state.container?.closest(".chat-pane");
          if (paneEl && !paneEl.classList.contains("focused")) {
            paneEl.classList.remove("stream-done-flash");
            void paneEl.offsetWidth;
            paneEl.classList.add("stream-done-flash");
            paneEl.addEventListener("animationend", () => paneEl.classList.remove("stream-done-flash"), { once: true });
          }
        },
        onAbort: () => {
          if (!streamingMsg) return;
          streamingMsg.streaming = false; streamingMsg.activeTool = null;
          if (!streamingMsg.text) streamingMsg.text = t("chat.interrupted");
          if (isVisible()) ctx.controllers.renderer.renderChat();
        },
      },
      onFinally: () => {
        document.removeEventListener("anima-tool-activity", _onSubordinateActivity);
        if (_subThrottleTimer) { clearTimeout(_subThrottleTimer); _subThrottleTimer = null; }
        for (const t of _toolDetailTimers.values()) clearTimeout(t);
        _toolDetailTimers.clear();
        try {
          if (streamingMsg && streamingMsg.streaming) {
            streamingMsg.streaming = false;
            if (!streamingMsg.text) {
              streamingMsg.text = streamingMsg.afterHeartbeatRelay ? t("chat.receive_failed") : t("chat.empty_response");
            }
            streamingMsg.afterHeartbeatRelay = false;
            renderFull();
          }

          const inputEl = $("chatPageInput");
          if (inputEl && state.selectedAnima === name) {
            inputEl.placeholder = t("chat.message_to", { name });
            saveDraft(name, inputEl.value || "", tid);
            const paneEl = ctx.state.container?.closest(".chat-pane");
            if (!paneEl || paneEl.classList.contains("focused")) {
              inputEl.focus();
            }
          }
          updateSendButton();
          ctx.controllers.anima.renderAnimaTabs();
          ctx.controllers.thread.renderThreadTabs();
        } finally {
          if (mgr.getPendingQueue(name, tid).length > 0) {
            const next = mgr.dequeue(name, tid);
            showPendingIndicator();
            if (mgr.getPendingQueue(name, tid).length === 0) hidePendingIndicator();
            setTimeout(() => sendChat(next.text, { images: next.images, displayImages: next.displayImages, targetAnima: name, targetThread: tid }), 150);
          }
        }
      },
    });

    ctx.controllers.renderer.renderChat();

    if (!success && error && error.name !== "AbortError") {
      logger.error("Chat stream error", { anima: name, error: error.message, name: error.name });
    }
  }

  async function resumeActiveStream(animaName) {
    const tid = state.selectedThreadId || "default";
    if (mgr.isStreamingFor(animaName, tid)) return;

    ctx.controllers.renderer.renderChat();

    const renderIfVisible = (msg) => {
      if (state.selectedAnima === animaName) ctx.controllers.renderer.renderStreamingBubble(msg);
    };
    const smartScroll = () => !ctx.controllers.renderer.isUserDetached();

    let streamingMsg = null;

    await mgr.resumeStream(animaName, tid, {
      callbacks: {
        onStreamCreated: msg => { streamingMsg = msg; ctx.controllers.renderer.renderChat(smartScroll()); updateSendButton(); },
        onTextDelta: text => { if (streamingMsg?.streaming) { streamingMsg.text += text; renderIfVisible(streamingMsg); } },
        onToolStart: toolName => { if (streamingMsg?.streaming) { streamingMsg.activeTool = toolName; renderIfVisible(streamingMsg); } },
        onToolEnd: () => { if (streamingMsg?.streaming) { streamingMsg.activeTool = null; renderIfVisible(streamingMsg); } },
        onThinkingStart: () => { if (streamingMsg?.streaming) { streamingMsg.thinkingText = ""; streamingMsg.thinking = true; renderIfVisible(streamingMsg); } },
        onThinkingDelta: text => { if (streamingMsg?.streaming) { streamingMsg.thinkingText = (streamingMsg.thinkingText || "") + text; renderIfVisible(streamingMsg); } },
        onThinkingEnd: () => { if (streamingMsg?.streaming) { streamingMsg.thinking = false; renderIfVisible(streamingMsg); } },
        onError: ({ message: errorMsg }) => { if (streamingMsg) { streamingMsg.text += `\n${t("chat.error_prefix")} ${errorMsg}`; streamingMsg.streaming = false; if (state.selectedAnima === animaName) ctx.controllers.renderer.renderChat(smartScroll()); } },
        onDone: ({ summary, images }) => {
          if (streamingMsg) {
            streamingMsg.text = summary || streamingMsg.text || t("chat.empty_response");
            streamingMsg.images = images || [];
            streamingMsg.streaming = false; streamingMsg.activeTool = null;
            if (state.selectedAnima === animaName) ctx.controllers.renderer.renderChat(smartScroll());
            ctx.controllers.renderer.markResponseComplete(animaName, tid);
          }
        },
      },
      onFinally: () => {
        try {
          if (streamingMsg?.streaming) {
            streamingMsg.streaming = false;
            if (!streamingMsg.text) streamingMsg.text = t("chat.empty_response");
            if (state.selectedAnima === animaName) ctx.controllers.renderer.renderChat(smartScroll());
          }
          updateSendButton();
          ctx.controllers.anima.renderAnimaTabs();
          ctx.controllers.thread.renderThreadTabs();
        } finally {
          if (mgr.getPendingQueue(animaName, tid).length > 0) {
            const next = mgr.dequeue(animaName, tid);
            showPendingIndicator();
            if (mgr.getPendingQueue(animaName, tid).length === 0) hidePendingIndicator();
            setTimeout(() => sendChat(next.text, { images: next.images, displayImages: next.displayImages, targetAnima: animaName, targetThread: tid }), 150);
          }
        }
      },
    });

    if (streamingMsg) {
      updateSendButton();
      ctx.controllers.anima.renderAnimaTabs();
      ctx.controllers.thread.renderThreadTabs();
      ctx.controllers.renderer.renderChat();
    }
  }

  return {
    submitChat, sendChat, resumeActiveStream,
    stopStreaming, addToQueue,
    showPendingIndicator, hidePendingIndicator,
    updateSendButton,
  };
}
