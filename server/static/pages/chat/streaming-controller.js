// ── Streaming / Send / Queue Controller ────────
import {
  saveDraft, clearDraft, chatInputMaxHeight,
  scheduleSaveChatUiState, CONSTANTS,
} from "./ctx.js";
import { TextAnimator, stripThinkTags } from "../../shared/chat/render-utils.js";
import { streamMeetingChat } from "../../shared/chat-stream.js";
import { clearReplyTarget, getReplyTarget, needsNotificationApi, quoteReplyMessage } from "../../shared/chat/reply-target.js";
import {
  ensureThink, restoreThink, thinkOnText, thinkToolStart, thinkToolDetail, thinkToolEnd,
  thinkDone, thinkError, thinkStop, emitThinkActivity, syncThinkAvatar, bindThinkingHandlers,
} from "../../shared/chat/thinking-bubble.js";
import { applyRowEmotion } from "../../shared/chat/emotion-face.js";

const SEND_BTN_ICONS = {
  send: `<svg class="chat-send-icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M12 19V5M5 12l7-7 7 7" /></svg>`,
  stop: `<svg class="chat-send-icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><rect x="5" y="5" width="14" height="14" rx="2.5" /></svg>`,
  interrupt: `<span class="chat-send-icon-group" aria-hidden="true"><svg class="chat-send-icon chat-send-icon-square" viewBox="0 0 24 24" focusable="false"><rect x="5" y="5" width="14" height="14" rx="2.5" /></svg><svg class="chat-send-icon" viewBox="0 0 24 24" focusable="false"><path d="M12 19V5M5 12l7-7 7 7" /></svg></span>`,
};

export function createStreamingController(ctx) {
  const $ = ctx.$;
  const { state, deps } = ctx;
  const { t, api, escapeHtml, logger, fetchActiveStream, fetchStreamProgress } = deps;
  const mgr = state.manager;

  mgr.addEventListener("stream-state-changed", () => {
    updateSendButton();
    ctx.controllers.anima?.renderAnimaTabs();
  });

  function isAnimaStreaming(name) { return mgr.isStreamingForAnima(name); }

  // ── Thinking bubble wiring ──────────────────────────────
  bindThinkingHandlers($("chatPageMessages"), { onStop: () => stopStreaming() });

  function _streamBubble(msg) {
    if (!msg?.streamId) return null;
    return $("chatPageMessages")?.querySelector(
      `.chat-bubble.assistant.streaming[data-stream-id="${CSS.escape(String(msg.streamId))}"]`,
    ) || null;
  }

  function _applyDoneEmotion(msg, animaName, emotion) {
    if (!msg?.think || !emotion) return;
    const avatar = $("chatPageMessages")?.querySelector(
      `.think-face[data-think-id="${CSS.escape(msg.think.id)}"]`,
    );
    const row = avatar?.closest(".chat-msg-row");
    if (row) applyRowEmotion(row, animaName, emotion);
  }

  function setSendButtonIcon(sendBtn, mode) {
    sendBtn.innerHTML = SEND_BTN_ICONS[mode] || SEND_BTN_ICONS.send;
    const labelKey = mode === "stop" ? "chat.stop_stream"
      : mode === "interrupt" ? "chat.interrupt_send" : "chat.send";
    sendBtn.setAttribute("aria-label", t(labelKey));
  }

  // ── Context Ring Indicator ──────────────────────────────
  const RING_CIRCUMFERENCE = 97.4; // 2 * PI * 15.5
  const _contextStateByAnima = {};

  function _setRingLabel(ring, pct) {
    const label = ring.querySelector(".context-ring-label");
    if (label) label.textContent = `${t("chat.presence.memory_label")} ${pct}%`;
  }

  function _applyRing(ring, ratio, inputTokens, contextWindow) {
    const fg = ring.querySelector(".context-ring-fg");
    if (!fg) return;
    const offset = RING_CIRCUMFERENCE * (1 - Math.min(ratio, 1));
    fg.style.strokeDashoffset = offset;

    let colorClass = "ctx-green";
    if (ratio >= 0.85) colorClass = "ctx-red";
    else if (ratio >= 0.70) colorClass = "ctx-yellow";
    ring.className = `context-ring-wrap ${colorClass}`;

    const pct = Math.round(ratio * 100);
    const inK = inputTokens >= 1000 ? `${(inputTokens / 1000).toFixed(0)}k` : String(inputTokens);
    const winK = contextWindow >= 1000 ? `${(contextWindow / 1000).toFixed(0)}k` : String(contextWindow);
    const usage = contextWindow ? `${pct}% (${inK} / ${winK})` : `${pct}%`;
    ring.title = `${t("chat.context_ring_title")}: ${usage}`;
    _setRingLabel(ring, pct);
  }

  function updateContextRing(data, animaName) {
    const ratio = data.contextUsageRatio || 0;
    const inputTokens = data.inputTokens || 0;
    const contextWindow = data.contextWindow || 0;

    const target = animaName || state.selectedAnima;
    if (target) _contextStateByAnima[target] = { ratio, inputTokens, contextWindow };

    if (!target || target === state.selectedAnima) {
      const ring = $("chatContextRing");
      if (ring) _applyRing(ring, ratio, inputTokens, contextWindow);
    }
  }

  function restoreContextRing(animaName) {
    const ring = $("chatContextRing");
    if (!ring) return;
    const saved = animaName ? _contextStateByAnima[animaName] : null;
    if (saved) {
      _applyRing(ring, saved.ratio, saved.inputTokens, saved.contextWindow);
    } else {
      const fg = ring.querySelector(".context-ring-fg");
      if (fg) fg.style.strokeDashoffset = RING_CIRCUMFERENCE;
      ring.className = "context-ring-wrap";
      ring.title = t("chat.context_ring_title");
      _setRingLabel(ring, 0);
    }
  }

  function updateSendButton() {
    const sendBtn = $("chatPageSendBtn");
    const inputVal = $("chatPageInput")?.value?.trim() || "";
    const hasInput = inputVal.length > 0;
    const attachmentManager = state.imageInputManager;
    const hasAttachments = Boolean((attachmentManager?.getImageCount() || 0) + (attachmentManager?.getFileCount() || 0));
    const filesProcessing = attachmentManager?.isProcessingFiles() || false;
    const meetingActive = ctx.controllers.meeting?.isActive?.();
    const name = state.selectedAnima;
    const tid = state.selectedThreadId;
    const streamCtx = name ? mgr.getStreamingContext(name) : null;
    const isChatStreaming = streamCtx && streamCtx.thread === tid;
    const pendingQueue = name ? mgr.getPendingQueue(name, tid) : [];
    const isMeetingStreaming = state.meetingStreaming === true;

    if (!sendBtn) return;

    sendBtn.classList.remove("stop", "interrupt");
    if (meetingActive) {
      setSendButtonIcon(sendBtn, "send");
      sendBtn.disabled = filesProcessing || (!hasInput && !hasAttachments) || isMeetingStreaming;
    } else if (!isChatStreaming) {
      setSendButtonIcon(sendBtn, "send");
      sendBtn.disabled = filesProcessing || !name || (!hasInput && !hasAttachments && pendingQueue.length === 0);
    } else if (hasInput || hasAttachments) {
      setSendButtonIcon(sendBtn, "send");
      sendBtn.disabled = filesProcessing;
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
      const files = p.files?.length ? ` <span style="opacity:0.6">(${t("chat.attach.file_count", { count: p.files.length })})</span>` : "";
      const fallback = p.files?.length ? t("chat.attach.files_only") : "(images only)";
      return `<div class="pending-queue-item" data-idx="${i}"><span class="pending-queue-item-num">${i + 1}.</span><span class="pending-queue-item-text">${txt || fallback}${img}${files}</span><button class="pending-queue-item-del" data-idx="${i}" type="button" aria-label="${escapeHtml(t("chat.queue_item_remove"))}">\u2715</button></div>`;
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
          if (removed.files?.length) {
            state.imageInputManager?.restoreFiles(removed.files, removed.displayFiles || []);
          }
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
    const hasFiles = state.imageInputManager && state.imageInputManager.getFileCount() > 0;
    if (!msg && !hasImages && !hasFiles) return;
    if (state.imageInputManager?.isProcessingFiles()) return;
    const name = state.selectedAnima;
    const tid = state.selectedThreadId;
    mgr.enqueue(name, tid, {
      text: msg,
      images: state.imageInputManager?.getPendingImages() || [],
      displayImages: state.imageInputManager?.getDisplayImages() || [],
      files: state.imageInputManager?.getPendingFiles() || [],
      displayFiles: state.imageInputManager?.getDisplayFiles() || [],
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
    const replyTarget = getReplyTarget(state);
    const targetIsCurrent = (!replyTarget?.animaName || replyTarget.animaName === state.selectedAnima)
      && (!replyTarget?.threadId || replyTarget.threadId === state.selectedThreadId);
    if (replyTarget && !targetIsCurrent) {
      clearReplyTarget(state);
      ctx.controllers.events?.clearReplyTarget();
    }
    const activeReplyTarget = targetIsCurrent ? replyTarget : null;
    const rawMsg = input.value.trim();
    // Ordinary notifications are answered in the chat itself, so the Anima replies in the conversation.
    if (needsNotificationApi(activeReplyTarget) && rawMsg) {
      sendNotificationReply(rawMsg, activeReplyTarget);
      return;
    }
    const msg = activeReplyTarget && rawMsg ? quoteReplyMessage(rawMsg, activeReplyTarget) : rawMsg;
    const hasImages = state.imageInputManager && state.imageInputManager.getImageCount() > 0;
    const hasFiles = state.imageInputManager && state.imageInputManager.getFileCount() > 0;
    if (state.imageInputManager?.isProcessingFiles()) return;
    const meetingActive = ctx.controllers.meeting?.isActive?.();
    const name = state.selectedAnima;
    const tid = state.selectedThreadId;
    const streamCtx = name ? mgr.getStreamingContext(name) : null;
    const isChatStreaming = streamCtx && streamCtx.thread === tid;
    const pendingQueue = name ? mgr.getPendingQueue(name, tid) : [];

    if (activeReplyTarget && (msg || hasImages || hasFiles)) {
      clearReplyTarget(state);
      ctx.controllers.events?.clearReplyTarget();
    }

    if (meetingActive && hasFiles) {
      state.imageInputManager?.setAttachmentError("chat.attach.meeting_unsupported");
      return;
    }

    if (meetingActive && (msg || hasImages)) {
      sendMeetingChat(msg, {
        images: state.imageInputManager?.getPendingImages() || [],
        displayImages: state.imageInputManager?.getDisplayImages() || [],
      });
      input.value = "";
      input.style.height = "auto";
      state.imageInputManager?.clearImages();
      return;
    }

    if (!isChatStreaming) {
      if (msg || hasImages || hasFiles) {
        mgr.enqueue(name, tid, {
          text: msg,
          images: state.imageInputManager?.getPendingImages() || [],
          displayImages: state.imageInputManager?.getDisplayImages() || [],
          files: state.imageInputManager?.getPendingFiles() || [],
          displayFiles: state.imageInputManager?.getDisplayFiles() || [],
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
      sendChat(next.text, { images: next.images, displayImages: next.displayImages, files: next.files, displayFiles: next.displayFiles });
      return;
    }

    if (msg || hasImages || hasFiles) {
      mgr.enqueue(name, tid, {
        text: msg,
        images: state.imageInputManager?.getPendingImages() || [],
        displayImages: state.imageInputManager?.getDisplayImages() || [],
        files: state.imageInputManager?.getPendingFiles() || [],
        displayFiles: state.imageInputManager?.getDisplayFiles() || [],
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

  async function sendNotificationReply(text, target) {
    const name = target.animaName || state.selectedAnima;
    const notificationId = target.notificationId || "";
    const tid = state.selectedThreadId;
    if (!name || !notificationId) return;
    const input = $("chatPageInput");
    try {
      await api(`/api/animas/${encodeURIComponent(name)}/notifications/reply`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ notification_id: notificationId, text }),
      });
      clearReplyTarget(state);
      ctx.controllers.events?.clearReplyTarget();
      if (input) {
        input.value = "";
        input.style.height = "auto";
      }
      saveDraft(name, "", tid);
      state.imageInputManager?.clearImages();
      await mgr.loadHistory(name, tid, CONSTANTS.HISTORY_PAGE_SIZE, state.fetchChatFeedHistory);
      if (state.selectedAnima === name && state.selectedThreadId === tid) {
        ctx.controllers.renderer.renderChat(!ctx.controllers.renderer.isUserDetached());
      }
      updateSendButton();
    } catch (error) {
      logger.warn("Notification reply failed", { anima: name, error: error?.message });
      ctx.controllers.events?.showReplyError?.(t("chat.line.reply_failed"));
    }
  }

  async function sendChat(message, overrideImages = null) {
    if (ctx.controllers.meeting?.isActive?.()) {
      if (overrideImages?.files?.length || state.imageInputManager?.getFileCount()) {
        state.imageInputManager?.setAttachmentError("chat.attach.meeting_unsupported");
        return;
      }
      sendMeetingChat(message, overrideImages);
      return;
    }

    const name = overrideImages?.targetAnima || state.selectedAnima;
    const images = overrideImages?.images || state.imageInputManager?.getPendingImages() || [];
    const displayImages = overrideImages?.displayImages || state.imageInputManager?.getDisplayImages() || [];
    const files = overrideImages?.files || state.imageInputManager?.getPendingFiles() || [];
    const displayFiles = overrideImages?.displayFiles || state.imageInputManager?.getDisplayFiles() || [];
    const tid = overrideImages?.targetThread || state.selectedThreadId;
    if (!name || (!message.trim() && images.length === 0 && files.length === 0)) return;
    if (mgr.isStreamingFor(name, tid)) {
      logger.warn("Blocked: this thread is already streaming", { anima: name, thread: tid });
      return;
    }

    const currentAnima = state.animas.find(p => p.name === name);
    const restoreUnsentInput = () => {
      const input = $("chatPageInput");
      if (input && !input.value && message) input.value = message;
    };
    if (currentAnima?.needs_repair || currentAnima?.bootstrap_state?.state === "needs_repair") {
      // The repair panel (with its recovery buttons) is the answer here;
      // keep the user's text so it can be sent once the Anima is back.
      restoreUnsentInput();
      ctx.controllers.renderer.renderChat();
      return;
    }
    if (currentAnima?.status === "bootstrapping" || currentAnima?.bootstrapping) {
      restoreUnsentInput();
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
    if (threadEntry && threadEntry.label === t("thread.new") && message.trim()) {
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

    const isVisible = () => state.selectedAnima === name && state.selectedThreadId === tid;

    const finalizeStreamError = (streamingMsg, errorMsg, recoveredText = "") => {
      if (_textAnimator) _textAnimator.flush();
      if (_thinkingAnimator) { _thinkingAnimator.flush(); _thinkingAnimator = null; }
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
      delete streamingMsg._displayText;
      delete streamingMsg._displayThinkingText;
      streamingMsg.streaming = false;
      streamingMsg.activeTool = null;
      if (isVisible()) ctx.controllers.renderer.renderChat();
    };

    const renderBubble = (streamingMsg, zone = "all") => { if (isVisible()) ctx.controllers.renderer.renderStreamingBubble(streamingMsg, zone); };
    const renderFull = () => { if (isVisible()) ctx.controllers.renderer.renderChat(!ctx.controllers.renderer.isUserDetached()); };

    const _toolDetailTimers = new Map();
    const _throttledToolDetail = (toolId) => {
      if (_toolDetailTimers.has(toolId)) return;
      renderBubble(streamingMsg, "tools");
      _toolDetailTimers.set(toolId, setTimeout(() => { _toolDetailTimers.delete(toolId); renderBubble(streamingMsg, "tools"); }, 200));
    };

    logger.debug(`_sendChat: starting stream for ${name} msg_len=${message.length}`);

    // ── RAF-based text rendering throttle ──
    // Coalesce rapid text_delta / thinking_delta into one render per frame
    // to avoid calling renderMarkdown(fullText) on every delta (O(n²) cost).
    let _rafPending = false;
    let _rafZone = "all";
    const _scheduleRender = (msg, zone = "text") => {
      if (!_rafPending) _rafZone = zone;
      else if (_rafZone !== zone) _rafZone = "all";
      if (_rafPending) return;
      _rafPending = true;
      requestAnimationFrame(() => {
        _rafPending = false;
        const z = _rafZone;
        _rafZone = "all";
        renderBubble(msg, z);
      });
    };

    // Use let + onStreamCreated to avoid TDZ: the const destructuring from
    // await would not be initialized when SSE callbacks fire during streaming.
    let streamingMsg = null;
    let _textAnimator = null;
    let _thinkingAnimator = null;
    const emitThink = (extra = {}) => emitThinkActivity({ anima: name, threadId: tid, msg: streamingMsg, ...extra });
    const syncAvatar = () => { if (isVisible()) syncThinkAvatar(_streamBubble(streamingMsg), streamingMsg); };

    const { success, error } = await mgr.sendChat(name, tid, message, {
      model: state.modelByThread[`${name}|${tid}`] || undefined,
      images,
      displayImages,
      files,
      displayFiles,
      callbacks: {
        onFilesReady: () => renderFull(),
        onStreamCreated: msg => {
          streamingMsg = msg;
          ensureThink(msg);
          _textAnimator = new TextAnimator({
            onUpdate: (displayText) => {
              if (!streamingMsg) return;
              streamingMsg._displayText = displayText;
              _scheduleRender(streamingMsg, "text");
            },
          });
          _textAnimator.start();
          renderFull();
          updateSendButton();
          emitThink();
        },
        onTextDelta: text => {
          if (!streamingMsg?.streaming) return;
          streamingMsg.afterHeartbeatRelay = false;
          streamingMsg.text += text;
          if (_textAnimator) _textAnimator.push(text);
          if (thinkOnText(streamingMsg)) {
            renderBubble(streamingMsg, "tools");
            syncAvatar();
            emitThink();
          }
        },
        onToolStart: (toolName, detail) => {
          if (!streamingMsg?.streaming) return;
          const { moved } = thinkToolStart(streamingMsg, toolName, detail?.tool_id);
          if (moved && _textAnimator) _textAnimator.start();
          streamingMsg.activeTool = toolName;
          if (!streamingMsg.toolHistory) streamingMsg.toolHistory = [];
          streamingMsg.toolHistory.push({
            tool_name: toolName,
            tool_id: detail?.tool_id || "",
            started_at: Date.now(),
          });
          renderBubble(streamingMsg, moved ? "all" : "tools");
          syncAvatar();
          emitThink();
          ctx.controllers.workIndicator?.onToolStart(toolName, detail);
        },
        onToolDetail: (_toolName, detailText, info) => {
          if (!streamingMsg?.streaming) return;
          thinkToolDetail(streamingMsg, info?.tool_id, detailText);
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
          ctx.controllers.workIndicator?.onToolDetail(_toolName, detailText, info);
        },
        onToolEnd: (detail) => {
          if (!streamingMsg?.streaming) return;
          streamingMsg.activeTool = null;
          thinkToolEnd(streamingMsg, detail);
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
          emitThink();
          ctx.controllers.workIndicator?.onToolEnd(detail);
        },
        onChainStart: () => {},
        onCompressionStart: () => { if (!streamingMsg?.streaming) return; streamingMsg.compressing = true; renderBubble(streamingMsg, "all"); emitThink(); },
        onCompressionEnd: () => { if (!streamingMsg?.streaming) return; streamingMsg.compressing = false; renderBubble(streamingMsg, "all"); emitThink(); },
        onHeartbeatRelayStart: ({ message: msg }) => {
          if (!streamingMsg?.streaming) return;
          streamingMsg.heartbeatRelay = true; streamingMsg.heartbeatText = "";
          renderBubble(streamingMsg, "text");
          ctx.controllers.activity.addLocalActivity("system", name, `${t("chat.heartbeat_relay")}: ${msg}`);
        },
        onHeartbeatRelay: ({ text }) => {
          if (!streamingMsg?.streaming) return;
          streamingMsg.heartbeatText = (streamingMsg.heartbeatText || "") + text;
          _scheduleRender(streamingMsg, "text");
        },
        onHeartbeatRelayDone: () => {
          if (!streamingMsg?.streaming) return;
          streamingMsg.heartbeatRelay = false; streamingMsg.heartbeatText = ""; streamingMsg.afterHeartbeatRelay = true;
          renderBubble(streamingMsg, "text");
        },
        onThinkingStart: () => {
          if (!streamingMsg?.streaming) return;
          streamingMsg.thinkingText = ""; streamingMsg.thinking = true;
          _thinkingAnimator = new TextAnimator({
            onUpdate: (displayText) => {
              if (!streamingMsg) return;
              streamingMsg._displayThinkingText = displayText;
              _scheduleRender(streamingMsg, "thinking");
            },
          });
          _thinkingAnimator.start();
          renderBubble(streamingMsg, "thinking");
          ctx.controllers.workIndicator?.onThinkingStart();
        },
        onThinkingDelta: text => {
          if (!streamingMsg?.streaming) return;
          streamingMsg.thinkingText = (streamingMsg.thinkingText || "") + text;
          if (_thinkingAnimator) _thinkingAnimator.push(text);
        },
        onThinkingEnd: () => {
          if (!streamingMsg?.streaming) return;
          if (_thinkingAnimator) { _thinkingAnimator.flush(); _thinkingAnimator = null; }
          delete streamingMsg._displayThinkingText;
          streamingMsg.thinking = false;
          renderBubble(streamingMsg, "thinking");
          ctx.controllers.workIndicator?.onThinkingEnd();
        },
        onContextUpdate: (ctxData) => { updateContextRing(ctxData, name); },
        onError: ({ message: errorMsg }) => {
          logger.debug(`onError: ${errorMsg}`);
          thinkError(streamingMsg);
          emitThink();
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
        onDone: ({ summary, emotion, images: doneImages, files: doneFiles, thinkingSummary, contextUsageRatio, inputTokens, contextWindow, contextThreshold }) => {
          if (_textAnimator) _textAnimator.flush();
          if (_thinkingAnimator) { _thinkingAnimator.flush(); _thinkingAnimator = null; }
          if (!streamingMsg) return;
          streamingMsg.activeTool = null;
          let text = summary || streamingMsg.text || "";
          const { thinking: strippedThinking, response: cleanResponse } = stripThinkTags(text);
          // Work-in-progress talk before the last tool stays in the steps, not the reply.
          const reply = thinkDone(streamingMsg, cleanResponse, { aligned: !!summary, emotion });
          streamingMsg.text = reply || t("chat.empty_response");
          streamingMsg.emotion = emotion || null;
          if (strippedThinking && !streamingMsg.thinkingText) {
            streamingMsg.thinkingText = strippedThinking;
          }
          delete streamingMsg._displayText;
          delete streamingMsg._displayThinkingText;
          streamingMsg.images = doneImages || [];
          streamingMsg.files = doneFiles || [];
          if (!streamingMsg.thinkingText && thinkingSummary) {
            streamingMsg.thinkingText = thinkingSummary;
          }
          streamingMsg.streaming = false;
          streamingMsg.heartbeatRelay = false; streamingMsg.heartbeatText = ""; streamingMsg.afterHeartbeatRelay = false;
          if (contextUsageRatio || contextWindow) {
            updateContextRing({ contextUsageRatio, inputTokens, contextWindow, threshold: contextThreshold }, name);
          }
          renderFull();
          _applyDoneEmotion(streamingMsg, name, emotion);
          emitThink({ emotion });
          ctx.controllers.activity.addLocalActivity("chat", name, `${t("chat.response_prefix")} ${streamingMsg.text.slice(0, 100)}`);
          ctx.controllers.renderer.markResponseComplete(name, tid);
          ctx.controllers.workIndicator?.onStreamSettled();

          const paneEl = ctx.state.container?.closest(".chat-pane");
          if (paneEl && !paneEl.classList.contains("focused")) {
            paneEl.classList.remove("stream-done-flash");
            void paneEl.offsetWidth;
            paneEl.classList.add("stream-done-flash");
            paneEl.addEventListener("animationend", () => paneEl.classList.remove("stream-done-flash"), { once: true });
          }
        },
        onAbort: () => {
          if (_textAnimator) _textAnimator.flush();
          if (_thinkingAnimator) { _thinkingAnimator.flush(); _thinkingAnimator = null; }
          if (!streamingMsg) return;
          delete streamingMsg._displayText;
          delete streamingMsg._displayThinkingText;
          streamingMsg.streaming = false; streamingMsg.activeTool = null;
          if (thinkStop(streamingMsg)) emitThink();
          if (!streamingMsg.text) streamingMsg.text = t("chat.interrupted");
          if (isVisible()) ctx.controllers.renderer.renderChat();
          ctx.controllers.workIndicator?.onStreamSettled();
        },
      },
      onFinally: () => {
        if (_textAnimator) { _textAnimator.stop(); _textAnimator = null; }
        if (_thinkingAnimator) { _thinkingAnimator.stop(); _thinkingAnimator = null; }
        ctx.controllers.workIndicator?.onStreamSettled();
        for (const t of _toolDetailTimers.values()) clearTimeout(t);
        _toolDetailTimers.clear();
        try {
          if (streamingMsg && streamingMsg.streaming) {
            streamingMsg.streaming = false;
            if (thinkStop(streamingMsg)) emitThink();
            delete streamingMsg._displayText;
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
            setTimeout(() => sendChat(next.text, { images: next.images, displayImages: next.displayImages, files: next.files, displayFiles: next.displayFiles, targetAnima: name, targetThread: tid }), 150);
          }
        }
      },
    });

    ctx.controllers.renderer.renderChat(!ctx.controllers.renderer.isUserDetached());

    if (!success && error && error.name !== "AbortError") {
      logger.error("Chat stream error", { anima: name, error: error.message, name: error.name });
    }
  }

  async function sendMeetingChat(message, overrideImages = null) {
    const room = ctx.controllers.meeting?.getRoom?.();
    if (!room?.room_id || (!message?.trim() && !(overrideImages?.images?.length))) return;
    if (state.meetingStreaming) {
      logger.warn("Blocked: meeting is already streaming");
      return;
    }

    const images = overrideImages?.images || state.imageInputManager?.getPendingImages() || [];
    const displayImages = overrideImages?.displayImages || state.imageInputManager?.getDisplayImages() || [];
    const roomId = room.room_id;
    const user = localStorage.getItem("animaworks_user") || "human";

    mgr.addMessage("meeting", roomId, {
      role: "user",
      text: message || "",
      images: displayImages,
      timestamp: new Date().toISOString(),
    });

    state.meetingStreaming = true;
    const abortController = new AbortController();

    const input = $("chatPageInput");
    if (input && !overrideImages) state.imageInputManager?.clearImages();
    ctx.controllers.renderer?.reattach?.();

    let currentStreamingMsg = null;
    let _meetingTextAnimator = null;
    const isVisible = () => Boolean(state.meetingMode && state.meetingRoom?.room_id === roomId);
    const renderBubble = (msg, zone = "all") => {
      if (isVisible()) ctx.controllers.renderer?.renderStreamingBubble?.(msg, zone);
    };
    const renderFull = () => {
      if (isVisible()) ctx.controllers.renderer?.renderChat?.(!ctx.controllers.renderer?.isUserDetached?.());
    };
    let _rafPending = false;
    let _rafZone = "all";
    const _scheduleRender = (msg, zone = "text") => {
      if (!_rafPending) _rafZone = zone;
      else if (_rafZone !== zone) _rafZone = "all";
      if (_rafPending) return;
      _rafPending = true;
      requestAnimationFrame(() => {
        _rafPending = false;
        const z = _rafZone;
        _rafZone = "all";
        renderBubble(msg, z);
      });
    };

    try {
      const body = JSON.stringify({
        message: message || "",
        from_person: user,
        ...(images.length ? { images } : {}),
      });

      await streamMeetingChat(roomId, body, abortController.signal, {
        onSpeakerStart: ({ speaker, role: speakerRole }) => {
          currentStreamingMsg = {
            role: "assistant",
            speaker,
            speakerRole: speakerRole || "participant",
            from_person: speaker,
            text: "",
            streaming: true,
            timestamp: new Date().toISOString(),
            streamId: `meeting-${roomId}-${speaker}-${Date.now()}`,
          };
          mgr.getSession("meeting", roomId).messages.push(currentStreamingMsg);
          renderFull();
        },
        onTextDelta: (text) => {
          if (!currentStreamingMsg?.streaming) return;
          currentStreamingMsg.text += text;
          if (!_meetingTextAnimator) {
            _meetingTextAnimator = new TextAnimator({
              onUpdate: (displayText) => {
                if (!currentStreamingMsg) return;
                currentStreamingMsg._displayText = displayText;
                _scheduleRender(currentStreamingMsg, "text");
              },
            });
            _meetingTextAnimator.start();
          }
          _meetingTextAnimator.push(text);
        },
        onSpeakerEnd: () => {
          if (_meetingTextAnimator) {
            _meetingTextAnimator.flush();
            _meetingTextAnimator.stop();
            _meetingTextAnimator = null;
          }
          if (currentStreamingMsg) {
            delete currentStreamingMsg._displayText;
            currentStreamingMsg.streaming = false;
            renderFull();
          }
          currentStreamingMsg = null;
        },
        onMeetingRedirect: ({ from, to, content }) => {
          if (_meetingTextAnimator) {
            _meetingTextAnimator.flush();
            _meetingTextAnimator.stop();
            _meetingTextAnimator = null;
          }
          if (currentStreamingMsg) {
            delete currentStreamingMsg._displayText;
            currentStreamingMsg.streaming = false;
            currentStreamingMsg = null;
          }
          mgr.getSession("meeting", roomId).messages.push({
            role: "assistant",
            speaker: from || "",
            speakerRole: from === room.chair ? "chair" : "participant",
            from_person: from || "",
            text: `@${to || ""} ${content || ""}`.trim(),
            timestamp: new Date().toISOString(),
          });
          renderFull();
        },
        onDone: () => {
          if (_meetingTextAnimator) {
            _meetingTextAnimator.flush();
            _meetingTextAnimator = null;
          }
          if (currentStreamingMsg) {
            delete currentStreamingMsg._displayText;
            currentStreamingMsg.streaming = false;
          }
          renderFull();
        },
        onError: ({ message: errorMsg }) => {
          if (currentStreamingMsg) {
            if (!currentStreamingMsg.text) currentStreamingMsg.text = "";
            currentStreamingMsg.text += `\n${t("chat.error_prefix")} ${errorMsg}`;
            delete currentStreamingMsg._displayText;
            currentStreamingMsg.streaming = false;
            renderFull();
          }
        },
      });
    } catch (err) {
      if (err.name !== "AbortError") {
        logger.error("Meeting chat stream error", { roomId, error: err.message });
        if (currentStreamingMsg) {
          currentStreamingMsg.text += `\n${t("chat.error_prefix")} ${err.message}`;
          currentStreamingMsg.streaming = false;
          renderFull();
        }
      }
    } finally {
      state.meetingStreaming = false;
      if (input && state.meetingMode) {
        input.placeholder = t("meeting.placeholder");
        input.focus();
      }
      updateSendButton();
      ctx.controllers.meeting?.updatePanel?.();
    }
  }

  async function resumeActiveStream(animaName) {
    const tid = state.selectedThreadId || "default";
    if (mgr.isStreamingFor(animaName, tid)) return;

    ctx.controllers.renderer.renderChat();

    const smartScroll = () => !ctx.controllers.renderer.isUserDetached();
    const renderBubbleR = (msg, zone = "all") => {
      if (state.selectedAnima === animaName && state.selectedThreadId === tid) ctx.controllers.renderer.renderStreamingBubble(msg, zone);
    };

    let _rafPendingR = false;
    let _rafZoneR = "all";
    const _scheduleRenderR = (msg, zone = "text") => {
      if (!_rafPendingR) _rafZoneR = zone;
      else if (_rafZoneR !== zone) _rafZoneR = "all";
      if (_rafPendingR) return;
      _rafPendingR = true;
      requestAnimationFrame(() => {
        _rafPendingR = false;
        const z = _rafZoneR;
        _rafZoneR = "all";
        renderBubbleR(msg, z);
      });
    };

    let streamingMsg = null;
    let _resumeAnimator = null;
    let _resumeThinkingAnimator = null;
    let resumeBase = "";
    const isVisibleR = () => state.selectedAnima === animaName && state.selectedThreadId === tid;
    const emitThink = (extra = {}) => emitThinkActivity({ anima: animaName, threadId: tid, msg: streamingMsg, ...extra });
    const syncAvatar = () => { if (isVisibleR()) syncThinkAvatar(_streamBubble(streamingMsg), streamingMsg); };

    await mgr.resumeStream(animaName, tid, {
      callbacks: {
        onStreamCreated: msg => {
          streamingMsg = msg;
          restoreThink(msg);
          resumeBase = msg.text || "";
          _resumeAnimator = new TextAnimator({
            onUpdate: (displayText) => {
              if (!streamingMsg) return;
              streamingMsg._displayText = resumeBase + displayText;
              _scheduleRenderR(streamingMsg, "text");
            },
          });
          _resumeAnimator.start();
          ctx.controllers.renderer.renderChat(smartScroll());
          updateSendButton();
          emitThink();
        },
        onTextDelta: text => {
          if (!streamingMsg?.streaming) return;
          streamingMsg.text += text;
          if (_resumeAnimator) _resumeAnimator.push(text);
          if (thinkOnText(streamingMsg)) {
            renderBubbleR(streamingMsg, "tools");
            syncAvatar();
            emitThink();
          }
        },
        onToolStart: (toolName, detail) => {
          if (!streamingMsg?.streaming) return;
          const { moved } = thinkToolStart(streamingMsg, toolName, detail?.tool_id);
          if (moved) {
            resumeBase = "";
            if (_resumeAnimator) _resumeAnimator.start();
          }
          streamingMsg.activeTool = toolName;
          renderBubbleR(streamingMsg, moved ? "all" : "tools");
          syncAvatar();
          emitThink();
          ctx.controllers.workIndicator?.onToolStart(toolName, detail);
        },
        onToolDetail: (toolName, detailText, info) => {
          if (!streamingMsg?.streaming) return;
          thinkToolDetail(streamingMsg, info?.tool_id, detailText);
          renderBubbleR(streamingMsg, "tools");
          ctx.controllers.workIndicator?.onToolDetail(toolName, detailText, info);
        },
        onToolEnd: (detail) => {
          if (!streamingMsg?.streaming) return;
          streamingMsg.activeTool = null;
          thinkToolEnd(streamingMsg, detail);
          renderBubbleR(streamingMsg, "tools");
          emitThink();
          ctx.controllers.workIndicator?.onToolEnd(detail);
        },
        onThinkingStart: () => {
          if (!streamingMsg?.streaming) return;
          streamingMsg.thinkingText = ""; streamingMsg.thinking = true;
          _resumeThinkingAnimator = new TextAnimator({
            onUpdate: (displayText) => {
              if (!streamingMsg) return;
              streamingMsg._displayThinkingText = displayText;
              _scheduleRenderR(streamingMsg, "thinking");
            },
          });
          _resumeThinkingAnimator.start();
          renderBubbleR(streamingMsg, "thinking");
          ctx.controllers.workIndicator?.onThinkingStart();
        },
        onThinkingDelta: text => {
          if (!streamingMsg?.streaming) return;
          streamingMsg.thinkingText = (streamingMsg.thinkingText || "") + text;
          if (_resumeThinkingAnimator) _resumeThinkingAnimator.push(text);
        },
        onThinkingEnd: () => {
          if (!streamingMsg?.streaming) return;
          if (_resumeThinkingAnimator) { _resumeThinkingAnimator.flush(); _resumeThinkingAnimator = null; }
          delete streamingMsg._displayThinkingText;
          streamingMsg.thinking = false;
          renderBubbleR(streamingMsg, "thinking");
          ctx.controllers.workIndicator?.onThinkingEnd();
        },
        onError: ({ message: errorMsg }) => {
          if (_resumeAnimator) _resumeAnimator.flush();
          if (_resumeThinkingAnimator) { _resumeThinkingAnimator.flush(); _resumeThinkingAnimator = null; }
          thinkError(streamingMsg);
          emitThink();
          if (streamingMsg) { streamingMsg.text += `\n${t("chat.error_prefix")} ${errorMsg}`; delete streamingMsg._displayText; delete streamingMsg._displayThinkingText; streamingMsg.streaming = false; if (state.selectedAnima === animaName && state.selectedThreadId === tid) ctx.controllers.renderer.renderChat(smartScroll()); }
          ctx.controllers.workIndicator?.onStreamSettled();
        },
        onContextUpdate: (ctxData) => { updateContextRing(ctxData, animaName); },
        onDone: ({ summary, emotion, images, files: doneFiles, thinkingSummary, contextUsageRatio, inputTokens, contextWindow, contextThreshold }) => {
          if (_resumeAnimator) _resumeAnimator.flush();
          if (_resumeThinkingAnimator) { _resumeThinkingAnimator.flush(); _resumeThinkingAnimator = null; }
          if (streamingMsg) {
            const reply = thinkDone(streamingMsg, summary || streamingMsg.text || "", { aligned: !!summary, emotion });
            streamingMsg.text = reply || t("chat.empty_response");
            streamingMsg.emotion = emotion || null;
            delete streamingMsg._displayText;
            delete streamingMsg._displayThinkingText;
            streamingMsg.images = images || [];
            streamingMsg.files = doneFiles || [];
            if (!streamingMsg.thinkingText && thinkingSummary) {
              streamingMsg.thinkingText = thinkingSummary;
            }
            streamingMsg.streaming = false; streamingMsg.activeTool = null;
            if (contextUsageRatio || contextWindow) {
              updateContextRing({ contextUsageRatio, inputTokens, contextWindow, threshold: contextThreshold }, animaName);
            }
            if (isVisibleR()) ctx.controllers.renderer.renderChat(smartScroll());
            _applyDoneEmotion(streamingMsg, animaName, emotion);
            emitThink({ emotion });
            ctx.controllers.renderer.markResponseComplete(animaName, tid);
          }
          ctx.controllers.workIndicator?.onStreamSettled();
        },
      },
      onFinally: () => {
        if (_resumeAnimator) { _resumeAnimator.stop(); _resumeAnimator = null; }
        if (_resumeThinkingAnimator) { _resumeThinkingAnimator.stop(); _resumeThinkingAnimator = null; }
        ctx.controllers.workIndicator?.onStreamSettled();
        try {
          if (streamingMsg?.streaming) {
            streamingMsg.streaming = false;
            if (thinkStop(streamingMsg)) emitThink();
            delete streamingMsg._displayText;
            delete streamingMsg._displayThinkingText;
            if (!streamingMsg.text) streamingMsg.text = t("chat.empty_response");
            if (state.selectedAnima === animaName && state.selectedThreadId === tid) ctx.controllers.renderer.renderChat(smartScroll());
          }
          updateSendButton();
          ctx.controllers.anima.renderAnimaTabs();
          ctx.controllers.thread.renderThreadTabs();
        } finally {
          if (mgr.getPendingQueue(animaName, tid).length > 0) {
            const next = mgr.dequeue(animaName, tid);
            showPendingIndicator();
            if (mgr.getPendingQueue(animaName, tid).length === 0) hidePendingIndicator();
            setTimeout(() => sendChat(next.text, { images: next.images, displayImages: next.displayImages, files: next.files, displayFiles: next.displayFiles, targetAnima: animaName, targetThread: tid }), 150);
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
    updateSendButton, updateContextRing, restoreContextRing,
  };
}
