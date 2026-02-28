// ── Image Input / Voice Chat Controller ────────
import { $ } from "./ctx.js";

export function createImageVoiceController(ctx) {
  const { state, deps } = ctx;
  const { createImageInput, initLightbox, initVoiceUI, updateVoiceUIAnima } = deps;

  function buildVoiceChatCallbacks(animaName) {
    return {
      addUserBubble(text) {
        const tid = state.selectedThreadId;
        if (!state.chatHistories[animaName]) state.chatHistories[animaName] = {};
        if (!state.chatHistories[animaName][tid]) state.chatHistories[animaName][tid] = [];
        state.chatHistories[animaName][tid].push({ role: "user", text, timestamp: new Date().toISOString() });
        ctx.controllers.renderer.renderChat();
      },
      addStreamingBubble() {
        const tid = state.selectedThreadId;
        if (!state.chatHistories[animaName]) state.chatHistories[animaName] = {};
        if (!state.chatHistories[animaName][tid]) state.chatHistories[animaName][tid] = [];
        const msg = { role: "assistant", text: "", streaming: true, activeTool: null, timestamp: new Date().toISOString() };
        state.chatHistories[animaName][tid].push(msg);
        ctx.controllers.renderer.renderChat();
        return msg;
      },
      updateStreamingBubble(msg) {
        ctx.controllers.renderer.renderStreamingBubble(msg);
      },
      finalizeStreamingBubble() {
        ctx.controllers.renderer.renderChat();
      },
    };
  }

  function updateVoiceAnima(animaName) {
    const wasActive = updateVoiceUIAnima(animaName);
    const chatInputForm = $("chatPageForm") || document.querySelector(".chat-input-form");
    if (chatInputForm && animaName) {
      initVoiceUI(chatInputForm, animaName, buildVoiceChatCallbacks(animaName), { autoConnect: wasActive });
    }
  }

  function initImageInputCtrl() {
    const chatMain = state.container?.querySelector(".chat-page-main");
    const previewEl = $("chatPagePreviewBar");
    const chatInput = $("chatPageInput");
    if (!chatMain || !previewEl || !chatInput) return;

    state.imageInputManager = createImageInput({
      container: chatMain,
      inputArea: chatInput,
      previewContainer: previewEl,
    });
    initLightbox();

    const chatInputFormEl = $("chatPageForm") || document.querySelector(".chat-input-form");
    if (chatInputFormEl && state.selectedAnima) {
      initVoiceUI(chatInputFormEl, state.selectedAnima, buildVoiceChatCallbacks(state.selectedAnima));
    }
  }

  return { initImageInput: initImageInputCtrl, updateVoiceAnima };
}
