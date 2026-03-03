// ── Bustup Overlay Controller ──────────────────
import { bustupCandidates, resolveAvatar, resolveCachedAvatar } from "../../modules/avatar-resolver.js";

export function createAvatarController(ctx) {
  const $ = ctx.$;
  const { state, deps } = ctx;
  const { escapeHtml } = deps;

  async function updateAvatar() {
    const container = $("chatPageAvatar");
    if (!container || !state.selectedAnima) {
      if (container) container.innerHTML = "";
      state.bustupUrl = null;
      return;
    }

    state.bustupUrl = null;
    const name = state.selectedAnima;
    const url = await resolveCachedAvatar(name, bustupCandidates(), "S");
    if (url) {
      state.bustupUrl = url;
      container.innerHTML = `<img src="${escapeHtml(url)}" alt="${escapeHtml(name)}" class="anima-avatar-img">`;
      container.style.cursor = "pointer";
      return;
    }
    container.style.cursor = "";
    container.innerHTML = `<div class="anima-avatar-placeholder">${escapeHtml(name.charAt(0).toUpperCase())}</div>`;
  }

  async function showBustupOverlay() {
    if (!state.selectedAnima) return;
    if (!state.bustupUrl) {
      state.bustupUrl = await resolveAvatar(state.selectedAnima, bustupCandidates());
    }
    if (!state.bustupUrl) return;
    removeBustupOverlay();

    const overlay = document.createElement("div");
    overlay.className = "bustup-overlay";
    overlay.id = "chatBustupOverlay";
    overlay.innerHTML = `<img class="bustup-overlay-img" src="${escapeHtml(state.bustupUrl)}" alt="${escapeHtml(state.selectedAnima)}">`;
    overlay.addEventListener("click", dismissBustupOverlay);
    document.body.appendChild(overlay);
    requestAnimationFrame(() => overlay.classList.add("visible"));
  }

  function dismissBustupOverlay() {
    const overlay = document.getElementById("chatBustupOverlay");
    if (!overlay) return;
    overlay.classList.remove("visible");
    overlay.classList.add("hiding");
    overlay.addEventListener("transitionend", () => overlay.remove(), { once: true });
  }

  function removeBustupOverlay() {
    document.querySelectorAll("#chatBustupOverlay").forEach(el => el.remove());
  }

  function onBustupEscape(e) {
    if (e.key === "Escape") dismissBustupOverlay();
  }

  return { updateAvatar, showBustupOverlay, dismissBustupOverlay, removeBustupOverlay, onBustupEscape };
}
