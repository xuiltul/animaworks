// ── Bustup Overlay Controller ──────────────────
import { $ } from "./ctx.js";

export function createAvatarController(ctx) {
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
    const candidates = ["avatar_bustup.png"];
    for (const filename of candidates) {
      const url = `/api/animas/${encodeURIComponent(name)}/assets/${encodeURIComponent(filename)}`;
      try {
        const resp = await fetch(url, { method: "HEAD" });
        if (resp.ok) {
          if (filename === "avatar_bustup.png") state.bustupUrl = url;
          container.innerHTML = `<img src="${escapeHtml(url)}" alt="${escapeHtml(name)}" class="anima-avatar-img">`;
          container.style.cursor = state.bustupUrl ? "pointer" : "";
          return;
        }
      } catch { /* next */ }
    }
    container.style.cursor = "";
    container.innerHTML = `<div class="anima-avatar-placeholder">${escapeHtml(name.charAt(0).toUpperCase())}</div>`;
  }

  async function showBustupOverlay() {
    if (!state.selectedAnima) return;
    if (!state.bustupUrl) {
      const url = `/api/animas/${encodeURIComponent(state.selectedAnima)}/assets/avatar_bustup.png`;
      try {
        const resp = await fetch(url, { method: "HEAD" });
        if (resp.ok) state.bustupUrl = url;
      } catch { /* noop */ }
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
    document.getElementById("chatBustupOverlay")?.remove();
  }

  function onBustupEscape(e) {
    if (e.key === "Escape") dismissBustupOverlay();
  }

  return { updateAvatar, showBustupOverlay, dismissBustupOverlay, removeBustupOverlay, onBustupEscape };
}
