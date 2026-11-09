// ── Presence panel: selected Anima's bust-up at the lower right of the talk area ──
import { presence, PANEL_EMOTION_TTL_MS } from "../../modules/anima-presence.js";
import { bustUrl } from "../../shared/chat/emotion-face.js";
import { BustupAnimator } from "./bustup-animator.js";

export const PRESENCE_PREF_KEY = "aw-chat-presence";
const MIN_WIDTH = 1200;
const SCROLL_PAUSE_PX = 240;

export function isPresenceEnabled() {
  try { return localStorage.getItem(PRESENCE_PREF_KEY) !== "off"; } catch { return true; }
}

export function setPresenceEnabled(on) {
  try { localStorage.setItem(PRESENCE_PREF_KEY, on ? "on" : "off"); } catch { /* storage blocked */ }
  window.dispatchEvent(new CustomEvent("aw:presence-pref"));
}

/** Text of the speech bubble above the bust-up (empty = hidden). */
export function sayText(t, entry) {
  if (entry.working || entry.calling) {
    const base = entry.label || t("chat.presence.thinking");
    return /[…。.]$/.test(base) ? base : `${base}…`;
  }
  if (entry.awaitingHuman) return t("chat.presence.waiting_say");
  return "";
}

export function createPresencePanel(ctx) {
  const $ = ctx.$;
  const { state, deps } = ctx;
  const { t } = deps;

  const mountTarget = typeof ctx.state.presenceMount === "function"
    ? ctx.state.presenceMount()
    : null;
  const area = mountTarget || $("chatPageMessages")?.parentElement;
  const areaIsMessages = !mountTarget;
  const messages = $("chatPageMessages");
  if (!area || !messages) return { refresh() {}, destroy() {} };

  const el = document.createElement("aside");
  el.className = "chat-presence";
  el.hidden = true;
  el.innerHTML = `
    <div class="chat-presence-say" aria-live="polite"></div>
    <div class="chat-presence-figure">
      <div class="chat-presence-frame"><div class="chat-presence-stage"></div></div>
      <button type="button" class="chat-presence-x" aria-label="${deps.escapeHtml(t("chat.presence.hide"))}" title="${deps.escapeHtml(t("chat.presence.hide"))}">&times;</button>
    </div>`;
  area.appendChild(el);

  const sayEl = el.querySelector(".chat-presence-say");
  const stage = el.querySelector(".chat-presence-stage");
  stage.setAttribute("role", "button");
  stage.setAttribute("tabindex", "0");
  stage.setAttribute("aria-label", t("chat.presence.open"));
  stage.title = t("chat.presence.open");

  let animator = null;
  let animatorFor = null;
  let currentEmotion = null;
  let paused = false;
  let timer = null;
  let destroyed = false;
  const reduceMotion = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

  function _splitActive() {
    return Boolean(state.container?.parentElement?.classList.contains("has-split"));
  }

  function _shouldShow() {
    if (state.embedded && state.embeddedVisible === false) return false;
    return isPresenceEnabled() && Boolean(state.selectedAnima)
      && window.innerWidth >= MIN_WIDTH && !_splitActive();
  }

  function _stopAnimator() {
    if (animator) { try { animator.destroy(); } catch { /* already gone */ } }
    animator = null;
    animatorFor = null;
  }

  function _renderStatic(name, emotion) {
    stage.innerHTML = "";
    const img = document.createElement("img");
    img.className = "chat-presence-static";
    img.alt = name;
    img.draggable = false;
    img.src = bustUrl(name, emotion, "P");
    stage.appendChild(img);
  }

  function _paint(name, emotion) {
    if (reduceMotion) {
      _stopAnimator();
      _renderStatic(name, emotion);
      currentEmotion = emotion;
      return;
    }
    if (!animator || animatorFor !== name) {
      _stopAnimator();
      stage.innerHTML = "";
      animator = new BustupAnimator(stage, name, "P");
      animatorFor = name;
      currentEmotion = null;
    }
    if (currentEmotion !== emotion) {
      currentEmotion = emotion;
      animator.setExpression(emotion);
    }
    // While reading older messages the picture stays as it is; only blinking stops.
    if (paused) animator.pause?.();
    else animator.resume?.();
  }

  function _openPortrait() {
    const name = state.selectedAnima;
    if (!name) return;
    ctx.controllers.avatar?.showBustupOverlay(name, presence.emotionFor(name, PANEL_EMOTION_TTL_MS));
  }

  function _onStageKeydown(event) {
    if (event.key !== "Enter" && event.key !== " ") return;
    event.preventDefault();
    _openPortrait();
  }

  function refresh() {
    if (destroyed) return;
    clearTimeout(timer);
    const show = _shouldShow();
    el.hidden = !show;
    if (areaIsMessages) area.classList.toggle("presence-on", show);
    if (!show) { _stopAnimator(); currentEmotion = null; return; }

    const name = state.selectedAnima;
    const entry = presence.get(name);
    const emotion = presence.emotionFor(name, PANEL_EMOTION_TTL_MS);
    const say = sayText(t, entry);
    sayEl.textContent = say;
    el.classList.toggle("idle", !say);
    el.dataset.emotion = emotion;
    _paint(name, emotion);

    // Come back to the neutral face once the reply emotion has expired.
    if (entry.lastEmotionAt && emotion !== "neutral" && !entry.working && !entry.awaitingHuman) {
      const left = entry.lastEmotionAt + PANEL_EMOTION_TTL_MS - Date.now();
      if (left > 0) timer = setTimeout(refresh, left + 50);
    }
  }

  function _onScroll() {
    const far = messages.scrollHeight - messages.scrollTop - messages.clientHeight > SCROLL_PAUSE_PX;
    if (far === paused) return;
    paused = far;
    if (animator) {
      if (paused) animator.pause?.();
      else animator.resume?.();
      return;
    }
    refresh();
  }

  el.querySelector(".chat-presence-x").addEventListener("click", () => setPresenceEnabled(false));
  stage.addEventListener("click", _openPortrait);
  stage.addEventListener("keydown", _onStageKeydown);
  messages.addEventListener("scroll", _onScroll, { passive: true });
  window.addEventListener("resize", refresh);
  window.addEventListener("aw:presence-pref", refresh);

  function destroy() {
    destroyed = true;
    clearTimeout(timer);
    _stopAnimator();
    messages.removeEventListener("scroll", _onScroll);
    stage.removeEventListener("click", _openPortrait);
    stage.removeEventListener("keydown", _onStageKeydown);
    window.removeEventListener("resize", refresh);
    window.removeEventListener("aw:presence-pref", refresh);
    el.remove();
    if (areaIsMessages) area.classList.remove("presence-on");
  }

  return { refresh, destroy };
}
