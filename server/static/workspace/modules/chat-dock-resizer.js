// ── Workspace Chat Dock resizer ──
// Drag or arrow-key the conversation/stage boundary, persisting the width.
import { t } from "/shared/i18n.js";

export const DOCK_WIDTH_KEY = "aw-workspace-dock-width";
export const MIN_DOCK_WIDTH = 340;
export const MAX_DOCK_WIDTH = 760;
export const MIN_STAGE_WIDTH = 320;
export const KEYBOARD_STEP = 20;

function defaultWidth() {
  return Math.max(380, Math.min(window.innerWidth * 0.38, 560));
}

function storedWidth() {
  try {
    const value = Number(localStorage.getItem(DOCK_WIDTH_KEY));
    return Number.isFinite(value) && value > 0 ? value : null;
  } catch {
    return null;
  }
}

function persistWidth(width) {
  try {
    localStorage.setItem(DOCK_WIDTH_KEY, String(Math.round(width)));
  } catch { /* storage may be disabled */ }
}

function clearStoredWidth() {
  try {
    localStorage.removeItem(DOCK_WIDTH_KEY);
  } catch { /* storage may be disabled */ }
}

export function initChatDockResizer({ dock, resizer, layout } = {}) {
  if (!dock || !resizer) return { refresh() {}, reset() {}, destroy() {} };

  const layoutEl = layout || dock.parentElement;
  // The width the user chose; null means "follow the window".
  let preferred = storedWidth();
  let width = preferred ?? defaultWidth();
  let dragging = false;
  let dragStartX = 0;
  let dragStartWidth = 0;
  let destroyed = false;

  function isNarrow() {
    return window.innerWidth <= 768;
  }

  function maxWidth() {
    const available = layoutEl?.clientWidth || window.innerWidth;
    const resizerWidth = resizer.offsetWidth || 0;
    return Math.max(
      MIN_DOCK_WIDTH,
      Math.min(MAX_DOCK_WIDTH, available - MIN_STAGE_WIDTH - resizerWidth),
    );
  }

  function clampWidth(value) {
    return Math.max(MIN_DOCK_WIDTH, Math.min(maxWidth(), Number(value) || defaultWidth()));
  }

  function applyWidth(value, { persist = false } = {}) {
    width = clampWidth(value);
    layoutEl?.style.setProperty("--ws-dock-width", `${Math.round(width)}px`);
    resizer.setAttribute("aria-valuenow", String(Math.round(width)));
    resizer.setAttribute("aria-valuemin", String(MIN_DOCK_WIDTH));
    resizer.setAttribute("aria-valuemax", String(Math.round(maxWidth())));
    if (persist) persistWidth(width);
  }

  function refresh() {
    if (destroyed) return;
    resizer.setAttribute("aria-label", t("ws.resize_dock"));
    const narrow = isNarrow();
    resizer.hidden = narrow || dock.hidden;
    if (narrow || dragging) return;
    applyWidth(preferred ?? defaultWidth());
  }

  function reset() {
    if (destroyed) return;
    clearStoredWidth();
    preferred = null;
    applyWidth(defaultWidth());
  }

  function onPointerDown(event) {
    if (resizer.hidden || event.button !== 0) return;
    event.preventDefault();
    dragging = true;
    dragStartX = event.clientX;
    dragStartWidth = width;
    resizer.classList.add("is-dragging");
    document.body.style.userSelect = "none";
    try {
      resizer.setPointerCapture(event.pointerId);
    } catch { /* pointer capture is best-effort */ }
  }

  function onPointerMove(event) {
    if (!dragging) return;
    applyWidth(dragStartWidth + event.clientX - dragStartX);
  }

  function endDrag(event) {
    if (!dragging) return;
    dragging = false;
    resizer.classList.remove("is-dragging");
    document.body.style.userSelect = "";
    try {
      resizer.releasePointerCapture(event.pointerId);
    } catch { /* capture may already be gone */ }
    preferred = width;
    persistWidth(width);
  }

  function onKeyDown(event) {
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
    event.preventDefault();
    const delta = event.key === "ArrowRight" ? KEYBOARD_STEP : -KEYBOARD_STEP;
    applyWidth(width + delta, { persist: true });
    preferred = width;
  }

  function onDoubleClick(event) {
    event.preventDefault();
    reset();
  }

  resizer.addEventListener("pointerdown", onPointerDown);
  resizer.addEventListener("pointermove", onPointerMove);
  resizer.addEventListener("pointerup", endDrag);
  resizer.addEventListener("pointercancel", endDrag);
  resizer.addEventListener("keydown", onKeyDown);
  resizer.addEventListener("dblclick", onDoubleClick);
  window.addEventListener("resize", refresh);

  const hiddenObserver = typeof MutationObserver !== "undefined"
    ? new MutationObserver(refresh)
    : null;
  hiddenObserver?.observe(dock, { attributes: true, attributeFilter: ["hidden"] });

  refresh();

  function destroy() {
    destroyed = true;
    hiddenObserver?.disconnect();
    window.removeEventListener("resize", refresh);
    resizer.removeEventListener("pointerdown", onPointerDown);
    resizer.removeEventListener("pointermove", onPointerMove);
    resizer.removeEventListener("pointerup", endDrag);
    resizer.removeEventListener("pointercancel", endDrag);
    resizer.removeEventListener("keydown", onKeyDown);
    resizer.removeEventListener("dblclick", onDoubleClick);
    document.body.style.userSelect = "";
  }

  return { refresh, reset, destroy };
}
