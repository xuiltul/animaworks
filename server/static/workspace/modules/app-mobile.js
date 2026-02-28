// ── Mobile Helpers ──────────────────────
// Viewport height fallback, timeline collapse, and mobile keyboard handling.

/** Set CSS custom property for viewport height fallback (svh-unsupported browsers). */
export function initViewportHeightFallback() {
  if (CSS.supports && CSS.supports('height', '100svh')) return;

  function setVh() {
    const vh = window.visualViewport?.height || window.innerHeight;
    document.documentElement.style.setProperty('--vh-fallback', `${vh}px`);
  }
  setVh();
  if (window.visualViewport) {
    window.visualViewport.addEventListener('resize', setVh);
  }
  window.addEventListener('resize', setVh);
}

/** Toggle timeline collapse on iPad-width viewports. */
export function initTimelineCollapseToggle() {
  document.addEventListener('click', (e) => {
    const btn = e.target.closest('.timeline-toggle-btn');
    if (!btn) return;
    const timeline = btn.closest('.ws-timeline');
    if (timeline) {
      timeline.classList.toggle('collapsed');
    }
  });
}

export function initMobileKeyboard() {
  const vv = window.visualViewport;

  function scrollInputIntoView() {
    const active = document.activeElement;
    if (!active?.matches(".ws-conv-input, .chat-input")) return;
    requestAnimationFrame(() => {
      active.scrollIntoView({ block: "nearest" });
    });
  }

  if (vv) {
    vv.addEventListener("resize", scrollInputIntoView);
  } else {
    let lastHeight = window.innerHeight;
    window.addEventListener("resize", () => {
      const current = window.innerHeight;
      if (Math.abs(current - lastHeight) > 100) {
        scrollInputIntoView();
      }
      lastHeight = current;
    });
  }
}
