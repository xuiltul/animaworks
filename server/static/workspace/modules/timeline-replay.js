/**
 * timeline-replay.js — 3D scene replay for timeline events.
 *
 * Replays a selected event by highlighting desks, showing message
 * effects, or opening message popups in the office3d scene.
 */

import { showMessage as showMessagePopup } from "./message-popup.js";
import { resolvePersons } from "./timeline-dom.js";

// ── Highlight helper (imported lazily from office3d) ──

let _highlightDesk = null;
let _clearHighlight = null;

export function ensureHighlightFns() {
  if (_highlightDesk) return;
  try {
    import("./office3d.js").then((mod) => {
      _highlightDesk = mod.highlightDesk;
      _clearHighlight = mod.clearHighlight;
    });
  } catch {
    // Will operate without highlight capability
  }
}

// ── Replay ─────────────────────────────────────────

/**
 * Replay a timeline event in the 3D scene.
 *
 * @param {object} event  — the timeline event
 * @param {HTMLElement} el — the clicked DOM element (for visual feedback)
 * @param {{ interactionManager: object|null }} ctx
 */
export function replayEvent(event, el, ctx) {
  const { type, anima, meta } = event;
  const { interactionManager } = ctx;

  // Visual feedback
  el.classList.add("replaying");
  setTimeout(() => el.classList.remove("replaying"), 2000);

  ensureHighlightFns();

  switch (type) {
    case "message":
    case "dm_received":
    case "dm_sent": {
      const p = resolvePersons(event);
      if (interactionManager && p.from && p.to) {
        interactionManager.showMessageEffect(p.from, p.to, p.text);
      } else if (_highlightDesk && (p.from || p.to)) {
        _highlightDesk(p.from || p.to);
        setTimeout(() => { if (_clearHighlight) _clearHighlight(); }, 3000);
      }
      if (meta && meta.message_id) {
        showMessagePopup(meta.message_id);
      }
      break;
    }

    case "chat":
    case "message_received":
    case "response_sent":
    case "board":
    case "channel_read":
    case "channel_post":
      if (_highlightDesk && anima) {
        _highlightDesk(anima);
        setTimeout(() => { if (_clearHighlight) _clearHighlight(); }, 3000);
      }
      break;

    case "heartbeat":
    case "heartbeat_start":
    case "heartbeat_end":
    case "heartbeat_reflection":
    case "cron":
    case "cron_executed":
      if (_highlightDesk && anima) {
        _highlightDesk(anima);
        setTimeout(() => { if (_clearHighlight) _clearHighlight(); }, 3000);
      }
      break;

    default:
      if (_highlightDesk && anima) {
        _highlightDesk(anima);
        setTimeout(() => { if (_clearHighlight) _clearHighlight(); }, 3000);
      }
      break;
  }
}
