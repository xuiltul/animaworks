/**
 * replay-ui.js — Replay control bar for the AnimaWorks Dashboard.
 *
 * Renders a seek bar and playback controls below the KPI bar in the org dashboard.
 * Layout: [◀◀] [▶/⏸] [▶▶] ──●──────────── 09:00 ─ 22:00  [速度: 50x]
 */

import { createLogger } from "../../shared/logger.js";
import { t } from "/shared/i18n.js";

const logger = createLogger("replay-ui");

// ── Constants ─────────────────────────────────────────────────────────────

const SPEED_CYCLE = [1, 5, 10, 50, 100, 200];
const SKIP_MS = 5 * 60 * 1000; // 5 minutes
const RANGE_OPTIONS = [
  { value: 1, label: "1h" },
  { value: 3, label: "3h" },
  { value: 6, label: "6h" },
  { value: 12, label: "12h" },
  { value: 24, label: "24h" },
];

// ── Helpers ───────────────────────────────────────────────────────────────

/**
 * Format millisecond timestamp to HH:MM in JST (Asia/Tokyo).
 * @param {number} ms - Unix timestamp in milliseconds
 * @returns {string} Formatted time string
 */
function formatTimeJST(ms) {
  const d = new Date(ms);
  return d.toLocaleTimeString("ja-JP", {
    hour: "2-digit",
    minute: "2-digit",
    timeZone: "Asia/Tokyo",
  });
}

function titleCase(value) {
  const text = String(value || "").trim();
  if (!text) return "";
  return text.charAt(0).toUpperCase() + text.slice(1);
}

function shortenText(value, maxLen = 42) {
  const text = String(value || "").trim();
  if (text.length <= maxLen) return text;
  return `${text.slice(0, Math.max(0, maxLen - 3)).trim()}...`;
}

// ── ReplayUI ───────────────────────────────────────────────────────────────

/**
 * Replay control bar for the org dashboard.
 * Inserts controls after the KPI bar within the org-canvas-root container.
 */
export class ReplayUI {
  /**
   * @param {object} opts
   * @param {HTMLElement} opts.container - DOM element (org-canvas-root) to insert controls into
   * @param {() => void} opts.onPlay - Called when play is requested
   * @param {() => void} opts.onPause - Called when pause is requested
   * @param {(timeMs: number) => void} opts.onSeek - Called when seek position changes
   * @param {(speed: number) => void} opts.onSpeedChange - Called when speed changes
   * @param {() => void} opts.onExit - Called when exiting replay mode
   * @param {(hours: number) => void} [opts.onRangeChange] - Called when time range changes
   * @param {number} [opts.initialHours=24] - Initial hours range
   */
  constructor({ container, onPlay, onPause, onSeek, onSpeedChange, onExit, onRangeChange, initialHours = 24 }) {
    this._container = container;
    this._onPlay = onPlay || (() => {});
    this._onPause = onPause || (() => {});
    this._onSeek = onSeek || (() => {});
    this._onSpeedChange = onSpeedChange || (() => {});
    this._onExit = onExit || (() => {});
    this._onRangeChange = onRangeChange || (() => {});
    this._initialHours = initialHours;

    this._startMs = 0;
    this._endMs = 1000;
    this._currentMs = 0;
    this._isPlaying = false;
    this._speedIndex = 0;
    this._isLoading = false;
    this._isDragging = false;

    this._root = null;
    this._playBtn = null;
    this._slider = null;
    this._timeStart = null;
    this._timeCurrent = null;
    this._timeEnd = null;
    this._speedBtn = null;
    this._rangeSelect = null;
    this._errorEl = null;
    this._narrativeEl = null;
    this._narrativeKind = null;
    this._narrativeLabel = null;
    this._narrativeSummary = null;
    this._narrativeRoute = null;
    this._narrativeGroup = null;
    this._narrativeStatus = null;
    this._narrativeCount = null;

    this._build();
  }

  _build() {
    this._root = document.createElement("div");
    this._root.className = "org-replay-bar";
    this._root.id = "orgReplayBar";

    const rangeOpts = RANGE_OPTIONS.map(o =>
      `<option value="${o.value}"${o.value === this._initialHours ? " selected" : ""}>${o.label}</option>`
    ).join("");

    this._root.innerHTML = `
      <div class="org-replay-row">
        <div class="org-replay-controls">
          <button class="org-replay-btn" id="replayExitBtn" title="Live mode" aria-label="${t("ws.replay.exit")}">✕</button>
          <button class="org-replay-btn" id="replayPrevBtn" title="Back 5min" aria-label="${t("ws.replay.prev")}">◀◀</button>
          <button class="org-replay-btn org-replay-btn--play" id="replayPlayBtn" title="Play" aria-label="${t("ws.replay.play")}">▶</button>
          <button class="org-replay-btn" id="replayNextBtn" title="Forward 5min" aria-label="${t("ws.replay.next")}">▶▶</button>
        </div>
        <div class="org-replay-seek">
          <span class="org-replay-time" id="replayTimeStart">--:--</span>
          <input type="range" class="org-replay-slider" id="replaySlider" min="0" max="1000" value="0" step="1" aria-label="${t("ws.replay.seek")}">
          <span class="org-replay-time" id="replayTimeCurrent">--:--</span>
          <span class="org-replay-time" id="replayTimeEnd">--:--</span>
        </div>
        <select class="org-replay-range" id="replayRangeSelect" title="遡り時間" aria-label="${t("ws.replay.range")}">${rangeOpts}</select>
        <button class="org-replay-speed" id="replaySpeedBtn" aria-label="${t("ws.replay.speed", { speed: "1x" })}">1x</button>
        <div class="org-replay-error" id="replayError" role="status" hidden></div>
      </div>
      <div class="org-replay-narrative" id="orgReplayNarrative">
        <div class="org-replay-narrative-main">
          <span class="org-replay-narrative-kind"></span>
          <span class="org-replay-narrative-label"></span>
          <span class="org-replay-narrative-summary"></span>
        </div>
        <div class="org-replay-narrative-meta">
          <span class="org-replay-narrative-route"></span>
          <span class="org-replay-narrative-group"></span>
          <span class="org-replay-narrative-status"></span>
          <span class="org-replay-narrative-count"></span>
        </div>
      </div>
    `;

    this._playBtn = this._root.querySelector("#replayPlayBtn");
    this._slider = this._root.querySelector("#replaySlider");
    this._timeStart = this._root.querySelector("#replayTimeStart");
    this._timeCurrent = this._root.querySelector("#replayTimeCurrent");
    this._timeEnd = this._root.querySelector("#replayTimeEnd");
    this._speedBtn = this._root.querySelector("#replaySpeedBtn");
    this._rangeSelect = this._root.querySelector("#replayRangeSelect");
    this._errorEl = this._root.querySelector("#replayError");
    this._narrativeEl = this._root.querySelector("#orgReplayNarrative");
    this._narrativeKind = this._root.querySelector(".org-replay-narrative-kind");
    this._narrativeLabel = this._root.querySelector(".org-replay-narrative-label");
    this._narrativeSummary = this._root.querySelector(".org-replay-narrative-summary");
    this._narrativeRoute = this._root.querySelector(".org-replay-narrative-route");
    this._narrativeGroup = this._root.querySelector(".org-replay-narrative-group");
    this._narrativeStatus = this._root.querySelector(".org-replay-narrative-status");
    this._narrativeCount = this._root.querySelector(".org-replay-narrative-count");

    const exitBtn = this._root.querySelector("#replayExitBtn");
    const prevBtn = this._root.querySelector("#replayPrevBtn");
    const nextBtn = this._root.querySelector("#replayNextBtn");

    // Play / Pause toggle
    this._playBtn.addEventListener("click", () => {
      if (this._isPlaying) {
        this._onPause();
      } else {
        this._onPlay();
      }
    });

    // Exit replay
    exitBtn.addEventListener("click", () => this._onExit());

    // Skip back 5 min
    prevBtn.addEventListener("click", () => {
      const target = Math.max(this._startMs, this._currentMs - SKIP_MS);
      this._onSeek(target);
    });

    // Skip forward 5 min
    nextBtn.addEventListener("click", () => {
      const target = Math.min(this._endMs, this._currentMs + SKIP_MS);
      this._onSeek(target);
    });

    // Slider: oninput for live seeking, onchange for final
    this._slider.addEventListener("input", () => {
      this._isDragging = true;
      const time = this._sliderToTime(Number(this._slider.value));
      this._timeCurrent.textContent = formatTimeJST(time);
      this._onSeek(time);
    });
    this._slider.addEventListener("change", () => {
      this._isDragging = false;
      const time = this._sliderToTime(Number(this._slider.value));
      this._onSeek(time);
    });

    // Speed cycle
    this._speedBtn.addEventListener("click", () => {
      this._speedIndex = (this._speedIndex + 1) % SPEED_CYCLE.length;
      const speed = SPEED_CYCLE[this._speedIndex];
      this._speedBtn.textContent = `${speed}x`;
      this._speedBtn.setAttribute("aria-label", t("ws.replay.speed", { speed: `${speed}x` }));
      this._onSpeedChange(speed);
    });

    // Range change
    this._rangeSelect.addEventListener("change", () => {
      const hours = Number(this._rangeSelect.value);
      if (hours > 0) this._onRangeChange(hours);
    });

    // Insert after KPI bar
    const kpiBar = this._container?.querySelector("#orgKpiBar");
    if (kpiBar) {
      kpiBar.after(this._root);
    } else {
      this._container?.prepend(this._root);
    }

    this._root.style.display = "none";
    logger.debug("ReplayUI built");
  }

  _sliderToTime(val) {
    const range = this._endMs - this._startMs;
    if (range <= 0) return this._startMs;
    return this._startMs + (val / 1000) * range;
  }

  _timeToSlider(ms) {
    const range = this._endMs - this._startMs;
    if (range <= 0) return 0;
    return ((ms - this._startMs) / range) * 1000;
  }

  /**
   * Show the replay control bar.
   */
  show() {
    if (this._root) this._root.style.display = "";
  }

  /**
   * Hide the replay control bar.
   */
  hide() {
    if (this._root) this._root.style.display = "none";
  }

  /**
   * Update slider position and current time display.
   * @param {number} currentMs - Current playback time in milliseconds
   * @param {number} [progress] - Optional 0–1 progress (used if time range not yet set)
   */
  updateTime(currentMs, progress) {
    this._currentMs = currentMs;
    this._timeCurrent.textContent = formatTimeJST(currentMs);

    if (!this._isDragging) {
      const range = this._endMs - this._startMs;
      const val = range > 0 ? this._timeToSlider(currentMs) : (progress ?? 0) * 1000;
      this._slider.value = String(Math.round(Math.max(0, Math.min(1000, val))));
    }
  }

  /**
   * Set the time range for the slider and display.
   * @param {number} startMs - Range start (milliseconds)
   * @param {number} endMs - Range end (milliseconds)
   */
  updateTimeRange(startMs, endMs) {
    this._startMs = startMs;
    this._endMs = endMs;
    this._timeStart.textContent = formatTimeJST(startMs);
    this._timeEnd.textContent = formatTimeJST(endMs);
    this._slider.min = "0";
    this._slider.max = "1000";
    this._slider.step = "1";
    this.updateTime(this._currentMs);
  }

  /**
   * Update play/pause button state.
   * @param {boolean} isPlaying - Whether playback is active
   */
  setPlaying(isPlaying) {
    this._isPlaying = isPlaying;
    this._playBtn.textContent = isPlaying ? "⏸" : "▶";
    this._playBtn.title = isPlaying ? "Pause" : "Play";
    this._playBtn.setAttribute("aria-label", t(isPlaying ? "ws.replay.pause" : "ws.replay.play"));
  }

  /**
   * Update speed display.
   * @param {number} speed - Current speed multiplier (1, 5, 10, 50, 100, 200)
   */
  setSpeed(speed) {
    const idx = SPEED_CYCLE.indexOf(speed);
    this._speedIndex = idx >= 0 ? idx : 0;
    this._speedBtn.textContent = `${speed}x`;
    this._speedBtn.setAttribute("aria-label", t("ws.replay.speed", { speed: `${speed}x` }));
  }

  /**
   * Show or hide loading indicator.
   * @param {boolean} isLoading - Whether loading state is active
   */
  setLoading(isLoading) {
    this._isLoading = isLoading;
    this._root?.classList.toggle("org-replay-bar--loading", isLoading);
  }

  /**
   * Show replay error message.
   * @param {string} message - Error text
   */
  setError(message) {
    if (!this._errorEl) return;
    this._errorEl.textContent = message || "Replay failed";
    this._errorEl.hidden = false;
  }

  /**
   * Clear replay error message.
   */
  clearError() {
    if (!this._errorEl) return;
    this._errorEl.textContent = "";
    this._errorEl.hidden = true;
  }

  _setOptionalText(el, text) {
    if (!el) return;
    const value = String(text || "").trim();
    el.textContent = value;
    el.hidden = value.length === 0;
  }

  /**
   * Update the narrative band from ReplayEngine semantic state.
   * @param {object} state - Narrative state emitted by ReplayEngine
   */
  updateNarrative(state = {}) {
    const event = state.currentEvent || null;
    const total = Number(state.totalEvents || 0);
    const currentIndex = Number(state.currentIndex ?? -1);

    if (!event) {
      this._setOptionalText(this._narrativeKind, "Replay");
      this._setOptionalText(this._narrativeLabel, "No activity at this time");
      this._setOptionalText(this._narrativeSummary, "");
      this._setOptionalText(this._narrativeRoute, "");
      this._setOptionalText(this._narrativeGroup, "");
      this._setOptionalText(this._narrativeStatus, "");
      this._setOptionalText(this._narrativeCount, total > 0 ? `0 / ${total}` : "0 / 0");
      return;
    }

    const actor = String(event.actor || "").trim();
    const target = String(event.target || "").trim();
    const route = actor && target ? `${actor} -> ${target}` : actor;
    const groupId = state.activeGroupId || event.group_id || "";
    const groupType = state.activeGroupType || event.group_type || "";
    const group = groupId ? `${groupType || "group"} ${shortenText(groupId)}` : "";
    const suppressed = Number(state.suppressedCount || 0);
    const count = `${Math.max(0, currentIndex) + 1} / ${total || 1}${suppressed > 0 ? `, ${suppressed} suppressed` : ""}`;

    this._setOptionalText(this._narrativeKind, titleCase(event.kind || "Replay"));
    this._setOptionalText(this._narrativeLabel, event.label || "Activity");
    this._setOptionalText(this._narrativeSummary, event.summary || "");
    this._setOptionalText(this._narrativeRoute, route);
    this._setOptionalText(this._narrativeGroup, group);
    this._setOptionalText(this._narrativeStatus, event.status || "");
    this._setOptionalText(this._narrativeCount, count);
  }

  /**
   * Remove DOM elements and clean up.
   */
  dispose() {
    this.updateNarrative({});
    this._root?.remove();
    this._root = null;
    this._playBtn = null;
    this._slider = null;
    this._timeStart = null;
    this._timeCurrent = null;
    this._timeEnd = null;
    this._speedBtn = null;
    this._rangeSelect = null;
    this._errorEl = null;
    this._narrativeEl = null;
    this._narrativeKind = null;
    this._narrativeLabel = null;
    this._narrativeSummary = null;
    this._narrativeRoute = null;
    this._narrativeGroup = null;
    this._narrativeStatus = null;
    this._narrativeCount = null;
    this._container = null;
    logger.debug("ReplayUI disposed");
  }
}
