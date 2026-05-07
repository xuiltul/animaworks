// ── Settings Page ────────────────────────────
import { t } from "/shared/i18n.js";
import { basePath } from "/shared/base-path.js";
import { applyTheme, applyDisplayMode, getDisplayMode, applyFontSize, getFontSize } from "../modules/app.js";

const _LS_ACTIVITY  = "aw-activity-level";
const _LS_SCHEDULE  = "aw-activity-schedule";
const _LS_ENTER_SEND = "aw-enter-to-send";

const THEMES = [
  { id: "default",   label: "Default",   colors: ["#374151", "#f5f5f5", "#fff"] },
  { id: "graphite",  label: "Graphite",  colors: ["#1f2937", "#f9fafb", "#fff"] },
  { id: "ocean",     label: "Ocean",     colors: ["#1e40af", "#f0f5ff", "#fff"] },
  { id: "forest",    label: "Forest",    colors: ["#166534", "#f0fdf4", "#fff"] },
  { id: "sunset",    label: "Sunset",    colors: ["#b45309", "#fffbeb", "#fff"] },
  { id: "rose",      label: "Rose",      colors: ["#be185d", "#fdf2f8", "#fff"] },
  { id: "lavender",  label: "Lavender",  colors: ["#7c3aed", "#f5f3ff", "#fff"] },
  { id: "nord",      label: "Nord",      colors: ["#5e81ac", "#eceff4", "#fff"] },
  { id: "monokai",   label: "Monokai",   colors: ["#a6e22e", "#1e1f1a", "#272822"] },
  { id: "midnight",  label: "Midnight",  colors: ["#60a5fa", "#0a1120", "#0f172a"] },
  { id: "solarized", label: "Solarized", colors: ["#268bd2", "#eee8d5", "#fdf6e3"] },
  { id: "business",  label: "Business",  colors: ["#1e40af", "#f8fafc", "#fff"] },
];

export function render(container) {
  const currentMode = getDisplayMode();
  const currentTheme = localStorage.getItem("aw-theme") || "default";

  container.innerHTML = `
    <div class="page-header">
      <h2>${t("settings.title")}</h2>
    </div>

    <section class="settings-section">
      <h3 class="settings-section-title">${t("settings.activity_level.title")}</h3>
      <p class="settings-section-desc">${t("settings.activity_level.desc")}</p>
      <div class="activity-level-control">
        <div class="activity-level-presets">
          <button class="preset-btn" data-level="30">\u{1F40C} ${t("settings.activity_level.eco")}</button>
          <button class="preset-btn" data-level="100">\u{1F4CA} ${t("settings.activity_level.normal")}</button>
          <button class="preset-btn" data-level="200">\u{1F680} ${t("settings.activity_level.fast")}</button>
          <button class="preset-btn" data-level="400">\u{26A1} ${t("settings.activity_level.boost")}</button>
        </div>
        <div class="activity-level-slider-row">
          <input type="range" min="10" max="400" value="100" step="10"
                 class="activity-level-slider" id="activityLevelSlider" />
          <div class="activity-level-value" id="activityLevelValue">100%</div>
        </div>
        <div class="activity-level-effect" id="activityLevelEffect"></div>

        <div class="night-mode-section">
          <label class="night-mode-toggle">
            <input type="checkbox" id="nightModeToggle" />
            \u{1F319} ${t("settings.night_mode.label")}
          </label>
          <div class="night-mode-settings" id="nightModeSettings" style="display:none">
            <div class="night-mode-time">
              <select id="nightStart">${_buildTimeOptions("22:00")}</select>
              <span class="night-mode-separator">～</span>
              <select id="nightEnd">${_buildTimeOptions("08:00")}</select>
            </div>
            <div class="night-mode-level-row">
              <span class="night-mode-level-label">${t("settings.night_mode.level_label")}</span>
              <input type="range" min="10" max="400" value="30" step="10"
                     class="activity-level-slider night-level-slider" id="nightLevelSlider" />
              <div class="activity-level-value night-level-value" id="nightLevelValue">30%</div>
            </div>
          </div>
        </div>
      </div>
    </section>

    <section class="settings-section">
      <h3 class="settings-section-title">${t("settings.mode.title")}</h3>
      <p class="settings-section-desc">${t("settings.mode.desc")}</p>
      <div class="settings-mode-cards">
        <button class="settings-mode-card ${currentMode === "anime" ? "active" : ""}" data-mode="anime">
          <div class="settings-mode-icon">
            <span class="nav-emoji">&#x1F338;</span>
          </div>
          <div class="settings-mode-info">
            <strong>${t("settings.mode.anime")}</strong>
            <span>${t("settings.mode.anime.desc")}</span>
          </div>
        </button>
        <button class="settings-mode-card ${currentMode === "realistic" ? "active" : ""}" data-mode="realistic">
          <div class="settings-mode-icon">
            <i data-lucide="building-2" style="width:24px;height:24px;"></i>
          </div>
          <div class="settings-mode-info">
            <strong>${t("settings.mode.realistic")}</strong>
            <span>${t("settings.mode.realistic.desc")}</span>
          </div>
        </button>
      </div>
    </section>

    <section class="settings-section">
      <h3 class="settings-section-title">${t("settings.theme.title")}</h3>
      <p class="settings-section-desc">${t("settings.theme.desc")}</p>
      <div class="settings-theme-grid" id="settingsThemeGrid">
        ${THEMES.map(th => `
          <button class="settings-theme-card ${th.id === currentTheme ? "active" : ""}" data-theme="${th.id}">
            <div class="settings-theme-preview">
              ${th.colors.map(c => `<span style="background:${c}"></span>`).join("")}
            </div>
            <span class="settings-theme-label">${th.label}</span>
          </button>
        `).join("")}
      </div>
    </section>

    <section class="settings-section">
      <h3 class="settings-section-title">${t("settings.font_size.title")}</h3>
      <p class="settings-section-desc">${t("settings.font_size.desc")}</p>
      <div class="activity-level-presets">
        <button class="preset-btn" data-font="12">${t("settings.font_size.small")}</button>
        <button class="preset-btn" data-font="14">${t("settings.font_size.medium")}</button>
        <button class="preset-btn" data-font="16">${t("settings.font_size.large")}</button>
        <button class="preset-btn" data-font="18">${t("settings.font_size.xl")}</button>
      </div>
      <div class="activity-level-slider-row">
        <input type="range" min="10" max="22" value="${getFontSize()}" step="1"
               class="activity-level-slider" id="fontSizeSlider" />
        <div class="activity-level-value" id="fontSizeValue">${getFontSize()}px</div>
      </div>
    </section>

    <section class="settings-section">
      <h3 class="settings-section-title">${t("settings.input.title")}</h3>
      <p class="settings-section-desc">${t("settings.input.desc")}</p>
      <label class="settings-checkbox-label">
        <input type="checkbox" id="enterToSendToggle" ${_isEnterToSend() ? "checked" : ""} />
        <span>${t("settings.input.enter_to_send")}</span>
      </label>
    </section>
  `;

  if (window.lucide) lucide.createIcons();

  container.querySelectorAll(".settings-mode-card").forEach(btn => {
    btn.addEventListener("click", () => _onModeChange(btn.dataset.mode, container));
  });
  container.querySelectorAll(".settings-theme-card").forEach(btn => {
    btn.addEventListener("click", () => _onThemeChange(btn.dataset.theme, container));
  });

  container.querySelector("#enterToSendToggle")?.addEventListener("change", (e) => {
    try { localStorage.setItem(_LS_ENTER_SEND, e.target.checked ? "true" : "false"); } catch { /* */ }
  });

  // Activity Level
  _initActivityLevel(container);

  // Font Size
  const fontSlider = container.querySelector("#fontSizeSlider");
  const fontValue  = container.querySelector("#fontSizeValue");
  function _applyFont(px) {
    applyFontSize(px);
    if (fontSlider) fontSlider.value = px;
    if (fontValue)  fontValue.textContent = `${px}px`;
  }
  fontSlider?.addEventListener("input", () => _applyFont(Number(fontSlider.value)));
  container.querySelectorAll("[data-font]").forEach(btn => {
    btn.addEventListener("click", () => _applyFont(Number(btn.dataset.font)));
  });
}

export function destroy() {}

// ── Time Options Builder ────────────────────

function _buildTimeOptions(selected) {
  let html = "";
  for (let h = 0; h < 24; h++) {
    for (let m = 0; m < 60; m += 30) {
      const val = `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}`;
      const sel = val === selected ? " selected" : "";
      html += `<option value="${val}"${sel}>${val}</option>`;
    }
  }
  return html;
}

// ── Internal ────────────────────────────────

async function _onModeChange(mode, container) {
  applyDisplayMode(mode);

  container.querySelectorAll(".settings-mode-card").forEach(c => {
    c.classList.toggle("active", c.dataset.mode === mode);
  });

  const autoTheme = mode === "realistic" ? "business" : "default";
  applyTheme(autoTheme);
  _updateThemeGrid(container, autoTheme);

  localStorage.removeItem("aw-workspace-view");

  try {
    await fetch(`${basePath}/api/settings/display-mode`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode }),
    });
  } catch { /* best-effort */ }
}

function _onThemeChange(theme, container) {
  applyTheme(theme);
  _updateThemeGrid(container, theme);
}

function _updateThemeGrid(container, theme) {
  container.querySelectorAll(".settings-theme-card").forEach(c => {
    c.classList.toggle("active", c.dataset.theme === theme);
  });
}

async function _initActivityLevel(container) {
  let schedule = [];
  let level = 100;
  let fromApi = false;
  try {
    const res = await fetch(`${basePath}/api/settings/activity-level`);
    if (res.ok) {
      const data = await res.json();
      level = data.activity_level || 100;
      schedule = data.activity_schedule || [];
      fromApi = true;
      _cacheActivityState(level, schedule);
    } else {
      console.warn("[Settings] GET activity-level failed:", res.status);
      ({ level, schedule } = _loadCachedActivityState());
      if (!schedule.length) _showSettingsStatus(container, t("settings.load_error"), true);
    }
  } catch (err) {
    console.warn("[Settings] GET activity-level error:", err);
    ({ level, schedule } = _loadCachedActivityState());
    if (!schedule.length) _showSettingsStatus(container, t("settings.load_error"), true);
  }

  const slider = container.querySelector("#activityLevelSlider");
  const display = container.querySelector("#activityLevelValue");
  if (slider) {
    slider.value = level;
    _updateActivityDisplay(display, level);
    _updateActivityEffect(container, level);
    _updatePresetButtons(container, level);
  }

  if (slider) {
    slider.addEventListener("input", (e) => {
      const val = parseInt(e.target.value, 10);
      const display = container.querySelector("#activityLevelValue");
      _updateActivityDisplay(display, val);
      _updateActivityEffect(container, val);
      _updatePresetButtons(container, val);
    });
    slider.addEventListener("change", (e) => {
      const val = parseInt(e.target.value, 10);
      _setActivityLevel(val, container);
    });
  }

  container.querySelectorAll(".preset-btn[data-level]").forEach(btn => {
    btn.addEventListener("click", () => {
      const val = parseInt(btn.dataset.level, 10);
      const slider = container.querySelector("#activityLevelSlider");
      const display = container.querySelector("#activityLevelValue");
      if (slider) slider.value = val;
      _updateActivityDisplay(display, val);
      _updateActivityEffect(container, val);
      _updatePresetButtons(container, val);
      _setActivityLevel(val, container);
    });
  });

  _initNightMode(container, schedule);
}

function _updateActivityDisplay(el, value) {
  if (!el) return;
  el.textContent = `${value}%`;
  if (value < 50) el.className = "activity-level-value level-low";
  else if (value <= 150) el.className = "activity-level-value level-normal";
  else if (value <= 300) el.className = "activity-level-value level-high";
  else el.className = "activity-level-value level-boost";
}

function _updateActivityEffect(container, level) {
  const el = container.querySelector("#activityLevelEffect");
  if (!el) return;
  const baseInterval = 30; // default
  const effInterval = Math.max(5, Math.round(baseInterval / (level / 100)));
  const baseTurns = 20;
  const effTurns = level >= 100 ? baseTurns : Math.max(3, Math.ceil(baseTurns * level / 100));
  el.textContent = t("settings.activity_level.effect", {
    interval: effInterval,
    turns: effTurns,
  });
}

function _updatePresetButtons(container, level) {
  container.querySelectorAll(".preset-btn[data-level]").forEach(btn => {
    btn.classList.toggle("active", parseInt(btn.dataset.level, 10) === level);
  });
}

async function _setActivityLevel(level, container) {
  _cacheActivityState(level, null);
  try {
    const res = await fetch(`${basePath}/api/settings/activity-level`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ activity_level: level }),
    });
    if (!res.ok) {
      console.warn("[Settings] PUT activity-level failed:", res.status);
      _showSettingsStatus(container, t("settings.save_error"), true);
    }
  } catch (err) {
    console.warn("[Settings] PUT activity-level error:", err);
    _showSettingsStatus(container, t("settings.save_error"), true);
  }

  // When night mode is ON, also update the schedule with the new day-level
  if (container) {
    const toggle = container.querySelector("#nightModeToggle");
    if (toggle && toggle.checked) {
      _saveNightMode(container, false);
    }
  }
}

// ── Night Mode ──────────────────────────────

function _initNightMode(container, schedule) {
  const toggle = container.querySelector("#nightModeToggle");
  const settings = container.querySelector("#nightModeSettings");
  const nightStart = container.querySelector("#nightStart");
  const nightEnd = container.querySelector("#nightEnd");
  const nightSlider = container.querySelector("#nightLevelSlider");
  const nightDisplay = container.querySelector("#nightLevelValue");
  if (!toggle || !settings) return;

  // Restore from schedule: 2 entries = night mode ON
  if (schedule.length >= 2) {
    toggle.checked = true;
    settings.style.display = "";
    // Entry with lower level is the night entry
    const sorted = [...schedule].sort((a, b) => a.level - b.level);
    const nightEntry = sorted[0];
    const dayEntry = sorted[1];
    if (nightStart) nightStart.value = nightEntry.start;
    if (nightEnd) nightEnd.value = nightEntry.end;
    if (nightSlider) nightSlider.value = nightEntry.level;
    if (nightDisplay) _updateActivityDisplay(nightDisplay, nightEntry.level);

    // Sync main slider to day entry level
    const mainSlider = container.querySelector("#activityLevelSlider");
    const mainDisplay = container.querySelector("#activityLevelValue");
    if (mainSlider && dayEntry) {
      mainSlider.value = dayEntry.level;
      _updateActivityDisplay(mainDisplay, dayEntry.level);
      _updateActivityEffect(container, dayEntry.level);
      _updatePresetButtons(container, dayEntry.level);
    }
  }

  toggle.addEventListener("change", () => {
    if (toggle.checked) {
      settings.style.display = "";
      _saveNightMode(container, true);
    } else {
      settings.style.display = "none";
      _clearNightMode(container);
    }
  });

  if (nightSlider) {
    nightSlider.addEventListener("input", () => {
      const val = parseInt(nightSlider.value, 10);
      _updateActivityDisplay(nightDisplay, val);
    });
    nightSlider.addEventListener("change", () => _saveNightMode(container, false));
  }

  if (nightStart) nightStart.addEventListener("change", () => _saveNightMode(container, false));
  if (nightEnd) nightEnd.addEventListener("change", () => _saveNightMode(container, false));
}

async function _saveNightMode(container, revertOnFail) {
  const nightStart = container.querySelector("#nightStart");
  const nightEnd = container.querySelector("#nightEnd");
  const nightSlider = container.querySelector("#nightLevelSlider");
  const mainSlider = container.querySelector("#activityLevelSlider");
  if (!nightStart || !nightEnd || !nightSlider || !mainSlider) return;

  const ns = nightStart.value;
  const ne = nightEnd.value;
  if (ns === ne) return;

  const nightLevel = parseInt(nightSlider.value, 10);
  const dayLevel = parseInt(mainSlider.value, 10);

  const schedule = [
    { start: ne, end: ns, level: dayLevel },
    { start: ns, end: ne, level: nightLevel },
  ];

  _cacheActivityState(dayLevel, schedule);

  try {
    const res = await fetch(`${basePath}/api/settings/activity-schedule`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ activity_schedule: schedule }),
    });
    if (!res.ok) {
      console.warn("[Settings] PUT activity-schedule failed:", res.status);
      _showSettingsStatus(container, t("settings.save_error"), true);
      if (revertOnFail) {
        _revertNightModeToggle(container, false);
        _cacheActivityState(dayLevel, []);
      }
    }
  } catch (err) {
    console.warn("[Settings] PUT activity-schedule error:", err);
    _showSettingsStatus(container, t("settings.save_error"), true);
    if (revertOnFail) {
      _revertNightModeToggle(container, false);
      _cacheActivityState(dayLevel, []);
    }
  }
}

async function _clearNightMode(container) {
  const mainSlider = container && container.querySelector("#activityLevelSlider");
  const curLevel = mainSlider ? parseInt(mainSlider.value, 10) : 100;
  const prevSchedule = _loadCachedActivityState().schedule;
  _cacheActivityState(curLevel, []);

  try {
    const res = await fetch(`${basePath}/api/settings/activity-schedule`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ activity_schedule: [] }),
    });
    if (!res.ok) {
      console.warn("[Settings] PUT clear schedule failed:", res.status);
      if (container) {
        _showSettingsStatus(container, t("settings.save_error"), true);
        _revertNightModeToggle(container, true);
        _cacheActivityState(curLevel, prevSchedule);
      }
    }
  } catch (err) {
    console.warn("[Settings] PUT clear schedule error:", err);
    if (container) {
      _showSettingsStatus(container, t("settings.save_error"), true);
      _revertNightModeToggle(container, true);
      _cacheActivityState(curLevel, prevSchedule);
    }
  }
}

// ── Enter-to-Send ───────────────────────────

function _isEnterToSend() {
  try { return localStorage.getItem(_LS_ENTER_SEND) === "true"; } catch { return false; }
}

// ── localStorage Cache ─────────────────────

function _cacheActivityState(level, schedule) {
  try {
    if (level != null) localStorage.setItem(_LS_ACTIVITY, String(level));
    if (schedule != null) localStorage.setItem(_LS_SCHEDULE, JSON.stringify(schedule));
  } catch { /* quota / private mode */ }
}

function _loadCachedActivityState() {
  try {
    const level = parseInt(localStorage.getItem(_LS_ACTIVITY), 10) || 100;
    const raw = localStorage.getItem(_LS_SCHEDULE);
    const schedule = raw ? JSON.parse(raw) : [];
    return { level, schedule: Array.isArray(schedule) ? schedule : [] };
  } catch {
    return { level: 100, schedule: [] };
  }
}

// ── UI Helpers ──────────────────────────────

function _revertNightModeToggle(container, checked) {
  const toggle = container.querySelector("#nightModeToggle");
  const settings = container.querySelector("#nightModeSettings");
  if (toggle) toggle.checked = checked;
  if (settings) settings.style.display = checked ? "" : "none";
}

function _showSettingsStatus(container, msg, isError) {
  if (!container) return;
  let el = container.querySelector("#settingsStatus");
  if (!el) {
    el = document.createElement("div");
    el.id = "settingsStatus";
    const header = container.querySelector(".page-header");
    if (header) header.after(el);
    else container.prepend(el);
  }
  el.textContent = msg;
  el.className = isError ? "settings-status settings-status-error" : "settings-status";
  clearTimeout(el._timer);
  el._timer = setTimeout(() => el.remove(), 5000);
}
