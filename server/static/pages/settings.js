// ── Settings Page (tab host: general | activity | api | users) ─
import { api } from "../modules/api.js";
import { escapeHtml } from "../modules/state.js";
import { createPageTabs } from "../shared/page-tabs.js";
import { t } from "/shared/i18n.js";
import { applyTheme, applyDisplayMode, getDisplayMode, applyFontSize, getFontSize } from "../modules/app.js";

const _LS_ACTIVITY  = "aw-activity-level";
const _LS_SCHEDULE  = "aw-activity-schedule";
const _LS_ENTER_SEND = "aw-enter-to-send";

let _pageTabs = null;
let _activeTabModule = null;
let _activeTab = "general";

const _TABS = [
  { id: "general", labelKey: "settings.tab_general" },
  { id: "activity", labelKey: "settings.tab_activity" },
  { id: "api", labelKey: "settings.tab_api" },
  { id: "users", labelKey: "settings.tab_users" },
];

/**
 * Resolve settings tab from router subPath.
 * @param {string} [subPath]
 * @returns {"general"|"activity"|"api"|"users"}
 */
function resolveSettingsTab(subPath) {
  const head = String(subPath || "")
    .split("/")
    .filter(Boolean)[0] || "";
  return _TABS.some((tab) => tab.id === head) ? head : "general";
}

/**
 * @param {string} tabId
 * @returns {string}
 */
function buildSettingsTabHash(tabId) {
  return tabId === "general" || !_TABS.some((tab) => tab.id === tabId)
    ? "#/settings"
    : `#/settings/${tabId}`;
}

function _destroyActiveTab() {
  if (_activeTabModule && typeof _activeTabModule.destroy === "function") {
    try {
      _activeTabModule.destroy();
    } catch {
      /* ignore */
    }
  }
  _activeTabModule = null;
  if (_pageTabs) {
    try {
      _pageTabs.destroy();
    } catch {
      /* ignore */
    }
    _pageTabs = null;
  }
}

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

/**
 * @param {HTMLElement} container
 * @param {{ subPath?: string }} [opts]
 */
export async function render(container, { subPath } = {}) {
  _destroyActiveTab();
  _activeTab = resolveSettingsTab(subPath);

  container.innerHTML = `
    <div class="page-header">
      <h2>${t("settings.title")}</h2>
    </div>
    <div id="settingsPageTabs"></div>
    <div id="settingsTabContent">
      <div class="loading-placeholder">${t("common.loading")}</div>
    </div>
  `;

  const tabsHost = document.getElementById("settingsPageTabs");
  if (tabsHost) {
    _pageTabs = createPageTabs({
      tabs: _TABS.map((tab) => ({ id: tab.id, label: t(tab.labelKey) })),
      container: tabsHost,
      activeId: _activeTab,
      onChange: (id) => {
        if (id === _activeTab) return;
        window.location.hash = buildSettingsTabHash(id);
      },
    });
  }

  const content = document.getElementById("settingsTabContent");
  if (!content) return;

  if (_activeTab === "users") {
    try {
      const mod = await import("./users.js");
      _activeTabModule = mod;
      content.innerHTML = "";
      if (typeof mod.render === "function") {
        await mod.render(content);
      }
    } catch (err) {
      console.error("[Settings] Failed to load users tab:", err);
      content.innerHTML = `<div class="page-error">${t("router.page_load_failed")}</div>`;
    }
    return;
  }

  if (_activeTab === "activity") {
    _renderActivity(content);
    return;
  }

  if (_activeTab === "api") {
    _renderApiAuth(content);
    return;
  }

  _renderGeneral(content);
}

function _renderApiAuth(container) {
  container.innerHTML = `
    <section class="settings-section" id="settingsApiAuthSection">
      <h3 class="settings-section-title">${t("settings.api_auth.title")}</h3>
      <p class="settings-section-desc">${t("settings.api_auth.desc")}</p>

      <h4 class="settings-subsection-title">${t("settings.api_auth.api_keys")}</h4>
      <div class="settings-auth-block" id="settingsApiKeys">
        <div class="loading-placeholder">${t("common.loading")}</div>
      </div>

      <h4 class="settings-subsection-title">${t("settings.api_auth.anthropic_title")}</h4>
      <div class="settings-auth-block" id="anthropicAuthSettings">
        <div class="loading-placeholder">${t("common.loading")}</div>
      </div>

      <h4 class="settings-subsection-title">${t("settings.api_auth.openai_title")}</h4>
      <div class="settings-auth-block" id="openaiAuthSettings">
        <div class="loading-placeholder">${t("common.loading")}</div>
      </div>

      <h4 class="settings-subsection-title">${t("settings.api_auth.cli_title")}</h4>
      <div class="settings-auth-block" id="cliToolsAuth">
        <div class="loading-placeholder">${t("common.loading")}</div>
      </div>

      <h4 class="settings-subsection-title">${t("settings.api_auth.auth_settings")}</h4>
      <div class="settings-auth-block" id="authSettings">
        <div class="loading-placeholder">${t("common.loading")}</div>
      </div>
    </section>
  `;

  _loadApiKeys();
  _loadAnthropicAuthSettings();
  _loadOpenAIAuthSettings();
  _loadCliToolsAuth();
  _loadAuthSettings();
}

function _renderActivity(container) {
  container.innerHTML = `
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
                 class="activity-level-slider" id="activityLevelSlider"
                 aria-label="${escapeHtml(t("settings.activity_level.slider_label"))}" />
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
              <select id="nightStart" aria-label="${escapeHtml(t("settings.night_mode.start_label"))}">${_buildTimeOptions("22:00")}</select>
              <span class="night-mode-separator">～</span>
              <select id="nightEnd" aria-label="${escapeHtml(t("settings.night_mode.end_label"))}">${_buildTimeOptions("08:00")}</select>
            </div>
            <div class="night-mode-level-row">
              <span class="night-mode-level-label">${t("settings.night_mode.level_label")}</span>
              <input type="range" min="10" max="400" value="30" step="10"
                     class="activity-level-slider night-level-slider" id="nightLevelSlider"
                     aria-label="${escapeHtml(t("settings.night_mode.level_slider_label"))}" />
              <div class="activity-level-value night-level-value" id="nightLevelValue">30%</div>
            </div>
          </div>
        </div>
      </div>
    </section>
  `;

  _initActivityLevel(container);
}

function _renderGeneral(container) {
  const currentMode = getDisplayMode();
  const currentTheme = localStorage.getItem("aw-theme") || "default";

  container.innerHTML = `
    <section class="settings-section">
      <h3 class="settings-section-title">${t("settings.mode.title")}</h3>
      <p class="settings-section-desc">${t("settings.mode.desc")}</p>
      <div class="settings-mode-cards">
        <button class="settings-mode-card ${currentMode === "anime" ? "active" : ""}" data-mode="anime">
          <div class="settings-mode-icon">
            <span class="settings-mode-emoji" aria-hidden="true">&#x1F338;</span>
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
          <button class="settings-theme-card ${th.id === currentTheme ? "active" : ""}" data-theme="${th.id}" aria-label="${escapeHtml(t("settings.theme.apply", { name: th.label }))}">
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
               class="activity-level-slider" id="fontSizeSlider"
               aria-label="${escapeHtml(t("settings.font_size.slider_label"))}" />
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
      <label class="settings-checkbox-label">
        <input type="checkbox" id="chatPresenceToggle" ${_isPresenceOn() ? "checked" : ""} />
        <span>${t("chat.presence.setting_label")}</span>
      </label>
      <p class="settings-section-desc">${t("chat.presence.setting_hint")}</p>
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

  container.querySelector("#chatPresenceToggle")?.addEventListener("change", (e) => {
    try { localStorage.setItem("aw-chat-presence", e.target.checked ? "on" : "off"); } catch { /* */ }
    window.dispatchEvent(new CustomEvent("aw:presence-pref"));
  });

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

export function destroy() {
  _destroyActiveTab();
  _activeTab = "general";
}

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
    await api("/api/settings/display-mode", {
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
    const data = await api("/api/settings/activity-level");
    level = data.activity_level || 100;
    schedule = data.activity_schedule || [];
    fromApi = true;
    _cacheActivityState(level, schedule);
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
    await api("/api/settings/activity-level", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ activity_level: level }),
    });
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
    await api("/api/settings/activity-schedule", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ activity_schedule: schedule }),
    });
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
    await api("/api/settings/activity-schedule", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ activity_schedule: [] }),
    });
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

function _isPresenceOn() {
  try { return localStorage.getItem("aw-chat-presence") !== "off"; } catch { return true; }
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

// ── API Keys (from config) ─────────────────

async function _loadApiKeys() {
  const keysEl = document.getElementById("settingsApiKeys");
  if (!keysEl) return;

  try {
    const config = await api("/api/system/config");
    const rows = [];
    function flattenConfig(obj, prefix = "") {
      for (const [key, val] of Object.entries(obj)) {
        const fullKey = prefix ? `${prefix}.${key}` : key;
        if (val && typeof val === "object" && !Array.isArray(val)) {
          flattenConfig(val, fullKey);
        } else {
          let displayVal = String(val);
          if (/key|secret|token|password/i.test(key) && displayVal.length > 4) {
            displayVal = displayVal.slice(0, 4) + "****";
          }
          rows.push({ key: fullKey, val: displayVal });
        }
      }
    }
    flattenConfig(config);

    // Only credentials belong here: "max_tokens" or "token_budget_monthly"
    // are tuning knobs, not keys, and an unset optional key is not an error.
    const isCredential = (fullKey) => {
      const leaf = fullKey.split(".").pop() || "";
      // *_env / *_vault_key only name where a secret is looked up.
      if (/tokens|budget|^max_|_max_|_env$|vault_key$/i.test(leaf)) return false;
      return /(^|_)(api_key|key|secret|password|token)$/i.test(leaf);
    };
    const keyEntries = rows.filter(r => isCredential(r.key));
    if (keyEntries.length > 0) {
      keysEl.innerHTML = keyEntries.map(k => {
        const configured = k.val && k.val !== "null" && k.val !== "None" && k.val !== "";
        return `
          <div class="settings-auth-row">
            <span class="settings-auth-icon">${configured ? "\u2705" : "\u2796"}</span>
            <span class="settings-auth-label">${escapeHtml(k.key)}</span>
            <code class="settings-auth-value">${escapeHtml(k.val)}</code>
          </div>
        `;
      }).join("");
    } else {
      keysEl.innerHTML = `<div class="loading-placeholder">${t("settings.api_auth.api_keys_not_in_config")}</div>`;
    }
  } catch {
    keysEl.innerHTML = `<div class="loading-placeholder">${t("settings.api_auth.config_unavailable")}</div>`;
  }
}

// ── Auth Settings (mode summary) ───────────
// Password change / user management live in the Users tab (#/settings/users).

async function _loadAuthSettings() {
  const el = document.getElementById("authSettings");
  if (!el) return;

  try {
    const me = await api("/api/auth/me");

    const modeLabel = {
      local_trust: t("settings.api_auth.auth_local_trust"),
      password: t("settings.api_auth.auth_password"),
      multi_user: t("settings.api_auth.auth_multi_user"),
    };
    el.innerHTML = `
      <div style="margin-bottom: 0.75rem;">
        <strong>${t("settings.api_auth.auth_mode")}:</strong> <code>${escapeHtml(modeLabel[me.auth_mode] || me.auth_mode || t("common.unknown"))}</code>
      </div>
      <p style="margin:0; color:var(--text-secondary, #666); font-size:0.85rem;">
        <a href="#/settings/users">${t("settings.api_auth.manage_in_users")}</a>
      </p>
    `;
  } catch {
    el.innerHTML = `<div class="loading-placeholder">${t("settings.api_auth.auth_fetch_failed")}</div>`;
  }
}

// ── Anthropic Auth Settings ─────────────────

async function _loadAnthropicAuthSettings() {
  const el = document.getElementById("anthropicAuthSettings");
  if (!el) return;

  try {
    const state = await api("/api/settings/anthropic-auth");
    const modeLabel = state.auth_mode === "claude_code_login"
      ? t("settings.api_auth.anthropic_mode_subscription")
      : t("settings.api_auth.anthropic_mode_api");

    const runtimeBadges = [
      {
        label: t("settings.api_auth.anthropic_env_key"),
        ok: !!state.env_api_key_configured,
      },
      {
        label: t("settings.api_auth.anthropic_config_key"),
        ok: !!state.config_api_key_configured,
      },
      {
        label: t("settings.api_auth.anthropic_claude_code"),
        ok: !!state.claude_code_available,
      },
    ];

    el.innerHTML = `
      <div style="margin-bottom: 1rem;">
        <strong>${t("settings.api_auth.anthropic_current")}:</strong>
        <code>${escapeHtml(modeLabel)}</code>
      </div>
      <div style="margin-bottom: 1rem;">
        <strong>${t("settings.api_auth.anthropic_saved")}:</strong>
        <span>${state.config_present ? "\u2705" : "\u274C"}</span>
      </div>
      <div style="display:grid; gap:0.5rem; margin-bottom: 1.25rem;">
        ${runtimeBadges.map(item => `
          <div class="settings-auth-row">
            <span class="settings-auth-icon">${item.ok ? "\u2705" : "\u274C"}</span>
            <span>${escapeHtml(item.label)}</span>
          </div>
        `).join("")}
      </div>

      <form id="anthropicAuthForm" class="users-form settings-auth-form">
        <label style="display:flex; flex-direction:column; gap:0.35rem;">
          <span>${t("settings.api_auth.anthropic_mode_label")}</span>
          <select id="anthropicAuthMode">
            <option value="api_key"${state.auth_mode === "api_key" ? " selected" : ""}>${t("settings.api_auth.anthropic_mode_api")}</option>
            <option value="claude_code_login"${state.auth_mode === "claude_code_login" ? " selected" : ""}>${t("settings.api_auth.anthropic_mode_subscription")}</option>
          </select>
        </label>
        <label id="anthropicApiKeyWrap" style="display:${state.auth_mode === "api_key" ? "flex" : "none"}; flex-direction:column; gap:0.35rem;">
          <span>${t("settings.api_auth.anthropic_api_key")}</span>
          <input type="password" id="anthropicApiKeyInput" placeholder="sk-ant-...">
          <small style="color:var(--text-secondary, #666);">${t("settings.api_auth.anthropic_api_key_hint")}</small>
        </label>
        <div id="anthropicAuthResult" class="login-error hidden"></div>
        <button type="submit" class="btn-primary">${t("settings.api_auth.anthropic_save")}</button>
      </form>
    `;

    const modeEl = document.getElementById("anthropicAuthMode");
    const apiKeyWrap = document.getElementById("anthropicApiKeyWrap");
    modeEl?.addEventListener("change", () => {
      if (apiKeyWrap) {
        apiKeyWrap.style.display = modeEl.value === "api_key" ? "flex" : "none";
      }
    });

    const form = document.getElementById("anthropicAuthForm");
    form?.addEventListener("submit", async (e) => {
      e.preventDefault();
      const result = document.getElementById("anthropicAuthResult");
      const authMode = document.getElementById("anthropicAuthMode").value;
      const apiKey = document.getElementById("anthropicApiKeyInput")?.value || "";

      if (authMode === "api_key" && !apiKey.trim()) {
        result.style.color = "#ef4444";
        result.textContent = t("settings.api_auth.anthropic_api_key_required");
        result.classList.remove("hidden");
        return;
      }

      try {
        await api("/api/settings/anthropic-auth", {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            auth_mode: authMode,
            api_key: apiKey,
          }),
        });

        result.style.color = "#22c55e";
        result.textContent = t("settings.api_auth.anthropic_saved_success");
        result.classList.remove("hidden");
        await _loadApiKeys();
        await _loadAnthropicAuthSettings();
      } catch (err) {
        result.style.color = "#ef4444";
        result.textContent = err.status
          ? err.message || t("settings.api_auth.anthropic_save_failed")
          : t("settings.api_auth.network_error");
        result.classList.remove("hidden");
      }
    });
  } catch {
    el.innerHTML = `<div class="loading-placeholder">${t("settings.api_auth.anthropic_fetch_failed")}</div>`;
  }
}

// ── CLI tools (read-only detection) ─────────

async function _loadCliToolsAuth() {
  const el = document.getElementById("cliToolsAuth");
  if (!el) return;

  try {
    const env = await api("/api/setup/environment");
    const rows = [
      { label: t("settings.api_auth.cli_claude_code"), ok: !!env.claude_code_available },
      { label: t("settings.api_auth.cli_codex_cli"), ok: !!env.codex_cli_available },
      { label: t("settings.api_auth.cli_codex_login"), ok: !!env.codex_login_available },
      { label: t("settings.api_auth.cli_cursor_agent"), ok: !!env.cursor_agent_available },
      { label: t("settings.api_auth.cli_cursor_auth"), ok: !!env.cursor_agent_authenticated },
      { label: t("settings.api_auth.cli_gemini_cli"), ok: !!env.gemini_cli_available },
      { label: t("settings.api_auth.cli_gemini_auth"), ok: !!env.gemini_authenticated },
    ];
    el.innerHTML = `
      <div style="display:grid; gap:0.5rem;">
        ${rows.map(r => `
          <div class="settings-auth-row">
            <span class="settings-auth-icon">${r.ok ? "\u2705" : "\u274C"}</span>
            <span>${escapeHtml(r.label)}</span>
          </div>
        `).join("")}
      </div>
    `;
  } catch {
    el.innerHTML = `<div class="loading-placeholder">${t("settings.api_auth.network_error")}</div>`;
  }
}

// ── OpenAI / Codex Auth ───────────────────

async function _loadOpenAIAuthSettings() {
  const el = document.getElementById("openaiAuthSettings");
  if (!el) return;

  try {
    const state = await api("/api/settings/openai-auth");
    const modeLabel = state.auth_mode === "codex_login"
      ? t("settings.api_auth.openai_mode_codex")
      : t("settings.api_auth.openai_mode_api");

    const runtimeBadges = [
      {
        label: t("settings.api_auth.openai_env_key"),
        ok: !!state.env_api_key_configured,
      },
      {
        label: t("settings.api_auth.openai_codex_cli"),
        ok: !!state.codex_cli_available,
      },
      {
        label: t("settings.api_auth.openai_codex_login"),
        ok: !!state.codex_login_available,
      },
    ];

    el.innerHTML = `
      <div style="margin-bottom: 1rem;">
        <strong>${t("settings.api_auth.openai_current")}:</strong>
        <code>${escapeHtml(modeLabel)}</code>
      </div>
      <div style="margin-bottom: 1rem;">
        <strong>${t("settings.api_auth.openai_saved")}:</strong>
        <span>${state.config_present ? t("settings.api_auth.openai_saved_yes") : t("settings.api_auth.openai_saved_no")}</span>
      </div>
      <div style="display:grid; gap:0.5rem; margin-bottom: 1.25rem;">
        ${runtimeBadges.map(item => `
          <div class="settings-auth-row">
            <span class="settings-auth-icon">${item.ok ? "\u2705" : "\u274C"}</span>
            <span>${escapeHtml(item.label)}</span>
          </div>
        `).join("")}
      </div>

      <form id="openaiAuthForm" class="users-form settings-auth-form">
        <label style="display:flex; flex-direction:column; gap:0.35rem;">
          <span>${t("settings.api_auth.openai_mode_label")}</span>
          <select id="openaiAuthMode">
            <option value="api_key"${state.auth_mode === "api_key" ? " selected" : ""}>${t("settings.api_auth.openai_mode_api")}</option>
            <option value="codex_login"${state.auth_mode === "codex_login" ? " selected" : ""}>${t("settings.api_auth.openai_mode_codex")}</option>
          </select>
        </label>
        <label id="openaiApiKeyWrap" style="display:${state.auth_mode === "api_key" ? "flex" : "none"}; flex-direction:column; gap:0.35rem;">
          <span>${t("settings.api_auth.openai_api_key")}</span>
          <input type="password" id="openaiApiKeyInput" placeholder="${t("settings.api_auth.openai_api_key_placeholder")}">
          <small style="color:var(--text-secondary, #666);">${t("settings.api_auth.openai_api_key_hint")}</small>
        </label>
        <div id="openaiAuthResult" class="login-error hidden"></div>
        <button type="submit" class="btn-primary">${t("settings.api_auth.openai_save")}</button>
      </form>
    `;

    const modeEl = document.getElementById("openaiAuthMode");
    const apiKeyWrap = document.getElementById("openaiApiKeyWrap");
    modeEl?.addEventListener("change", () => {
      if (apiKeyWrap) {
        apiKeyWrap.style.display = modeEl.value === "api_key" ? "flex" : "none";
      }
    });

    const form = document.getElementById("openaiAuthForm");
    form?.addEventListener("submit", async (e) => {
      e.preventDefault();
      const result = document.getElementById("openaiAuthResult");
      const authMode = document.getElementById("openaiAuthMode").value;
      const apiKey = document.getElementById("openaiApiKeyInput")?.value || "";

      if (authMode === "api_key" && !apiKey.trim()) {
        result.style.color = "#ef4444";
        result.textContent = t("settings.api_auth.openai_api_key_required");
        result.classList.remove("hidden");
        return;
      }

      try {
        await api("/api/settings/openai-auth", {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            auth_mode: authMode,
            api_key: apiKey,
          }),
        });

        result.style.color = "#22c55e";
        result.textContent = t("settings.api_auth.openai_saved_success");
        result.classList.remove("hidden");
        await _loadApiKeys();
        await _loadOpenAIAuthSettings();
      } catch (err) {
        result.style.color = "#ef4444";
        result.textContent = err.status
          ? err.message || t("settings.api_auth.openai_save_failed")
          : t("settings.api_auth.network_error");
        result.classList.remove("hidden");
      }
    });
  } catch {
    el.innerHTML = `<div class="loading-placeholder">${t("settings.api_auth.openai_fetch_failed")}</div>`;
  }
}
