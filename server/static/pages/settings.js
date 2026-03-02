// ── Settings Page ────────────────────────────
import { t } from "/shared/i18n.js";
import { applyTheme, applyDisplayMode, getDisplayMode } from "../modules/app.js";

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
  `;

  if (window.lucide) lucide.createIcons();

  container.querySelectorAll(".settings-mode-card").forEach(btn => {
    btn.addEventListener("click", () => _onModeChange(btn.dataset.mode, container));
  });
  container.querySelectorAll(".settings-theme-card").forEach(btn => {
    btn.addEventListener("click", () => _onThemeChange(btn.dataset.theme, container));
  });
}

export function destroy() {}

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
    await fetch("/api/settings/display-mode", {
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
