/* ── Step 2: Environment + API Keys ───────── */

import { t } from "../setup.js";

let container = null;
let envData = {
  claude_code_available: false,
  python_version: "",
  os_info: "",
};
let selectedProvider = "";
let apiKey = "";
let apiKeyValid = null; // null = unchecked, true/false
let ollamaUrl = "http://localhost:11434";
let imageKeys = {
  novelai: "",
  fal: "",
  meshy: "",
};
let imageKeyStatus = {
  novelai: null,
  fal: null,
  meshy: null,
};

const PROVIDERS = [
  { id: "claude_code", keyRequired: false },
  { id: "anthropic", keyRequired: true, keyEnv: "ANTHROPIC_API_KEY" },
  { id: "openai", keyRequired: true, keyEnv: "OPENAI_API_KEY" },
  { id: "google", keyRequired: true, keyEnv: "GOOGLE_API_KEY" },
  { id: "ollama", keyRequired: false },
];

export function initEnvironmentStep(el) {
  container = el;
  render();
  fetchEnvironment();
}

async function fetchEnvironment() {
  try {
    const res = await fetch("/api/setup/environment");
    if (res.ok) {
      const data = await res.json();
      envData = { ...envData, ...data };

      // Auto-select provider based on detection
      if (envData.claude_code_available && !selectedProvider) {
        selectedProvider = "claude_code";
      }
      render();
    }
  } catch {
    // Use defaults
  }
}

function render() {
  const provider = PROVIDERS.find((p) => p.id === selectedProvider);
  const needsKey = provider?.keyRequired;
  const isOllama = selectedProvider === "ollama";

  container.innerHTML = `
    <h2 data-i18n="env.title">${t("env.title")}</h2>
    <p style="color: #8888aa; font-size: 0.85rem; margin-top: 4px;" data-i18n="env.desc">${t("env.desc")}</p>

    <div class="env-section" style="margin-top: 20px;">
      <div class="env-section-title" data-i18n="env.detection">${t("env.detection")}</div>
      <div class="env-detection">
        <div class="env-detection-icon">${envData.claude_code_available ? "\u2705" : "\u2b1c"}</div>
        <div class="env-detection-text">
          <div class="env-detection-name" data-i18n="env.claude_code">${t("env.claude_code")}</div>
          <div class="env-detection-status ${envData.claude_code_available ? "found" : "not-found"}"
               data-i18n="env.claude_code.${envData.claude_code_available ? "found" : "notfound"}">
            ${envData.claude_code_available ? t("env.claude_code.found") : t("env.claude_code.notfound")}
          </div>
        </div>
      </div>
    </div>

    <div class="env-section">
      <div class="env-section-title" data-i18n="env.provider.title">${t("env.provider.title")}</div>
      <div class="provider-cards">
        ${renderProviders()}
      </div>
      ${needsKey ? renderApiKeyInput() : ""}
      ${isOllama ? renderOllamaInput() : ""}
    </div>

    <div class="env-section">
      <div class="env-section-title" data-i18n="env.imagegen.title">${t("env.imagegen.title")}</div>
      <div class="image-key-section">
        ${renderImageKey("novelai", t("env.novelai"), t("env.novelai.desc"), t("env.recommended"))}
        ${renderImageKey("fal", t("env.fal"), t("env.fal.desc"), t("env.optional"))}
        ${renderImageKey("meshy", t("env.meshy"), t("env.meshy.desc"), t("env.optional"))}
      </div>
    </div>

    <div id="envError"></div>
  `;

  bindEvents();
}

function renderProviders() {
  return PROVIDERS.map((p) => {
    // Hide claude_code option if not detected
    if (p.id === "claude_code" && !envData.claude_code_available) return "";

    const selected = p.id === selectedProvider ? " selected" : "";
    const badge = p.id === "claude_code" && envData.claude_code_available
      ? `<span class="provider-badge">${t("env.recommended")}</span>`
      : "";

    return `
      <div class="provider-card${selected}" data-provider="${p.id}">
        <div class="provider-radio"></div>
        <div>
          <div class="provider-name">${t(`env.provider.${p.id}`)}</div>
          <div class="provider-desc">${t(`env.provider.${p.id}.desc`)}</div>
        </div>
        ${badge}
      </div>
    `;
  }).join("");
}

function renderApiKeyInput() {
  let statusHtml = "";
  if (apiKeyValid === true) {
    statusHtml = `<div class="validation-status valid">\u2713 ${t("env.apikey.valid")}</div>`;
  } else if (apiKeyValid === false) {
    statusHtml = `<div class="validation-status invalid">\u2717 ${t("env.apikey.invalid")}</div>`;
  }

  return `
    <div class="api-key-section">
      <label class="form-label" data-i18n="env.apikey">${t("env.apikey")}</label>
      <div class="api-key-row">
        <input type="password" class="api-key-input" id="apiKeyInput"
               data-i18n-placeholder="env.apikey.placeholder"
               placeholder="${t("env.apikey.placeholder")}" value="${escapeAttr(apiKey)}">
        <button class="btn-validate" id="btnValidateKey" data-i18n="btn.validate">${t("btn.validate")}</button>
      </div>
      <div id="apiKeyStatus">${statusHtml}</div>
    </div>
  `;
}

function renderOllamaInput() {
  return `
    <div class="api-key-section">
      <label class="form-label" data-i18n="env.ollama.url">${t("env.ollama.url")}</label>
      <div class="api-key-row">
        <input type="text" class="api-key-input" id="ollamaUrlInput"
               data-i18n-placeholder="env.ollama.url.placeholder"
               placeholder="${t("env.ollama.url.placeholder")}" value="${escapeAttr(ollamaUrl)}">
        <button class="btn-validate" id="btnValidateOllama" data-i18n="btn.validate">${t("btn.validate")}</button>
      </div>
      <div id="ollamaStatus"></div>
    </div>
  `;
}

function renderImageKey(id, label, hint, badge) {
  const val = imageKeys[id] || "";
  const status = imageKeyStatus[id];
  let statusHtml = "";
  if (status === true) {
    statusHtml = `<div class="validation-status valid">\u2713 ${t("env.apikey.valid")}</div>`;
  } else if (status === false) {
    statusHtml = `<div class="validation-status invalid">\u2717 ${t("env.apikey.invalid")}</div>`;
  }

  const badgeClass = badge === t("env.recommended") ? "provider-badge" : "image-key-optional";
  return `
    <div class="image-key-item">
      <div class="image-key-label">
        ${label}
        <span class="${badgeClass}">${badge}</span>
      </div>
      <div class="form-hint">${hint}</div>
      <div class="api-key-row" style="margin-top: 4px;">
        <input type="password" class="api-key-input" data-image-key="${id}"
               data-i18n-placeholder="env.apikey.placeholder"
               placeholder="${t("env.apikey.placeholder")}" value="${escapeAttr(val)}">
      </div>
      <div class="image-key-status" data-image-status="${id}">${statusHtml}</div>
    </div>
  `;
}

function bindEvents() {
  // Provider selection
  container.querySelectorAll(".provider-card").forEach((card) => {
    card.addEventListener("click", () => {
      selectedProvider = card.dataset.provider;
      apiKeyValid = null;
      render();
    });
  });

  // API key validation
  const validateBtn = container.querySelector("#btnValidateKey");
  if (validateBtn) {
    validateBtn.addEventListener("click", () => validateApiKey());
  }

  const apiInput = container.querySelector("#apiKeyInput");
  if (apiInput) {
    apiInput.addEventListener("input", (e) => {
      apiKey = e.target.value;
      apiKeyValid = null;
    });
  }

  // Ollama URL
  const ollamaInput = container.querySelector("#ollamaUrlInput");
  if (ollamaInput) {
    ollamaInput.addEventListener("input", (e) => {
      ollamaUrl = e.target.value;
    });
  }

  const validateOllama = container.querySelector("#btnValidateOllama");
  if (validateOllama) {
    validateOllama.addEventListener("click", () => validateOllamaUrl());
  }

  // Image keys
  container.querySelectorAll("[data-image-key]").forEach((input) => {
    input.addEventListener("input", (e) => {
      const key = e.target.dataset.imageKey;
      imageKeys[key] = e.target.value;
      imageKeyStatus[key] = null;
    });

    input.addEventListener("blur", (e) => {
      const key = e.target.dataset.imageKey;
      if (imageKeys[key]) validateImageKey(key);
    });
  });
}

async function validateApiKey() {
  if (!apiKey.trim()) return;

  const statusEl = container.querySelector("#apiKeyStatus");
  statusEl.innerHTML = `<div class="validation-status checking"><span class="loading-spinner"></span> ${t("btn.validating")}</div>`;

  try {
    const res = await fetch("/api/setup/validate-key", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ provider: selectedProvider, api_key: apiKey }),
    });
    const data = await res.json();
    apiKeyValid = data.valid;

    if (data.valid) {
      statusEl.innerHTML = `<div class="validation-status valid">\u2713 ${t("env.apikey.valid")}</div>`;
    } else {
      statusEl.innerHTML = `<div class="validation-status invalid">\u2717 ${data.message || t("env.apikey.invalid")}</div>`;
    }
  } catch {
    statusEl.innerHTML = `<div class="validation-status invalid">\u2717 ${t("error.network")}</div>`;
    apiKeyValid = false;
  }
}

async function validateOllamaUrl() {
  const statusEl = container.querySelector("#ollamaStatus");
  statusEl.innerHTML = `<div class="validation-status checking"><span class="loading-spinner"></span> ${t("btn.validating")}</div>`;

  try {
    const res = await fetch("/api/setup/validate-key", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ provider: "ollama", ollama_url: ollamaUrl }),
    });
    const data = await res.json();
    if (data.valid) {
      statusEl.innerHTML = `<div class="validation-status valid">\u2713 ${t("env.apikey.valid")}</div>`;
    } else {
      statusEl.innerHTML = `<div class="validation-status invalid">\u2717 ${data.message || t("env.apikey.invalid")}</div>`;
    }
  } catch {
    statusEl.innerHTML = `<div class="validation-status invalid">\u2717 ${t("error.network")}</div>`;
  }
}

async function validateImageKey(key) {
  const statusEl = container.querySelector(`[data-image-status="${key}"]`);
  if (!statusEl) return;

  statusEl.innerHTML = `<div class="validation-status checking"><span class="loading-spinner"></span> ${t("btn.validating")}</div>`;

  try {
    const res = await fetch("/api/setup/validate-key", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ provider: key, api_key: imageKeys[key] }),
    });
    const data = await res.json();
    imageKeyStatus[key] = data.valid;

    if (data.valid) {
      statusEl.innerHTML = `<div class="validation-status valid">\u2713 ${t("env.apikey.valid")}</div>`;
    } else {
      statusEl.innerHTML = `<div class="validation-status invalid">\u2717 ${data.message || t("env.apikey.invalid")}</div>`;
    }
  } catch {
    statusEl.innerHTML = "";
  }
}

export function validateEnvironment() {
  const errorEl = container.querySelector("#envError");

  if (!selectedProvider) {
    errorEl.innerHTML = `<div class="error-message">${t("error.apikey_required")}</div>`;
    return false;
  }

  const provider = PROVIDERS.find((p) => p.id === selectedProvider);
  if (provider?.keyRequired && !apiKey.trim()) {
    errorEl.innerHTML = `<div class="error-message">${t("error.apikey_required")}</div>`;
    return false;
  }

  if (errorEl) errorEl.innerHTML = "";
  return true;
}

export function getEnvironmentData() {
  return {
    provider: selectedProvider,
    api_key: apiKey || undefined,
    ollama_url: selectedProvider === "ollama" ? ollamaUrl : undefined,
    image_keys: {
      novelai_token: imageKeys.novelai || undefined,
      fal_key: imageKeys.fal || undefined,
      meshy_api_key: imageKeys.meshy || undefined,
    },
  };
}

function escapeAttr(s) {
  return s.replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/'/g, "&#39;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
