/* ── Step 2: Environment + API Keys ───────── */

import { basePath } from "/shared/base-path.js";
import { t } from "../setup.js";

let container = null;
let envData = {
  claude_code_available: false,
  claude_code_authenticated: false,
  claude_subscription_type: null,
  codex_cli_available: false,
  codex_login_available: false,
  cursor_agent_available: false,
  cursor_agent_authenticated: false,
  gemini_cli_available: false,
  gemini_authenticated: false,
};
let selectedProvider = "";
let otherExpanded = false;
let apiKey = "";
let apiKeyValid = null;
let claudeCodeValid = null;
let claudeCodeCode = null;
let codexLoginValid = null;
let codexLoginCode = null;
let codexDeviceLogin = null;
let cursorAgentValid = null;
let geminiCliValid = null;
let ollamaUrl = "http://localhost:11434";
let selectedImageStyle = "realistic";
let imageKeys = { novelai: "", fal: "", meshy: "" };
let imageKeyStatus = { novelai: null, fal: null, meshy: null };

const PROVIDERS = [
  { id: "anthropic", keyRequired: true },
  { id: "openai", keyRequired: true },
  { id: "google", keyRequired: true },
  { id: "cursor_agent", keyRequired: false },
  { id: "gemini_cli", keyRequired: false },
  { id: "ollama", keyRequired: false },
];

export function initEnvironmentStep(el) {
  container = el;
  render();
  fetchEnvironment();
}

async function fetchEnvironment() {
  try {
    const res = await fetch(`${basePath}/api/setup/environment`);
    if (!res.ok) return;
    envData = { ...envData, ...(await res.json()) };

    if (!selectedProvider) {
      if (envData.claude_code_authenticated) selectedProvider = "claude_code";
      else if (envData.codex_login_available) selectedProvider = "codex";
      else if (envData.claude_code_available) selectedProvider = "claude_code";
      else if (envData.codex_cli_available) selectedProvider = "codex";
    }
    render();
  } catch {
    // Use the default detection state.
  }
}

function render() {
  const isOllama = selectedProvider === "ollama";

  container.innerHTML = `
    <h2 data-i18n="env.title">${t("env.title")}</h2>
    <p style="color: #8888aa; font-size: 0.85rem; margin-top: 4px;" data-i18n="env.desc">${t("env.desc")}</p>

    <div class="env-section" style="margin-top: 20px;">
      <div class="env-detection-header">
        <div class="env-section-title" data-i18n="env.detection">${t("env.detection")}</div>
        <button class="btn-validate" id="btnRecheck" data-i18n="btn.recheck">${t("btn.recheck")}</button>
      </div>
      <div class="env-detection-list">
        ${renderDetectionRow("env.claude_code", envData.claude_code_available, envData.claude_code_authenticated, envData.claude_subscription_type)}
        ${renderDetectionRow("env.codex_cli", envData.codex_cli_available, envData.codex_login_available)}
        ${renderDetectionRow("env.cursor_agent", envData.cursor_agent_available, envData.cursor_agent_authenticated)}
        ${renderDetectionRow("env.gemini_cli", envData.gemini_cli_available, envData.gemini_authenticated)}
      </div>
      ${!envData.claude_code_available && !envData.codex_cli_available ? renderInstallHint() : ""}
    </div>

    <div class="env-section">
      ${renderProviders()}
    </div>

    <div class="env-section">
      <div class="env-section-title" data-i18n="env.imagegen.title">${t("env.imagegen.title")}</div>
      <div class="env-section-desc" data-i18n="env.imagegen.desc">${t("env.imagegen.desc")}</div>
      <div class="env-section-desc" data-i18n="env.imagestyle.desc">${t("env.imagestyle.desc")}</div>
      <div class="image-style-cards">
        <div class="image-style-card${selectedImageStyle === "realistic" ? " selected" : ""}" data-style="realistic">
          <div class="image-style-radio"></div>
          <div>
            <div class="image-style-name" data-i18n="env.imagestyle.realistic">${t("env.imagestyle.realistic")}</div>
            <div class="image-style-desc" data-i18n="env.imagestyle.realistic.desc">${t("env.imagestyle.realistic.desc")}</div>
          </div>
        </div>
        <div class="image-style-card${selectedImageStyle === "anime" ? " selected" : ""}" data-style="anime">
          <div class="image-style-radio"></div>
          <div>
            <div class="image-style-name" data-i18n="env.imagestyle.anime">${t("env.imagestyle.anime")}</div>
            <div class="image-style-desc" data-i18n="env.imagestyle.anime.desc">${t("env.imagestyle.anime.desc")}</div>
          </div>
        </div>
      </div>
      <div class="image-key-section">${renderImageKeysForStyle()}</div>
    </div>

    <div id="envError"></div>
  `;

  bindEvents();
}

function renderDetectionRow(nameKey, available, authenticated, plan = null) {
  let icon = "\u2b1c";
  let statusClass = "not-found";
  let status = t("env.status.not_installed");
  if (authenticated) {
    icon = "\u2705";
    statusClass = "found";
    status = plan ? t("env.status.logged_in_plan").replace("{plan}", plan) : t("env.status.logged_in");
  } else if (available) {
    icon = "\u26a0\ufe0f";
    statusClass = "warn";
    status = t("env.status.not_logged_in");
  }

  return `
    <div class="env-detection">
      <div class="env-detection-icon">${icon}</div>
      <div class="env-detection-text">
        <div class="env-detection-name">${t(nameKey)}</div>
        <div class="env-detection-status ${statusClass}">${status}</div>
      </div>
    </div>
  `;
}

function renderInstallHint() {
  return `
    <div class="env-install-hint">
      <strong>${t("env.install.title")}</strong>
      <div>${formatInlineCode(t("env.install.claude"))}</div>
      <div>${formatInlineCode(t("env.install.codex"))}</div>
    </div>
  `;
}

const SUBSCRIPTION_PROVIDERS = ["claude_code", "codex"];

function renderProviders() {
  const subscription = [];
  if (envData.claude_code_available) subscription.push(renderProviderCard("claude_code", envData.claude_code_authenticated));
  if (envData.codex_cli_available) {
    subscription.push(renderProviderCard("codex", envData.codex_login_available && !envData.claude_code_authenticated));
  }

  const other = PROVIDERS
    .filter((provider) => provider.id !== "cursor_agent" || envData.cursor_agent_available)
    .filter((provider) => provider.id !== "gemini_cli" || envData.gemini_cli_available)
    .map((provider) => renderProviderCard(provider.id))
    .join("");

  const details = renderProviderDetails();
  const selectedIsSubscription = SUBSCRIPTION_PROVIDERS.includes(selectedProvider);
  const collapsible = subscription.length > 0;
  const expanded = !collapsible || otherExpanded || (selectedProvider && !selectedIsSubscription);
  const toggle = collapsible
    ? `<button type="button" class="provider-group-toggle${expanded ? " expanded" : ""}" id="btnToggleOther" aria-expanded="${expanded}">
         <span class="provider-group-caret">${expanded ? "\u25be" : "\u25b8"}</span>${t("env.provider.group.other")}
       </button>`
    : `<div class="provider-group-title">${t("env.provider.group.other")}</div>`;

  return `
    ${subscription.length ? `<div class="provider-group-title">${t("env.provider.group.subscription")}</div><div class="provider-cards">${subscription.join("")}</div>${selectedIsSubscription ? details : ""}` : ""}
    ${toggle}
    ${expanded ? `<div class="provider-cards">${other}</div>${selectedIsSubscription ? "" : details}` : ""}
  `;
}

function renderProviderCard(providerId, recommended = false) {
  const selected = providerId === selectedProvider ? " selected" : "";
  const badge = recommended ? `<span class="provider-badge">${t("env.recommended")}</span>` : "";
  return `
    <div class="provider-card${selected}" data-provider="${providerId}">
      <div class="provider-radio"></div>
      <div>
        <div class="provider-name">${t(`env.provider.${providerId}`)}</div>
        <div class="provider-desc">${t(`env.provider.${providerId}.desc`)}</div>
      </div>
      ${badge}
    </div>
  `;
}

function renderProviderDetails() {
  if (selectedProvider === "claude_code") return renderClaudeCodeDetails();
  if (selectedProvider === "codex") return renderCodexDetails();
  if (selectedProvider === "cursor_agent") return renderCursorAgentStatus();
  if (selectedProvider === "gemini_cli") return renderGeminiCliStatus();
  if (selectedProvider === "ollama") return renderOllamaInput();
  if (["anthropic", "openai", "google"].includes(selectedProvider)) return renderApiKeyInput();
  return "";
}

function renderApiKeyInput() {
  const statusHtml = apiKeyValid === true
    ? `<div class="validation-status valid">\u2713 ${t("env.apikey.valid")}</div>`
    : apiKeyValid === false ? `<div class="validation-status invalid">\u2717 ${t("env.apikey.invalid")}</div>` : "";
  return `
    <div class="api-key-section">
      <label class="form-label" for="apiKeyInput" data-i18n="env.apikey">${t("env.apikey")}</label>
      <div class="api-key-row">
        <input type="password" class="api-key-input" id="apiKeyInput" data-i18n-placeholder="env.apikey.placeholder" placeholder="${t("env.apikey.placeholder")}" value="${escapeAttr(apiKey)}">
        <button class="btn-validate" id="btnValidateKey" data-i18n="btn.validate">${t("btn.validate")}</button>
      </div>
      <div id="apiKeyStatus">${statusHtml}</div>
    </div>
  `;
}

function renderCliStatus(prefix, available, authenticated, valid, code, plan = null) {
  if (valid === true || authenticated) {
    const text = prefix === "claude_code"
      ? t("env.claude_code.status.ready")
      : t(`env.${prefix}.status.ready`);
    return `<div class="validation-status valid">\u2713 ${text}</div>`;
  }
  const statusCode = code || (available ? "not_logged_in" : "not_installed");
  const text = prefix === "claude_code"
    ? t(`env.claude_code.status.${statusCode}`)
    : t(`env.${prefix}.status.${statusCode}`);
  return `<div class="validation-status invalid">\u2717 ${text}</div>`;
}

function renderClaudeCodeDetails() {
  const authenticated = envData.claude_code_authenticated || claudeCodeValid === true;
  return `
    <div class="api-key-section">
      ${renderCliStatus("claude_code", envData.claude_code_available, authenticated, claudeCodeValid, claudeCodeCode)}
      ${!authenticated ? `<div class="form-hint">${formatInlineCode(t("env.claude_code.login_hint"))}</div><button class="btn-validate" id="btnValidateClaudeCode" data-i18n="btn.reverify">${t("btn.reverify")}</button>` : ""}
    </div>
  `;
}

function renderCodexDetails() {
  const authenticated = envData.codex_login_available || codexLoginValid === true;
  return `
    <div class="api-key-section">
      ${renderCliStatus("codex", envData.codex_cli_available, authenticated, codexLoginValid, codexLoginCode)}
      ${!authenticated ? `<div class="form-hint">${formatInlineCode(t("env.codex.login_hint"))}</div>
        <div class="api-key-row">
          <button class="btn-validate" id="btnStartCodexBrowserLogin" data-i18n="btn.browser_login">${t("btn.browser_login")}</button>
          <button class="btn-validate" id="btnValidateCodexLogin" data-i18n="btn.reverify">${t("btn.reverify")}</button>
        </div>
        <div id="codexDeviceLoginInfo">${renderCodexDeviceLoginInfo()}</div>` : ""}
    </div>
  `;
}

function renderCursorAgentStatus() {
  return `
    <div class="api-key-section">
      ${renderCliStatus("cursor_agent", envData.cursor_agent_available, envData.cursor_agent_authenticated, cursorAgentValid)}
      <button class="btn-validate" id="btnValidateCursorAgent" data-i18n="btn.validate">${t("btn.validate")}</button>
    </div>
  `;
}

function renderGeminiCliStatus() {
  return `
    <div class="api-key-section">
      ${renderCliStatus("gemini_cli", envData.gemini_cli_available, envData.gemini_authenticated, geminiCliValid)}
      <button class="btn-validate" id="btnValidateGeminiCli" data-i18n="btn.validate">${t("btn.validate")}</button>
    </div>
  `;
}

function renderCodexDeviceLoginInfo() {
  if (!codexDeviceLogin) return "";
  if (codexDeviceLogin.already_logged_in) {
    return `<div class="validation-status valid">\u2713 ${escapeHtml(codexDeviceLogin.message || t("env.codex.status.ready"))}</div>`;
  }
  if (!codexDeviceLogin.ok) {
    return `<div class="validation-status invalid">\u2717 ${escapeHtml(codexDeviceLogin.message || t("env.codex.browser.failed"))}</div>`;
  }
  const loginUrl = escapeHtml(codexDeviceLogin.login_url || "");
  const deviceCode = escapeHtml(codexDeviceLogin.device_code || "");
  return `
    <div class="validation-status checking">${t("env.codex.browser.instructions")}</div>
    <div style="margin-top:8px;font-size:0.9rem;">
      <div><strong>${t("env.codex.browser.url")}</strong> <a href="${loginUrl}" target="_blank" rel="noopener noreferrer">${loginUrl}</a></div>
      <div style="margin-top:4px;"><strong>${t("env.codex.browser.code")}</strong> <code>${deviceCode}</code></div>
    </div>
  `;
}

function renderOllamaInput() {
  return `
    <div class="api-key-section">
      <label class="form-label" for="ollamaUrlInput" data-i18n="env.ollama.url">${t("env.ollama.url")}</label>
      <div class="api-key-row">
        <input type="text" class="api-key-input" id="ollamaUrlInput" data-i18n-placeholder="env.ollama.url.placeholder" placeholder="${t("env.ollama.url.placeholder")}" value="${escapeAttr(ollamaUrl)}">
        <button class="btn-validate" id="btnValidateOllama" data-i18n="btn.validate">${t("btn.validate")}</button>
      </div>
      <div id="ollamaStatus"></div>
    </div>
  `;
}

function renderImageKeysForStyle() {
  if (selectedImageStyle === "realistic") {
    return renderImageKey("fal", t("env.fal"), t("env.fal.desc"), t("env.optional"));
  }
  return [
    renderImageKey("novelai", t("env.novelai"), t("env.novelai.desc"), t("env.recommended")),
    renderImageKey("fal", t("env.fal"), t("env.fal.desc"), t("env.optional")),
    renderImageKey("meshy", t("env.meshy"), t("env.meshy.desc"), t("env.optional")),
  ].join("");
}

function renderImageKey(id, label, hint, badge) {
  const status = imageKeyStatus[id];
  const statusHtml = status === true
    ? `<div class="validation-status valid">\u2713 ${t("env.apikey.valid")}</div>`
    : status === false ? `<div class="validation-status invalid">\u2717 ${t("env.apikey.invalid")}</div>` : "";
  const badgeClass = badge === t("env.recommended") ? "provider-badge" : "image-key-optional";
  return `
    <div class="image-key-item">
      <div class="image-key-label">${label}<span class="${badgeClass}">${badge}</span></div>
      <div class="form-hint">${hint}</div>
      <div class="api-key-row" style="margin-top: 4px;">
        <input type="password" class="api-key-input" data-image-key="${id}" data-i18n-placeholder="env.apikey.placeholder" placeholder="${t("env.apikey.placeholder")}" value="${escapeAttr(imageKeys[id] || "")}">
      </div>
      <div class="image-key-status" data-image-status="${id}">${statusHtml}</div>
    </div>
  `;
}

function bindEvents() {
  container.querySelectorAll(".image-style-card").forEach((card) => {
    card.addEventListener("click", () => {
      selectedImageStyle = card.dataset.style;
      render();
    });
  });

  container.querySelectorAll(".provider-card").forEach((card) => {
    card.addEventListener("click", () => {
      selectedProvider = card.dataset.provider;
      otherExpanded = !SUBSCRIPTION_PROVIDERS.includes(selectedProvider);
      apiKeyValid = null;
      claudeCodeValid = null;
      claudeCodeCode = null;
      codexLoginValid = null;
      codexLoginCode = null;
      codexDeviceLogin = null;
      cursorAgentValid = null;
      geminiCliValid = null;
      render();
    });
  });

  const toggleOther = container.querySelector("#btnToggleOther");
  if (toggleOther) toggleOther.addEventListener("click", () => {
    otherExpanded = !otherExpanded;
    render();
  });

  const recheck = container.querySelector("#btnRecheck");
  if (recheck) recheck.addEventListener("click", fetchEnvironment);

  const validateKey = container.querySelector("#btnValidateKey");
  if (validateKey) validateKey.addEventListener("click", validateApiKey);
  const validateClaude = container.querySelector("#btnValidateClaudeCode");
  if (validateClaude) validateClaude.addEventListener("click", validateClaudeCodeLogin);
  const validateCodex = container.querySelector("#btnValidateCodexLogin");
  if (validateCodex) validateCodex.addEventListener("click", validateCodexLogin);
  const startCodex = container.querySelector("#btnStartCodexBrowserLogin");
  if (startCodex) startCodex.addEventListener("click", startCodexBrowserLogin);
  const validateCursor = container.querySelector("#btnValidateCursorAgent");
  if (validateCursor) validateCursor.addEventListener("click", validateCursorAgent);
  const validateGemini = container.querySelector("#btnValidateGeminiCli");
  if (validateGemini) validateGemini.addEventListener("click", validateGeminiCli);
  const validateOllamaButton = container.querySelector("#btnValidateOllama");
  if (validateOllamaButton) validateOllamaButton.addEventListener("click", validateOllamaUrl);

  const apiInput = container.querySelector("#apiKeyInput");
  if (apiInput) apiInput.addEventListener("input", (event) => {
    apiKey = event.target.value;
    apiKeyValid = null;
  });
  const ollamaInput = container.querySelector("#ollamaUrlInput");
  if (ollamaInput) ollamaInput.addEventListener("input", (event) => { ollamaUrl = event.target.value; });

  container.querySelectorAll("[data-image-key]").forEach((input) => {
    input.addEventListener("input", (event) => {
      const key = event.target.dataset.imageKey;
      imageKeys[key] = event.target.value;
      imageKeyStatus[key] = null;
    });
    input.addEventListener("blur", (event) => {
      const key = event.target.dataset.imageKey;
      if (imageKeys[key]) validateImageKey(key);
    });
  });
}

async function postValidation(payload) {
  const res = await fetch(`${basePath}/api/setup/validate-key`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return res.json();
}

async function validateApiKey() {
  if (!apiKey.trim()) return;
  const statusEl = container.querySelector("#apiKeyStatus");
  statusEl.innerHTML = `<div class="validation-status checking"><span class="loading-spinner"></span> ${t("btn.validating")}</div>`;
  try {
    const data = await postValidation({ provider: selectedProvider, api_key: apiKey });
    apiKeyValid = data.valid;
    statusEl.innerHTML = data.valid
      ? `<div class="validation-status valid">\u2713 ${t("env.apikey.valid")}</div>`
      : `<div class="validation-status invalid">\u2717 ${t("env.apikey.invalid")}</div>`;
  } catch {
    apiKeyValid = false;
    statusEl.innerHTML = `<div class="validation-status invalid">\u2717 ${t("error.network")}</div>`;
  }
}

async function validateClaudeCodeLogin() {
  const statusEl = container.querySelector(".api-key-section .validation-status");
  if (statusEl) statusEl.innerHTML = `<span class="loading-spinner"></span> ${t("btn.validating")}`;
  try {
    const data = await postValidation({ provider: "claude_code" });
    claudeCodeValid = data.valid;
    claudeCodeCode = data.code;
    if (data.valid) envData.claude_code_authenticated = true;
    render();
  } catch {
    claudeCodeValid = false;
    claudeCodeCode = "not_logged_in";
    render();
  }
}

async function validateCodexLogin() {
  const section = container.querySelector(".api-key-section");
  if (section) section.querySelector(".validation-status").innerHTML = `<span class="loading-spinner"></span> ${t("btn.validating")}`;
  try {
    const data = await postValidation({ provider: "codex" });
    codexLoginValid = data.valid;
    codexLoginCode = data.code;
    if (data.valid) envData.codex_login_available = true;
    render();
  } catch {
    codexLoginValid = false;
    codexLoginCode = "not_logged_in";
    render();
  }
}

async function startCodexBrowserLogin() {
  const infoEl = container.querySelector("#codexDeviceLoginInfo");
  if (!infoEl) return;
  infoEl.innerHTML = `<div class="validation-status checking"><span class="loading-spinner"></span> ${t("btn.validating")}</div>`;
  try {
    const res = await fetch(`${basePath}/api/setup/codex/device-login`, { method: "POST", headers: { "Content-Type": "application/json" } });
    codexDeviceLogin = await res.json();
    if (codexDeviceLogin.login_url) window.open(codexDeviceLogin.login_url, "_blank", "noopener,noreferrer");
    infoEl.innerHTML = renderCodexDeviceLoginInfo();
  } catch {
    codexDeviceLogin = { ok: false, message: t("error.network") };
    infoEl.innerHTML = renderCodexDeviceLoginInfo();
  }
}

async function validateCursorAgent() {
  await validateCli("cursor_agent", "cursorAgentValid", "cursor_agent_authenticated");
}

async function validateGeminiCli() {
  await validateCli("gemini_cli", "geminiCliValid", "gemini_authenticated");
}

async function validateCli(provider, validName, authName) {
  try {
    const data = await postValidation({ provider });
    if (validName === "cursorAgentValid") cursorAgentValid = data.valid;
    if (validName === "geminiCliValid") geminiCliValid = data.valid;
    envData[authName] = !!data.valid;
    render();
  } catch {
    if (validName === "cursorAgentValid") cursorAgentValid = false;
    if (validName === "geminiCliValid") geminiCliValid = false;
    render();
  }
}

async function validateOllamaUrl() {
  const statusEl = container.querySelector("#ollamaStatus");
  statusEl.innerHTML = `<div class="validation-status checking"><span class="loading-spinner"></span> ${t("btn.validating")}</div>`;
  try {
    const data = await postValidation({ provider: "ollama", ollama_url: ollamaUrl });
    statusEl.innerHTML = data.valid
      ? `<div class="validation-status valid">\u2713 ${t("env.apikey.valid")}</div>`
      : `<div class="validation-status invalid">\u2717 ${t("env.apikey.invalid")}</div>`;
  } catch {
    statusEl.innerHTML = `<div class="validation-status invalid">\u2717 ${t("error.network")}</div>`;
  }
}

async function validateImageKey(key) {
  const statusEl = container.querySelector(`[data-image-status="${key}"]`);
  if (!statusEl) return;
  statusEl.innerHTML = `<div class="validation-status checking"><span class="loading-spinner"></span> ${t("btn.validating")}</div>`;
  try {
    const data = await postValidation({ provider: key, api_key: imageKeys[key] });
    imageKeyStatus[key] = data.valid;
    statusEl.innerHTML = data.valid
      ? `<div class="validation-status valid">\u2713 ${t("env.apikey.valid")}</div>`
      : `<div class="validation-status invalid">\u2717 ${t("env.apikey.invalid")}</div>`;
  } catch {
    statusEl.innerHTML = "";
  }
}

export function validateEnvironment() {
  const errorEl = container.querySelector("#envError");
  if (!selectedProvider) {
    errorEl.innerHTML = `<div class="error-message">${formatInlineCode(t("error.provider_required"))}</div>`;
    return false;
  }
  if (selectedProvider === "claude_code" && !(envData.claude_code_authenticated || claudeCodeValid === true)) {
    errorEl.innerHTML = `<div class="error-message">${formatInlineCode(t("error.claude_code_login_required"))}</div>`;
    return false;
  }
  if (selectedProvider === "codex" && !(envData.codex_login_available || codexLoginValid === true)) {
    errorEl.innerHTML = `<div class="error-message">${formatInlineCode(t("error.codex_login_required"))}</div>`;
    return false;
  }
  if (selectedProvider === "cursor_agent" && !(envData.cursor_agent_authenticated || cursorAgentValid === true)) {
    errorEl.innerHTML = `<div class="error-message">${formatInlineCode(t("error.cursor_agent_auth_required"))}</div>`;
    return false;
  }
  if (selectedProvider === "gemini_cli" && !(envData.gemini_authenticated || geminiCliValid === true)) {
    errorEl.innerHTML = `<div class="error-message">${formatInlineCode(t("error.gemini_auth_required"))}</div>`;
    return false;
  }
  const provider = PROVIDERS.find((item) => item.id === selectedProvider);
  if (provider?.keyRequired && !apiKey.trim()) {
    errorEl.innerHTML = `<div class="error-message">${formatInlineCode(t("error.apikey_required"))}</div>`;
    return false;
  }
  errorEl.innerHTML = "";
  return true;
}

export function getEnvironmentData() {
  return {
    provider: selectedProvider,
    auth_mode: selectedProvider === "claude_code" ? "claude_code_login" : selectedProvider === "codex" ? "codex_login" : "api_key",
    api_key: apiKey || undefined,
    ollama_url: selectedProvider === "ollama" ? ollamaUrl : undefined,
    image_style: selectedImageStyle,
    image_keys: {
      novelai_token: imageKeys.novelai || undefined,
      fal_key: imageKeys.fal || undefined,
      meshy_api_key: imageKeys.meshy || undefined,
    },
  };
}

function formatInlineCode(text) {
  return text.replace(/`([^`]+)`/g, "<code>$1</code>");
}

function escapeAttr(s) {
  return String(s).replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/'/g, "&#39;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function escapeHtml(s) {
  return escapeAttr(String(s));
}
