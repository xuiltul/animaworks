/* ── Step 4: Confirmation ──────────────────── */

import { t, goToStep } from "../setup.js";

let confirmPanel = null;

export function initConfirmStep(panel) {
  confirmPanel = panel;

  panel.innerHTML = `
    <h2 class="step-section-title">${t("confirm.title")}</h2>
    <p class="step-section-desc">${t("confirm.desc")}</p>
    <div class="confirm-card" id="confirmSummary"></div>
  `;
}

export function populateConfirm(data) {
  if (!confirmPanel) return;

  const locale = data.language?.locale || "ja";
  const userName = data.userinfo?.username || "-";
  const userDisplayName = data.userinfo?.display_name || "";
  const provider = data.environment?.provider || "-";
  const leaderName = data.leader?.name || "-";
  const imageKeys = data.environment?.image_keys || {};

  // Build API key summary rows
  const keyEntries = Object.entries(imageKeys).filter(([, v]) => v);
  const keySummary = keyEntries.length > 0
    ? keyEntries.map(([k, v]) => `
        <div class="confirm-row">
          <span class="confirm-key">${k}</span>
          <span class="confirm-value masked">${maskKey(v)}</span>
        </div>
      `).join("")
    : `<div class="confirm-row">
        <span class="confirm-key">${t("confirm.apikeys")}</span>
        <span class="confirm-value">${t("confirm.not_configured")}</span>
      </div>`;

  confirmPanel.innerHTML = `
    <h2 class="step-section-title">${t("confirm.title")}</h2>
    <p class="step-section-desc">${t("confirm.desc")}</p>

    <div class="confirm-card">
      <div class="confirm-section">
        <div class="confirm-section-title">${t("confirm.language")}
          <button class="btn-edit-step" data-step="0">${t("confirm.edit")}</button>
        </div>
        <div class="confirm-row">
          <span class="confirm-key">${t("confirm.language")}</span>
          <span class="confirm-value">${locale === "ja" ? t("lang.ja") : t("lang.en")}</span>
        </div>
      </div>

      <div class="confirm-section">
        <div class="confirm-section-title">${t("confirm.userinfo")}
          <button class="btn-edit-step" data-step="1">${t("confirm.edit")}</button>
        </div>
        <div class="confirm-row">
          <span class="confirm-key">${t("confirm.username")}</span>
          <span class="confirm-value">${userName}</span>
        </div>
        ${userDisplayName ? `<div class="confirm-row">
          <span class="confirm-key">${t("confirm.displayname")}</span>
          <span class="confirm-value">${userDisplayName}</span>
        </div>` : ""}
      </div>

      <div class="confirm-section">
        <div class="confirm-section-title">${t("confirm.provider")}
          <button class="btn-edit-step" data-step="2">${t("confirm.edit")}</button>
        </div>
        <div class="confirm-row">
          <span class="confirm-key">${t("confirm.provider")}</span>
          <span class="confirm-value">${t(`env.provider.${provider}`) || provider}</span>
        </div>
        ${keySummary}
      </div>

      <div class="confirm-section">
        <div class="confirm-section-title">${t("confirm.leader")}
          <button class="btn-edit-step" data-step="3">${t("confirm.edit")}</button>
        </div>
        <div class="confirm-row">
          <span class="confirm-key">${t("confirm.leader")}</span>
          <span class="confirm-value">${leaderName}</span>
        </div>
      </div>
    </div>
  `;

  // Wire up edit buttons
  confirmPanel.querySelectorAll(".btn-edit-step").forEach((btn) => {
    btn.addEventListener("click", () => {
      const step = parseInt(btn.dataset.step, 10);
      goToStep(step);
    });
  });
}

export async function completeSetup(data) {
  if (!confirmPanel) return;

  // Build API payload from wizard data
  const payload = {
    locale: data.language?.locale || "ja",
    credentials: {},
    anima: { name: data.leader?.name },
    user: data.userinfo ? {
      username: data.userinfo.username,
      display_name: data.userinfo.display_name || "",
      bio: data.userinfo.bio || "",
    } : undefined,
  };

  // Add credentials from environment step
  const env = data.environment || {};
  if (env.provider && env.api_key) {
    payload.credentials[env.provider] = { api_key: env.api_key };
  }
  // Add image generation keys
  const imageKeys = env.image_keys || {};
  for (const [key, value] of Object.entries(imageKeys)) {
    if (value) {
      payload.credentials[key] = { api_key: value };
    }
  }

  try {
    const res = await fetch("/api/setup/complete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || t("confirm.error"));
    }

    // Persist username to localStorage so the dashboard auto-logs in
    if (data.userinfo?.username) {
      localStorage.setItem("animaworks_user", data.userinfo.username);
    }

    // Show success screen
    confirmPanel.innerHTML = `
      <div class="confirm-success visible">
        <div class="success-icon">\u2705</div>
        <h2 class="success-title">${t("confirm.success")}</h2>
        <p class="success-message">${t("confirm.success.desc")}</p>
      </div>
    `;

    // Hide nav buttons
    const nav = document.querySelector(".step-nav");
    if (nav) nav.style.display = "none";

    // Redirect to dashboard after a short delay
    setTimeout(() => {
      window.location.href = "/";
    }, 2000);
  } catch (e) {
    const errorDiv = confirmPanel.querySelector("#confirmError") || document.createElement("div");
    errorDiv.id = "confirmError";
    errorDiv.innerHTML = `<div class="error-message">${e.message || t("confirm.error")}</div>`;
    if (!confirmPanel.querySelector("#confirmError")) {
      confirmPanel.appendChild(errorDiv);
    }
    throw e;
  }
}

function maskKey(key) {
  if (!key || key.length < 8) return "\u2022\u2022\u2022\u2022\u2022\u2022\u2022\u2022";
  return key.slice(0, 4) + "\u2022".repeat(Math.min(key.length - 8, 16)) + key.slice(-4);
}
