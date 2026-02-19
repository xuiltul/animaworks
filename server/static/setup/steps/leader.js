/* ── Step 3: Leader Creation ──────────────── */

import { t } from "../setup.js";

let leaderName = "";

export function initLeaderStep(panel) {
  panel.innerHTML = `
    <h2 class="step-section-title" data-i18n="leader.title">${t("leader.title")}</h2>
    <p class="step-section-desc" data-i18n="leader.desc">${t("leader.desc")}</p>
    <div class="leader-name-input">
      <label class="form-label" data-i18n="leader.label">${t("leader.label")}</label>
      <input type="text" class="form-input" id="leaderNameInput"
        pattern="[a-zA-Z]+"
        data-i18n-placeholder="leader.placeholder"
        placeholder="${t("leader.placeholder")}"
        value="${leaderName}"
        autocomplete="off" />
      <p class="form-hint" data-i18n="leader.hint">${t("leader.hint")}</p>
      <p class="error-message" id="leaderError" style="display: none;"></p>
    </div>
  `;

  const input = panel.querySelector("#leaderNameInput");
  input.addEventListener("input", () => {
    leaderName = input.value.trim().toLowerCase();
    // Clear error on input
    const errorEl = panel.querySelector("#leaderError");
    if (errorEl) errorEl.style.display = "none";
  });
}

export function validateLeader() {
  const namePattern = /^[a-zA-Z]+$/;
  if (!leaderName || !namePattern.test(leaderName)) {
    const errorEl = document.querySelector("#leaderError");
    if (errorEl) {
      errorEl.textContent = t("leader.error");
      errorEl.style.display = "block";
    }
    return false;
  }
  return true;
}

export function getLeaderData() {
  return {
    name: leaderName.toLowerCase(),
  };
}
