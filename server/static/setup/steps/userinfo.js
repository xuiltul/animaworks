/* ── Step 2: User Information ────────────── */

import { t } from "../setup.js";

let container = null;
let username = "";
let displayName = "";
let bio = "";

export function initUserInfoStep(el) {
  container = el;
  render();
}

function render() {
  container.innerHTML = `
    <h2 class="step-section-title" data-i18n="userinfo.title">${t("userinfo.title")}</h2>
    <p class="step-section-desc" data-i18n="userinfo.desc">${t("userinfo.desc")}</p>

    <div class="form-group">
      <label class="form-label" data-i18n="userinfo.name">${t("userinfo.name")}</label>
      <input type="text" class="form-input" id="userinfoName"
        pattern="[a-zA-Z0-9_]+"
        data-i18n-placeholder="userinfo.name.placeholder"
        placeholder="${t("userinfo.name.placeholder")}"
        value="${escapeAttr(username)}"
        autocomplete="off" />
      <p class="form-hint" data-i18n="userinfo.name.hint">${t("userinfo.name.hint")}</p>
      <p class="error-message" id="userinfoError" style="display: none;"></p>
    </div>

    <div class="form-group">
      <label class="form-label" data-i18n="userinfo.displayname">${t("userinfo.displayname")}</label>
      <input type="text" class="form-input" id="userinfoDisplayName"
        data-i18n-placeholder="userinfo.displayname.placeholder"
        placeholder="${t("userinfo.displayname.placeholder")}"
        value="${escapeAttr(displayName)}"
        autocomplete="off" />
    </div>

    <div class="form-group">
      <label class="form-label" data-i18n="userinfo.bio">${t("userinfo.bio")}</label>
      <textarea class="form-input userinfo-bio" id="userinfoBio"
        data-i18n-placeholder="userinfo.bio.placeholder"
        placeholder="${t("userinfo.bio.placeholder")}"
        rows="4">${escapeAttr(bio)}</textarea>
    </div>
  `;

  // Bind events
  const nameInput = container.querySelector("#userinfoName");
  nameInput.addEventListener("input", (e) => {
    username = e.target.value.trim().toLowerCase();
    const errorEl = container.querySelector("#userinfoError");
    if (errorEl) errorEl.style.display = "none";
  });

  const displayInput = container.querySelector("#userinfoDisplayName");
  displayInput.addEventListener("input", (e) => {
    displayName = e.target.value.trim();
  });

  const bioInput = container.querySelector("#userinfoBio");
  bioInput.addEventListener("input", (e) => {
    bio = e.target.value;
  });
}

export function validateUserInfo() {
  const namePattern = /^[a-zA-Z0-9_]+$/;
  if (!username || !namePattern.test(username)) {
    const errorEl = document.querySelector("#userinfoError");
    if (errorEl) {
      errorEl.textContent = t("userinfo.error");
      errorEl.style.display = "block";
    }
    return false;
  }
  return true;
}

export function getUserInfoData() {
  return {
    username: username.toLowerCase(),
    display_name: displayName,
    bio: bio,
  };
}

function escapeAttr(s) {
  return s.replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/'/g, "&#39;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
