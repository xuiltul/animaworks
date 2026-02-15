/* ── Setup Wizard — Main Logic ─────────────── */

import { initLanguageStep, getLanguageData } from "./steps/language.js";
import { initEnvironmentStep, getEnvironmentData, validateEnvironment } from "./steps/environment.js";
import { initUserInfoStep, getUserInfoData, validateUserInfo } from "./steps/userinfo.js";
import { initLeaderStep, getLeaderData, validateLeader } from "./steps/leader.js";
import { initConfirmStep, populateConfirm, completeSetup } from "./steps/confirm.js";

// ── State ───────────────────────────────────

const state = {
  currentStep: 0,
  totalSteps: 5,
  locale: "ja",
  translations: {},
};

// ── i18n ────────────────────────────────────

async function loadTranslations(locale) {
  try {
    const res = await fetch(`/setup/i18n/${locale}.json`);
    if (!res.ok) throw new Error(`Failed to load ${locale} translations`);
    state.translations = await res.json();
    state.locale = locale;
    applyTranslations();
  } catch (err) {
    console.warn("i18n load error:", err);
  }
}

function applyTranslations() {
  document.querySelectorAll("[data-i18n]").forEach((el) => {
    const key = el.getAttribute("data-i18n");
    const val = state.translations[key];
    if (val) el.textContent = val;
  });

  // Update placeholder attributes
  document.querySelectorAll("[data-i18n-placeholder]").forEach((el) => {
    const key = el.getAttribute("data-i18n-placeholder");
    const val = state.translations[key];
    if (val) el.placeholder = val;
  });
}

export function t(key) {
  return state.translations[key] || key;
}

export function getLocale() {
  return state.locale;
}

export async function setLocale(locale) {
  state.locale = locale;
  await loadTranslations(locale);
}

// ── Step Navigation ─────────────────────────

const stepItems = document.querySelectorAll(".step-item");
const stepConnectors = document.querySelectorAll(".step-connector");
const btnBack = document.getElementById("btnBack");
const btnNext = document.getElementById("btnNext");
const stepIndicator = document.getElementById("stepIndicator");
const stepContent = document.getElementById("stepContent");

function updateStepUI() {
  const step = state.currentStep;

  // Update progress bar
  stepItems.forEach((item, i) => {
    item.classList.remove("active", "done");
    if (i === step) item.classList.add("active");
    else if (i < step) item.classList.add("done");
  });

  stepConnectors.forEach((conn, i) => {
    conn.classList.toggle("done", i < step);
  });

  // Update panels
  document.querySelectorAll(".step-panel").forEach((panel, i) => {
    panel.classList.toggle("active", i === step);
  });

  // Update buttons
  btnBack.disabled = step === 0;
  stepIndicator.textContent = `${step + 1} / ${state.totalSteps}`;

  if (step === state.totalSteps - 1) {
    btnNext.textContent = t("btn.complete");
    btnNext.classList.add("btn-complete");
  } else {
    btnNext.textContent = t("btn.next");
    btnNext.classList.remove("btn-complete");
  }
}

async function goNext() {
  const step = state.currentStep;

  // Validate current step
  if (step === 1) {
    const valid = validateUserInfo();
    if (!valid) return;
  } else if (step === 2) {
    const valid = validateEnvironment();
    if (!valid) return;
  } else if (step === 3) {
    const valid = validateLeader();
    if (!valid) return;
  }

  if (step === state.totalSteps - 1) {
    // Complete setup
    btnNext.disabled = true;
    try {
      await completeSetup(gatherAllData());
    } finally {
      btnNext.disabled = false;
    }
    return;
  }

  state.currentStep = Math.min(step + 1, state.totalSteps - 1);

  // Prepare next step data
  if (state.currentStep === 4) {
    populateConfirm(gatherAllData());
  }

  updateStepUI();
}

function goBack() {
  state.currentStep = Math.max(state.currentStep - 1, 0);
  updateStepUI();
}

export function goToStep(step) {
  if (step >= 0 && step < state.totalSteps) {
    state.currentStep = step;
    updateStepUI();
  }
}

function gatherAllData() {
  return {
    language: getLanguageData(),
    userinfo: getUserInfoData(),
    environment: getEnvironmentData(),
    leader: getLeaderData(),
  };
}

// ── Initialization ──────────────────────────

async function init() {
  // Create step panels
  stepContent.innerHTML = `
    <div class="step-panel active" id="stepLanguage"></div>
    <div class="step-panel" id="stepUserInfo"></div>
    <div class="step-panel" id="stepEnvironment"></div>
    <div class="step-panel" id="stepLeader"></div>
    <div class="step-panel" id="stepConfirm"></div>
  `;

  // Load default translations
  await loadTranslations(state.locale);

  // Initialize step modules
  initLanguageStep(document.getElementById("stepLanguage"));
  initUserInfoStep(document.getElementById("stepUserInfo"));
  initEnvironmentStep(document.getElementById("stepEnvironment"));
  initLeaderStep(document.getElementById("stepLeader"));
  initConfirmStep(document.getElementById("stepConfirm"));

  // Bind navigation
  btnNext.addEventListener("click", goNext);
  btnBack.addEventListener("click", goBack);

  updateStepUI();
}

init();
