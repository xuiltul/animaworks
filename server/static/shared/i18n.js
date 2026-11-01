/* ── Shared i18n Module ────────────────────── */

import { api } from "../modules/api.js";
import { basePath } from "/shared/base-path.js";

let _translations = {};
let _locale = 'ja';
let _initPromise = null;

export async function initI18n() {
  if (_initPromise) return _initPromise;
  _initPromise = _doInit();
  return _initPromise;
}

async function _doInit() {
  try {
    const data = await api("/api/system/config", { redirectOnUnauthorized: false });
    _locale = data.locale || 'ja';
  } catch {
    const stored = localStorage.getItem('animaworks-locale');
    _locale = stored || (navigator.language?.startsWith('en') ? 'en' : 'ja');
  }
  localStorage.setItem('animaworks-locale', _locale);
  await loadTranslations(_locale);
}

export async function loadTranslations(locale) {
  _locale = locale;
  document.documentElement.lang = locale;
  try {
    const res = await fetch(`${basePath}/i18n/${locale}.json`, { cache: 'no-store' });
    _translations = await res.json();
  } catch {
    if (locale !== 'ja') {
      try {
        const res = await fetch(`${basePath}/i18n/ja.json`, { cache: 'no-store' });
        _translations = await res.json();
      } catch { /* use empty dict */ }
    }
  }
}

export function t(key, params = {}) {
  let text = _translations[key] || key;
  for (const [k, v] of Object.entries(params)) {
    text = text.replaceAll(`{${k}}`, v);
  }
  return text;
}

export function applyTranslations() {
  document.querySelectorAll('[data-i18n]').forEach(el => {
    const val = t(el.dataset.i18n);
    if (val !== el.dataset.i18n) el.textContent = val;
  });
  document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
    const val = t(el.dataset.i18nPlaceholder);
    if (val !== el.dataset.i18nPlaceholder) el.placeholder = val;
  });
  document.querySelectorAll('[data-i18n-title]').forEach(el => {
    const val = t(el.dataset.i18nTitle);
    if (val !== el.dataset.i18nTitle) el.title = val;
  });
  document.querySelectorAll('[data-i18n-aria-label]').forEach(el => {
    const val = t(el.dataset.i18nAriaLabel);
    if (val !== el.dataset.i18nAriaLabel) el.setAttribute('aria-label', val);
  });
}

export function getLocale() {
  return _locale;
}
