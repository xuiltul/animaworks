/* ── Shared i18n Module ────────────────────── */

let _translations = {};
let _locale = 'ja';
let _initPromise = null;

export async function initI18n() {
  if (_initPromise) return _initPromise;
  _initPromise = _doInit();
  return _initPromise;
}

async function _doInit() {
  const stored = localStorage.getItem('animaworks-locale');
  if (stored) {
    _locale = stored;
  } else {
    try {
      const res = await fetch('/api/system/config');
      const data = await res.json();
      _locale = data.locale || 'ja';
    } catch {
      _locale = navigator.language?.startsWith('en') ? 'en' : 'ja';
    }
    localStorage.setItem('animaworks-locale', _locale);
  }
  await loadTranslations(_locale);
}

export async function loadTranslations(locale) {
  _locale = locale;
  try {
    const res = await fetch(`/i18n/${locale}.json`);
    _translations = await res.json();
  } catch {
    if (locale !== 'ja') {
      try {
        const res = await fetch('/i18n/ja.json');
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
}

export function getLocale() {
  return _locale;
}
