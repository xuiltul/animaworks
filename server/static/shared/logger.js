// ── Frontend Unified Logger ──────────────────────────────────
// Dual output: console + server (via fetch, with sendBeacon fallback on unload).
// Usage:
//   import { createLogger } from '../shared/logger.js';
//   const logger = createLogger('websocket');
//   logger.info('Connected', { url });

const LOG_LEVELS = { DEBUG: 0, INFO: 1, WARN: 2, ERROR: 3 };
const FLUSH_INTERVAL = 5000;
const MAX_BUFFER_SIZE = 100;
const SERVER_ENDPOINT = '/api/system/frontend-logs';

// URL parameter override: ?log=debug (persists to localStorage)
try {
  const _urlLog = new URLSearchParams(location.search).get('log');
  if (_urlLog) {
    const _lvl = _urlLog.toUpperCase();
    if (_lvl in LOG_LEVELS) {
      localStorage.setItem('animaworks_log_level', _lvl);
    }
  }
} catch { /* ignore */ }

let _buffer = [];
let _flushTimer = null;
let _sessionId = null;
let _flushListenersAttached = false;
let _flushing = false;

function _getSessionId() {
  if (!_sessionId) {
    _sessionId = sessionStorage.getItem('animaworks_session_id');
    if (!_sessionId) {
      // crypto.randomUUID requires secure context (HTTPS or localhost).
      // Fall back to crypto.getRandomValues for HTTP + LAN IP access.
      if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
        _sessionId = crypto.randomUUID().slice(0, 12);
      } else {
        const arr = new Uint8Array(6);
        crypto.getRandomValues(arr);
        _sessionId = Array.from(arr, b => b.toString(16).padStart(2, '0')).join('');
      }
      sessionStorage.setItem('animaworks_session_id', _sessionId);
    }
  }
  return _sessionId;
}

// Primary flush: fetch with response checking and buffer recovery
async function _flush() {
  if (_buffer.length === 0 || _flushing) return;
  _flushing = true;
  const entries = _buffer.splice(0);
  try {
    const res = await fetch(SERVER_ENDPOINT, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(entries),
    });
    if (!res.ok) {
      console.debug(`[logger] flush failed: HTTP ${res.status}`);
      _restoreBuffer(entries);
    }
  } catch (err) {
    console.debug(`[logger] flush error: ${err.message}`);
    _restoreBuffer(entries);
  } finally {
    _flushing = false;
  }
}

// Unload flush: sendBeacon (fire-and-forget, reliable during page unload)
function _flushBeacon() {
  if (_buffer.length === 0) return;
  const entries = _buffer.splice(0);
  navigator.sendBeacon(
    SERVER_ENDPOINT,
    new Blob([JSON.stringify(entries)], { type: 'application/json' })
  );
}

function _restoreBuffer(entries) {
  // Prepend failed entries back to buffer, respecting max size
  const combined = entries.concat(_buffer);
  _buffer.length = 0;
  const start = Math.max(0, combined.length - MAX_BUFFER_SIZE);
  for (let i = start; i < combined.length; i++) {
    _buffer.push(combined[i]);
  }
}

function _ensureFlushTimer() {
  if (_flushTimer) return;
  _flushTimer = setInterval(_flush, FLUSH_INTERVAL);
  if (!_flushListenersAttached) {
    _flushListenersAttached = true;
    // Use sendBeacon for page hide/unload (reliable during navigation)
    window.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'hidden') _flushBeacon();
    });
    window.addEventListener('beforeunload', _flushBeacon);
  }
}

class Logger {
  constructor(module) {
    this.module = module;
    _ensureFlushTimer();
  }

  _log(level, ...args) {
    const threshold = localStorage.getItem('animaworks_log_level') || 'INFO';
    if (LOG_LEVELS[level] < LOG_LEVELS[threshold]) return;

    const ts = new Date().toISOString();
    const prefix = `[${ts}] [${level}] [${this.module}]`;

    // 1. Console output
    const consoleFn = level === 'ERROR' ? 'error' : level === 'WARN' ? 'warn' : 'log';
    console[consoleFn](prefix, ...args);

    // 2. Buffer for server send
    const message = args.map(a =>
      typeof a === 'object' ? JSON.stringify(a) : String(a)
    ).join(' ');

    const entry = {
      ts,
      level,
      module: this.module,
      msg: message,
      session_id: _getSessionId(),
      url: location.href,
      ua: navigator.userAgent.slice(0, 100),
    };

    _buffer.push(entry);
    if (_buffer.length > MAX_BUFFER_SIZE) _buffer.shift();

    // ERROR: flush immediately
    if (level === 'ERROR') _flush();
  }

  debug(...args) { this._log('DEBUG', ...args); }
  info(...args)  { this._log('INFO', ...args); }
  warn(...args)  { this._log('WARN', ...args); }
  error(...args) { this._log('ERROR', ...args); }
}

export function createLogger(module) { return new Logger(module); }
