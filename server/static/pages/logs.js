// ── Log Viewer ──────────────────────────────
import { api } from "../modules/api.js";
import { escapeHtml } from "../modules/state.js";

let _container = null;
let _eventSource = null;
let _streaming = false;
let _autoScroll = true;
let _levelFilters = { DEBUG: true, INFO: true, WARNING: true, ERROR: true };

export function render(container) {
  _container = container;
  _streaming = false;
  _autoScroll = true;
  _levelFilters = { DEBUG: true, INFO: true, WARNING: true, ERROR: true };

  container.innerHTML = `
    <div class="page-header">
      <h2>ログビューア</h2>
    </div>

    <div style="display:flex; gap:1rem; align-items:center; margin-bottom:1rem; flex-wrap:wrap;">
      <label style="font-weight:500;">ログファイル:</label>
      <select id="logFileSelect" style="flex:1; max-width:400px; padding:0.4rem; border:1px solid var(--border-color, #ddd); border-radius:0.375rem;">
        <option value="">ファイルを選択...</option>
      </select>
      <button class="btn-secondary" id="logStreamToggle">ストリーミング開始</button>
    </div>

    <div style="display:flex; gap:0.5rem; margin-bottom:1rem; flex-wrap:wrap;">
      <span style="font-weight:500; margin-right:0.5rem;">レベル:</span>
      <label style="display:flex; align-items:center; gap:0.25rem;">
        <input type="checkbox" class="log-level-filter" data-level="DEBUG" checked> DEBUG
      </label>
      <label style="display:flex; align-items:center; gap:0.25rem;">
        <input type="checkbox" class="log-level-filter" data-level="INFO" checked> INFO
      </label>
      <label style="display:flex; align-items:center; gap:0.25rem;">
        <input type="checkbox" class="log-level-filter" data-level="WARNING" checked> WARNING
      </label>
      <label style="display:flex; align-items:center; gap:0.25rem;">
        <input type="checkbox" class="log-level-filter" data-level="ERROR" checked> ERROR
      </label>
    </div>

    <div class="card">
      <div class="card-body">
        <pre id="logContent" style="max-height:600px; overflow-y:auto; white-space:pre-wrap; word-break:break-word; margin:0; font-size:0.8rem; line-height:1.4;">
ログファイルを選択してください</pre>
      </div>
    </div>
  `;

  _loadFileList();
  _bindEvents();
}

export function destroy() {
  _stopStreaming();
  _container = null;
}

// ── Event Binding ──────────────────────────

function _bindEvents() {
  if (!_container) return;

  const select = document.getElementById("logFileSelect");
  if (select) {
    select.addEventListener("change", () => {
      _stopStreaming();
      if (select.value) {
        _loadLogContent(select.value);
      }
    });
  }

  const streamBtn = document.getElementById("logStreamToggle");
  if (streamBtn) {
    streamBtn.addEventListener("click", () => {
      if (_streaming) {
        _stopStreaming();
        streamBtn.textContent = "ストリーミング開始";
      } else {
        _startStreaming();
        streamBtn.textContent = "ストリーミング停止";
      }
    });
  }

  _container.querySelectorAll(".log-level-filter").forEach(cb => {
    cb.addEventListener("change", () => {
      _levelFilters[cb.dataset.level] = cb.checked;
      _applyFilters();
    });
  });
}

// ── Data Loading ───────────────────────────

async function _loadFileList() {
  const select = document.getElementById("logFileSelect");
  if (!select) return;

  try {
    const data = await api("/api/system/logs");
    const files = data.files || data || [];

    if (Array.isArray(files) && files.length > 0) {
      let opts = '<option value="">ファイルを選択...</option>';
      for (const f of files) {
        const name = typeof f === "string" ? f : (f.name || f.filename || "");
        if (name) {
          opts += `<option value="${escapeHtml(name)}">${escapeHtml(name)}</option>`;
        }
      }
      select.innerHTML = opts;
    } else {
      select.innerHTML = '<option value="">ログファイルなし</option>';
    }
  } catch {
    select.innerHTML = '<option value="">ログAPIが未実装です。Phase 3で追加予定。</option>';
    const content = document.getElementById("logContent");
    if (content) content.textContent = "ログAPIが未実装です。Phase 3で追加予定。";
  }
}

async function _loadLogContent(filename) {
  const content = document.getElementById("logContent");
  if (!content) return;

  content.textContent = "読み込み中...";

  try {
    const data = await api(`/api/system/logs/${encodeURIComponent(filename)}?offset=0&limit=200`);
    const lines = data.lines || data.content || [];

    if (Array.isArray(lines)) {
      content.dataset.allLines = JSON.stringify(lines);
      _applyFilters();
    } else if (typeof lines === "string") {
      content.dataset.allLines = JSON.stringify(lines.split("\n"));
      _applyFilters();
    } else {
      content.textContent = JSON.stringify(data, null, 2);
    }
  } catch (err) {
    content.textContent = `ログの読み込みに失敗しました: ${err.message}`;
  }
}

// ── Filtering ──────────────────────────────

function _applyFilters() {
  const content = document.getElementById("logContent");
  if (!content || !content.dataset.allLines) return;

  try {
    const allLines = JSON.parse(content.dataset.allLines);
    const filtered = allLines.filter(line => {
      const text = typeof line === "string" ? line : (line.message || JSON.stringify(line));
      // Check if line matches any enabled level
      for (const [level, enabled] of Object.entries(_levelFilters)) {
        if (!enabled && text.includes(level)) return false;
      }
      return true;
    });

    content.innerHTML = filtered.map(line => {
      const text = typeof line === "string" ? line : (line.message || JSON.stringify(line));
      return _colorizeLogLine(text);
    }).join("\n");

    if (_autoScroll) {
      content.scrollTop = content.scrollHeight;
    }
  } catch {
    // Leave content as-is
  }
}

function _formatLogLine(text) {
  try {
    const obj = JSON.parse(text);
    if (obj.ts && obj.level && obj.msg) {
      const ts = obj.ts.replace(/T/, " ").replace(/\+.+$/, "");
      return `${ts} [${obj.level}] ${obj.msg}`;
    }
  } catch { /* not JSON, use as-is */ }
  return text;
}

function _colorizeLogLine(text) {
  const formatted = _formatLogLine(text);
  const escaped = escapeHtml(formatted);
  if (formatted.includes("ERROR")) return `<span style="color:#ef4444;">${escaped}</span>`;
  if (formatted.includes("WARNING")) return `<span style="color:#f59e0b;">${escaped}</span>`;
  if (formatted.includes("DEBUG")) return `<span style="color:#9ca3af;">${escaped}</span>`;
  return escaped;
}

// ── Streaming ──────────────────────────────

function _startStreaming() {
  _stopStreaming();
  _streaming = true;

  try {
    _eventSource = new EventSource("/api/system/logs/stream");

    _eventSource.onmessage = (evt) => {
      const content = document.getElementById("logContent");
      if (!content) return;

      try {
        const data = JSON.parse(evt.data);
        const text = data.line || data.message || evt.data;

        // Check filters
        for (const [level, enabled] of Object.entries(_levelFilters)) {
          if (!enabled && text.includes(level)) return;
        }

        const line = document.createElement("div");
        line.innerHTML = _colorizeLogLine(text);
        content.appendChild(line);

        // Cap at 1000 lines
        while (content.children.length > 1000) {
          content.removeChild(content.firstChild);
        }

        if (_autoScroll) {
          content.scrollTop = content.scrollHeight;
        }
      } catch {
        // Raw text
        const content2 = document.getElementById("logContent");
        if (content2) {
          content2.textContent += "\n" + evt.data;
          if (_autoScroll) content2.scrollTop = content2.scrollHeight;
        }
      }
    };

    _eventSource.onerror = () => {
      _stopStreaming();
      const btn = document.getElementById("logStreamToggle");
      if (btn) btn.textContent = "ストリーミング開始";
      const content = document.getElementById("logContent");
      if (content) content.textContent += "\n[ストリーミング接続が切断されました]";
    };
  } catch {
    _streaming = false;
    const content = document.getElementById("logContent");
    if (content) content.textContent += "\n[ストリーミングの開始に失敗しました]";
  }
}

function _stopStreaming() {
  if (_eventSource) {
    _eventSource.close();
    _eventSource = null;
  }
  _streaming = false;
}
