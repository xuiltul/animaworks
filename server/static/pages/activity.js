// ── Activity Timeline Page ──────────────────
import { api } from "../modules/api.js";
import { state, escapeHtml, smartTimestamp, renderMarkdown } from "../modules/state.js";
import { getIcon, getDisplaySummary, TYPE_CATEGORIES } from "../shared/activity-types.js";

let _refreshInterval = null;
let _events = [];
let _total = 0;
let _offset = 0;
let _hasMore = false;
let _expandedId = null;
let _selectedAnima = "";
let _selectedTypes = [];

const LIMIT = 200;

// ── Render ─────────────────────────────────

export function render(container) {
  _events = [];
  _total = 0;
  _offset = 0;
  _hasMore = false;
  _expandedId = null;
  _selectedAnima = "";
  _selectedTypes = [];

  container.innerHTML = `
    <div class="activity-page">
      <div class="activity-header">
        <h2>アクティビティタイムライン</h2>
        <span class="activity-count" id="activityCount"></span>
      </div>

      <div class="activity-filters">
        <select class="activity-anima-select" id="activityAnimaSelect">
          <option value="">全Anima</option>
        </select>
        <div id="activityTypeChips" style="display:flex; gap:0.35rem; flex-wrap:wrap;"></div>
      </div>

      <div class="activity-list" id="activityList">
        <div class="loading-placeholder">読み込み中...</div>
      </div>

      <div id="activityLoadMoreWrap"></div>
    </div>
  `;

  _buildAnimaSelect();
  _buildTypeChips();
  _loadEvents(true);
  _refreshInterval = setInterval(() => _loadEvents(true), 30000);
}

export function destroy() {
  if (_refreshInterval) {
    clearInterval(_refreshInterval);
    _refreshInterval = null;
  }
}

// ── Filter UI ──────────────────────────────

function _buildAnimaSelect() {
  const sel = document.getElementById("activityAnimaSelect");
  if (!sel) return;

  const animas = state.animas || [];
  for (const a of animas) {
    const opt = document.createElement("option");
    opt.value = a.name;
    opt.textContent = a.name;
    sel.appendChild(opt);
  }

  sel.addEventListener("change", () => {
    _selectedAnima = sel.value;
    _offset = 0;
    _events = [];
    _expandedId = null;
    _loadEvents(true);
  });
}

function _buildTypeChips() {
  const wrap = document.getElementById("activityTypeChips");
  if (!wrap) return;

  wrap.innerHTML = "";
  for (let i = 0; i < TYPE_CATEGORIES.length; i++) {
    const chip = TYPE_CATEGORIES[i];
    const btn = document.createElement("button");
    btn.className = "activity-type-chip" + (i === 0 ? " active" : "");
    btn.textContent = chip.label;
    btn.dataset.index = i;

    btn.addEventListener("click", () => {
      // Toggle active state
      const allChip = wrap.querySelector('[data-index="0"]');

      if (i === 0) {
        // "All" clicked — deactivate everything else
        for (const b of wrap.querySelectorAll(".activity-type-chip")) {
          b.classList.remove("active");
        }
        btn.classList.add("active");
        _selectedTypes = [];
      } else {
        // Specific chip toggled
        allChip.classList.remove("active");
        btn.classList.toggle("active");

        // Collect active types
        _selectedTypes = [];
        for (const b of wrap.querySelectorAll(".activity-type-chip.active")) {
          const idx = parseInt(b.dataset.index);
          if (idx > 0) {
            _selectedTypes.push(...TYPE_CATEGORIES[idx].types);
          }
        }

        // If nothing selected, revert to "All"
        if (_selectedTypes.length === 0) {
          allChip.classList.add("active");
        }
      }

      _offset = 0;
      _events = [];
      _expandedId = null;
      _loadEvents(true);
    });

    wrap.appendChild(btn);
  }
}

// ── Data Loading ───────────────────────────

async function _loadEvents(reset) {
  if (reset) {
    _offset = 0;
    _events = [];
    _expandedId = null;
  }

  let url = `/api/activity/recent?hours=48&limit=${LIMIT}&offset=${_offset}`;
  if (_selectedAnima) {
    url += `&anima=${encodeURIComponent(_selectedAnima)}`;
  }
  if (_selectedTypes.length > 0) {
    url += `&event_type=${encodeURIComponent(_selectedTypes.join(","))}`;
  }

  const list = document.getElementById("activityList");

  try {
    const data = await api(url);
    const newEvents = data.events || [];
    _total = data.total ?? 0;
    _hasMore = data.has_more ?? false;

    if (reset) {
      _events = newEvents;
    } else {
      _events = _events.concat(newEvents);
    }

    _offset = _events.length;
    _renderList();
    _updateCount();
    _renderLoadMore();
  } catch (err) {
    if (list) {
      list.innerHTML = `<div class="activity-empty">読み込み失敗: ${escapeHtml(err.message)}</div>`;
    }
  }
}

// ── Rendering ──────────────────────────────

function _updateCount() {
  const el = document.getElementById("activityCount");
  if (el) {
    el.textContent = `[${_events.length}/${_total}]`;
  }
}

function _renderList() {
  const list = document.getElementById("activityList");
  if (!list) return;

  if (_events.length === 0) {
    list.innerHTML = '<div class="activity-empty">アクティビティはまだありません</div>';
    return;
  }

  list.innerHTML = "";
  for (const evt of _events) {
    const row = _createRow(evt);
    list.appendChild(row);

    if (evt.id === _expandedId) {
      const detail = _createDetail(evt);
      list.appendChild(detail);
    }
  }
}

function _createRow(evt) {
  const row = document.createElement("div");
  row.className = "activity-row" + (evt.id === _expandedId ? " expanded" : "");

  const icon = getIcon(evt.type);
  const time = smartTimestamp(evt.ts);
  const anima = evt.anima || "";
  const summary = getDisplaySummary(evt);

  row.innerHTML =
    `<span class="activity-row-time">${escapeHtml(time)}</span>` +
    `<span class="activity-row-icon">${icon}</span>` +
    `<span class="activity-row-anima">${escapeHtml(anima)}</span>` +
    `<span class="activity-row-summary">${escapeHtml(summary)}</span>`;

  row.addEventListener("click", () => {
    if (_expandedId === evt.id) {
      _expandedId = null;
    } else {
      _expandedId = evt.id;
    }
    _renderList();
  });

  return row;
}

function _createDetail(evt) {
  const detail = document.createElement("div");
  detail.className = "activity-detail";

  let html = "";

  // Content (fallback to summary for heartbeat etc. where content is empty)
  const detailText = evt.content || (evt.summary && evt.summary.length > 80 ? evt.summary : "");
  if (detailText) {
    html += `<div class="activity-detail-content activity-markdown">${renderMarkdown(detailText)}</div>`;
  }

  // Metadata fields
  const fields = [
    ["Type", evt.type],
    ["From", evt.from_person],
    ["To", evt.to_person],
    ["Channel", evt.channel],
    ["Tool", evt.tool],
    ["Via", evt.via],
  ];

  for (const [label, value] of fields) {
    if (value) {
      html += `<div class="activity-detail-field"><span class="activity-detail-label">${label}:</span>${escapeHtml(value)}</div>`;
    }
  }

  // Meta (JSON)
  if (evt.meta && Object.keys(evt.meta).length > 0) {
    html += `<div class="activity-detail-field"><span class="activity-detail-label">Meta:</span></div>`;
    html += `<div class="activity-meta">${escapeHtml(JSON.stringify(evt.meta, null, 2))}</div>`;
  }

  if (!html) {
    html = '<div class="activity-detail-field" style="color:var(--text-secondary,#aaa);">詳細情報なし</div>';
  }

  detail.innerHTML = html;
  return detail;
}

function _renderLoadMore() {
  const wrap = document.getElementById("activityLoadMoreWrap");
  if (!wrap) return;

  if (_hasMore) {
    wrap.innerHTML = `<button class="activity-load-more" id="activityLoadMoreBtn">もっと読み込む (${_events.length}/${_total}件)</button>`;
    const btn = document.getElementById("activityLoadMoreBtn");
    if (btn) {
      btn.addEventListener("click", () => {
        btn.disabled = true;
        btn.textContent = "読み込み中...";
        _loadEvents(false);
      });
    }
  } else {
    wrap.innerHTML = "";
  }
}
