// ── Activity Timeline (swimlane) ─────────────
// Loaded as the default tab of activity.js tab host.

import { api } from "../modules/api.js";
import { escapeHtml } from "../modules/state.js";
import { GROUP_TYPE_CATEGORIES } from "../shared/activity-types.js";
import { renderRunningTasksStrip } from "../shared/activity-context.js";
import { t } from "/shared/i18n.js";
import { basePath } from "/shared/base-path.js";
import { renderSwimlane } from "./activity/swimlane.js";
import { renderGroupDetail } from "./activity/group-detail.js";

let _refreshInterval = null;
let _resizeObserver = null;
let _groups = [];
let _totalGroups = 0;
let _totalEvents = 0;
let _groupOffset = 0;
let _hasMore = false;
let _selectedAnima = "";
/** @type {string[]} selected group types (trigger-based) */
let _selectedGroupTypes = [];
/** @type {1|3|6|24|48} */
let _rangeHours = 1;
/** @type {string|null} */
let _selectedGroupId = null;
/** @type {string[]} */
let _animaNames = [];
let _d3LoadPromise = null;

const RANGE_OPTIONS = [1, 3, 6, 24, 48];

function _groupLimitForHours(hours) {
  // Prefer denser pages for shorter windows so the chart is not sparse
  if (hours <= 24) return 200;
  return 200;
}

// ── d3 (scaleTime only) ────────────────────

function _ensureD3() {
  if (typeof window !== "undefined" && window.d3?.scaleTime) {
    return Promise.resolve(window.d3);
  }
  if (_d3LoadPromise) return _d3LoadPromise;
  _d3LoadPromise = new Promise((resolve) => {
    if (typeof document === "undefined") {
      resolve(null);
      return;
    }
    const existing = document.querySelector("script[data-activity-d3]");
    if (existing) {
      existing.addEventListener("load", () => resolve(window.d3 || null));
      existing.addEventListener("error", () => resolve(null));
      if (window.d3) resolve(window.d3);
      return;
    }
    const script = document.createElement("script");
    script.src = `${basePath}/vendor/d3.v7.min.js`;
    script.dataset.activityD3 = "1";
    script.async = true;
    script.onload = () => resolve(window.d3 || null);
    script.onerror = () => resolve(null);
    document.head.appendChild(script);
  });
  return _d3LoadPromise;
}

// ── Render ─────────────────────────────────

export function render(container) {
  _groups = [];
  _totalGroups = 0;
  _totalEvents = 0;
  _groupOffset = 0;
  _hasMore = false;
  _selectedAnima = "";
  _selectedGroupTypes = [];
  _rangeHours = 1;
  _selectedGroupId = null;
  _animaNames = [];

  container.innerHTML = `
    <div class="activity-page activity-page--swimlane">
      <div class="activity-header">
        <h2>${t("activity.page_title")}</h2>
        <span class="activity-count" id="activityCount"></span>
      </div>

      <div class="running-tasks-strip" id="activityRunningTasks" hidden></div>

      <div class="activity-filters">
        <select class="activity-anima-select" id="activityAnimaSelect">
          <option value="">${t("activity.all_animas")}</option>
        </select>
        <div id="activityTypeChips" style="display:flex; gap:0.35rem; flex-wrap:wrap;"></div>
        <div class="activity-range-chips" id="activityRangeChips" role="group" aria-label="${t("activity.swimlane_range_label")}"></div>
      </div>

      <div class="swimlane-wrap" id="activitySwimlaneWrap">
        <div class="swimlane-scroll" id="activitySwimlaneScroll">
          <svg class="swimlane-svg" id="activitySwimlaneSvg" xmlns="http://www.w3.org/2000/svg"></svg>
        </div>
        <div class="swimlane-empty" id="activitySwimlaneEmpty" hidden>${t("activity.empty")}</div>
      </div>

      <div id="activityLoadMoreWrap"></div>

      <div class="swimlane-detail" id="activityGroupDetail" hidden></div>
    </div>
  `;

  _buildAnimaSelect();
  _buildTypeChips();
  _buildRangeChips();
  _bindDetailClose();
  _observeResize();

  _ensureD3().finally(() => {
    _loadEvents(true);
  });
  _loadRunningTasks();
  _refreshInterval = setInterval(() => {
    _loadEvents(true);
    _loadRunningTasks();
  }, 30000);
}

export function destroy() {
  if (_refreshInterval) {
    clearInterval(_refreshInterval);
    _refreshInterval = null;
  }
  if (_resizeObserver) {
    _resizeObserver.disconnect();
    _resizeObserver = null;
  }
  _groups = [];
  _selectedGroupId = null;
}

// ── Filter UI ──────────────────────────────

async function _buildAnimaSelect() {
  const sel = document.getElementById("activityAnimaSelect");
  if (!sel) return;

  try {
    const animas = await api("/api/animas");
    _animaNames = (animas || []).map((a) => a.name).filter(Boolean);
    for (const name of _animaNames) {
      const opt = document.createElement("option");
      opt.value = name;
      opt.textContent = name;
      sel.appendChild(opt);
    }
  } catch (err) {
    console.error("Failed to load animas for activity filter:", err);
  }

  sel.addEventListener("change", () => {
    _selectedAnima = sel.value;
    _groupOffset = 0;
    _groups = [];
    _selectedGroupId = null;
    _loadEvents(true);
    _loadRunningTasks();
  });
}

function _buildTypeChips() {
  const wrap = document.getElementById("activityTypeChips");
  if (!wrap) return;

  wrap.innerHTML = "";
  for (let i = 0; i < GROUP_TYPE_CATEGORIES.length; i++) {
    const chip = GROUP_TYPE_CATEGORIES[i];
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "activity-type-chip" + (i === 0 ? " active" : "");
    btn.textContent = chip.i18nKey ? t(chip.i18nKey) : chip.label;
    btn.dataset.index = String(i);

    btn.addEventListener("click", () => {
      const allChip = wrap.querySelector('[data-index="0"]');

      if (i === 0) {
        for (const b of wrap.querySelectorAll(".activity-type-chip")) {
          b.classList.remove("active");
        }
        btn.classList.add("active");
        _selectedGroupTypes = [];
      } else {
        allChip?.classList.remove("active");
        btn.classList.toggle("active");

        _selectedGroupTypes = [];
        for (const b of wrap.querySelectorAll(".activity-type-chip.active")) {
          const idx = parseInt(b.dataset.index, 10);
          if (idx > 0 && GROUP_TYPE_CATEGORIES[idx]?.groupTypes) {
            _selectedGroupTypes.push(...GROUP_TYPE_CATEGORIES[idx].groupTypes);
          }
        }

        if (_selectedGroupTypes.length === 0) {
          allChip?.classList.add("active");
        }
      }

      _groupOffset = 0;
      _groups = [];
      _selectedGroupId = null;
      _loadEvents(true);
    });

    wrap.appendChild(btn);
  }
}

function _buildRangeChips() {
  const wrap = document.getElementById("activityRangeChips");
  if (!wrap) return;
  wrap.innerHTML = "";

  for (const hours of RANGE_OPTIONS) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "activity-range-chip" + (hours === _rangeHours ? " active" : "");
    btn.dataset.hours = String(hours);
    btn.textContent = t(`activity.swimlane_range_${hours}h`);
    btn.addEventListener("click", () => {
      if (_rangeHours === hours) return;
      _rangeHours = hours;
      for (const b of wrap.querySelectorAll(".activity-range-chip")) {
        b.classList.toggle("active", parseInt(b.dataset.hours, 10) === hours);
      }
      _groupOffset = 0;
      _groups = [];
      _selectedGroupId = null;
      _loadEvents(true);
    });
    wrap.appendChild(btn);
  }
}

function _bindDetailClose() {
  const detail = document.getElementById("activityGroupDetail");
  if (!detail) return;
  detail.addEventListener("swimlane-detail-close", () => {
    _selectedGroupId = null;
    _paintSwimlane();
  });
}

function _observeResize() {
  const wrap = document.getElementById("activitySwimlaneWrap");
  if (!wrap || typeof ResizeObserver === "undefined") return;
  _resizeObserver = new ResizeObserver(() => {
    _paintSwimlane();
  });
  _resizeObserver.observe(wrap);
}

// ── Data Loading ───────────────────────────

async function _loadEvents(reset) {
  if (reset) {
    _groupOffset = 0;
    // Keep previously selected group id for restore after redraw
  }

  const limit = _groupLimitForHours(_rangeHours);
  let url = `/api/activity/recent?hours=${_rangeHours}&grouped=true&group_limit=${limit}&group_offset=${_groupOffset}`;
  if (_selectedAnima) {
    url += `&anima=${encodeURIComponent(_selectedAnima)}`;
  }
  if (_selectedGroupTypes.length > 0) {
    url += `&group_type=${encodeURIComponent(_selectedGroupTypes.join(","))}`;
  }

  const wrap = document.getElementById("activitySwimlaneWrap");

  try {
    const data = await api(url);
    const newGroups = data.groups || [];
    _totalGroups = data.total_groups ?? 0;
    _totalEvents = data.total_events ?? 0;
    _hasMore = data.has_more ?? false;

    if (reset) {
      _groups = newGroups;
    } else {
      // Append older groups (dedupe by id)
      const seen = new Set(_groups.map((g) => g.id));
      for (const g of newGroups) {
        if (!seen.has(g.id)) _groups.push(g);
      }
    }

    _groupOffset = _groups.length;
    _paintSwimlane();
    _updateCount();
    _renderLoadMore();
    _restoreDetail();
  } catch (err) {
    if (wrap) {
      const empty = document.getElementById("activitySwimlaneEmpty");
      if (empty) {
        empty.hidden = false;
        empty.textContent = `${t("activity.load_failed")}: ${escapeHtml(err.message)}`;
      }
    }
  }
}

async function _loadRunningTasks() {
  const strip = document.getElementById("activityRunningTasks");
  if (!strip) return;
  let url = "/api/activity/running-tasks";
  if (_selectedAnima) url += `?anima=${encodeURIComponent(_selectedAnima)}`;
  try {
    const data = await api(url);
    renderRunningTasksStrip(strip, data, t);
  } catch (err) {
    console.error("Failed to load running activity tasks:", err);
  }
}

// ── Rendering ──────────────────────────────

function _updateCount() {
  const el = document.getElementById("activityCount");
  if (el) {
    el.textContent = t("activity.count_display", { groups: _groups.length, events: _totalEvents });
  }
}

function _timeRange() {
  const nowMs = Date.now();
  const hours = _rangeHours;
  return {
    startMs: nowMs - hours * 3600_000,
    endMs: nowMs,
    hours,
  };
}

function _paintSwimlane() {
  const svg = document.getElementById("activitySwimlaneSvg");
  const empty = document.getElementById("activitySwimlaneEmpty");
  const scroll = document.getElementById("activitySwimlaneScroll");
  if (!svg) return;

  if (_groups.length === 0) {
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    if (empty) {
      empty.hidden = false;
      empty.textContent = t("activity.empty");
    }
    if (scroll) scroll.hidden = true;
    return;
  }

  if (empty) empty.hidden = true;
  if (scroll) scroll.hidden = false;

  const width = scroll?.clientWidth || svg.parentElement?.clientWidth || 800;
  const animaOrder = _selectedAnima ? [_selectedAnima] : _animaNames;

  renderSwimlane(svg, _groups, {
    timeRange: _timeRange(),
    nowMs: Date.now(),
    selectedGroupId: _selectedGroupId,
    animaOrder,
    width,
    onSelectGroup: (grp) => {
      if (!grp) {
        _selectedGroupId = null;
        renderGroupDetail(document.getElementById("activityGroupDetail"), null);
        _paintSwimlane();
        return;
      }
      _selectedGroupId = grp.id;
      renderGroupDetail(document.getElementById("activityGroupDetail"), grp);
      _paintSwimlane();
    },
  });
}

function _restoreDetail() {
  const detail = document.getElementById("activityGroupDetail");
  if (!_selectedGroupId || !detail) {
    renderGroupDetail(detail, null);
    return;
  }
  const grp = _groups.find((g) => g.id === _selectedGroupId);
  if (grp) {
    renderGroupDetail(detail, grp);
  } else {
    _selectedGroupId = null;
    renderGroupDetail(detail, null);
  }
}

function _renderLoadMore() {
  const wrap = document.getElementById("activityLoadMoreWrap");
  if (!wrap) return;

  if (_hasMore) {
    wrap.innerHTML = `<button type="button" class="activity-load-more" id="activityLoadMoreBtn">${t("activity.swimlane_load_earlier", { current: _groups.length, total: _totalGroups })}</button>`;
    const btn = document.getElementById("activityLoadMoreBtn");
    if (btn) {
      btn.addEventListener("click", () => {
        btn.disabled = true;
        btn.textContent = t("common.loading");
        _loadEvents(false);
      });
    }
  } else {
    wrap.innerHTML = "";
  }
}
