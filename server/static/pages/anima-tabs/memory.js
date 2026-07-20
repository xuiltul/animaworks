// ── Anima detail tab: Memory Browser ────────
// Absorbed from pages/memory.js (Phase 4).
// Sub-tabs: episodes / knowledge / procedures / graph / calendar.

import { api } from "../../modules/api.js";
import { escapeAttr, escapeHtml, renderMarkdown, renderSafeMarkdown } from "../../modules/state.js";
import { createPageTabs } from "../../shared/page-tabs.js";
import { basePath } from "/shared/base-path.js";
import { getLocale, t } from "/shared/i18n.js";

let _selectedAnima = null;
let _activeTab = "episodes";
let _viewMode = "list"; // "list" | "content"
let _container = null;
let _d3LoadPromise = null;
let _graphSimulation = null;
let _viewRequestId = 0;
let _detailRequestId = 0;
let _calendarCursor = _startOfMonth(new Date());
let _subTabs = null;

const _TAB_META = {
  episodes: { icon: "📝", labelKey: "chat.memory_episodes" },
  knowledge: { icon: "📘", labelKey: "chat.memory_knowledge" },
  procedures: { icon: "📑", labelKey: "chat.memory_procedures" },
  graph: { icon: "🕸️", labelKey: "memory.graph_tab" },
  calendar: { icon: "📅", labelKey: "memory.calendar_tab" },
};

const _GRAPH_NODE_COLORS = {
  knowledge: "#4f7cac",
  procedures: "#d97706",
};

function _extractStatsCount(value) {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (value && typeof value === "object") {
    const count = value.count;
    if (typeof count === "number" && Number.isFinite(count)) return count;
  }
  return 0;
}

function _tabLabel(tab, count = null) {
  const meta = _TAB_META[tab];
  if (!meta) return "";
  const base = `${meta.icon} ${t(meta.labelKey)}`;
  return count === null ? base : `${base} (${count})`;
}

function _setTabLabel(tab, count = null) {
  const btn =
    _subTabs?.el?.querySelector(`.page-tab[data-tab="${tab}"]`) ||
    _container?.querySelector(`.page-tab[data-tab="${tab}"]`);
  if (!btn) return;
  btn.textContent = _tabLabel(tab, count);
}

function _resetTabLabels() {
  for (const tab of Object.keys(_TAB_META)) _setTabLabel(tab, null);
}

function _startOfMonth(value) {
  return new Date(value.getFullYear(), value.getMonth(), 1);
}

function _setActiveTab(tab) {
  _activeTab = tab;
  if (_subTabs) {
    _subTabs.setActive(tab);
  } else {
    _container?.querySelectorAll(".page-tab").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.tab === tab);
    });
  }
}

function _stopGraphSimulation() {
  if (_graphSimulation) {
    _graphSimulation.stop();
    _graphSimulation = null;
  }
}

function _loadD3() {
  if (window.d3) return Promise.resolve(window.d3);
  if (_d3LoadPromise) return _d3LoadPromise;

  _d3LoadPromise = new Promise((resolve, reject) => {
    const existing = document.querySelector("script[data-memory-d3]");
    const script = existing || document.createElement("script");

    const handleLoad = () => {
      if (window.d3) {
        resolve(window.d3);
      } else {
        script.remove();
        _d3LoadPromise = null;
        reject(new Error(t("memory.graph_d3_failed")));
      }
    };
    const handleError = () => {
      script.remove();
      _d3LoadPromise = null;
      reject(new Error(t("memory.graph_d3_failed")));
    };

    script.addEventListener("load", handleLoad, { once: true });
    script.addEventListener("error", handleError, { once: true });
    if (!existing) {
      script.src = `${basePath}/vendor/d3.v7.min.js`;
      script.async = true;
      script.dataset.memoryD3 = "true";
      document.head.appendChild(script);
    }
  });

  return _d3LoadPromise;
}

/**
 * @param {HTMLElement} container
 * @param {{ animaName: string }} opts
 */
export function render(container, { animaName } = {}) {
  _container = container;
  _selectedAnima = animaName || null;
  _activeTab = "episodes";
  _viewMode = "list";
  _calendarCursor = _startOfMonth(new Date());
  _viewRequestId += 1;

  if (_subTabs) {
    try {
      _subTabs.destroy();
    } catch {
      /* ignore */
    }
    _subTabs = null;
  }

  container.innerHTML = `
    <div id="memorySubTabsHost"></div>
    <div class="card memory-page-card">
      <div class="card-body memory-page-content" id="memoryMainContent">
        <div class="loading-placeholder">${_selectedAnima ? t("common.loading") : t("assets.select_anima")}</div>
      </div>
    </div>
  `;

  const tabsHost = container.querySelector("#memorySubTabsHost");
  if (tabsHost) {
    _subTabs = createPageTabs({
      tabs: Object.keys(_TAB_META).map((tab) => ({
        id: tab,
        label: _tabLabel(tab),
      })),
      container: tabsHost,
      activeId: _activeTab,
      onChange: (id) => {
        if (_activeTab === "graph" && id !== "graph") _stopGraphSimulation();
        _activeTab = id;
        _viewMode = "list";
        _loadActiveView();
      },
    });
    _subTabs.el.classList.add("memory-page-tabs");
  }

  _resetTabLabels();
  _updateTabCounts();
  _loadActiveView();
}

export function destroy() {
  _viewRequestId += 1;
  _detailRequestId += 1;
  _stopGraphSimulation();
  if (_subTabs) {
    try {
      _subTabs.destroy();
    } catch {
      /* ignore */
    }
    _subTabs = null;
  }
  _container = null;
  _selectedAnima = null;
}

// ── Data Loading ───────────────────────────

async function _updateTabCounts() {
  if (!_selectedAnima) {
    _resetTabLabels();
    return;
  }

  let epCount = 0;
  let knCount = 0;
  let prCount = 0;

  try {
    const stats = await api(`/api/animas/${encodeURIComponent(_selectedAnima)}/memory/stats`);
    epCount = _extractStatsCount(stats.episodes);
    knCount = _extractStatsCount(stats.knowledge);
    prCount = _extractStatsCount(stats.procedures);
  } catch {
    try {
      const [episodes, knowledge, procedures] = await Promise.all([
        api(`/api/animas/${encodeURIComponent(_selectedAnima)}/episodes`).catch(() => ({ files: [] })),
        api(`/api/animas/${encodeURIComponent(_selectedAnima)}/knowledge`).catch(() => ({ files: [] })),
        api(`/api/animas/${encodeURIComponent(_selectedAnima)}/procedures`).catch(() => ({ files: [] })),
      ]);
      epCount = (episodes.files || []).length;
      knCount = (knowledge.files || []).length;
      prCount = (procedures.files || []).length;
    } catch {
      /* keep zero counts */
    }
  }
  _setTabLabel("episodes", epCount);
  _setTabLabel("knowledge", knCount);
  _setTabLabel("procedures", prCount);
}

function _loadActiveView() {
  _viewRequestId += 1;
  if (_activeTab === "graph") return _loadGraph(_viewRequestId);
  if (_activeTab === "calendar") return _loadCalendar(_viewRequestId);
  return _loadFileList(_viewRequestId);
}

async function _loadFileList(requestId = ++_viewRequestId) {
  const content = document.getElementById("memoryMainContent");
  if (!content) return;

  if (!_selectedAnima) {
    content.innerHTML = `<div class="loading-placeholder">${t("assets.select_anima")}</div>`;
    return;
  }

  content.innerHTML = `<div class="loading-placeholder">${t("common.loading")}</div>`;
  const endpoint = `/api/animas/${encodeURIComponent(_selectedAnima)}/${_activeTab}`;

  try {
    const data = await api(endpoint);
    if (requestId !== _viewRequestId || !_container) return;
    const files = data.files || [];

    if (files.length === 0) {
      content.innerHTML = `<div class="loading-placeholder">${t("memory.no_files")}</div>`;
      return;
    }

    content.innerHTML = files
      .map(
        (file) => `
      <div class="memory-file-item memory-page-file" data-file="${escapeAttr(file)}">
        ${escapeHtml(file)}
      </div>
    `,
      )
      .join("");

    content.querySelectorAll(".memory-file-item").forEach((item) => {
      item.addEventListener("click", () => _loadFileContent(item.dataset.file));
    });
  } catch (err) {
    if (requestId !== _viewRequestId) return;
    content.innerHTML = `<div class="loading-placeholder">${t("memory.fetch_failed")}: ${escapeHtml(err.message)}</div>`;
  }
}

async function _loadFileContent(file, memoryType = _activeTab, returnTab = _activeTab) {
  const content = document.getElementById("memoryMainContent");
  if (!content || !_selectedAnima) return;

  _viewRequestId += 1;
  const requestId = _viewRequestId;
  _viewMode = "content";
  _stopGraphSimulation();

  content.innerHTML = `
    <div>
      <button class="btn-secondary" id="memoryBackToList">&larr; ${t("animas.back")}</button>
      <h3 class="memory-file-title">${escapeHtml(file)}</h3>
      <div id="memoryFileBody" class="loading-placeholder">${t("common.loading")}</div>
    </div>
  `;

  document.getElementById("memoryBackToList")?.addEventListener("click", () => {
    _viewMode = "list";
    _setActiveTab(returnTab);
    _loadActiveView();
  });

  const endpoint = `/api/animas/${encodeURIComponent(_selectedAnima)}/${memoryType}/${encodeURIComponent(file)}`;

  try {
    const data = await api(endpoint);
    if (requestId !== _viewRequestId) return;
    const body = document.getElementById("memoryFileBody");
    if (body) {
      const raw = data.content || t("chat.no_content");
      body.className = "memory-file-body";
      body.innerHTML = renderMarkdown(raw);
    }
  } catch (err) {
    if (requestId !== _viewRequestId) return;
    const body = document.getElementById("memoryFileBody");
    if (body) {
      body.className = "loading-placeholder";
      body.textContent = `${t("memory.fetch_failed")}: ${err.message}`;
    }
  }
}

// ── Graph View ─────────────────────────────

async function _loadGraph(requestId) {
  const content = document.getElementById("memoryMainContent");
  if (!content) return;
  _stopGraphSimulation();

  if (!_selectedAnima) {
    content.innerHTML = `<div class="loading-placeholder">${t("assets.select_anima")}</div>`;
    return;
  }

  content.innerHTML = `<div class="loading-placeholder">${t("memory.graph_loading")}</div>`;
  try {
    const [data, d3] = await Promise.all([
      api(`/api/animas/${encodeURIComponent(_selectedAnima)}/memory/graph`),
      _loadD3(),
    ]);
    if (requestId !== _viewRequestId || _activeTab !== "graph" || !_container) return;
    _renderGraphView(data, d3);
  } catch (err) {
    if (requestId !== _viewRequestId) return;
    content.innerHTML = `<div class="loading-placeholder">${t("memory.fetch_failed")}: ${escapeHtml(err.message)}</div>`;
  }
}

function _renderGraphView(data, d3) {
  const content = document.getElementById("memoryMainContent");
  if (!content) return;
  const nodes = Array.isArray(data.nodes) ? data.nodes : [];
  const edges = Array.isArray(data.edges) ? data.edges : [];

  if (nodes.length === 0) {
    content.innerHTML = `<div class="memory-empty-state">${t("memory.empty")}</div>`;
    return;
  }

  content.innerHTML = `
    <div class="memory-graph-view">
      <div class="memory-graph-toolbar">
        <label for="memorySimilarityThreshold">
          ${t("memory.similarity_threshold")}
          <output id="memorySimilarityValue">0.00</output>
        </label>
        <input id="memorySimilarityThreshold" type="range" min="0" max="1" step="0.05" value="0">
        <div class="memory-graph-badges">
          ${data.partial ? `<span class="memory-status-badge partial">${t("memory.graph_partial")}</span>` : ""}
          ${data.edges_capped ? `<span class="memory-status-badge capped">${t("memory.graph_edges_capped")}</span>` : ""}
        </div>
      </div>
      <div class="memory-graph-layout">
        <div class="memory-graph-stage" id="memoryGraphStage" aria-label="${escapeAttr(t("memory.graph_aria"))}"></div>
        <aside class="memory-graph-detail" id="memoryGraphDetail">
          <div class="memory-detail-placeholder">${t("memory.graph_select_node")}</div>
        </aside>
      </div>
    </div>
  `;

  const slider = document.getElementById("memorySimilarityThreshold");
  const output = document.getElementById("memorySimilarityValue");
  const updateOutput = () => {
    if (output) output.textContent = Number(slider?.value || 0).toFixed(2);
  };
  const draw = () => {
    const threshold = Number(slider?.value || 0);
    updateOutput();
    _drawGraph(d3, nodes, edges, threshold);
  };
  slider?.addEventListener("input", updateOutput);
  slider?.addEventListener("change", draw);
  draw();
}

function _drawGraph(d3, rawNodes, rawEdges, threshold) {
  const stage = document.getElementById("memoryGraphStage");
  if (!stage) return;
  _stopGraphSimulation();
  stage.replaceChildren();

  const nodes = rawNodes.map((node) => ({ ...node }));
  const nodeIds = new Set(nodes.map((node) => node.id));
  const edges = rawEdges
    .filter((edge) => edge.link_type === "explicit" || Number(edge.similarity || 0) >= threshold)
    .filter((edge) => nodeIds.has(String(edge.source)) && nodeIds.has(String(edge.target)))
    .map((edge) => ({ ...edge, source: String(edge.source), target: String(edge.target) }));
  const degrees = new Map(nodes.map((node) => [node.id, 0]));
  for (const edge of edges) {
    degrees.set(edge.source, (degrees.get(edge.source) || 0) + 1);
    degrees.set(edge.target, (degrees.get(edge.target) || 0) + 1);
  }

  const bounds = stage.getBoundingClientRect();
  const width = Math.max(480, Math.floor(bounds.width || 760));
  const height = Math.max(420, Math.floor(bounds.height || 520));
  const radius = (node) => 7 + Math.min(12, Math.sqrt(degrees.get(node.id) || 0) * 3);

  const svg = d3
    .select(stage)
    .append("svg")
    .attr("viewBox", [0, 0, width, height])
    .attr("role", "img")
    .attr("aria-label", t("memory.graph_aria"));
  const viewport = svg.append("g");
  svg.call(
    d3
      .zoom()
      .scaleExtent([0.25, 5])
      .on("zoom", (event) => {
        viewport.attr("transform", event.transform);
      }),
  );

  const links = viewport
    .append("g")
    .attr("class", "memory-graph-links")
    .selectAll("line")
    .data(edges)
    .join("line")
    .attr("class", (edge) => (edge.link_type === "explicit" ? "explicit" : "similarity"))
    .attr("stroke-opacity", (edge) =>
      edge.link_type === "explicit"
        ? 0.75
        : 0.18 + Math.max(0, Math.min(1, Number(edge.similarity || 0))) * 0.62,
    );

  const nodeGroups = viewport
    .append("g")
    .attr("class", "memory-graph-nodes")
    .selectAll("g")
    .data(nodes)
    .join("g")
    .attr("tabindex", 0)
    .attr("role", "button")
    .attr("aria-label", (node) => `${node.stem}, ${node.memory_type}`)
    .on("click", (_event, node) => _showNodeDetail(node))
    .on("keydown", (event, node) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        _showNodeDetail(node);
      }
    });

  nodeGroups
    .append("circle")
    .attr("r", radius)
    .attr("fill", (node) => _GRAPH_NODE_COLORS[node.memory_type] || "#6b7280");
  nodeGroups
    .append("text")
    .attr("x", (node) => radius(node) + 4)
    .attr("y", 4)
    .text((node) => node.stem);
  nodeGroups.append("title").text((node) => node.stem);

  const simulation = d3
    .forceSimulation(nodes)
    .force("link", d3.forceLink(edges).id((node) => node.id).distance(82).strength(0.35))
    .force("charge", d3.forceManyBody().strength(-180))
    .force("center", d3.forceCenter(width / 2, height / 2))
    .force(
      "collision",
      d3.forceCollide().radius((node) => radius(node) + 12),
    )
    .on("tick", () => {
      links
        .attr("x1", (edge) => edge.source.x)
        .attr("y1", (edge) => edge.source.y)
        .attr("x2", (edge) => edge.target.x)
        .attr("y2", (edge) => edge.target.y);
      nodeGroups.attr("transform", (node) => `translate(${node.x},${node.y})`);
    });

  nodeGroups.call(
    d3
      .drag()
      .on("start", (event, node) => {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        node.fx = node.x;
        node.fy = node.y;
      })
      .on("drag", (event, node) => {
        node.fx = event.x;
        node.fy = event.y;
      })
      .on("end", (event, node) => {
        if (!event.active) simulation.alphaTarget(0);
        node.fx = null;
        node.fy = null;
      }),
  );

  _graphSimulation = simulation;
}

function _formatMetadataValue(value) {
  if (value === null || value === undefined || value === "") return t("memory.not_available");
  if (typeof value === "number") return value.toFixed(2).replace(/\.00$/, "");
  return String(value);
}

function _episodeDate(value) {
  const match = String(value || "").match(/\d{4}-\d{2}-\d{2}/);
  return match ? match[0] : null;
}

function _withoutFrontmatter(content) {
  return String(content || "")
    .replace(/^---\s*\n[\s\S]*?\n---\s*\n?/, "")
    .trim();
}

async function _showNodeDetail(node) {
  const panel = document.getElementById("memoryGraphDetail");
  if (!panel || !_selectedAnima) return;
  const requestId = ++_detailRequestId;
  const sourceEpisodes = Array.isArray(node.source_episodes)
    ? node.source_episodes.map(_episodeDate).filter(Boolean)
    : [];

  panel.innerHTML = `
    <h3>${escapeHtml(node.stem)}</h3>
    <dl class="memory-detail-meta">
      <div><dt>${t("memory.detail_type")}</dt><dd>${escapeHtml(t(`memory.type_${node.memory_type}`))}</dd></div>
      <div><dt>${t("memory.detail_created")}</dt><dd>${escapeHtml(_formatMetadataValue(node.created_at))}</dd></div>
      <div><dt>${t("memory.detail_updated")}</dt><dd>${escapeHtml(_formatMetadataValue(node.updated_at))}</dd></div>
      <div><dt>${t("memory.detail_confidence")}</dt><dd>${escapeHtml(_formatMetadataValue(node.confidence))}</dd></div>
    </dl>
    <section>
      <h4>${t("memory.detail_excerpt")}</h4>
      <div class="memory-detail-excerpt loading-placeholder" id="memoryDetailExcerpt">${t("common.loading")}</div>
    </section>
    <section>
      <h4>${t("memory.detail_sources")}</h4>
      <div class="memory-source-links">
        ${
          sourceEpisodes.length > 0
            ? sourceEpisodes.map((date) => `<button type="button" data-episode-date="${date}">${date}</button>`).join("")
            : `<span>${t("memory.detail_no_sources")}</span>`
        }
      </div>
    </section>
  `;

  panel.querySelectorAll("[data-episode-date]").forEach((button) => {
    button.addEventListener("click", () => _openEpisodeFromGraph(button.dataset.episodeDate));
  });

  const memoryType = node.memory_type === "procedures" ? "procedures" : "knowledge";
  try {
    const data = await api(
      `/api/animas/${encodeURIComponent(_selectedAnima)}/${memoryType}/${encodeURIComponent(node.stem)}`,
    );
    if (requestId !== _detailRequestId) return;
    const excerpt = document.getElementById("memoryDetailExcerpt");
    if (!excerpt) return;
    const raw = _withoutFrontmatter(data.content);
    const shortened = raw.length > 700 ? `${raw.slice(0, 700)}…` : raw;
    excerpt.className = "memory-detail-excerpt";
    excerpt.innerHTML = shortened ? renderSafeMarkdown(shortened) : escapeHtml(t("chat.no_content"));
  } catch (err) {
    if (requestId !== _detailRequestId) return;
    const excerpt = document.getElementById("memoryDetailExcerpt");
    if (excerpt) {
      excerpt.className = "memory-detail-excerpt loading-placeholder";
      excerpt.textContent = `${t("memory.fetch_failed")}: ${err.message}`;
    }
  }
}

function _openEpisodeFromGraph(date) {
  const parsed = new Date(`${date}T00:00:00`);
  if (!Number.isNaN(parsed.getTime())) _calendarCursor = _startOfMonth(parsed);
  _setActiveTab("calendar");
  _loadFileContent(date, "episodes", "calendar");
}

// ── Calendar View ──────────────────────────

function _calendarMonthLabel(year, month) {
  return new Intl.DateTimeFormat(getLocale(), { year: "numeric", month: "long" }).format(
    new Date(year, month - 1, 1),
  );
}

async function _loadCalendar(requestId) {
  const content = document.getElementById("memoryMainContent");
  if (!content) return;
  if (!_selectedAnima) {
    content.innerHTML = `<div class="loading-placeholder">${t("assets.select_anima")}</div>`;
    return;
  }

  const year = _calendarCursor.getFullYear();
  const month = _calendarCursor.getMonth() + 1;
  content.innerHTML = `<div class="loading-placeholder">${t("common.loading")}</div>`;
  try {
    const data = await api(
      `/api/animas/${encodeURIComponent(_selectedAnima)}/episodes/calendar?year=${year}&month=${month}`,
    );
    if (requestId !== _viewRequestId || _activeTab !== "calendar" || !_container) return;
    _renderCalendar(data);
  } catch (err) {
    if (requestId !== _viewRequestId) return;
    content.innerHTML = `<div class="loading-placeholder">${t("memory.fetch_failed")}: ${escapeHtml(err.message)}</div>`;
  }
}

function _renderCalendar(data) {
  const content = document.getElementById("memoryMainContent");
  if (!content) return;
  const firstWeekday = new Date(data.year, data.month - 1, 1).getDay();
  const today = new Date();
  const todayKey = [
    today.getFullYear(),
    String(today.getMonth() + 1).padStart(2, "0"),
    String(today.getDate()).padStart(2, "0"),
  ].join("-");
  const weekdayKeys = ["sun", "mon", "tue", "wed", "thu", "fri", "sat"];
  const blanks = Array.from(
    { length: firstWeekday },
    () => `<div class="memory-calendar-day blank" aria-hidden="true"></div>`,
  ).join("");
  const days = (data.days || [])
    .map((day) => {
      const dayNumber = Number(day.date.slice(-2));
      const classes = ["memory-calendar-day"];
      if (day.has_episode) classes.push("has-episode");
      if (day.date === todayKey) classes.push("today");
      if (day.has_episode) {
        return `<button type="button" class="${classes.join(" ")}" data-date="${day.date}" aria-label="${escapeAttr(t("memory.calendar_open", { date: day.date }))}">
        <span>${dayNumber}</span><span class="memory-calendar-dot" aria-hidden="true"></span>
      </button>`;
      }
      return `<div class="${classes.join(" ")}"><span>${dayNumber}</span></div>`;
    })
    .join("");

  content.innerHTML = `
    <div class="memory-calendar-view">
      <div class="memory-calendar-nav">
        <button type="button" class="btn-secondary" id="memoryCalendarPrev" aria-label="${escapeAttr(t("memory.calendar_previous"))}">&larr;</button>
        <h3>${escapeHtml(_calendarMonthLabel(data.year, data.month))}</h3>
        <button type="button" class="btn-secondary" id="memoryCalendarNext" aria-label="${escapeAttr(t("memory.calendar_next"))}">&rarr;</button>
      </div>
      <div class="memory-calendar-grid">
        ${weekdayKeys.map((key) => `<div class="memory-calendar-weekday">${t(`memory.weekday_${key}`)}</div>`).join("")}
        ${blanks}${days}
      </div>
    </div>
  `;

  document.getElementById("memoryCalendarPrev")?.addEventListener("click", () => {
    _calendarCursor = new Date(data.year, data.month - 2, 1);
    _loadActiveView();
  });
  document.getElementById("memoryCalendarNext")?.addEventListener("click", () => {
    _calendarCursor = new Date(data.year, data.month, 1);
    _loadActiveView();
  });
  content.querySelectorAll("[data-date]").forEach((button) => {
    button.addEventListener("click", () => _loadFileContent(button.dataset.date, "episodes", "calendar"));
  });
}
