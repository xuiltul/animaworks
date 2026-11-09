import { basePath } from "../../shared/base-path.js";

const D3_SCRIPT_SELECTOR = "script[data-memory-popover-d3]";
const MEMORY_TABS = ["graph", "state", "files", "days", "activity", "routine"];
const KIND_ORDER = ["anima", "person", "service", "place", "topic"];
const INSTANCES = new Set();
let _d3LoadPromise = null;
let _instanceId = 0;

export function pickSayLine(markdown) {
  for (const raw of String(markdown ?? "").split(/\r?\n/)) {
    let line = raw.trim().replace(/^>\s?/, "").replace(/^[-*+]\s+/, "").trim();
    if (!line || /^#{1,6}\s/.test(line)) continue;
    if (!line || /^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}\s*$/.test(line) || /^[-=*_#\s]+$/.test(line)) continue;
    const chars = [...line];
    return chars.length > 60 ? chars.slice(0, 59).join("") + "…" : line;
  }
  return "";
}

export function bubbleRadius(count, maxCount) {
  const max = Math.max(1, Number(maxCount) || 1);
  const value = Math.max(0, Math.min(max, Number(count) || 0));
  return 10 + 36 * Math.sqrt(value / max);
}

export function kindColorVar(kind) {
  const normalized = String(kind || "").toLowerCase();
  if (normalized === "anima" || normalized === "user") return "--aw-mem-k-anima";
  if (normalized === "person") return "--aw-mem-k-person";
  if (normalized === "service") return "--aw-mem-k-service";
  if (normalized === "place") return "--aw-mem-k-place";
  return "--aw-mem-k-topic";
}

export function splitStateSections(markdown) {
  const sections = [];
  let current = null;
  for (const line of String(markdown ?? "").split(/\r?\n/)) {
    const heading = line.match(/^##\s+(.+?)\s*#*\s*$/);
    if (heading) {
      if (current) sections.push({ heading: current.heading, body: current.body.join("\n").trim() });
      current = { heading: heading[1].trim(), body: [] };
    } else if (current) {
      current.body.push(line);
    } else if (line.trim()) {
      current = { heading: "", body: [line] };
    }
  }
  if (current) sections.push({ heading: current.heading, body: current.body.join("\n").trim() });
  return sections;
}

export function graphBounds(nodes, maxCount) {
  const box = { left: Infinity, right: -Infinity, top: Infinity, bottom: -Infinity };
  for (const node of nodes) {
    if (!Number.isFinite(node.x) || !Number.isFinite(node.y)) continue;
    const r = node.kind === "center" ? 44 : bubbleRadius(node.count, maxCount);
    box.left = Math.min(box.left, node.x - r - 40);
    box.right = Math.max(box.right, node.x + r + 40);
    box.top = Math.min(box.top, node.y - r - 8);
    box.bottom = Math.max(box.bottom, node.y + r + 22);
  }
  if (!Number.isFinite(box.left)) return { left: -1, right: 1, top: -1, bottom: 1 };
  return box;
}

export function tailLeft(anchorRect, popRect) {
  const anchorCenter = Number(anchorRect?.left || 0) + Number(anchorRect?.width || 0) / 2;
  const left = Number(popRect?.left || 0);
  const width = Number(popRect?.width || (Number(popRect?.right || 0) - left) || 0);
  return Math.max(24, Math.min(Math.max(24, width - 24), anchorCenter - left));
}

function _count(value) {
  if (typeof value === "number") return value;
  return Number(value?.count) || 0;
}

function _faceUrl(name) {
  return (basePath || "") + "/api/animas/" + encodeURIComponent(name) + "/assets/face/neutral?size=S";
}

function _kindGroup(kind) {
  return kind === "anima" || kind === "user" ? "anima" : KIND_ORDER.includes(kind) ? kind : "topic";
}

function _date(value) {
  if (!value) return "";
  return String(value).match(/\d{4}-\d{2}-\d{2}/)?.[0] || String(value);
}

function _firstMarkdownLine(markdown) {
  const text = String(markdown ?? "").replace(/^---\s*\r?\n[\s\S]*?\r?\n---\s*\r?\n?/, "");
  for (const raw of text.split(/\r?\n/)) {
    let line = raw.trim();
    if (!line || /^#{1,6}\s/.test(line)) continue;
    line = line.replace(/^[-*+]\s+/, "").replace(/^>\s?/, "").replace(/[\x60*_~]/g, "").trim();
    if (line) return line;
  }
  return "";
}

function _fileMetadata(content, fallbackTitle) {
  const text = String(content ?? "");
  const frontmatter = text.match(/^---\s*\r?\n([\s\S]*?)\r?\n---\s*(?:\r?\n|$)/);
  const updated = frontmatter?.[1]?.match(/^\s*(?:updated_at|updated|modified_at)\s*:\s*["']?([^"'\r\n]+)["']?\s*$/m)?.[1]?.trim() || "";
  const heading = text.match(/^#\s+(.+?)\s*#*\s*$/m)?.[1]?.trim();
  return { title: heading || String(fallbackTitle || ""), updated: _date(updated), excerpt: _firstMarkdownLine(text) };
}

function _getD3(doc, win) {
  if (win?.d3) return Promise.resolve(win.d3);
  if (_d3LoadPromise) return _d3LoadPromise;
  _d3LoadPromise = new Promise((resolve, reject) => {
    const existing = doc.querySelector(D3_SCRIPT_SELECTOR);
    const script = existing || doc.createElement("script");
    const onLoad = () => {
      if (win?.d3) resolve(win.d3);
      else {
        script.remove();
        _d3LoadPromise = null;
        reject(new Error("D3 unavailable"));
      }
    };
    const onError = () => {
      script.remove();
      _d3LoadPromise = null;
      reject(new Error("D3 unavailable"));
    };
    script.addEventListener("load", onLoad, { once: true });
    script.addEventListener("error", onError, { once: true });
    if (!existing) {
      script.src = (basePath || "") + "/vendor/d3.v7.min.js";
      script.async = true;
      script.dataset.memoryPopoverD3 = "true";
      doc.head.appendChild(script);
    }
  });
  return _d3LoadPromise;
}

function _formatCount(value) {
  return Number(value || 0).toLocaleString();
}

export function createMemoryPopover(ctx) {
  const doc = ctx.document || globalThis.document;
  const win = ctx.window || globalThis.window || globalThis;
  const { state, deps } = ctx;
  const { api, t, escapeHtml } = deps;
  const instanceId = ++_instanceId;
  const cache = new Map();
  let root = null;
  let dialog = null;
  let scrim = null;
  let open = false;
  let activeName = null;
  let activeAnchor = null;
  let activeTab = "graph";
  let activeDays = "";
  let activeEntity = null;
  let activeGraph = null;
  let simulation = null;
  let graphFit = null;
  let activityTimer = null;
  let fileObserver = null;
  let listenersBound = false;
  let documentKeyHandler = null;
  let offKinds = new Set();

  function entry(name) {
    if (!cache.has(name)) cache.set(name, { values: new Map(), pending: new Map(), fileContents: new Map() });
    return cache.get(name);
  }

  function q(selector) {
    return root?.querySelector(selector) || null;
  }

  function qa(selector) {
    return root ? [...root.querySelectorAll(selector)] : [];
  }

  function html(value) {
    return escapeHtml(String(value ?? ""));
  }

  function translated(key, fallback) {
    const value = t(key);
    return value === key ? fallback : value;
  }

  function ensureShell() {
    if (root) return;
    root = doc.createElement("div");
    root.className = "aw-memory-overlay";
    root.hidden = true;
    const tabButtons = MEMORY_TABS.map((tab) =>
      '<button type="button" class="aw-memory-tab" data-memory-tab="' + tab + '" role="tab" aria-selected="' + (tab === "graph" ? "true" : "false") + '">' +
      html(t("chat.memory.tab." + tab)) + "</button>",
    ).join("");
    const kindButtons = KIND_ORDER.map((kind) =>
      '<button type="button" class="aw-memory-chip" data-memory-kind="' + kind + '" aria-pressed="true">' +
      '<i></i>' + html(t("chat.memory.kind." + kind)) + "</button>",
    ).join("");
    root.innerHTML = [
      '<div class="aw-memory-scrim" data-memory-action="scrim" aria-hidden="true"></div>',
      '<section class="aw-memory-dialog" role="dialog" aria-modal="true" aria-label="' + html(t("chat.memory.dialog_label")) + '">',
      '<header class="aw-memory-head">',
      '<div class="aw-memory-identity"><span class="aw-memory-face-wrap"><img class="aw-memory-face" alt=""><span class="aw-memory-face-fallback" aria-hidden="true"></span></span>',
      '<div class="aw-memory-title"><div class="aw-memory-title-line"><strong class="aw-memory-name"></strong><span class="aw-memory-role"></span></div><div class="aw-memory-say"></div></div></div>',
      '<div class="aw-memory-stats">',
      '<div><strong data-memory-stat="facts">—</strong><span>' + html(t("chat.memory.stat.facts")) + '</span></div>',
      '<div><strong data-memory-stat="knowledge">—</strong><span>' + html(t("chat.memory.stat.knowledge")) + '</span></div>',
      '<div><strong data-memory-stat="procedures">—</strong><span>' + html(t("chat.memory.stat.procedures")) + '</span></div>',
      '<div><strong data-memory-stat="episodes">—</strong><span>' + html(t("chat.memory.stat.episodes")) + '</span></div>',
      '</div><button type="button" class="aw-memory-close" data-memory-action="close" aria-label="' + html(t("chat.memory.close")) + '">×</button></header>',
      '<nav class="aw-memory-tabs" role="tablist">' + tabButtons + '</nav>',
      '<div class="aw-memory-content">',
      '<section class="aw-memory-pane aw-memory-pane-graph" data-memory-pane="graph" role="tabpanel">',
      '<div class="aw-memory-graph-tools"><div class="aw-memory-kinds">' + kindButtons + '</div>',
      '<select class="aw-memory-period" aria-label="' + html(t("chat.memory.graph.period")) + '"><option value="">' + html(t("chat.memory.graph.period.all")) + '</option><option value="30">' + html(t("chat.memory.graph.period.30")) + '</option><option value="7">' + html(t("chat.memory.graph.period.7")) + '</option></select>',
      '<input class="aw-memory-search" type="search" placeholder="' + html(t("chat.memory.graph.search")) + '" aria-label="' + html(t("chat.memory.graph.search")) + '"></div>',
      '<div class="aw-memory-graph-layout"><div class="aw-memory-stage"><svg class="aw-memory-svg" role="img" aria-label="' + html(t("chat.memory.graph.aria")) + '"></svg><div class="aw-memory-graph-fallback" hidden></div>',
      '<div class="aw-memory-hint">' + html(t("chat.memory.graph.hint")) + '</div></div><aside class="aw-memory-detail aw-memory-detail-entity" hidden></aside></div></section>',
      '<section class="aw-memory-pane aw-memory-pane-state" data-memory-pane="state" role="tabpanel" hidden><div class="aw-memory-state"></div></section>',
      '<section class="aw-memory-pane aw-memory-pane-files" data-memory-pane="files" role="tabpanel" hidden>',
      '<div class="aw-memory-file-tools"><button type="button" class="aw-memory-chip is-active" data-memory-file-kind="all">' + html(t("chat.memory.files.all")) + '</button><button type="button" class="aw-memory-chip" data-memory-file-kind="knowledge">' + html(t("chat.memory.files.knowledge")) + '</button><button type="button" class="aw-memory-chip" data-memory-file-kind="procedures">' + html(t("chat.memory.files.procedures")) + '</button>',
      '<input class="aw-memory-file-search" type="search" placeholder="' + html(t("chat.memory.files.search")) + '" aria-label="' + html(t("chat.memory.files.search")) + '"></div>',
      '<div class="aw-memory-file-list"></div><div class="aw-memory-file-detail" hidden></div></section>',
      '<section class="aw-memory-pane aw-memory-pane-days" data-memory-pane="days" role="tabpanel" hidden><div class="aw-memory-days"></div><div class="aw-memory-episode-detail" hidden></div></section>',
      '<section class="aw-memory-pane aw-memory-pane-activity" data-memory-pane="activity" role="tabpanel" hidden><div class="aw-memory-activity activity-feed"></div></section>',
      '<section class="aw-memory-pane aw-memory-pane-routine" data-memory-pane="routine" role="tabpanel" hidden><div class="aw-memory-routine"></div></section>',
      "</div></section>",
    ].join("");
    scrim = root.querySelector(".aw-memory-scrim");
    dialog = root.querySelector(".aw-memory-dialog");
    doc.body.appendChild(root);
    bindListeners();
  }

  function bindListeners() {
    if (listenersBound) return;
    listenersBound = true;
    root.addEventListener("click", (event) => {
      const target = event.target;
      if (target.closest?.('[data-memory-action="close"]') || target.closest?.('[data-memory-action="scrim"]')) {
        close();
        return;
      }
      const tabButton = target.closest?.("[data-memory-tab]");
      if (tabButton) {
        switchTab(tabButton.dataset.memoryTab);
        return;
      }
      const entityButton = target.closest?.("[data-memory-entity]");
      if (entityButton) {
        selectEntity(entityButton.dataset.memoryEntity);
        return;
      }
      const relatedButton = target.closest?.("[data-memory-related]");
      if (relatedButton) {
        selectEntity(relatedButton.dataset.memoryRelated);
        return;
      }
      const kindButton = target.closest?.("[data-memory-kind]");
      if (kindButton) {
        const kind = kindButton.dataset.memoryKind;
        if (offKinds.has(kind)) offKinds.delete(kind);
        else offKinds.add(kind);
        kindButton.classList.toggle("is-off", offKinds.has(kind));
        kindButton.setAttribute("aria-pressed", offKinds.has(kind) ? "false" : "true");
        paintGraphFilters();
        return;
      }
      const fileButton = target.closest?.("[data-memory-file]");
      if (fileButton) {
        openFile(fileButton.dataset.memoryType, fileButton.dataset.memoryFile, fileButton.dataset.memoryTitle);
        return;
      }
      const episodeButton = target.closest?.("[data-memory-episode]");
      if (episodeButton) {
        openEpisode(episodeButton.dataset.memoryEpisode);
        return;
      }
      if (target.closest?.("[data-memory-back]")) {
        showFileList(target.closest("[data-memory-back]").dataset.memoryBack);
        return;
      }
      if (target.closest?.('[data-memory-action="ask"]')) askAboutActiveEntity();
      const fileKind = target.closest?.("[data-memory-file-kind]");
      if (fileKind) {
        qa("[data-memory-file-kind]").forEach((button) =>
          button.classList.toggle("is-active", button === fileKind),
        );
        applyFileFilters();
      }
    });
    root.addEventListener("change", (event) => {
      if (event.target.matches?.(".aw-memory-period")) {
        activeDays = event.target.value;
        const data = entry(activeName).values.get(graphKey(activeDays));
        if (data) renderGraph(activeName, data);
        void loadGraph(activeName, activeDays, entry(activeName).values.has(graphKey(activeDays)));
      } else if (event.target.matches?.("[data-memory-file-kind]")) applyFileFilters();
    });
    root.addEventListener("input", (event) => {
      if (event.target.matches?.(".aw-memory-search")) paintGraphFilters();
      if (event.target.matches?.(".aw-memory-file-search")) applyFileFilters();
    });
    scrim.addEventListener("click", close);
    documentKeyHandler = (event) => {
      if (open && event.key === "Escape") close();
    };
    doc.addEventListener("keydown", documentKeyHandler);
    win.addEventListener?.("resize", onViewportResize);
  }

  function graphKey(days) {
    return "graph:" + (days || "all");
  }

  function loadValue(name, key, fetcher, refresh = false) {
    const current = entry(name);
    if (current.pending.has(key)) return current.pending.get(key);
    if (!refresh && current.values.has(key)) return Promise.resolve(current.values.get(key));
    const request = Promise.resolve().then(fetcher).then((value) => {
      current.values.set(key, value);
      return value;
    }).finally(() => current.pending.delete(key));
    current.pending.set(key, request);
    return request;
  }

  function apiPath(name, path) {
    return "/api/animas/" + encodeURIComponent(name) + "/" + path;
  }

  function currentFor(name, tab = activeTab) {
    return open && activeName === name && activeTab === tab;
  }

  function setLoading(tab) {
    const target = {
      graph: q(".aw-memory-graph-fallback"),
      state: q(".aw-memory-state"),
      files: q(".aw-memory-file-list"),
      days: q(".aw-memory-days"),
      activity: q(".aw-memory-activity"),
      routine: q(".aw-memory-routine"),
    }[tab];
    if (target) {
      target.hidden = false;
      target.innerHTML = '<div class="aw-memory-placeholder">' + html(t("chat.memory.loading")) + "</div>";
    }
  }

  function setPaneError(tab, message) {
    const target = {
      graph: q(".aw-memory-graph-fallback"),
      state: q(".aw-memory-state"),
      files: q(".aw-memory-file-list"),
      days: q(".aw-memory-days"),
      activity: q(".aw-memory-activity"),
      routine: q(".aw-memory-routine"),
    }[tab];
    if (target) {
      target.hidden = false;
      target.innerHTML = '<div class="aw-memory-placeholder">' + html(message || t("chat.memory.load_failed")) + "</div>";
    }
  }

  function readStateValue(data) {
    if (typeof data?.state === "string") return data.state;
    if (data?.state && typeof data.state === "object") return JSON.stringify(data.state, null, 2);
    return "";
  }

  function renderHeader(name) {
    if (!root || activeName !== name) return;
    const current = entry(name);
    const anima = state.animas?.find((item) => item.name === name) || {};
    const stateData = current.values.get("state");
    const stats = current.values.get("stats");
    const graph = current.values.get(graphKey(activeDays)) || current.values.get(graphKey(""));
    const role = state.animaSummary?.role || anima.role || "";
    const face = q(".aw-memory-face");
    q(".aw-memory-name").textContent = name;
    q(".aw-memory-role").textContent = role;
    q(".aw-memory-say").textContent = pickSayLine(readStateValue(stateData)) || translated("chat.memory.say_empty", "");
    if (face) {
      face.src = _faceUrl(name);
      face.alt = name;
      face.onerror = () => {
        face.hidden = true;
        q(".aw-memory-face-fallback")?.classList.add("is-visible");
      };
      face.onload = () => {
        face.hidden = false;
        q(".aw-memory-face-fallback")?.classList.remove("is-visible");
      };
    }
    const values = {
      facts: graph?.totals?.facts,
      knowledge: stats ? _count(stats.knowledge) : null,
      procedures: stats ? _count(stats.procedures) : null,
      episodes: stats ? _count(stats.episodes) : null,
    };
    for (const [key, value] of Object.entries(values)) {
      const item = q('[data-memory-stat="' + key + '"]');
      if (item) item.textContent = value == null ? "—" : _formatCount(value);
    }
  }

  function refreshHeader(name) {
    const current = entry(name);
    renderHeader(name);
    void Promise.all([
      loadValue(name, "stats", () => api(apiPath(name, "memory/stats")), current.values.has("stats")).catch(() => null),
      loadValue(name, "state", () => api(apiPath(name, "state")), current.values.has("state")).catch(() => null),
    ]).then(() => {
      if (open && activeName === name) renderHeader(name);
    });
  }

  function switchTab(tab) {
    if (!MEMORY_TABS.includes(tab)) return;
    stopActivityRefresh();
    activeTab = tab;
    qa("[data-memory-tab]").forEach((button) => {
      const selected = button.dataset.memoryTab === tab;
      button.classList.toggle("is-active", selected);
      button.setAttribute("aria-selected", selected ? "true" : "false");
    });
    qa("[data-memory-pane]").forEach((pane) => { pane.hidden = pane.dataset.memoryPane !== tab; });
    if (tab === "files") showFileList("files");
    if (tab === "days") showFileList("days");
    const current = entry(activeName);
    if (tab === "graph") {
      activeDays = q(".aw-memory-period")?.value || activeDays || "";
      const cached = current.values.get(graphKey(activeDays));
      if (cached) renderGraph(activeName, cached);
      void loadGraph(activeName, activeDays, false);
    } else {
      void loadTab(tab, false);
    }
    if (tab === "activity") startActivityRefresh();
  }

  function loadTab(tab, refresh = false) {
    const name = activeName;
    if (!name || tab === "graph") return Promise.resolve(null);
    const current = entry(name);
    const cached = current.values.get(tab);
    if (cached !== undefined) renderTab(tab, name, cached);
    else setLoading(tab);
    let fetcher;
    if (tab === "state") {
      fetcher = () => api(apiPath(name, "state"));
    } else if (tab === "files") {
      fetcher = async () => {
        const [knowledge, procedures] = await Promise.all([
          api(apiPath(name, "knowledge")),
          api(apiPath(name, "procedures")),
        ]);
        return { knowledge: knowledge.files || [], procedures: procedures.files || [] };
      };
    } else if (tab === "days") {
      fetcher = async () => {
        const data = await api(apiPath(name, "episodes"));
        return [...(data.files || [])].sort((a, b) => String(b).localeCompare(String(a)));
      };
    } else if (tab === "activity") {
      fetcher = async () => {
        if (!ctx.controllers.activity?.loadActivity) return [];
        return await ctx.controllers.activity.loadActivity({ target: q(".aw-memory-activity"), animaName: name }) || [];
      };
    } else if (tab === "routine") {
      fetcher = async () => {
        const [heartbeat, cron] = await Promise.all([
          api(apiPath(name, "heartbeat")).catch(() => ({ content: "" })),
          api(apiPath(name, "cron")).catch(() => ({ content: "" })),
        ]);
        return { heartbeat: heartbeat.content || "", cron: cron.content || "" };
      };
    } else {
      return Promise.resolve(null);
    }
    return loadValue(name, tab, fetcher, refresh).then((value) => {
      if (currentFor(name, tab)) renderTab(tab, name, value);
      return value;
    }).catch((error) => {
      if (currentFor(name, tab)) setPaneError(tab, error?.message || t("chat.memory.load_failed"));
      return null;
    });
  }

  function renderTab(tab, name, value) {
    if (tab === "state") renderState(value);
    else if (tab === "files") renderFiles(value);
    else if (tab === "days") renderDays(value);
    else if (tab === "activity") {
      const target = q(".aw-memory-activity");
      if (ctx.controllers.activity?.renderActivity) ctx.controllers.activity.renderActivity(value, target);
      else if (target) target.innerHTML = value.length
        ? value.map((item) => '<div class="activity-entry">' + html(item.summary || item.text || "") + "</div>").join("")
        : '<div class="activity-empty">' + html(t("activity.empty")) + "</div>";
    } else if (tab === "routine") renderRoutine(value);
  }

  function renderState(data) {
    const target = q(".aw-memory-state");
    if (!target) return;
    const sections = splitStateSections(readStateValue(data));
    if (!sections.length) {
      target.innerHTML = '<div class="aw-memory-placeholder">' + html(t("animas.no_state")) + "</div>";
      return;
    }
    target.innerHTML = '<div class="aw-memory-state-grid">' + sections.map((section) =>
      '<section class="aw-memory-card">' +
      (section.heading ? "<h3>" + html(section.heading) + "</h3>" : "") +
      '<div class="aw-memory-markdown">' + (deps.renderMarkdown ? deps.renderMarkdown(section.body, activeName) : html(section.body)) + "</div></section>",
    ).join("") + "</div>";
  }

  function normalizeFile(type, raw) {
    if (typeof raw === "string") return { type, file: raw, title: raw };
    const file = raw?.file || raw?.name || raw?.stem || raw?.topic || "";
    return {
      type,
      file,
      title: raw?.title || raw?.label || file,
      updated: _date(raw?.updated_at || raw?.updated || ""),
      excerpt: raw?.excerpt || raw?.preview || "",
    };
  }

  function renderFiles(data) {
    const list = q(".aw-memory-file-list");
    if (!list) return;
    const files = [
      ...(data.knowledge || []).map((file) => normalizeFile("knowledge", file)),
      ...(data.procedures || []).map((file) => normalizeFile("procedures", file)),
    ];
    if (!files.length) {
      list.innerHTML = '<div class="aw-memory-placeholder">' + html(t("memory.no_files")) + "</div>";
      return;
    }
    list.innerHTML = '<div class="aw-memory-cards">' + files.map((file, index) =>
      '<button type="button" class="aw-memory-file-card" data-memory-file="' + html(file.file) + '" data-memory-type="' + file.type + '" data-memory-title="' + html(file.title) + '" data-memory-index="' + index + '" data-memory-kind-filter="' + file.type + '">' +
      '<small class="aw-memory-file-meta"><span class="aw-memory-file-type">' + html(t("chat.memory.files." + file.type)) + '</span><span class="aw-memory-file-updated">' + html(file.updated || t("chat.memory.unknown_date")) + '</span></small>' +
      '<strong class="aw-memory-file-title">' + html(file.title) + '</strong><span class="aw-memory-file-preview">' + html(file.excerpt || t("chat.memory.preview_loading")) + '</span></button>',
    ).join("") + "</div>";
    qa(".aw-memory-file-card").forEach((card, index) => {
      if (files[index].excerpt || files[index].updated) return;
      watchFileCard(card, files[index]);
    });
    applyFileFilters();
  }

  function watchFileCard(card, file) {
    const observerCtor = win?.IntersectionObserver;
    if (observerCtor) {
      if (!fileObserver) {
        fileObserver = new observerCtor((records) => {
          for (const record of records) {
            if (!record.isIntersecting) continue;
            fileObserver.unobserve(record.target);
            const index = Number(record.target.dataset.memoryIndex);
            const data = entry(activeName).values.get("files");
            const currentFiles = [
              ...(data?.knowledge || []).map((item) => normalizeFile("knowledge", item)),
              ...(data?.procedures || []).map((item) => normalizeFile("procedures", item)),
            ];
            if (currentFiles[index]) void loadFilePreview(activeName, record.target, currentFiles[index]);
          }
        }, { root: q(".aw-memory-file-list"), rootMargin: "100px" });
      }
      fileObserver.observe(card);
      return;
    }
    if (Number(card.dataset.memoryIndex) < 12) void loadFilePreview(activeName, card, file);
  }

  async function loadFilePreview(name, card, file) {
    const content = await loadFileContent(name, file.type, file.file).catch(() => null);
    if (!content || !card.isConnected) return;
    const metadata = _fileMetadata(content, file.title);
    const title = card.querySelector(".aw-memory-file-title");
    const updated = card.querySelector(".aw-memory-file-updated");
    const preview = card.querySelector(".aw-memory-file-preview");
    if (title) title.textContent = metadata.title || file.title;
    if (updated) updated.textContent = metadata.updated || t("chat.memory.unknown_date");
    if (preview) preview.textContent = metadata.excerpt || t("chat.no_content");
    card.dataset.memoryTitle = metadata.title || file.title;
  }

  function applyFileFilters() {
    const selected = q("[data-memory-file-kind].is-active")?.dataset.memoryFileKind || "all";
    const query = q(".aw-memory-file-search")?.value?.trim().toLocaleLowerCase() || "";
    qa(".aw-memory-file-card").forEach((card) => {
      const text = (card.dataset.memoryTitle + " " + card.dataset.memoryFile).toLocaleLowerCase();
      card.hidden = (selected !== "all" && card.dataset.memoryKindFilter !== selected) || Boolean(query && !text.includes(query));
    });
  }

  function showFileList(backTo = "files") {
    if (backTo === "days") {
      const list = q(".aw-memory-days");
      const detail = q(".aw-memory-episode-detail");
      if (detail) detail.hidden = true;
      if (list) list.hidden = false;
      return;
    }
    const list = q(".aw-memory-file-list");
    const detail = q(".aw-memory-file-detail");
    if (detail) detail.hidden = true;
    if (list) list.hidden = false;
  }

  async function openFile(type, file, title = file) {
    if (!activeName || !file) return;
    const name = activeName;
    switchTabWithoutLoad("files");
    void loadTab("files", entry(name).values.has("files"));
    const list = q(".aw-memory-file-list");
    const detail = q(".aw-memory-file-detail");
    if (list) list.hidden = true;
    if (detail) {
      detail.hidden = false;
      detail.innerHTML = '<button type="button" class="aw-memory-back" data-memory-back="files">&larr; ' + html(t("chat.memory.back")) + '</button><h3>' + html(title) + '</h3><div class="aw-memory-markdown">' + html(t("chat.memory.loading")) + "</div>";
    }
    const content = await loadFileContent(name, type, file).catch((error) => "[Error] " + error.message);
    if (!open || activeName !== name || activeTab !== "files" || !detail) return;
    const body = detail.querySelector(".aw-memory-markdown");
    if (body) body.innerHTML = deps.renderMarkdown ? deps.renderMarkdown(content || t("chat.no_content"), name) : html(content);
  }

  function switchTabWithoutLoad(tab) {
    stopActivityRefresh();
    activeTab = tab;
    qa("[data-memory-tab]").forEach((button) => {
      const selected = button.dataset.memoryTab === tab;
      button.classList.toggle("is-active", selected);
      button.setAttribute("aria-selected", selected ? "true" : "false");
    });
    qa("[data-memory-pane]").forEach((pane) => { pane.hidden = pane.dataset.memoryPane !== tab; });
    if (tab === "activity") startActivityRefresh();
  }

  function loadFileContent(name, type, file) {
    const current = entry(name);
    const key = type + ":" + file;
    if (current.fileContents.has(key)) return Promise.resolve(current.fileContents.get(key));
    if (current.pending.has("content:" + key)) return current.pending.get("content:" + key);
    const endpoint = apiPath(name, type + "/" + encodeURIComponent(file));
    const request = api(endpoint).then((data) => {
      const content = data?.content || "";
      current.fileContents.set(key, content);
      return content;
    }).finally(() => current.pending.delete("content:" + key));
    current.pending.set("content:" + key, request);
    return request;
  }

  function renderDays(dates) {
    const target = q(".aw-memory-days");
    if (!target) return;
    target.hidden = false;
    if (!dates?.length) {
      target.innerHTML = '<div class="aw-memory-placeholder">' + html(t("chat.memory.days.empty")) + "</div>";
      return;
    }
    target.innerHTML = '<div class="aw-memory-timeline">' + dates.map((date) =>
      '<button type="button" class="aw-memory-day" data-memory-episode="' + html(date) + '"><time>' + html(date) + '</time><span>' + html(t("chat.memory.days.open")) + "</span></button>",
    ).join("") + "</div>";
  }

  async function openEpisode(date) {
    if (!activeName) return;
    const name = activeName;
    const list = q(".aw-memory-days");
    const detail = q(".aw-memory-episode-detail");
    if (list) list.hidden = true;
    if (detail) {
      detail.hidden = false;
      detail.innerHTML = '<button type="button" class="aw-memory-back" data-memory-back="days">&larr; ' + html(t("chat.memory.back")) + '</button><h3>' + html(date) + '</h3><div class="aw-memory-markdown">' + html(t("chat.memory.loading")) + "</div>";
    }
    const content = await loadFileContent(name, "episodes", date).catch((error) => "[Error] " + error.message);
    if (!open || activeName !== name || activeTab !== "days" || !detail) return;
    const body = detail.querySelector(".aw-memory-markdown");
    if (body) body.innerHTML = deps.renderMarkdown ? deps.renderMarkdown(content || t("chat.no_content"), name) : html(content);
  }

  function renderRoutine(data) {
    const target = q(".aw-memory-routine");
    if (!target) return;
    const card = (title, content, empty) => '<section class="aw-memory-card"><h3>' + html(title) + '</h3><div class="aw-memory-markdown">' +
      (content ? (deps.renderMarkdown ? deps.renderMarkdown(content, activeName) : html(content)) : "<p>" + html(empty) + "</p>") + "</div></section>";
    target.innerHTML = '<div class="aw-memory-state-grid">' +
      card(t("chat.memory.routine.heartbeat"), data.heartbeat, t("chat.heartbeat_not_configured")) +
      card(t("chat.memory.routine.cron"), data.cron, t("chat.cron_not_configured")) + "</div>";
  }

  function loadGraph(name, days, refresh = false) {
    if (!name) return Promise.resolve(null);
    const key = graphKey(days);
    const cached = entry(name).values.get(key);
    if (cached) renderGraph(name, cached);
    else if (currentFor(name, "graph")) setLoading("graph");
    const query = days ? "?days=" + encodeURIComponent(days) + "&limit=40" : "?limit=40";
    return loadValue(name, key, () => api(apiPath(name, "memory/connections") + query), refresh)
      .then((data) => {
        if (currentFor(name, "graph") && activeDays === days) {
          void renderGraph(name, data);
          renderHeader(name);
        }
        return data;
      }).catch((error) => {
        if (currentFor(name, "graph")) setPaneError("graph", error?.message || t("chat.memory.graph.load_failed"));
        return null;
      });
  }

  async function renderGraph(name, data) {
    activeGraph = data;
    const svgEl = q(".aw-memory-svg");
    const fallback = q(".aw-memory-graph-fallback");
    if (!svgEl || !fallback || activeName !== name) return;
    if (simulation) {
      simulation.stop();
      simulation = null;
    }
    const nodes = (data.nodes || []).filter((node) => node?.id && node.id !== name).map((node) => ({
      ...node,
      label: node.label || node.id,
      kind: node.kind || "topic",
      count: Number(node.count) || 0,
    }));
    if (!nodes.length) {
      svgEl.hidden = true;
      fallback.hidden = false;
      fallback.innerHTML = '<div class="aw-memory-placeholder">' + html(t("chat.memory.graph.empty")) + "</div>";
      return;
    }
    svgEl.hidden = false;
    try {
      const d3 = await _getD3(doc, win);
      if (!open || activeName !== name || activeTab !== "graph") return;
      fallback.hidden = true;
      drawGraph(d3, name, data, nodes, svgEl);
    } catch {
      if (!open || activeName !== name || activeTab !== "graph") return;
      svgEl.hidden = true;
      fallback.hidden = false;
      fallback.innerHTML = '<div class="aw-memory-fallback-list">' + nodes.map((node) =>
        '<button type="button" data-memory-entity="' + html(node.id) + '" data-memory-node-kind="' + _kindGroup(node.kind) + '"><i style="--aw-memory-node:var(' + kindColorVar(node.kind) + ')"></i><span>' + html(node.label) + '</span><small>' + html(_formatCount(node.count)) + "</small></button>",
      ).join("") + "</div>";
    }
  }

  function nodeFace(node, name) {
    if (node.id === "__aw_center__" || node.id === name) return _faceUrl(name);
    if (node.kind === "anima") return _faceUrl(node.id);
    if (node.kind !== "user") return "";
    const label = doc.getElementById("currentUserLabel");
    const namedUser = state.users?.find((item) => (item.name || item.username) === node.id);
    if (label && label.textContent.trim() === node.id) {
      const image = label.querySelector("img") || label.parentElement?.querySelector("img");
      if (image?.src) return image.src;
    }
    return namedUser?.avatar_url || namedUser?.avatar || "";
  }

  // Zoom out (never in) so every bubble and its label stays inside the stage, below the toolbar.
  function fitGraph(svg, nodes, maxCount, fallbackWidth, fallbackHeight) {
    // The stage narrows when the detail drawer opens, so measure it every time.
    const size = svg.node()?.getBoundingClientRect?.();
    const width = size?.width > 0 ? size.width : fallbackWidth;
    const height = size?.height > 0 ? size.height : fallbackHeight;
    graphFit = () => fitGraph(svg, nodes, maxCount, fallbackWidth, fallbackHeight);
    const box = graphBounds(nodes, maxCount);
    const top = 48;
    // Keep the anima (fixed at 0,0) in the middle of the stage.
    const halfW = Math.max(-box.left, box.right);
    const halfH = Math.max(-box.top, box.bottom);
    const scale = Math.max(1, (2 * halfW) / width, (2 * halfH) / (height - top));
    const w = width * scale;
    const h = (height - top) * scale;
    svg.attr("viewBox", [-w / 2, -h / 2 - top * scale, w, h + top * scale]);
  }

  function drawGraph(d3, name, data, entityNodes, svgEl) {
    const stage = q(".aw-memory-stage");
    const rect = stage?.getBoundingClientRect() || { width: 560, height: 520 };
    const width = Math.max(300, rect.width || 560);
    const height = Math.max(280, rect.height || 520);
    const center = { id: "__aw_center__", label: name, kind: "center", count: 1, fx: 0, fy: 0 };
    const nodes = [center, ...entityNodes.map((node) => ({ ...node }))];
    const byId = new Map(nodes.map((node) => [node.id, node]));
    const links = entityNodes.map((node) => ({ source: center.id, target: node.id, weight: node.count, spoke: true }));
    for (const link of data.links || []) {
      if (byId.has(link.source) && byId.has(link.target)) {
        links.push({ source: link.source, target: link.target, weight: Number(link.weight) || 1, spoke: false });
      }
    }
    const maxCount = Math.max(1, ...entityNodes.map((node) => node.count));
    const svg = d3.select(svgEl);
    svg.attr("viewBox", [-width / 2, -height / 2 + 16, width, height]);
    svg.selectAll("*").remove();
    const group = svg.append("g");
    const edgeSelection = group.append("g").selectAll("line").data(links).join("line")
      .attr("class", "aw-memory-edge")
      .attr("stroke-width", (link) => link.spoke ? 1 : Math.max(1, Math.sqrt(link.weight) / 2));
    const nodeSelection = group.append("g").selectAll("g").data(nodes).join("g")
      .attr("class", (node) => node.kind === "center" ? "aw-memory-node is-center" : "aw-memory-node")
      .on("mouseenter", (event, node) => hoverNode(node, nodes, links, nodeSelection, edgeSelection))
      .on("mouseleave", () => hoverNode(null, nodes, links, nodeSelection, edgeSelection))
      .on("click", (event, node) => { if (node.kind !== "center") selectEntity(node.id); });
    nodeSelection.each(function (node, index) {
      const selection = d3.select(this);
      const radius = node.kind === "center" ? 38 : bubbleRadius(node.count, maxCount);
      const color = "var(" + kindColorVar(node.kind) + ")";
      const image = nodeFace(node, name);
      selection.append("circle").attr("class", "aw-memory-node-bubble").attr("r", radius)
        .attr("fill", node.kind === "center" ? "var(--aw-color-accent)" : color).attr("fill-opacity", image ? 1 : 0.9);
      if (image) {
        const clipId = "aw-memory-clip-" + instanceId + "-" + index;
        selection.append("clipPath").attr("id", clipId).append("circle").attr("r", radius - 1);
        selection.append("image").attr("href", image).attr("x", -radius + 1).attr("y", -radius + 1)
          .attr("width", (radius - 1) * 2).attr("height", (radius - 1) * 2)
          .attr("clip-path", "url(#" + clipId + ")").style("pointer-events", "none");
      }
      if (node.kind === "center") {
        selection.append("text").attr("class", "aw-memory-node-label is-center").attr("dy", radius + 17).text(node.label);
      } else {
        const inside = radius >= 28 && !image;
        selection.append("text").attr("class", "aw-memory-node-label").attr("dy", inside ? -1 : radius + 13)
          .text(node.label).classed("is-inside", inside);
        if (inside) selection.append("text").attr("class", "aw-memory-node-count").attr("dy", 13).text(_formatCount(node.count));
      }
    });
    simulation = d3.forceSimulation(nodes)
      .force("link", d3.forceLink(links).id((node) => node.id)
        .distance((link) => link.spoke ? 205 - Math.min(105, link.weight / 14) : 72)
        .strength((link) => link.spoke ? 0.14 : 0.27))
      .force("charge", d3.forceManyBody().strength(-70))
      .force("collide", d3.forceCollide((node) => (node.kind === "center" ? 44 : bubbleRadius(node.count, maxCount)) + 13))
      .force("x", d3.forceX().strength(0.035)).force("y", d3.forceY().strength(0.055))
      .on("tick", () => {
        edgeSelection.attr("x1", (link) => link.source.x).attr("y1", (link) => link.source.y)
          .attr("x2", (link) => link.target.x).attr("y2", (link) => link.target.y);
        nodeSelection.attr("transform", (node) => "translate(" + node.x + "," + node.y + ")");
        paintGraphFilters();
        fitGraph(svg, nodes, maxCount, width, height);
      });
    nodeSelection.call(d3.drag()
      .on("start", (event, node) => {
        if (!event.active) simulation.alphaTarget(0.2).restart();
        node.fx = node.x;
        node.fy = node.y;
      })
      .on("drag", (event, node) => { node.fx = event.x; node.fy = event.y; })
      .on("end", (event, node) => {
        if (!event.active) simulation.alphaTarget(0);
        if (node.kind !== "center") {
          node.fx = null;
          node.fy = null;
        }
      }));
    paintGraphFilters();
    if (activeEntity && byId.has(activeEntity)) nodeSelection.classed("is-selected", (node) => node.id === activeEntity);
  }

  function hoverNode(node, nodes, links, nodeSelection, edgeSelection) {
    const focus = node || nodes.find((item) => item.id === activeEntity);
    if (!focus || focus.kind === "center") {
      nodeSelection.classed("is-dim", false);
      edgeSelection.classed("is-hot", false);
      paintGraphFilters();
      return;
    }
    const neighbors = new Set([focus.id]);
    for (const link of links) {
      const source = typeof link.source === "object" ? link.source.id : link.source;
      const target = typeof link.target === "object" ? link.target.id : link.target;
      if (source === focus.id) neighbors.add(target);
      if (target === focus.id) neighbors.add(source);
    }
    nodeSelection.classed("is-dim", (item) => !neighbors.has(item.id));
    edgeSelection.classed("is-hot", (link) => !link.spoke && (link.source.id === focus.id || link.target.id === focus.id));
  }

  function paintGraphFilters() {
    const query = q(".aw-memory-search")?.value?.trim().toLocaleLowerCase() || "";
    const d3 = win?.d3;
    const svg = q(".aw-memory-svg");
    if (d3 && svg) {
      d3.select(svg).selectAll(".aw-memory-node").each(function (node) {
        const hiddenKind = node.kind !== "center" && offKinds.has(_kindGroup(node.kind));
        const dimSearch = query && node.kind !== "center" && !String(node.label || node.id).toLocaleLowerCase().includes(query);
        d3.select(this).style("display", hiddenKind ? "none" : null).style("opacity", dimSearch ? 0.18 : null);
      });
      d3.select(svg).selectAll(".aw-memory-edge").style("display", (link) => {
        const sourceKind = typeof link.source === "object" ? link.source.kind : "topic";
        const targetKind = typeof link.target === "object" ? link.target.kind : "topic";
        const sourceHidden = sourceKind !== "center" && offKinds.has(_kindGroup(sourceKind));
        const targetHidden = targetKind !== "center" && offKinds.has(_kindGroup(targetKind));
        return sourceHidden || targetHidden ? "none" : null;
      });
    }
    qa(".aw-memory-fallback-list button").forEach((button) => {
      const kind = button.dataset.memoryNodeKind || "topic";
      const matches = button.textContent.toLocaleLowerCase().includes(query);
      button.hidden = offKinds.has(kind) || !matches;
    });
  }

  function selectEntity(id) {
    if (!activeName || !id) return;
    const node = activeGraph?.nodes?.find((item) => item.id === id) || { id, label: id, kind: "topic", count: 0 };
    activeEntity = id;
    const drawer = q(".aw-memory-detail-entity");
    if (!drawer) return;
    drawer.hidden = false;
    q(".aw-memory-graph-layout")?.classList.add("has-detail");
    win.requestAnimationFrame?.(() => graphFit?.());
    drawer.innerHTML = '<div class="aw-memory-detail-loading">' + html(t("chat.memory.loading")) + "</div>";
    if (simulation && q(".aw-memory-svg") && win?.d3) {
      win.d3.select(q(".aw-memory-svg")).selectAll(".aw-memory-node").classed("is-selected", (item) => item.id === id);
    }
    const current = entry(activeName);
    const key = "entity:" + id;
    const cached = current.values.get(key);
    if (cached) renderEntityDetail(node, cached);
    const name = activeName;
    void loadValue(name, key, () =>
      api(apiPath(name, "memory/connections/entity?name=" + encodeURIComponent(id))),
    false).then((detail) => {
      if (open && activeName === name && activeEntity === id) renderEntityDetail(node, detail);
    }).catch((error) => {
      if (activeEntity === id) drawer.innerHTML = '<div class="aw-memory-placeholder">' + html(error?.message || t("chat.memory.load_failed")) + "</div>";
    });
  }

  function renderEntityDetail(node, detail) {
    const drawer = q(".aw-memory-detail-entity");
    if (!drawer) return;
    const kind = detail.kind || node.kind || "topic";
    const facts = [...(detail.facts || [])].sort((a, b) => String(b.recorded_at || "").localeCompare(String(a.recorded_at || "")));
    const related = detail.related || [];
    const files = detail.files || [];
    const fileType = (file) => file.memory_type === "procedure" || file.memory_type === "procedures" ? "procedures" : "knowledge";
    const kindLabel = t("chat.memory.kind." + _kindGroup(kind));
    const image = nodeFace({ ...node, kind }, activeName);
    const dot = image ? '<img src="' + html(image) + '" alt="">' : '<span>' + html(String(detail.entity || node.label || node.id).slice(0, 1)) + "</span>";
    drawer.innerHTML = [
      '<div class="aw-memory-detail-head"><span class="aw-memory-detail-dot" style="--aw-memory-node:var(' + kindColorVar(kind) + ')">' + dot + '</span><div><strong>' + html(detail.entity || node.label || node.id) + '</strong><small>' + html(kindLabel) + " · " + html(t("chat.memory.detail.fact_count", { count: _formatCount(detail.count ?? node.count) })) + (detail.last_seen ? " · " + html(t("chat.memory.detail.last_seen", { date: _date(detail.last_seen) })) : "") + "</small></div></div>",
      '<section class="aw-memory-detail-section"><h3>' + html(t("chat.memory.detail.facts")) + '</h3><div class="aw-memory-facts">' +
      (facts.length ? facts.map((fact) => '<article class="aw-memory-fact"><time>' + html(_date(fact.recorded_at)) + '</time><p>' + html(fact.text) + "</p></article>").join("") : '<p class="aw-memory-muted">' + html(t("chat.memory.detail.no_facts")) + "</p>") +
      "</div></section>",
      '<section class="aw-memory-detail-section"><h3>' + html(t("chat.memory.detail.related")) + '</h3><div class="aw-memory-related">' +
      (related.length ? related.map((item) => '<button type="button" class="aw-memory-chip" data-memory-related="' + html(item.id) + '" data-memory-related-kind="' + _kindGroup(item.kind) + '"><i></i>' + html(item.id) + "</button>").join("") : '<span class="aw-memory-muted">' + html(t("chat.memory.detail.no_related")) + "</span>") +
      "</div></section>",
      '<section class="aw-memory-detail-section"><h3>' + html(t("chat.memory.detail.files")) + '</h3><div class="aw-memory-related">' +
      (files.length ? files.map((file) => '<button type="button" class="aw-memory-file-link" data-memory-file="' + html(file.file) + '" data-memory-type="' + fileType(file) + '" data-memory-title="' + html(file.title || file.file) + '"><small>' + html(t("chat.memory.files." + fileType(file))) + "</small>" + html(file.title || file.file) + "</button>").join("") : '<span class="aw-memory-muted">' + html(t("chat.memory.detail.no_files")) + "</span>") +
      "</div></section>",
      '<button type="button" class="aw-memory-ask" data-memory-action="ask">' + html(t("chat.memory.detail.ask", { anima: activeName, name: detail.entity || node.label || node.id })) + "</button>",
    ].join("");
  }

  function askAboutActiveEntity() {
    if (!activeEntity) return;
    const input = ctx.$?.("chatPageInput") || ctx.$root?.("chatPageInput");
    if (!input) return;
    const label = activeGraph?.nodes?.find((item) => item.id === activeEntity)?.label || activeEntity;
    input.value = t("chat.memory.ask_text", { name: label });
    input.focus();
    close();
  }

  function startActivityRefresh() {
    stopActivityRefresh();
    activityTimer = win.setInterval?.(() => {
      if (open && activeTab === "activity") void loadTab("activity", true);
    }, 30000) || null;
  }

  function stopActivityRefresh() {
    if (activityTimer != null) win.clearInterval?.(activityTimer);
    activityTimer = null;
  }

  function onViewportResize() {
    if (open) position(activeAnchor);
  }

  function position(anchorEl) {
    const rect = anchorEl?.getBoundingClientRect?.();
    if (!dialog || !rect) return;
    const width = Number(win.innerWidth) || 1024;
    const height = Number(win.innerHeight) || 768;
    const fullScreen = width <= 768;
    dialog.classList.toggle("is-fullscreen", fullScreen);
    if (fullScreen) {
      dialog.style.left = "0px";
      dialog.style.top = "0px";
      dialog.style.width = "100vw";
      dialog.style.height = "100dvh";
      dialog.style.setProperty("--aw-memory-tail-left", "50%");
      return;
    }
    const popWidth = Math.min(960, width - 24);
    const popHeight = Math.min(670, height - 32);
    // An anima in the left rail: open to its right, with the tail on the left edge.
    let beside = rect.right < width * 0.3 && rect.right + 14 + Math.min(popWidth, 480) <= width;
    let anchorRight = rect.right;
    if (ctx.state.embedded) {
      // In the Workspace dock, use the dock's right edge so the memory dialog
      // opens onto the stage without covering the conversation.
      const dock = ctx.state.container?.closest?.(".ws-chat-dock");
      const dockRect = dock?.getBoundingClientRect?.();
      if (dockRect && dockRect.right + 14 + 480 <= width) {
        anchorRight = dockRect.right;
        beside = true;
      }
    }
    dialog.classList.toggle("is-beside-anchor", beside);
    if (beside) {
      const sideWidth = Math.min(popWidth, width - anchorRight - 26);
      const sideTop = Math.max(12, Math.min(height - popHeight - 12, rect.top + rect.height / 2 - 48));
      dialog.style.left = (anchorRight + 14) + "px";
      dialog.style.top = sideTop + "px";
      dialog.style.width = sideWidth + "px";
      dialog.style.height = popHeight + "px";
      const tailY = Math.max(24, Math.min(popHeight - 24, rect.top + rect.height / 2 - sideTop));
      dialog.style.setProperty("--aw-memory-tail-y", tailY + "px");
      dialog.classList.remove("is-above-anchor");
      return;
    }
    const left = Math.max(12, Math.min(width - popWidth - 12, rect.left + rect.width / 2 - popWidth / 2));
    let top = rect.bottom + 14;
    if (top + popHeight > height - 12) top = Math.max(12, rect.top - popHeight - 14);
    dialog.style.left = left + "px";
    dialog.style.top = top + "px";
    dialog.style.width = popWidth + "px";
    dialog.style.height = popHeight + "px";
    dialog.style.setProperty("--aw-memory-tail-left", tailLeft(rect, { left, width: popWidth }) + "px");
    dialog.style.setProperty("--aw-memory-tail-top", top > rect.top ? "-9px" : "auto");
    dialog.classList.toggle("is-above-anchor", top < rect.top);
  }

  function openPopover(anchorEl) {
    const name = state.selectedAnima;
    if (!name) return;
    for (const instance of INSTANCES) {
      if (instance !== controls && instance.isOpen()) instance.close();
    }
    ensureShell();
    activeName = name;
    activeAnchor = anchorEl || activeAnchor;
    activeTab = "graph";
    activeDays = "";
    activeEntity = null;
    activeGraph = null;
    offKinds = new Set();
    open = true;
    if (fileObserver) {
      fileObserver.disconnect();
      fileObserver = null;
    }
    root.hidden = false;
    scrim.hidden = false;
    q(".aw-memory-file-detail").hidden = true;
    q(".aw-memory-file-list").hidden = false;
    q(".aw-memory-episode-detail").hidden = true;
    q(".aw-memory-days").hidden = false;
    q(".aw-memory-detail-entity").hidden = true;
    q(".aw-memory-graph-layout").classList.remove("has-detail");
    win.requestAnimationFrame?.(() => graphFit?.());
    q(".aw-memory-period").value = "";
    q(".aw-memory-search").value = "";
    qa("[data-memory-kind]").forEach((button) => {
      button.classList.remove("is-off");
      button.setAttribute("aria-pressed", "true");
    });
    position(activeAnchor);
    switchTabWithoutLoad("graph");
    renderHeader(name);
    refreshHeader(name);
    const cached = entry(name).values.has(graphKey(""));
    void loadGraph(name, "", cached);
    q(".aw-memory-close")?.focus?.();
  }

  function close() {
    if (!open) return;
    open = false;
    stopActivityRefresh();
    if (simulation) {
      simulation.stop();
      simulation = null;
    }
    if (fileObserver) {
      fileObserver.disconnect();
      fileObserver = null;
    }
    if (root) root.hidden = true;
  }

  function destroy() {
    close();
    if (documentKeyHandler) doc.removeEventListener("keydown", documentKeyHandler);
    win.removeEventListener?.("resize", onViewportResize);
    root?.remove();
    root = null;
    dialog = null;
    scrim = null;
    INSTANCES.delete(controls);
  }

  const controls = { open: openPopover, close, isOpen: () => open, destroy };
  INSTANCES.add(controls);
  return controls;
}
