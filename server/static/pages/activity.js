// ── Activity page (tab host) ─────────────────
// Tabs: timeline (default) | report | logs
// Hash routes: #/activity | #/activity/report | #/activity/logs

import { createPageTabs } from "../shared/page-tabs.js";
import { t } from "/shared/i18n.js";

let _pageTabs = null;
let _activeTabModule = null;
let _activeTab = "timeline";
let _container = null;

/** Tab definitions: id used in subPath (timeline is default empty). */
const _TABS = [
  { id: "timeline", labelKey: "activity.tab_timeline" },
  { id: "report", labelKey: "activity.tab_report" },
  { id: "logs", labelKey: "activity.tab_logs" },
];

/**
 * Resolve activity tab id from router subPath.
 * Pure helper — exported for unit tests.
 * @param {string} [subPath]
 * @returns {"timeline"|"report"|"logs"}
 */
export function resolveActivityTab(subPath) {
  const head = String(subPath || "")
    .split("/")
    .filter(Boolean)[0] || "";
  if (head === "report") return "report";
  if (head === "logs") return "logs";
  return "timeline";
}

/**
 * Build hash for an activity tab (reload / bookmark safe).
 * @param {string} tabId
 * @returns {string}
 */
export function buildActivityTabHash(tabId) {
  if (tabId === "report") return "#/activity/report";
  if (tabId === "logs") return "#/activity/logs";
  return "#/activity";
}

/**
 * Module loader for a tab (relative paths). Exported for tests.
 * @param {string} tabId
 * @returns {() => Promise<{ render: Function, destroy?: Function }>}
 */
export function activityTabLoader(tabId) {
  if (tabId === "report") return () => import("./activity-report.js");
  if (tabId === "logs") return () => import("./logs.js");
  return () => import("./activity-timeline.js");
}

function _destroyActiveTab() {
  if (_activeTabModule && typeof _activeTabModule.destroy === "function") {
    try {
      _activeTabModule.destroy();
    } catch {
      /* ignore */
    }
  }
  _activeTabModule = null;
  if (_pageTabs) {
    try {
      _pageTabs.destroy();
    } catch {
      /* ignore */
    }
    _pageTabs = null;
  }
}

/**
 * @param {HTMLElement} container
 * @param {{ subPath?: string }} [opts]
 */
export async function render(container, { subPath } = {}) {
  _container = container;
  _destroyActiveTab();
  _activeTab = resolveActivityTab(subPath);

  container.innerHTML = `
    <div class="page-header">
      <h2>${t("nav.activity")}</h2>
    </div>
    <div id="activityPageTabs"></div>
    <div id="activityTabContent">
      <div class="loading-placeholder">${t("common.loading")}</div>
    </div>
  `;

  const tabsHost = document.getElementById("activityPageTabs");
  if (tabsHost) {
    _pageTabs = createPageTabs({
      tabs: _TABS.map((tab) => ({ id: tab.id, label: t(tab.labelKey) })),
      container: tabsHost,
      activeId: _activeTab,
      onChange: (id) => {
        if (id === _activeTab) return;
        window.location.hash = buildActivityTabHash(id);
      },
    });
  }

  await _loadTab(_activeTab);
}

async function _loadTab(tabId) {
  const content = document.getElementById("activityTabContent");
  if (!content) return;

  if (_activeTabModule && typeof _activeTabModule.destroy === "function") {
    try {
      _activeTabModule.destroy();
    } catch {
      /* ignore */
    }
  }
  _activeTabModule = null;
  content.innerHTML = `<div class="loading-placeholder">${t("common.loading")}</div>`;

  try {
    const mod = await activityTabLoader(tabId)();
    _activeTabModule = mod;
    content.innerHTML = "";
    if (typeof mod.render === "function") {
      await mod.render(content);
    }
  } catch (err) {
    console.error("[Activity] Failed to load tab:", tabId, err);
    content.innerHTML = `<div class="page-error">${t("router.page_load_failed")}</div>`;
  }
}

export function destroy() {
  _destroyActiveTab();
  _container = null;
  _activeTab = "timeline";
}
