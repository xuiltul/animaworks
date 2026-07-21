/* ── Hash Router ───────────────────────────── */

import { applyTranslations, t } from "/shared/i18n.js";

// ── Route Registry ──────────────────────────

const routes = {};
let currentPage = null;
let containerEl = null;

// Permanent hash redirects (legacy paths → new destinations).
// Keys are path prefixes (without '#'). Values are full hashes including '#'.
// Note: /setup is handled by an explicit block in handleRoute (kept for
// compatibility with existing structure tests) and is also listed here.
export const REDIRECTS = {
  "/processes": "#/animas",
  "/server": "#/",
  "/setup": "#/settings",
  "/memory": "#/animas",
  "/assets": "#/animas",
  "/activity-report": "#/activity/report",
  "/logs": "#/activity/logs",
  "/users": "#/settings/users",
};

/**
 * Resolve a permanent redirect for a path (no leading '#').
 * Matches exact path or path under a redirect prefix.
 * @param {string} path - e.g. "/processes" or "/setup/foo"
 * @returns {string|null} target hash (e.g. "#/animas") or null
 */
export function resolveRedirect(path) {
  if (!path) return null;
  for (const [from, to] of Object.entries(REDIRECTS)) {
    if (path === from || path.startsWith(from + "/")) {
      return to;
    }
  }
  return null;
}

/**
 * Match a path against registered route keys (longest prefix wins).
 * @param {string} path - e.g. "/animas/sakura/process"
 * @param {string[]} [routeKeys] - defaults to currently registered routes
 * @returns {{ route: string, subPath: string, navPath: string } | null}
 */
export function resolveRouteMatch(path, routeKeys) {
  const keys = routeKeys || Object.keys(routes);
  if (keys.includes(path)) {
    return { route: path, subPath: "", navPath: path };
  }

  let best = null;
  for (const route of keys) {
    if (path.startsWith(route + "/")) {
      if (!best || route.length > best.route.length) {
        best = {
          route,
          subPath: decodeURIComponent(path.slice(route.length + 1)),
          navPath: route,
        };
      }
    }
  }
  return best;
}

/**
 * Parse anima detail subPath into name + tab.
 * e.g. "sakura" → { name: "sakura", tab: "overview" }
 *      "sakura/process" → { name: "sakura", tab: "process" }
 * @param {string} subPath
 * @returns {{ name: string|null, tab: string }}
 */
export function parseAnimaSubPath(subPath) {
  if (!subPath) return { name: null, tab: "overview" };
  const parts = String(subPath).split("/").filter(Boolean);
  const name = parts[0] || null;
  const tab = parts[1] || "overview";
  return { name, tab };
}

// ── Public API ──────────────────────────────

/**
 * Initialize the hash router.
 * @param {string} containerId - ID of the DOM element where pages render
 */
export function initRouter(containerId) {
  containerEl = document.getElementById(containerId);
  if (!containerEl) {
    console.error(`[Router] Container #${containerId} not found`);
    return;
  }

  // Register all routes
  registerRoutes();

  // Listen for hash changes
  window.addEventListener("hashchange", handleRoute);
  window.addEventListener("load", handleRoute);

  // Trigger initial route
  handleRoute();
}

/**
 * Programmatically navigate to a hash route.
 * @param {string} hash - Target hash (e.g. "#/chat")
 */
export function navigateTo(hash) {
  window.location.hash = hash;
}

// ── Route Registration ──────────────────────

// Cache-bust suffix for ES module dynamic imports.
// Uses a timestamp so every page load fetches the latest code.
const _v = "?v=" + Date.now();

function registerRoutes() {
  routes["/"] = () => import("../pages/home.js" + _v);
  routes["/activity"] = () => import("../pages/activity.js" + _v);
  routes["/chat"] = () => import("../pages/chat.js" + _v);
  routes["/board"] = () => import("../pages/board.js" + _v);
  routes["/task-board"] = () => import("../pages/task-board.js" + _v);
  // /users removed — redirected to #/settings/users (see REDIRECTS)
  routes["/animas"] = () => import("../pages/animas.js" + _v);
  // /processes removed — redirected to #/animas (see REDIRECTS)
  // /server removed — redirected to #/ (see REDIRECTS)
  // /memory removed — redirected to #/animas (see REDIRECTS)
  // /assets removed — redirected to #/animas (see REDIRECTS)
  // /logs removed — redirected to #/activity/logs (see REDIRECTS)
  // /activity-report removed — redirected to #/activity/report (see REDIRECTS)
  routes["/settings"] = () => import("../pages/settings.js" + _v);
}

// ── Route Handler ───────────────────────────

async function handleRoute() {
  const hash = window.location.hash || "#/chat";
  const path = hash.slice(1) || "/chat"; // Remove leading '#'

  // Legacy SPA #/setup was merged into #/settings (API & Authentication)
  if (path === "/setup" || path.startsWith("/setup/")) {
    window.location.hash = "#/settings";
    return;
  }

  // Permanent redirects (e.g. #/processes → #/animas)
  const redirect = resolveRedirect(path);
  if (redirect) {
    window.location.hash = redirect;
    return;
  }

  // Try exact match first, then longest prefix match for parameterized routes
  const matched = resolveRouteMatch(path);
  let loader = matched ? routes[matched.route] : null;
  let subPath = matched ? matched.subPath : "";
  let navPath = matched ? matched.navPath : path;

  if (!loader) {
    // Fallback to chat
    window.location.hash = "#/chat";
    return;
  }

  // Destroy previous page
  if (currentPage && typeof currentPage.destroy === "function") {
    try {
      currentPage.destroy();
    } catch (err) {
      console.error("[Router] Error destroying page:", err);
    }
  }

  // Clear container
  containerEl.innerHTML = "";

  // Update active nav item
  updateActiveNav(navPath);

  // Load and render new page
  try {
    const mod = await loader();
    currentPage = mod;
    if (typeof mod.render === "function") {
      await mod.render(containerEl, { subPath });
    } else {
      console.error(`[Router] Page module for "${path}" has no render() function`);
      containerEl.innerHTML = `<div class="page-error">${t("router.page_load_failed")}</div>`;
    }
    applyTranslations();
  } catch (err) {
    console.error(`[Router] Failed to load page "${path}":`, err);
    containerEl.innerHTML = `<div class="page-error">${t("router.page_load_failed")}</div>`;
    currentPage = null;
    applyTranslations();
  }
}

// ── Navigation Highlight ────────────────────

function updateActiveNav(path) {
  const navItems = document.querySelectorAll(".nav-item");
  navItems.forEach((item) => {
    const route = item.dataset.route;
    if (route) {
      item.classList.toggle("active", route === path);
    }
  });
}
