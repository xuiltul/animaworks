/* ── Hash Router ───────────────────────────── */

// ── Route Registry ──────────────────────────

const routes = {};
let currentPage = null;
let containerEl = null;

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
// Increment on code changes to force browser re-fetch.
const _v = "?v=20260219a";

function registerRoutes() {
  routes["/"] = () => import("../pages/home.js" + _v);
  routes["/activity"] = () => import("../pages/activity.js" + _v);
  routes["/chat"] = () => import("../pages/chat.js" + _v);
  routes["/board"] = () => import("../pages/board.js" + _v);
  routes["/setup"] = () => import("../pages/setup.js" + _v);
  routes["/users"] = () => import("../pages/users.js" + _v);
  routes["/animas"] = () => import("../pages/animas.js" + _v);
  routes["/processes"] = () => import("../pages/processes.js" + _v);
  routes["/server"] = () => import("../pages/server-page.js" + _v);
  routes["/memory"] = () => import("../pages/memory.js" + _v);
  routes["/logs"] = () => import("../pages/logs.js" + _v);
  routes["/assets"] = () => import("../pages/assets.js" + _v);
  routes["/tool-prompts"] = () => import("../pages/tool-prompts.js" + _v);
}

// ── Route Handler ───────────────────────────

async function handleRoute() {
  const hash = window.location.hash || "#/chat";
  const path = hash.slice(1) || "/chat"; // Remove leading '#'

  const loader = routes[path];
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
  updateActiveNav(path);

  // Load and render new page
  try {
    const mod = await loader();
    currentPage = mod;
    if (typeof mod.render === "function") {
      await mod.render(containerEl);
    } else {
      console.error(`[Router] Page module for "${path}" has no render() function`);
      containerEl.innerHTML = '<div class="page-error">ページの読み込みに失敗しました</div>';
    }
  } catch (err) {
    console.error(`[Router] Failed to load page "${path}":`, err);
    containerEl.innerHTML = '<div class="page-error">ページの読み込みに失敗しました</div>';
    currentPage = null;
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
