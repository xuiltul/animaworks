/* ── Generic Page Tabs ─────────────────────────
 * Reusable tab bar using existing .page-tabs / .page-tab CSS.
 *
 * Usage:
 *   import { createPageTabs } from "/shared/page-tabs.js";
 *   const tabs = createPageTabs({
 *     tabs: [{ id: "overview", label: "Overview" }, { id: "process", label: "Process" }],
 *     container: el,
 *     activeId: "overview",
 *     onChange: (tabId) => { ... },
 *   });
 *   tabs.setActive("process");
 *   tabs.destroy();
 */

/**
 * @typedef {{ id: string, label: string }} PageTabDef
 *
 * @param {object} options
 * @param {PageTabDef[]} options.tabs - Tab definitions (id + display label)
 * @param {HTMLElement} options.container - Parent element to append the tab bar into
 * @param {(tabId: string) => void} [options.onChange] - Called when user selects a tab
 * @param {string} [options.activeId] - Initially active tab id (defaults to first tab)
 * @returns {{
 *   el: HTMLElement,
 *   setActive: (tabId: string) => void,
 *   getActive: () => string,
 *   destroy: () => void,
 * }}
 */
export function createPageTabs({ tabs, container, onChange, activeId } = {}) {
  if (!Array.isArray(tabs) || tabs.length === 0) {
    throw new Error("createPageTabs: tabs must be a non-empty array");
  }
  if (!container) {
    throw new Error("createPageTabs: container is required");
  }

  let active = activeId && tabs.some((t) => t.id === activeId) ? activeId : tabs[0].id;

  const el = document.createElement("div");
  el.className = "page-tabs";
  el.setAttribute("role", "tablist");

  function _renderButtons() {
    el.innerHTML = tabs
      .map((tab) => {
        const isActive = tab.id === active;
        return (
          `<button type="button" role="tab" class="page-tab${isActive ? " active" : ""}"` +
          ` data-tab="${_escapeAttr(tab.id)}"` +
          ` aria-selected="${isActive ? "true" : "false"}">` +
          `${tab.label}</button>`
        );
      })
      .join("");

    el.querySelectorAll(".page-tab").forEach((btn) => {
      btn.addEventListener("click", () => {
        const id = btn.dataset.tab;
        if (!id || id === active) return;
        setActive(id);
        if (typeof onChange === "function") onChange(id);
      });
    });
  }

  function setActive(tabId) {
    if (!tabs.some((t) => t.id === tabId)) return;
    active = tabId;
    el.querySelectorAll(".page-tab").forEach((btn) => {
      const isActive = btn.dataset.tab === tabId;
      btn.classList.toggle("active", isActive);
      btn.setAttribute("aria-selected", isActive ? "true" : "false");
    });
  }

  function getActive() {
    return active;
  }

  function destroy() {
    el.remove();
  }

  _renderButtons();
  container.appendChild(el);

  return { el, setActive, getActive, destroy };
}

function _escapeAttr(str) {
  return String(str ?? "")
    .replace(/&/g, "&amp;")
    .replace(/"/g, "&quot;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}
