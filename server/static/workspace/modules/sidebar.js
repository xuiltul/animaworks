// ── Right Panel Tabs ──────────────────────
// Extracted from app.js: activateRightTab + related state

import { getState, setState } from "./state.js";
import { loadSessions } from "./session.js";
import { loadActivityHistory } from "./activity.js";

let _dom = {};
let _onBoardInit = null;

/**
 * Initialize sidebar with DOM refs and board callback.
 * @param {object} dom - { tabState, tabActivity, tabBoard, tabHistory, paneState, paneActivity, paneBoard, paneHistory }
 * @param {object} opts
 * @param {function} opts.onBoardInit - Called when board tab is first activated
 */
export function initSidebar(dom, opts = {}) {
  _dom = dom;
  _onBoardInit = opts.onBoardInit || null;

  [_dom.tabState, _dom.tabActivity, _dom.tabBoard, _dom.tabHistory].forEach(btn => {
    btn?.addEventListener("click", () => activateRightTab(btn.dataset.tab));
  });
}

export function activateRightTab(tab) {
  setState({ activeRightTab: tab });

  [_dom.tabState, _dom.tabActivity, _dom.tabBoard, _dom.tabHistory].forEach(btn => {
    btn?.classList.toggle("active", btn.dataset.tab === tab);
  });

  [_dom.paneState, _dom.paneActivity, _dom.paneBoard, _dom.paneHistory].forEach(pane => {
    if (pane) pane.style.display = pane.dataset.pane === tab ? "" : "none";
  });

  if (tab === "history") loadSessions();
  if (tab === "board" && _onBoardInit) _onBoardInit();
  if (tab === "activity") loadActivityHistory();
}
