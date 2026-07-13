// ── Activity Feed ──────────────────────
// Extracted from app.js: addActivity + loadActivityHistory

import { escapeHtml } from "./utils.js";
import { getIcon } from "../../shared/activity-types.js";
import {
  createContextBadge,
  decorateContextElement,
  updateLiveParallelIndicators,
} from "../../shared/activity-context.js";
import { basePath } from "/shared/base-path.js";
import { createLogger } from "../../shared/logger.js";
import { t } from "/shared/i18n.js";

const logger = createLogger("ws-activity");

let _activityHistoryLoaded = false;
let _paneActivity = null;

export function initActivity(paneActivityEl) {
  _paneActivity = paneActivityEl;
}

export function addActivity(type, animaName, summary, isoTs, ctx = "") {
  if (!_paneActivity) return;

  const d = isoTs ? new Date(isoTs) : new Date();
  const ts = d.toLocaleTimeString("ja-JP", { hour: "2-digit", minute: "2-digit" });
  const icon = getIcon(type);

  const entry = document.createElement("div");
  entry.className = "activity-entry";
  decorateContextElement(entry, ctx);
  entry.dataset.activityAnima = animaName;
  entry.dataset.activityTs = String(d.getTime());
  entry.innerHTML = `
    <span class="activity-time">${ts}</span>
    <span class="activity-icon">${icon}</span>
    <span class="activity-anima">${escapeHtml(animaName)}</span>
    <span class="activity-summary">${escapeHtml(summary)}</span>`;
  const contextBadge = createContextBadge(ctx);
  if (contextBadge) entry.querySelector(".activity-summary")?.before(contextBadge);

  _paneActivity.prepend(entry);
  if (window.lucide) lucide.createIcons({ nodes: [entry] });

  while (_paneActivity.children.length > 200) {
    _paneActivity.removeChild(_paneActivity.lastChild);
  }
  updateLiveParallelIndicators(_paneActivity, t);
}

export async function loadActivityHistory() {
  if (_activityHistoryLoaded || !_paneActivity) return;
  _activityHistoryLoaded = true;

  try {
    const res = await fetch(`${basePath}/api/activity/recent?hours=24&limit=50&offset=0`);
    if (!res.ok) return;
    const data = await res.json();
    const events = data.events || [];

    for (const evt of events) {
      const type = evt.type || "system";
      const anima = evt.anima || evt.animas || "";
      const summary = evt.summary || evt.content || "";
      const ts = evt.ts || evt.timestamp || "";
      addActivity(type, typeof anima === "string" ? anima : String(anima), summary, ts, evt.ctx || "");
    }
  } catch (err) {
    logger.error("Failed to load activity history", { error: err.message });
  }
}

export function resetActivityHistory() {
  _activityHistoryLoaded = false;
}
