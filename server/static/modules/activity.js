/* ── Activity Feed ─────────────────────────── */

import { dom, nowTimeStr, escapeHtml } from "./state.js";
import { getIcon } from "../shared/activity-types.js";
import {
  createContextBadge,
  decorateContextElement,
  updateLiveParallelIndicators,
} from "../shared/activity-context.js";
import { t } from "/shared/i18n.js";

let activityEmpty = true;

export function addActivity(type, animaName, summary, ctx = "") {
  const feed = dom.activityFeed || document.getElementById("activityFeed");
  if (!feed) return; // Activity feed not in DOM (page not active)

  if (activityEmpty) {
    feed.innerHTML = "";
    activityEmpty = false;
  }

  const icon = getIcon(type);
  const entry = document.createElement("div");
  entry.className = "activity-entry";
  decorateContextElement(entry, ctx);
  entry.dataset.activityAnima = animaName;
  entry.dataset.activityTs = String(Date.now());
  entry.innerHTML = `
    <span class="activity-icon">${icon}</span>
    <span class="activity-time">${nowTimeStr()}</span>
    <div class="activity-body">
      <span class="activity-anima">${escapeHtml(animaName)}</span>
      <span class="activity-summary"> ${escapeHtml(summary)}</span>
    </div>`;
  const contextBadge = createContextBadge(ctx);
  if (contextBadge) entry.querySelector(".activity-summary")?.before(contextBadge);
  feed.appendChild(entry);
  if (window.lucide) lucide.createIcons({ nodes: [entry] });
  feed.scrollTop = feed.scrollHeight;

  // Cap at 200 entries
  while (feed.children.length > 200) {
    feed.removeChild(feed.firstChild);
  }
  updateLiveParallelIndicators(feed, t);
}
