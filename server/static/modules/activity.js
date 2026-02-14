/* ── Activity Feed ─────────────────────────── */

import { dom, nowTimeStr, escapeHtml } from "./state.js";

const TYPE_ICONS = {
  heartbeat: "\uD83D\uDC93",   // heart
  cron: "\u23F0",               // alarm clock
  chat: "\uD83D\uDCAC",        // speech bubble
  system: "\u2699\uFE0F",      // gear
};

let activityEmpty = true;

export function addActivity(type, personName, summary) {
  const feed = dom.activityFeed || document.getElementById("activityFeed");
  if (!feed) return; // Activity feed not in DOM (page not active)

  if (activityEmpty) {
    feed.innerHTML = "";
    activityEmpty = false;
  }

  const icon = TYPE_ICONS[type] || TYPE_ICONS.system;
  const entry = document.createElement("div");
  entry.className = "activity-entry";
  entry.innerHTML = `
    <span class="activity-icon">${icon}</span>
    <span class="activity-time">${nowTimeStr()}</span>
    <div class="activity-body">
      <span class="activity-person">${escapeHtml(personName)}</span>
      <span class="activity-summary"> ${escapeHtml(summary)}</span>
    </div>`;
  feed.appendChild(entry);
  feed.scrollTop = feed.scrollHeight;

  // Cap at 200 entries
  while (feed.children.length > 200) {
    feed.removeChild(feed.firstChild);
  }
}
