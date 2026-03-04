// ── Activity Feed / Anima State Controller ────
export function createActivityController(ctx) {
  const $root = ctx.$root;
  const { state, deps } = ctx;
  const { api, t, escapeHtml, timeStr, getIcon, getDisplaySummary } = deps;

  function addLocalActivity(type, animaName, summary) {
    const feed = $root("chatActivityFeed");
    if (!feed) return;

    const empty = feed.querySelector(".activity-empty");
    if (empty) empty.remove();

    const icon = getIcon(type);
    const ts = new Date().toLocaleTimeString("ja-JP", { hour: "2-digit", minute: "2-digit", second: "2-digit" });

    const entry = document.createElement("div");
    entry.className = "activity-entry";
    entry.innerHTML = `
      <span class="activity-icon">${icon}</span>
      <span class="activity-time">${ts}</span>
      <div class="activity-body">
        <span class="activity-anima">${escapeHtml(animaName)}</span>
        <span class="activity-summary"> ${escapeHtml(summary)}</span>
      </div>`;
    feed.appendChild(entry);
    feed.scrollTop = feed.scrollHeight;

    while (feed.children.length > 200) feed.removeChild(feed.firstChild);
  }

  async function loadActivity() {
    if (!state.selectedAnima) return;
    try {
      const data = await api(`/api/activity/recent?hours=6&limit=50&anima=${encodeURIComponent(state.selectedAnima)}`);
      const events = data.events || [];
      const feed = $root("chatActivityFeed");
      if (!feed) return;

      if (events.length === 0) {
        feed.innerHTML = `<div class="activity-empty">${t("activity.empty")}</div>`;
        return;
      }

      feed.innerHTML = events.slice(0, 50).map(evt => {
        const icon = getIcon(evt.type);
        const ts = timeStr(evt.ts);
        const summary = getDisplaySummary(evt);
        return `
          <div class="activity-entry">
            <span class="activity-icon">${icon}</span>
            <span class="activity-time">${escapeHtml(ts)}</span>
            <div class="activity-body">
              <span class="activity-anima">${escapeHtml(evt.anima || "")}</span>
              <span class="activity-summary"> ${escapeHtml(summary)}</span>
            </div>
          </div>`;
      }).join("");
    } catch { /* keep existing */ }
  }

  function renderAnimaState() {
    const el = $root("chatAnimaState");
    if (!el) return;
    const d = state.animaDetail;
    if (!d || !d.state) { el.textContent = t("animas.no_state"); return; }
    el.textContent = typeof d.state === "string" ? d.state : JSON.stringify(d.state, null, 2);
  }

  return { addLocalActivity, loadActivity, renderAnimaState };
}
