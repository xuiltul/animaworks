/* ── System Status ─────────────────────────── */

import { t } from "/shared/i18n.js";
import { state, dom } from "./state.js";
import { api } from "./api.js";

export async function loadSystemStatus() {
  try {
    const data = await api("/api/system/status");
    const animaCount = data.animas || 0;
    const schedulerRunning = data.scheduler_running;
    const dot = dom.systemStatus.querySelector(".status-dot");
    if (schedulerRunning) {
      dot.className = "status-dot status-idle";
      dom.systemStatusText.textContent = t("status.scheduler_running", { count: animaCount });
    } else {
      dot.className = "status-dot status-error";
      dom.systemStatusText.textContent = t("status.scheduler_stopped");
    }
  } catch {
    updateSystemStatus();
  }
}

export function updateSystemStatus() {
  const dot = dom.systemStatus.querySelector(".status-dot");
  if (state.wsConnected) {
    dot.className = "status-dot status-idle";
    dom.systemStatusText.textContent = t("status.connected", { count: state.animas.length });
  } else {
    dot.className = "status-dot status-offline";
    dom.systemStatusText.textContent = t("status.reconnecting");
  }
}
