/* ── State & DOM refs ──────────────────────── */

export const state = {
  currentUser: localStorage.getItem("animaworks_user") || null,
  persons: [],            // PersonStatus[]
  selectedPerson: null,   // string (name)
  personDetail: null,     // full detail object
  chatHistories: {},      // { [name]: [{role, text}] }
  activeMemoryTab: "episodes",
  activeRightTab: "state",
  wsConnected: false,
  sessionList: null,      // cached session list for selected person
};

const $id = (id) => document.getElementById(id);

export const dom = {
  systemStatus: $id("systemStatus"),
  systemStatusText: $id("systemStatusText"),
  loginScreen: $id("loginScreen"),
  userList: $id("userList"),
  guestLoginBtn: $id("guestLoginBtn"),
  userInfo: $id("userInfo"),
  currentUserLabel: $id("currentUserLabel"),
  logoutBtn: $id("logoutBtn"),
};

// ── Helpers ────────────────────────────────

export function timeStr(isoOrTs) {
  if (!isoOrTs) return "--:--";
  const d = new Date(isoOrTs);
  if (isNaN(d.getTime())) return "--:--";
  return d.toLocaleTimeString("ja-JP", { hour: "2-digit", minute: "2-digit" });
}

export function nowTimeStr() {
  return new Date().toLocaleTimeString("ja-JP", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

export function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

export function statusClass(status) {
  if (!status) return "status-offline";
  const s = status.toLowerCase();
  if (s === "idle" || s === "running") return "status-idle";
  if (s === "thinking" || s === "processing" || s === "busy") return "status-thinking";
  if (s === "error") return "status-error";
  return "status-offline";
}

export function renderMarkdown(text) {
  try {
    return marked.parse(text, { breaks: true });
  } catch {
    return escapeHtml(text);
  }
}
