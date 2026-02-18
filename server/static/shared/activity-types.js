// ── Unified Activity Type Definitions ──────────
// Canonical event type → icon mapping, shared across all activity views.

// ── Type icons ──────────────────────────────────
export const TYPE_ICONS = {
  // API detailed types (primary)
  message_received: "📨",
  response_sent:    "💬",
  channel_read:     "📖",
  channel_post:     "📢",
  dm_received:      "📩",
  dm_sent:          "✉️",
  human_notify:     "📣",
  tool_use:         "🔧",
  heartbeat_start:  "🔄",
  heartbeat_end:    "✅",
  cron_executed:    "⏰",
  memory_write:     "📝",
  error:            "⚠️",
  issue_resolved:   "🎯",
  // WebSocket simplified types (fallback for real-time events)
  message:      "📩",
  heartbeat:    "💓",
  cron:         "⏰",
  chat:         "💬",
  board:        "📋",
  notification: "🔔",
  status:       "🔵",
  system:       "⚙️",
  session:      "📄",
};

const FALLBACK_ICON = "⚙️";

export function getIcon(type) {
  return TYPE_ICONS[type] || FALLBACK_ICON;
}

// ── Filter categories (detailed API types) ──────
export const TYPE_CATEGORIES = [
  { label: "All", types: [] },
  { label: "💬 MSG", types: ["message_received", "response_sent", "dm_received", "dm_sent"] },
  { label: "📢 CH", types: ["channel_read", "channel_post"] },
  { label: "🔄 HB", types: ["heartbeat_start", "heartbeat_end"] },
  { label: "⏰ CRON", types: ["cron_executed"] },
  { label: "🔧 Tool", types: ["tool_use"] },
  { label: "📝 Mem", types: ["memory_write"] },
  { label: "📣 Notify", types: ["human_notify"] },
  { label: "⚠️ Err", types: ["error", "issue_resolved"] },
];

// ── Type-based default summaries ────────────────
const TYPE_DEFAULTS = {
  message_received: "メッセージ受信",
  response_sent: "応答送信",
  channel_read: "チャネル確認",
  channel_post: "チャネル投稿",
  dm_received: "DM受信",
  dm_sent: "DM送信",
  human_notify: "人間通知",
  tool_use: "ツール実行",
  heartbeat_start: "定期巡回開始",
  heartbeat_end: "定期巡回完了",
  cron_executed: "スケジュール実行",
  memory_write: "記憶書き込み",
  error: "エラー",
  issue_resolved: "解決済み",
};

export function getDisplaySummary(evt) {
  if (evt.summary) return evt.summary;
  if (evt.content) {
    return evt.content.length > 200 ? evt.content.slice(0, 200) + "…" : evt.content;
  }
  return TYPE_DEFAULTS[evt.type] || "";
}

// ── Event normalizer (for workspace timeline compatibility) ──
export function normalizeEvent(evt) {
  const out = { ...evt };
  // animas (array) → anima (string)
  if (Array.isArray(out.animas) && !out.anima) {
    out.anima = out.animas.join(", ");
  }
  // timestamp → ts
  if (out.timestamp && !out.ts) {
    out.ts = out.timestamp;
  }
  // metadata → meta
  if (out.metadata && !out.meta) {
    out.meta = out.metadata;
  }
  return out;
}
