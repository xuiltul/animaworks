// ── Unified Activity Type Definitions ──────────
// Canonical event type → icon mapping, shared across all activity views.

// ── Type icons ──────────────────────────────────
export const TYPE_ICONS = {
  message_received: { emoji: "📨", lucide: "mail" },
  response_sent:    { emoji: "💬", lucide: "message-circle" },
  channel_read:     { emoji: "📖", lucide: "book-open" },
  channel_post:     { emoji: "📢", lucide: "megaphone" },
  dm_received:      { emoji: "📩", lucide: "inbox" },
  dm_sent:          { emoji: "✉️", lucide: "send" },
  human_notify:     { emoji: "📣", lucide: "bell-ring" },
  tool_use:         { emoji: "🔧", lucide: "wrench" },
  tool_result:      { emoji: "📋", lucide: "clipboard-check" },
  heartbeat_start:       { emoji: "🔄", lucide: "refresh-cw" },
  heartbeat_end:         { emoji: "💓", lucide: "heart-pulse" },
  heartbeat_reflection:  { emoji: "💭", lucide: "brain" },
  cron_executed:    { emoji: "⏰", lucide: "clock" },
  memory_write:     { emoji: "📝", lucide: "pencil" },
  error:            { emoji: "⚠️", lucide: "alert-triangle" },
  issue_resolved:   { emoji: "🎯", lucide: "check-circle" },
  message:      { emoji: "📩", lucide: "inbox" },
  heartbeat:    { emoji: "💓", lucide: "heart-pulse" },
  cron:         { emoji: "⏰", lucide: "clock" },
  chat:         { emoji: "💬", lucide: "message-circle" },
  board:        { emoji: "📋", lucide: "clipboard-list" },
  notification: { emoji: "🔔", lucide: "bell" },
  status:       { emoji: "🔵", lucide: "circle" },
  system:       { emoji: "⚙️", lucide: "settings" },
  session:      { emoji: "📄", lucide: "file-text" },
};

export function getIcon(type) {
  const entry = TYPE_ICONS[type] || { emoji: "📌", lucide: "pin" };
  const isBusiness = document.body.classList.contains('theme-business');
  if (isBusiness) {
    return `<i data-lucide="${entry.lucide}" style="width:16px;height:16px;display:inline-block;vertical-align:middle;"></i>`;
  }
  return entry.emoji;
}

// ── Filter categories (detailed API types) ──────
export const TYPE_CATEGORIES = [
  { label: "All", types: [] },
  { label: "💬 MSG", types: ["message_received", "response_sent", "dm_received", "dm_sent"] },
  { label: "📢 CH", types: ["channel_read", "channel_post"] },
  { label: "🔄 HB", types: ["heartbeat_start", "heartbeat_end", "heartbeat_reflection"] },
  { label: "⏰ CRON", types: ["cron_executed"] },
  { label: "🔧 Tool", types: ["tool_use", "tool_result"] },
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
  tool_result: "ツール結果",
  heartbeat_start: "定期巡回開始",
  heartbeat_end: "定期巡回完了",
  heartbeat_reflection: "巡回振り返り",
  cron_executed: "スケジュール実行",
  memory_write: "記憶書き込み",
  error: "エラー",
  issue_resolved: "解決済み",
};

export function getDisplaySummary(evt) {
  if (evt.type === "tool_use") {
    return evt.tool || "ツール実行";
  }
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
