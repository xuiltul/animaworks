// ── Unified Activity Type Definitions ──────────
// Canonical event type → icon mapping, shared across all activity views.

import { t } from "/shared/i18n.js";

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
  { label: "All", i18nKey: "activity.filter_all", types: [] },
  { label: "💬 MSG", i18nKey: "activity.filter_msg", types: ["message_received", "response_sent", "dm_received", "dm_sent"] },
  { label: "📢 CH", i18nKey: "activity.filter_ch", types: ["channel_read", "channel_post"] },
  { label: "🔄 HB", i18nKey: "activity.filter_hb", types: ["heartbeat_start", "heartbeat_end", "heartbeat_reflection"] },
  { label: "⏰ CRON", i18nKey: "activity.filter_cron", types: ["cron_executed"] },
  { label: "🔧 Tool", i18nKey: "activity.filter_tool", types: ["tool_use", "tool_result"] },
  { label: "📝 Mem", i18nKey: "activity.filter_mem", types: ["memory_write"] },
  { label: "📣 Notify", i18nKey: "activity.filter_notify", types: ["human_notify"] },
  { label: "⚠️ Err", i18nKey: "activity.filter_err", types: ["error", "issue_resolved"] },
];

// ── Type-based default summaries (i18n keys) ────
const TYPE_DEFAULT_KEYS = {
  message_received: "activity_types.message_received",
  response_sent: "activity_types.response_sent",
  channel_read: "activity_types.channel_read",
  channel_post: "activity_types.channel_post",
  dm_received: "activity_types.dm_received",
  dm_sent: "activity_types.dm_sent",
  human_notify: "activity_types.human_notify",
  tool_use: "activity_types.tool_use",
  tool_result: "activity_types.tool_result",
  heartbeat_start: "activity_types.heartbeat_start",
  heartbeat_end: "activity_types.heartbeat_end",
  heartbeat_reflection: "activity_types.heartbeat_reflection",
  cron_executed: "activity_types.cron_executed",
  memory_write: "activity_types.memory_write",
  error: "activity_types.error",
  issue_resolved: "activity_types.issue_resolved",
};

export function getDisplaySummary(evt) {
  if (evt.type === "tool_use") {
    return evt.tool || t("activity_types.tool_use");
  }
  if (evt.summary) return evt.summary;
  if (evt.content) {
    return evt.content.length > 200 ? evt.content.slice(0, 200) + "…" : evt.content;
  }
  const key = TYPE_DEFAULT_KEYS[evt.type];
  return key ? t(key) : "";
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
