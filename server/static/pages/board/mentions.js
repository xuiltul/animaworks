// ── Board mention audience ───────────────────
// Shared rules for who a board mention reaches. The composer uses the same
// judge for its @ suggestions and for the "who will be notified" preview, and
// the delivery footer uses it to explain why someone was not reached.
import { t } from "/shared/i18n.js";

const MENTION_RE = /@([a-zA-Z][a-zA-Z0-9_-]{0,30})/g;

/** Return the unique anima names mentioned in *text* (case preserved). */
export function parseMentions(text) {
  const names = [];
  const seen = new Set();
  MENTION_RE.lastIndex = 0;
  let match = MENTION_RE.exec(String(text || ""));
  while (match) {
    const name = match[1];
    const key = name.toLowerCase();
    if (!seen.has(key)) {
      seen.add(key);
      names.push(name);
    }
    match = MENTION_RE.exec(String(text || ""));
  }
  return names;
}

/**
 * Can this board call *name*? Returns `{ ok: true }` or
 * `{ ok: false, reason }`. Unknown names return null.
 */
export function mentionAudience(store, channel, name) {
  const anima = store.anima(name);
  if (!anima) return null;
  if (anima.enabled === false || store.statusOf(name) === "stopped") {
    return { ok: false, reason: t("board.mention_unreachable_disabled") };
  }
  const access = channel?.access || "open";
  if (access === "members") {
    const members = Array.isArray(channel.members) ? channel.members : [];
    if (!members.includes(name)) {
      return { ok: false, reason: t("board.mention_unreachable_member") };
    }
    return { ok: true };
  }
  const company = String(channel?.company || "").trim();
  if (company && String(anima.company || "").trim() !== company) {
    return { ok: false, reason: t("board.mention_unreachable_company") };
  }
  return { ok: true };
}

/** Mention names in *text* that exist in the anima directory. */
export function knownMentions(store, text) {
  return parseMentions(text).filter((name) => store.hasAnima(name));
}

/** Animas that can be called from *channel* (suggestion candidates). */
export function callableAnimas(store, channel) {
  const rows = store.state.animas.filter((anima) => mentionAudience(store, channel, anima.name)?.ok);
  return rows.sort((a, b) => String(a.name).localeCompare(String(b.name)));
}

/** Animas that can be added as members of *channel*. */
export function memberCandidates(store, channel) {
  const members = new Set(Array.isArray(channel?.members) ? channel.members : []);
  const company = String(channel?.company || "").trim();
  return store.state.animas
    .filter((anima) => !members.has(anima.name))
    .filter((anima) => !company || String(anima.company || "").trim() === company)
    .sort((a, b) => String(a.name).localeCompare(String(b.name)));
}
