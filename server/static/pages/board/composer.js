// ── Board composer ───────────────────────────
// Writes posts, suggests @-targets that are actually reachable from the
// current board, previews Markdown, and keeps a per-board draft in
// localStorage. The page owns submission; this module owns the input UX.
import { t } from "/shared/i18n.js";
import { escapeAttr, escapeHtml, renderMarkdown } from "../../modules/state.js";
import { avatarMarkup, statusLabel } from "./avatar.js";
import { decorateMarkdown } from "./markdown.js";
import { callableAnimas, knownMentions, mentionAudience } from "./mentions.js";

const DRAFT_PREFIX = "animaworks-board-draft:v1";
const SUGGEST_RE = /@([a-zA-Z0-9_-]*)$/;

export function createComposer({ store, onSubmit }) {
  const form = document.createElement("form");
  form.className = "board-composer";
  form.id = "boardComposer";
  form.setAttribute("data-board-composer", "");
  form.innerHTML = `
    <div class="board-composer-suggest" id="boardComposerSuggest" role="listbox" aria-label="${escapeHtml(t("board.suggest_label"))}" hidden></div>
    <div class="board-composer-wrap">
      <textarea id="boardComposerInput" class="board-composer-input" rows="1" autocomplete="off" placeholder="${escapeHtml(t("board.message_placeholder"))}" aria-label="${escapeHtml(t("board.message_input"))}"></textarea>
      <div class="board-composer-preview board-markdown" id="boardComposerPreview" hidden></div>
      <div class="board-composer-targets" id="boardComposerTargets" aria-live="polite" hidden></div>
      <div class="board-composer-tools">
        <button type="button" class="board-composer-preview-toggle" id="boardComposerPreviewToggle" aria-pressed="false">${escapeHtml(t("board.preview"))}</button>
        <span class="board-composer-draft-note" id="boardComposerDraftNote"></span>
      </div>
    </div>
    <button type="submit" class="board-composer-send">${escapeHtml(t("board.send"))}</button>
    <div class="board-composer-error" hidden></div>`;

  const textarea = form.querySelector("#boardComposerInput");
  const sendButton = form.querySelector(".board-composer-send");
  const errorBox = form.querySelector(".board-composer-error");
  const suggestBox = form.querySelector("#boardComposerSuggest");
  const previewBox = form.querySelector("#boardComposerPreview");
  const previewToggle = form.querySelector("#boardComposerPreviewToggle");
  const targetsBox = form.querySelector("#boardComposerTargets");
  const draftNote = form.querySelector("#boardComposerDraftNote");

  const markdownCtx = {
    t,
    isAnima: (name) => store.hasAnima(name),
    hasChannel: (name) => store.hasChannel(name),
    avatarUrl: (name) => store.avatarUrl(name),
    statusOf: (name) => store.statusOf(name),
    statusLabel: (name) => t(`board.mention_status_${store.statusOf(name)}`),
  };

  let context = { type: null, name: null, writable: true, channel: null };
  let activeKey = "";
  let replyTo = null;
  let writable = true;
  let sending = false;
  let previewOn = false;
  let suggestItems = [];
  let suggestIndex = 0;

  function draftKey() {
    if (context.type !== "channel" || !context.name) return "";
    return `${DRAFT_PREFIX}:channel:${context.name}`;
  }

  function readDraft(key) {
    if (!key) return "";
    try {
      return localStorage.getItem(key) || "";
    } catch {
      return "";
    }
  }

  function writeDraft(key, value) {
    if (!key) return;
    try {
      if (value) localStorage.setItem(key, value);
      else localStorage.removeItem(key);
    } catch {
      // Drafting is best-effort (private mode / quota).
    }
  }

  function resize() {
    textarea.style.height = "auto";
    textarea.style.height = `${Math.min(textarea.scrollHeight, 160)}px`;
  }

  function setError(message) {
    if (!message) {
      errorBox.hidden = true;
      errorBox.textContent = "";
      return;
    }
    errorBox.hidden = false;
    errorBox.textContent = message;
  }

  function setDraftNote(message) {
    draftNote.textContent = message || "";
  }

  // ── @ suggestions ────────────────────────
  function closeSuggest() {
    suggestItems = [];
    suggestIndex = 0;
    suggestBox.hidden = true;
    suggestBox.innerHTML = "";
  }

  function currentQuery() {
    const caret = textarea.selectionStart ?? textarea.value.length;
    const before = textarea.value.slice(0, caret);
    const match = before.match(SUGGEST_RE);
    if (!match) return null;
    return { text: match[1].toLowerCase(), start: caret - match[0].length, end: caret };
  }

  function suggestHead() {
    const channel = context.channel;
    if (channel?.access === "members") return t("board.suggest_head_members");
    if (channel?.company) return t("board.suggest_head_company", { company: channel.company });
    return t("board.suggest_head_all");
  }

  function renderSuggest() {
    if (suggestItems.length === 0) {
      closeSuggest();
      return;
    }
    const items = suggestItems.map((name, index) => {
      const anima = store.anima(name);
      const status = store.statusOf(name);
      const active = index === suggestIndex;
      return `<button type="button" class="board-suggest-item${active ? " board-is-active" : ""}" role="option" aria-selected="${active}" data-board-suggest-name="${escapeAttr(name)}">
        ${avatarMarkup({ name, kind: "anima", url: store.avatarUrl(name), size: "xs" })}
        <span class="board-suggest-name">@${escapeHtml(name)}</span>
        <span class="board-suggest-role">${escapeHtml(anima?.role || "")}</span>
        <span class="board-suggest-status status-${escapeHtml(status)}">${escapeHtml(statusLabel(status))}</span>
      </button>`;
    });
    suggestBox.innerHTML = `<div class="board-suggest-head">${escapeHtml(suggestHead())}</div>${items.join("")}`;
    suggestBox.hidden = false;
  }

  function updateSuggest() {
    if (!writable || !context.channel) {
      closeSuggest();
      return;
    }
    const query = currentQuery();
    if (!query) {
      closeSuggest();
      return;
    }
    suggestItems = callableAnimas(store, context.channel)
      .map((anima) => anima.name)
      .filter((name) => name.toLowerCase().startsWith(query.text));
    if (suggestItems.length === 0) {
      closeSuggest();
      return;
    }
    suggestIndex = Math.min(suggestIndex, suggestItems.length - 1);
    renderSuggest();
  }

  function pick(name) {
    const query = currentQuery();
    if (!query) return;
    const before = textarea.value.slice(0, query.start);
    const after = textarea.value.slice(query.end);
    const inserted = `@${name} `;
    textarea.value = `${before}${inserted}${after}`;
    const caret = before.length + inserted.length;
    textarea.setSelectionRange(caret, caret);
    closeSuggest();
    textarea.focus();
    handleInput();
  }

  // ── Targets / preview ────────────────────
  function updateTargets() {
    if (!writable || !context.channel) {
      targetsBox.hidden = true;
      targetsBox.innerHTML = "";
      return;
    }
    const names = knownMentions(store, textarea.value);
    if (names.length === 0) {
      targetsBox.hidden = true;
      targetsBox.innerHTML = "";
      return;
    }
    const items = names.map((name) => {
      const judge = mentionAudience(store, context.channel, name);
      const avatar = avatarMarkup({ name, kind: "anima", url: store.avatarUrl(name), size: "xs" });
      if (judge?.ok) {
        return `<span class="board-target">${avatar}<b class="board-target-name">${escapeHtml(name)}</b><span class="board-target-status">${escapeHtml(statusLabel(store.statusOf(name)))}</span></span>`;
      }
      return `<span class="board-target board-target-ng">${avatar}<b class="board-target-name">${escapeHtml(name)}</b><span class="board-target-reason">${escapeHtml(judge?.reason || t("board.mention_unreachable_member"))}</span></span>`;
    });
    targetsBox.innerHTML = `<span class="board-targets-label">${escapeHtml(t("board.notify_targets"))}</span>${items.join("")}`;
    targetsBox.hidden = false;
  }

  function renderPreview() {
    previewBox.hidden = !previewOn;
    textarea.hidden = previewOn;
    previewToggle.setAttribute("aria-pressed", String(previewOn));
    previewToggle.classList.toggle("board-is-active", previewOn);
    if (!previewOn) {
      previewBox.innerHTML = "";
      return;
    }
    const text = textarea.value.trim();
    if (!text) {
      previewBox.innerHTML = `<p class="board-composer-preview-empty">${escapeHtml(t("board.preview_empty"))}</p>`;
      return;
    }
    try {
      previewBox.innerHTML = decorateMarkdown(renderMarkdown(text), markdownCtx);
    } catch {
      previewBox.textContent = text;
    }
  }

  function setPreview(open) {
    previewOn = Boolean(open);
    renderPreview();
    resize();
  }

  function handleInput() {
    resize();
    updateTargets();
    updateSuggest();
    writeDraft(draftKey(), textarea.value);
    if (textarea.value && draftNote.textContent !== t("board.draft_restored")) {
      setDraftNote(t("board.draft_saved"));
    } else if (!textarea.value) {
      setDraftNote("");
    }
    if (previewOn) renderPreview();
  }

  function loadDraft() {
    const key = draftKey();
    const value = readDraft(key);
    textarea.value = value;
    replyTo = null;
    setDraftNote(value ? t("board.draft_restored") : "");
    updateTargets();
    closeSuggest();
    if (previewOn) renderPreview();
    resize();
  }

  // ── Submit / quote ───────────────────────
  async function submit() {
    if (!writable || sending) return;
    const text = textarea.value.trim();
    if (!text) return;
    sending = true;
    textarea.disabled = true;
    sendButton.disabled = true;
    setError("");
    const options = replyTo ? { replyTo } : {};
    const result = await onSubmit(text, options);
    sending = false;
    textarea.disabled = false;
    sendButton.disabled = false;
    if (result?.ok) {
      textarea.value = "";
      replyTo = null;
      writeDraft(draftKey(), "");
      setDraftNote("");
      setPreview(false);
      updateTargets();
      closeSuggest();
      resize();
      textarea.focus();
    } else {
      setError(`${t("board.send_failed")}${result?.error ? `: ${result.error}` : ""}`);
      textarea.focus();
    }
  }

  function quote(message) {
    if (!message || !writable) return;
    const first = String(message.text || "").split("\n").find((line) => line.trim() !== "");
    const quoted = `> ${(first || "").trim()}`;
    const line = store.hasAnima(message.from) ? `@${message.from} ` : "";
    textarea.value = `${quoted}\n${line}`;
    replyTo = message.id !== undefined && message.id !== null && message.id !== "" ? String(message.id) : null;
    handleInput();
    textarea.focus();
    const caret = textarea.value.length;
    textarea.setSelectionRange(caret, caret);
  }

  // ── Wiring ───────────────────────────────
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    void submit();
  });

  textarea.addEventListener("keydown", (event) => {
    if (event.isComposing || event.keyCode === 229) return;
    if (!suggestBox.hidden && suggestItems.length > 0) {
      if (event.key === "ArrowDown") {
        event.preventDefault();
        suggestIndex = (suggestIndex + 1) % suggestItems.length;
        renderSuggest();
        return;
      }
      if (event.key === "ArrowUp") {
        event.preventDefault();
        suggestIndex = (suggestIndex - 1 + suggestItems.length) % suggestItems.length;
        renderSuggest();
        return;
      }
      if (event.key === "Enter" || event.key === "Tab") {
        event.preventDefault();
        pick(suggestItems[suggestIndex]);
        return;
      }
      if (event.key === "Escape") {
        event.preventDefault();
        closeSuggest();
        return;
      }
    }
    if (event.key !== "Enter") return;
    if (event.ctrlKey || event.metaKey || !event.shiftKey) {
      event.preventDefault();
      void submit();
    }
  });

  textarea.addEventListener("input", handleInput);
  textarea.addEventListener("blur", () => {
    setTimeout(() => {
      if (!form.contains(document.activeElement)) closeSuggest();
    }, 120);
  });
  suggestBox.addEventListener("mousedown", (event) => {
    const item = event.target.closest("[data-board-suggest-name]");
    if (!item) return;
    event.preventDefault();
    pick(item.dataset.boardSuggestName);
  });
  previewToggle.addEventListener("click", () => setPreview(!previewOn));

  function setContext({ type, name, writable: nextWritable, channel }) {
    writable = Boolean(nextWritable);
    form.hidden = !writable;
    const key = type === "channel" && name ? `${type}:${name}` : `${type || ""}:${name || ""}`;
    const changed = key !== activeKey;
    context = { type, name, writable, channel: channel || null };
    activeKey = key;

    if (type === "channel" && writable) {
      textarea.placeholder = t("board.message_post", { channel: name || "" });
    } else {
      textarea.placeholder = t("board.message_placeholder");
    }

    if (changed) loadDraft();
    updateTargets();
    if (!suggestBox.hidden) updateSuggest();
    if (previewOn) renderPreview();
  }

  /** Re-evaluate suggestions/targets after the anima directory loads. */
  function refresh() {
    updateTargets();
    if (!suggestBox.hidden) updateSuggest();
    if (previewOn) renderPreview();
  }

  resize();

  return {
    element: form,
    setContext,
    refresh,
    quote,
    focus: () => textarea.focus(),
    destroy: () => {
      form.remove();
    },
  };
}
