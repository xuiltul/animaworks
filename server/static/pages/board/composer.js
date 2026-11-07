// ── Board composer ───────────────────────────
import { t } from "/shared/i18n.js";
import { escapeHtml } from "../../modules/state.js";

export function createComposer({ onSubmit }) {
  const form = document.createElement("form");
  form.className = "board-composer";
  form.id = "boardComposer";
  form.setAttribute("data-board-composer", "");
  form.innerHTML = `
    <textarea id="boardComposerInput" class="board-composer-input" rows="1" autocomplete="off" placeholder="${escapeHtml(t("board.message_placeholder"))}" aria-label="${escapeHtml(t("board.message_input"))}"></textarea>
    <button type="submit" class="board-composer-send">${escapeHtml(t("board.send"))}</button>
    <div class="board-composer-error" hidden></div>`;

  const textarea = form.querySelector("#boardComposerInput");
  const sendButton = form.querySelector(".board-composer-send");
  const errorBox = form.querySelector(".board-composer-error");
  let writable = true;
  let sending = false;

  function resize() {
    textarea.style.height = "auto";
    textarea.style.height = `${Math.min(textarea.scrollHeight, 120)}px`;
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

  async function submit() {
    if (!writable || sending) return;
    const text = textarea.value.trim();
    if (!text) return;
    sending = true;
    textarea.disabled = true;
    sendButton.disabled = true;
    setError("");
    const result = await onSubmit(text);
    sending = false;
    textarea.disabled = false;
    sendButton.disabled = false;
    if (result?.ok) {
      textarea.value = "";
      resize();
      textarea.focus();
    } else {
      setError(`${t("board.send_failed")}${result?.error ? `: ${result.error}` : ""}`);
      textarea.focus();
    }
  }

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    void submit();
  });

  textarea.addEventListener("keydown", (event) => {
    if (event.isComposing || event.keyCode === 229) return;
    if (event.key !== "Enter") return;
    if (event.ctrlKey || event.metaKey || !event.shiftKey) {
      event.preventDefault();
      void submit();
    }
  });

  textarea.addEventListener("input", resize);

  function setContext({ type, name, writable: nextWritable }) {
    writable = Boolean(nextWritable);
    form.hidden = !writable;
    if (type === "channel" && writable) {
      textarea.placeholder = t("board.message_post", { channel: name || "" });
    }
  }

  resize();

  return {
    element: form,
    setContext,
    focus: () => textarea.focus(),
    destroy: () => {
      form.remove();
    },
  };
}
