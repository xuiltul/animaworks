// ── Text Artifact Popup ──────────────────────────────────
// Shared handler for file: code block artifacts.
// Uses event delegation (attached once to document).

let _initialized = false;

const _contentStore = new Map();
let _idCounter = 0;

/**
 * Register a large content entry, returning a store key.
 */
export function registerArtifactContent(content) {
  const id = `artifact-${++_idCounter}`;
  _contentStore.set(id, content);
  return id;
}

window.__registerArtifactContent = registerArtifactContent;

/**
 * Initialize click handler for .text-artifact-card elements.
 * Safe to call multiple times (attaches once).
 */
export function initTextArtifactHandlers() {
  if (_initialized) return;
  _initialized = true;

  document.addEventListener("click", (e) => {
    const card = e.target.closest(".text-artifact-card");
    if (!card) return;

    const filename = card.dataset.filename || "untitled.txt";
    const storeId = card.dataset.contentId;
    const content = storeId
      ? (_contentStore.get(storeId) || "")
      : (card.dataset.content || "");

    _openModal(filename, content);
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") _closeModal();
  });
}

function _openModal(filename, content) {
  _closeModal();

  const overlay = document.createElement("div");
  overlay.className = "text-artifact-modal-overlay";
  overlay.addEventListener("click", (e) => {
    if (e.target === overlay) _closeModal();
  });

  overlay.innerHTML = `
    <div class="text-artifact-modal">
      <div class="text-artifact-modal-header">
        <span class="text-artifact-modal-filename">${_esc(filename)}</span>
        <button class="text-artifact-modal-close" aria-label="Close">&times;</button>
      </div>
      <div class="text-artifact-modal-body">
        <textarea class="text-artifact-modal-textarea" spellcheck="false">${_esc(content)}</textarea>
      </div>
      <div class="text-artifact-modal-footer">
        <button class="text-artifact-btn text-artifact-btn-copy">Copy</button>
        <button class="text-artifact-btn text-artifact-btn-download">Download</button>
      </div>
    </div>
  `;

  document.body.appendChild(overlay);

  overlay.querySelector(".text-artifact-modal-close").addEventListener("click", _closeModal);

  overlay.querySelector(".text-artifact-btn-copy").addEventListener("click", (e) => {
    const textarea = overlay.querySelector(".text-artifact-modal-textarea");
    const btn = e.currentTarget;
    if (navigator.clipboard?.writeText) {
      navigator.clipboard.writeText(textarea.value).then(() => {
        btn.textContent = "Copied!";
        setTimeout(() => { btn.textContent = "Copy"; }, 1500);
      }).catch(() => {
        _fallbackCopy(textarea, btn);
      });
    } else {
      _fallbackCopy(textarea, btn);
    }
  });

  overlay.querySelector(".text-artifact-btn-download").addEventListener("click", () => {
    const textarea = overlay.querySelector(".text-artifact-modal-textarea");
    const blob = new Blob([textarea.value], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  });
}

function _closeModal() {
  document.querySelector(".text-artifact-modal-overlay")?.remove();
}

function _fallbackCopy(textarea, btn) {
  textarea.select();
  try {
    document.execCommand("copy");
    btn.textContent = "Copied!";
  } catch {
    btn.textContent = "Failed";
  }
  setTimeout(() => { btn.textContent = "Copy"; }, 1500);
}

function _esc(str) {
  const d = document.createElement("div");
  d.textContent = str;
  return d.innerHTML;
}
