// ── Memory Browser Controller ──────────────────
import { $ } from "./ctx.js";

export function createMemoryController(ctx) {
  const { state, deps } = ctx;
  const { api, t, escapeHtml } = deps;

  async function loadMemoryTab() {
    const fileList = $("chatMemoryFileList");
    if (!fileList) return;

    if (!state.selectedAnima) {
      fileList.innerHTML = `<div class="loading-placeholder">${t("chat.anima_select_first")}</div>`;
      return;
    }

    fileList.innerHTML = `<div class="loading-placeholder">${t("common.loading")}</div>`;
    const endpoint = `/api/animas/${encodeURIComponent(state.selectedAnima)}/${state.activeMemoryTab}`;

    try {
      const data = await api(endpoint);
      const files = data.files || [];
      if (files.length === 0) {
        fileList.innerHTML = `<div class="loading-placeholder">${t("memory.no_files")}</div>`;
        return;
      }
      fileList.innerHTML = files.map(f =>
        `<div class="memory-file-item" data-file="${escapeHtml(f)}" data-tab="${state.activeMemoryTab}">${escapeHtml(f)}</div>`,
      ).join("");

      fileList.querySelectorAll(".memory-file-item").forEach(item => {
        item.addEventListener("click", () => loadMemoryContent(item.dataset.tab, item.dataset.file));
      });
    } catch (err) {
      fileList.innerHTML = `<div class="loading-placeholder">${t("common.load_failed")}: ${escapeHtml(err.message)}</div>`;
    }
  }

  async function loadMemoryContent(tab, file) {
    if (!state.selectedAnima) return;
    const fileList = $("chatMemoryFileList");
    const contentArea = $("chatMemoryContentArea");
    const titleEl = $("chatMemoryContentTitle");
    const bodyEl = $("chatMemoryContentBody");

    if (fileList) fileList.style.display = "none";
    if (contentArea) contentArea.style.display = "";
    if (titleEl) titleEl.textContent = file;
    if (bodyEl) bodyEl.textContent = t("common.loading");

    const endpoint = `/api/animas/${encodeURIComponent(state.selectedAnima)}/${tab}/${encodeURIComponent(file)}`;
    try {
      const data = await api(endpoint);
      if (bodyEl) bodyEl.textContent = data.content || t("chat.no_content");
    } catch (err) {
      if (bodyEl) bodyEl.textContent = `${t("chat.error_prefix")} ${err.message}`;
    }
  }

  return { loadMemoryTab, loadMemoryContent };
}
