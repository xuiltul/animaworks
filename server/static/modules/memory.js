/* ── Memory Browser ────────────────────────── */

import { state, dom, escapeHtml } from "./state.js";
import { api } from "./api.js";
import { t } from "/shared/i18n.js";

export function activateMemoryTab(tab) {
  state.activeMemoryTab = tab;
  document.querySelectorAll(".memory-tab").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.tab === tab);
  });
  // Hide content detail, show list
  const contentArea = dom.memoryContentArea || document.getElementById("memoryContentArea");
  const fileList = dom.memoryFileList || document.getElementById("memoryFileList");
  if (contentArea) contentArea.style.display = "none";
  if (fileList) fileList.style.display = "";
  loadMemoryTab(tab);
}

export async function loadMemoryTab(tab) {
  const fileList = dom.memoryFileList || document.getElementById("memoryFileList");
  if (!fileList) return; // Memory browser not in DOM

  const name = state.selectedAnima;
  if (!name) {
    fileList.innerHTML = `<div class="loading-placeholder">${t("assets.select_anima")}</div>`;
    return;
  }

  fileList.innerHTML = `<div class="loading-placeholder">${t("common.loading")}</div>`;

  let endpoint;
  if (tab === "episodes") endpoint = `/api/animas/${encodeURIComponent(name)}/episodes`;
  else if (tab === "knowledge") endpoint = `/api/animas/${encodeURIComponent(name)}/knowledge`;
  else endpoint = `/api/animas/${encodeURIComponent(name)}/procedures`;

  try {
    const data = await api(endpoint);
    const files = data.files || [];
    if (files.length === 0) {
      fileList.innerHTML = `<div class="loading-placeholder">${t("memory.no_files")}</div>`;
      return;
    }
    fileList.innerHTML = files.map((f) =>
      `<div class="memory-file-item" data-file="${escapeHtml(f)}" data-tab="${tab}">${escapeHtml(f)}</div>`
    ).join("");

    fileList.querySelectorAll(".memory-file-item").forEach((item) => {
      item.addEventListener("click", () => {
        loadMemoryContent(item.dataset.tab, item.dataset.file);
      });
    });
  } catch (err) {
    console.error("Failed to load memory files:", err);
    fileList.innerHTML = `<div class="loading-placeholder">${t("common.load_failed")}</div>`;
  }
}

async function loadMemoryContent(tab, file) {
  const name = state.selectedAnima;
  if (!name) return;

  const fileList = dom.memoryFileList || document.getElementById("memoryFileList");
  const contentArea = dom.memoryContentArea || document.getElementById("memoryContentArea");
  const contentTitle = dom.memoryContentTitle || document.getElementById("memoryContentTitle");
  const contentBody = dom.memoryContentBody || document.getElementById("memoryContentBody");
  if (!contentArea) return;

  let endpoint;
  if (tab === "episodes") endpoint = `/api/animas/${encodeURIComponent(name)}/episodes/${encodeURIComponent(file)}`;
  else if (tab === "knowledge") endpoint = `/api/animas/${encodeURIComponent(name)}/knowledge/${encodeURIComponent(file)}`;
  else endpoint = `/api/animas/${encodeURIComponent(name)}/procedures/${encodeURIComponent(file)}`;

  if (fileList) fileList.style.display = "none";
  contentArea.style.display = "";
  if (contentTitle) contentTitle.textContent = file;
  if (contentBody) contentBody.textContent = t("common.loading");

  try {
    const data = await api(endpoint);
    if (contentBody) contentBody.textContent = data.content || t("chat.no_content");
  } catch (err) {
    if (contentBody) contentBody.textContent = `[${t("tools.error")}] ${err.message}`;
  }
}
