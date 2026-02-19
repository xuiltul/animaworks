// ── Memory Browser ──────────────────────
// Tabbed viewer for Episodes / Knowledge / Procedures.

import { getState, setState } from "./state.js";
import * as api from "./api.js";
import { escapeHtml, stripMdExtension } from "./utils.js";

const TABS = [
  { key: "episodes", label: "Episodes" },
  { key: "knowledge", label: "Knowledge" },
  { key: "procedures", label: "Procedures" },
];

// Container reference for async operations
let _container = null;

// ── DOM Queries ──────────────────────

function findNodes() {
  if (!_container) return {};
  const browser = _container.querySelector(".memory-browser") || _container;
  return {
    fileList: browser.querySelector(".memory-file-list"),
    contentArea: browser.querySelector(".memory-content-area"),
    contentTitle: browser.querySelector(".memory-content-title"),
    contentBody: browser.querySelector(".memory-content-body"),
  };
}

function showFileList() {
  const { fileList, contentArea } = findNodes();
  if (fileList) fileList.style.display = "";
  if (contentArea) contentArea.style.display = "none";
}

function showContentArea() {
  const { fileList, contentArea } = findNodes();
  if (fileList) fileList.style.display = "none";
  if (contentArea) contentArea.style.display = "";
}

// ── Render ──────────────────────

/** Build the full memory browser DOM inside the given container. */
export function renderMemoryBrowser(container) {
  _container = container;
  const { activeMemoryTab } = getState();

  container.innerHTML = `
    <div class="memory-browser">
      <div class="memory-tabs">
        ${TABS.map(
          (t) =>
            `<button class="memory-tab${t.key === activeMemoryTab ? " active" : ""}"
                    data-tab="${t.key}">${t.label}</button>`
        ).join("")}
      </div>
      <div class="memory-file-list"></div>
      <div class="memory-content-area" style="display:none">
        <button class="memory-back-btn">&larr; Back</button>
        <div class="memory-content-title"></div>
        <div class="memory-content-body"></div>
      </div>
    </div>`;

  // Bind tab clicks
  container.querySelectorAll(".memory-tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      const tab = btn.dataset.tab;
      setState({ activeMemoryTab: tab });

      // Update active class
      container.querySelectorAll(".memory-tab").forEach((b) => {
        b.classList.toggle("active", b.dataset.tab === tab);
      });

      showFileList();
      loadMemoryTab(tab);
    });
  });

  // Bind back button
  container.querySelector(".memory-back-btn").addEventListener("click", () => {
    showFileList();
  });
}

/** Initialize: render + load the active tab. */
export function initMemory(container) {
  _container = container;
  renderMemoryBrowser(container);
  loadMemoryTab(getState().activeMemoryTab);
}

// ── Data Loading ──────────────────────

/** Load file list for the given memory tab. */
export async function loadMemoryTab(tab) {
  const { fileList } = findNodes();
  if (!fileList) return;

  const { selectedAnima } = getState();

  if (!selectedAnima) {
    fileList.innerHTML = '<div class="loading-placeholder">Anima を選択してください</div>';
    return;
  }

  fileList.innerHTML = '<div class="loading-placeholder">読み込み中...</div>';

  try {
    let data;
    if (tab === "episodes") data = await api.fetchEpisodes(selectedAnima);
    else if (tab === "knowledge") data = await api.fetchKnowledge(selectedAnima);
    else data = await api.fetchProcedures(selectedAnima);

    const files = data.files || [];
    if (files.length === 0) {
      fileList.innerHTML = '<div class="loading-placeholder">ファイルがありません</div>';
      return;
    }

    fileList.innerHTML = files
      .map(
        (f) =>
          `<div class="memory-file-item" data-file="${escapeHtml(f)}" data-tab="${tab}">${escapeHtml(stripMdExtension(f))}</div>`
      )
      .join("");

    fileList.querySelectorAll(".memory-file-item").forEach((item) => {
      item.addEventListener("click", () => {
        loadMemoryContent(item.dataset.tab, item.dataset.file);
      });
    });
  } catch (err) {
    console.error("Failed to load memory files:", err);
    fileList.innerHTML = `<div class="loading-placeholder">読み込み失敗: ${escapeHtml(err.message)}</div>`;
  }
}

/** Load and display a single memory file. */
async function loadMemoryContent(tab, file) {
  const { selectedAnima } = getState();
  if (!selectedAnima) return;

  const { contentTitle, contentBody } = findNodes();
  if (!contentTitle || !contentBody) return;

  showContentArea();
  contentTitle.textContent = stripMdExtension(file);
  contentBody.textContent = "読み込み中...";

  try {
    let data;
    if (tab === "episodes") data = await api.fetchEpisode(selectedAnima, file);
    else if (tab === "knowledge") data = await api.fetchKnowledgeTopic(selectedAnima, file);
    else data = await api.fetchProcedure(selectedAnima, file);

    contentBody.textContent = data.content || "(内容なし)";
  } catch (err) {
    console.error("Failed to load memory content:", err);
    contentBody.textContent = `[エラー] ${err.message}`;
  }
}
