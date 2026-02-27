// ── Prompt Settings Management ──────────────────
import { api } from "../modules/api.js";
import { t } from "/shared/i18n.js";

let _activeTab = "descriptions";

export function render(container) {
  container.innerHTML = `
    <div class="page-header">
      <h2>${t("tools.page_title")}</h2>
    </div>
    <div class="page-tabs" id="toolPromptTabs">
      <button class="page-tab active" data-tab="descriptions">${t("tools.tab_descriptions")}</button>
      <button class="page-tab" data-tab="guides">${t("tools.tab_guides")}</button>
      <button class="page-tab" data-tab="sections">${t("tools.tab_sections")}</button>
      <button class="page-tab" data-tab="preview">${t("tools.tab_preview")}</button>
    </div>
    <div id="toolPromptContent">
      <div class="loading-placeholder">${t("common.loading")}</div>
    </div>
  `;

  // Tab switching
  document.getElementById("toolPromptTabs").addEventListener("click", (e) => {
    const tab = e.target.dataset?.tab;
    if (!tab) return;
    _activeTab = tab;
    document.querySelectorAll("#toolPromptTabs .page-tab").forEach((el) => {
      el.classList.toggle("active", el.dataset.tab === tab);
    });
    _renderTab();
  });

  _renderTab();
}

export function destroy() {}

// ── Tab Rendering ──────────────────────────

async function _renderTab() {
  const content = document.getElementById("toolPromptContent");
  if (!content) return;
  content.innerHTML = `<div class="loading-placeholder">${t("common.loading")}</div>`;

  try {
    if (_activeTab === "descriptions") {
      await _renderDescriptions(content);
    } else if (_activeTab === "guides") {
      await _renderGuides(content);
    } else if (_activeTab === "sections") {
      await _renderSections(content);
    } else if (_activeTab === "preview") {
      await _renderPreview(content);
    }
  } catch (err) {
    content.innerHTML = `<div class="card"><div class="card-body">
      <div class="loading-placeholder">${t("tools.load_error")} ${_esc(err.message)}</div>
    </div></div>`;
  }
}

// ── Descriptions Tab ───────────────────────

async function _renderDescriptions(container) {
  const data = await api("/api/tool-prompts/descriptions");
  const descs = data.descriptions || [];

  if (descs.length === 0) {
    container.innerHTML = `<div class="card"><div class="card-body">
      <div class="loading-placeholder">${t("tools.no_descriptions")}</div>
    </div></div>`;
    return;
  }

  container.innerHTML = `
    <div class="card">
      <div class="card-header">
        <h3>${t("tools.desc_header")}</h3>
        <p style="color:var(--text-secondary);font-size:0.85rem;margin:0.25rem 0 0;">
          ${t("tools.desc_hint")}
        </p>
      </div>
      <div class="card-body">
        <table class="data-table" id="descTable">
          <thead>
            <tr>
              <th style="width:180px;">${t("tools.tool_name")}</th>
              <th>Description</th>
              <th style="width:100px;">${t("tools.save")}</th>
            </tr>
          </thead>
          <tbody>
            ${descs.map((d) => `
              <tr data-name="${_esc(d.name)}">
                <td><code>${_esc(d.name)}</code></td>
                <td>
                  <textarea class="desc-textarea" rows="3"
                    style="width:100%;font-family:inherit;font-size:0.9rem;resize:vertical;border:1px solid #ddd;border-radius:4px;padding:0.5rem;"
                  >${_esc(d.description)}</textarea>
                </td>
                <td style="text-align:center;">
                  <button class="btn-primary btn-save-desc" data-name="${_esc(d.name)}"
                    style="font-size:0.8rem;padding:0.3rem 0.8rem;">保存</button>
                  <div class="save-status" style="font-size:0.75rem;margin-top:0.25rem;"></div>
                </td>
              </tr>
            `).join("")}
          </tbody>
        </table>
      </div>
    </div>
  `;

  // Save handlers
  container.querySelectorAll(".btn-save-desc").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const name = btn.dataset.name;
      const row = btn.closest("tr");
      const textarea = row.querySelector(".desc-textarea");
      const status = row.querySelector(".save-status");
      const description = textarea.value.trim();

      if (!description) {
        status.textContent = "空にできません";
        status.style.color = "#dc2626";
        return;
      }

      btn.disabled = true;
      status.textContent = "保存中...";
      status.style.color = "#666";

      try {
        await api(`/api/tool-prompts/descriptions/${encodeURIComponent(name)}`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ description }),
        });
        status.textContent = "保存完了";
        status.style.color = "#16a34a";
      } catch (err) {
        status.textContent = "エラー";
        status.style.color = "#dc2626";
      } finally {
        btn.disabled = false;
        setTimeout(() => { status.textContent = ""; }, 3000);
      }
    });
  });
}

// ── Guides Tab ─────────────────────────────

async function _renderGuides(container) {
  const data = await api("/api/tool-prompts/guides");
  const guides = data.guides || [];

  if (guides.length === 0) {
    container.innerHTML = `<div class="card"><div class="card-body">
      <div class="loading-placeholder">ガイドがありません。animaworks init を実行してください。</div>
    </div></div>`;
    return;
  }

  const guideLabels = {
    s_builtin: t("tools.guide_s_builtin"),
    s_mcp: t("tools.guide_s_mcp"),
    non_s: t("tools.non_s_guide"),
  };

  container.innerHTML = guides.map((g) => `
    <div class="card" style="margin-bottom:1rem;" data-key="${_esc(g.key)}">
      <div class="card-header" style="display:flex;justify-content:space-between;align-items:center;">
        <div>
          <h3 style="margin:0;">${_esc(guideLabels[g.key] || g.key)}</h3>
          <code style="font-size:0.8rem;color:var(--text-secondary);">${_esc(g.key)}</code>
        </div>
        <div style="display:flex;align-items:center;gap:0.5rem;">
          <span class="guide-status" style="font-size:0.8rem;"></span>
          <button class="btn-primary btn-save-guide" data-key="${_esc(g.key)}"
            style="font-size:0.85rem;padding:0.4rem 1rem;">${t("tools.save")}</button>
        </div>
      </div>
      <div class="card-body" style="padding:0;">
        <textarea class="guide-textarea"
          style="width:100%;min-height:400px;font-family:'Menlo','Monaco','Courier New',monospace;font-size:0.85rem;
            resize:vertical;border:none;border-top:1px solid #eee;padding:1rem;line-height:1.5;box-sizing:border-box;"
        >${_esc(g.content)}</textarea>
      </div>
    </div>
  `).join("");

  // Save handlers
  container.querySelectorAll(".btn-save-guide").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const key = btn.dataset.key;
      const card = btn.closest(".card");
      const textarea = card.querySelector(".guide-textarea");
      const status = card.querySelector(".guide-status");
      const content = textarea.value.trim();

      if (!content) {
        status.textContent = t("tools.cannot_empty");
        status.style.color = "#dc2626";
        return;
      }

      btn.disabled = true;
      status.textContent = t("tools.saving");
      status.style.color = "#666";

      try {
        await api(`/api/tool-prompts/guides/${encodeURIComponent(key)}`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ content }),
        });
        status.textContent = t("tools.saved");
        status.style.color = "#16a34a";
      } catch (err) {
        status.textContent = t("tools.error");
        status.style.color = "#dc2626";
      } finally {
        btn.disabled = false;
        setTimeout(() => { status.textContent = ""; }, 3000);
      }
    });
  });
}

// ── Sections Tab ──────────────────────────

async function _renderSections(container) {
  const data = await api("/api/tool-prompts/sections");
  const sections = data.sections || [];

  if (sections.length === 0) {
    container.innerHTML = `<div class="card"><div class="card-body">
      <div class="loading-placeholder">${t("tools.no_sections")}</div>
    </div></div>`;
    return;
  }

  container.innerHTML = `
    <div style="margin-bottom:1rem;">
      <p style="color:var(--text-secondary);font-size:0.85rem;margin:0;">
        ${t("tools.sections_hint")}
      </p>
    </div>
  ` + sections.map((s) => `
    <div class="card" style="margin-bottom:1rem;" data-key="${_esc(s.key)}">
      <div class="card-header" style="display:flex;justify-content:space-between;align-items:center;">
        <div style="display:flex;align-items:center;gap:0.75rem;">
          <h3 style="margin:0;"><code>${_esc(s.key)}</code></h3>
          <span style="display:inline-block;padding:0.15rem 0.5rem;border-radius:4px;font-size:0.75rem;
            background:${_conditionColor(s.condition)};color:#fff;">
            ${_esc(_conditionLabel(s.condition))}
          </span>
        </div>
        <div style="display:flex;align-items:center;gap:0.5rem;">
          <span class="section-status" style="font-size:0.8rem;"></span>
          <button class="btn-primary btn-save-section" data-key="${_esc(s.key)}"
            data-condition="${_esc(s.condition || "")}"
            style="font-size:0.85rem;padding:0.4rem 1rem;">${t("tools.save")}</button>
        </div>
      </div>
      <div class="card-body" style="padding:0;">
        <textarea class="section-textarea"
          style="width:100%;min-height:300px;font-family:'Menlo','Monaco','Courier New',monospace;font-size:0.85rem;
            resize:vertical;border:none;border-top:1px solid #eee;padding:1rem;line-height:1.5;box-sizing:border-box;"
        >${_esc(s.content)}</textarea>
      </div>
    </div>
  `).join("");

  // Save handlers
  container.querySelectorAll(".btn-save-section").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const key = btn.dataset.key;
      const condition = btn.dataset.condition || null;
      const card = btn.closest(".card");
      const textarea = card.querySelector(".section-textarea");
      const status = card.querySelector(".section-status");
      const content = textarea.value.trim();

      if (!content) {
        status.textContent = t("tools.cannot_empty");
        status.style.color = "#dc2626";
        return;
      }

      btn.disabled = true;
      status.textContent = t("tools.saving");
      status.style.color = "#666";

      try {
        await api(`/api/tool-prompts/sections/${encodeURIComponent(key)}`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ content, condition: condition || null }),
        });
        status.textContent = t("tools.saved");
        status.style.color = "#16a34a";
      } catch (err) {
        status.textContent = t("tools.error");
        status.style.color = "#dc2626";
      } finally {
        btn.disabled = false;
        setTimeout(() => { status.textContent = ""; }, 3000);
      }
    });
  });
}

// ── Preview Tab ────────────────────────────

async function _renderPreview(container) {
  // Load anima list for dropdown
  let animas = [];
  try {
    const data = await api("/api/animas");
    animas = (data.animas || data || []).map((a) =>
      typeof a === "string" ? a : a.name
    );
  } catch (err) {
    // Non-fatal
  }

  container.innerHTML = `
    <div class="card" style="margin-bottom:1rem;">
      <div class="card-header"><h3>${t("tools.preview_schema")}</h3></div>
      <div class="card-body">
        <div style="display:flex;gap:0.5rem;align-items:center;margin-bottom:1rem;">
          <label>${t("tools.preview_format")}</label>
          <select id="schemaMode" style="padding:0.3rem 0.5rem;border:1px solid #ddd;border-radius:4px;">
            <option value="anthropic">Anthropic</option>
            <option value="litellm">LiteLLM</option>
            <option value="text">Text (Mode B)</option>
          </select>
          <button class="btn-primary" id="btnPreviewSchema" style="font-size:0.85rem;padding:0.35rem 1rem;">
            ${t("tools.preview_generate")}
          </button>
        </div>
        <pre id="schemaPreviewOutput"
          style="background:#f5f5f5;padding:1rem;border-radius:4px;overflow:auto;max-height:500px;
            font-size:0.8rem;line-height:1.4;white-space:pre-wrap;display:none;"></pre>
      </div>
    </div>

    <div class="card">
      <div class="card-header"><h3>${t("tools.preview_prompt")}</h3></div>
      <div class="card-body">
        <div style="display:flex;gap:0.5rem;align-items:center;margin-bottom:1rem;">
          <label>Anima:</label>
          <select id="previewAnima" style="padding:0.3rem 0.5rem;border:1px solid #ddd;border-radius:4px;">
            ${animas.map((n) => `<option value="${_esc(n)}">${_esc(n)}</option>`).join("")}
          </select>
          <button class="btn-primary" id="btnPreviewPrompt" style="font-size:0.85rem;padding:0.35rem 1rem;">
            ${t("tools.preview_build")}
          </button>
          <span id="promptPreviewStatus" style="font-size:0.8rem;color:#666;"></span>
        </div>
        <div id="promptPreviewMeta" style="display:none;margin-bottom:0.5rem;font-size:0.85rem;">
          <span class="status-badge--info" style="margin-right:0.5rem;" id="metaMode"></span>
          <span class="status-badge--info" style="margin-right:0.5rem;" id="metaTokens"></span>
          <span class="status-badge--info" id="metaChars"></span>
        </div>
        <pre id="promptPreviewOutput"
          style="background:#f5f5f5;padding:1rem;border-radius:4px;overflow:auto;max-height:700px;
            font-size:0.8rem;line-height:1.4;white-space:pre-wrap;display:none;"></pre>
      </div>
    </div>
  `;

  // Schema preview
  document.getElementById("btnPreviewSchema").addEventListener("click", async () => {
    const mode = document.getElementById("schemaMode").value;
    const output = document.getElementById("schemaPreviewOutput");
    output.style.display = "block";
    output.textContent = t("tools.generating");

    try {
      const result = await api("/api/tool-prompts/preview/schema", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode }),
      });

      if (mode === "text") {
        output.textContent = result.text;
      } else {
        output.textContent = JSON.stringify(result.tools, null, 2);
      }
    } catch (err) {
      output.textContent = t("tools.load_error") + err.message;
    }
  });

  // System prompt preview
  document.getElementById("btnPreviewPrompt").addEventListener("click", async () => {
    const animaName = document.getElementById("previewAnima").value;
    const output = document.getElementById("promptPreviewOutput");
    const meta = document.getElementById("promptPreviewMeta");
    const status = document.getElementById("promptPreviewStatus");

    if (!animaName) {
      status.textContent = t("tools.select_anima");
      return;
    }

    output.style.display = "block";
    output.textContent = t("tools.building");
    meta.style.display = "none";
    status.textContent = t("tools.building");

    try {
      const result = await api("/api/tool-prompts/preview/system-prompt", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ anima_name: animaName }),
      });

      output.textContent = result.system_prompt;
      meta.style.display = "block";
      document.getElementById("metaMode").textContent = `Mode: ${result.execution_mode}`;
      document.getElementById("metaTokens").textContent = `~${result.token_estimate} tokens`;
      document.getElementById("metaChars").textContent = `${result.char_count} chars`;
      status.textContent = "";
    } catch (err) {
      output.textContent = t("tools.load_error") + err.message;
      status.textContent = t("tools.error");
    }
  });
}

// ── Helpers ─────────────────────────────────

function _esc(str) {
  if (!str) return "";
  const el = document.createElement("span");
  el.textContent = str;
  return el.innerHTML;
}

function _conditionLabel(condition) {
  if (!condition) return t("tools.condition_always");
  const labels = {
    "mode:s": t("tools.condition_mode_s"),
    "mode:non_s": t("tools.non_s_mode"),
    "mode:a": t("tools.condition_mode_a"),
    "solo_top_level": t("tools.condition_solo_top"),
  };
  return labels[condition] || condition;
}

function _conditionColor(condition) {
  if (!condition) return "#6b7280";
  if (condition.startsWith("mode:")) return "#2563eb";
  if (condition === "solo_top_level") return "#7c3aed";
  return "#6b7280";
}
