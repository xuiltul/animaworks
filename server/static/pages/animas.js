// ── Anima Management ───────────────────────
import { api } from "../modules/api.js";
import { escapeHtml, statusClass, renderMarkdown } from "../modules/state.js";
import { t } from "/shared/i18n.js";

let _viewMode = "list"; // "list" | "detail"
let _selectedName = null;
let _container = null;
let _modelsCache = null;
let _toolsCache = null;

function _extractStatsCount(value) {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (value && typeof value === "object") {
    const count = value.count;
    if (typeof count === "number" && Number.isFinite(count)) return count;
  }
  return 0;
}

export function render(container, { subPath } = {}) {
  _container = container;
  if (subPath) {
    _viewMode = "detail";
    _selectedName = subPath;
    _showDetail(subPath);
  } else {
    _viewMode = "list";
    _selectedName = null;
    _renderList();
  }
}

export function destroy() {
  _container = null;
  _viewMode = "list";
  _selectedName = null;
}

// ── List View ──────────────────────────────

async function _renderList() {
  if (!_container) return;

  _container.innerHTML = `
    <div class="page-header">
      <h2>${t("nav.animas")}</h2>
    </div>
    <div id="animasListContent">
      <div class="loading-placeholder">${t("common.loading")}</div>
    </div>
  `;

  const content = document.getElementById("animasListContent");
  if (!content) return;

  try {
    const animas = await api("/api/animas");

    if (animas.length === 0) {
      content.innerHTML = `<div class="loading-placeholder">${t("animas.not_registered")}</div>`;
      return;
    }

    content.innerHTML = `
      <table class="data-table">
        <thead>
          <tr>
            <th>${t("animas.table_name")}</th>
            <th>${t("animas.table_status")}</th>
            <th>${t("animas.table_pid")}</th>
            <th>${t("animas.table_uptime")}</th>
            <th>${t("animas.table_actions")}</th>
          </tr>
        </thead>
        <tbody id="animasTableBody"></tbody>
      </table>
    `;

    const tbody = document.getElementById("animasTableBody");
    for (const p of animas) {
      const dotClass = statusClass(p.status);
      const statusLabel = p.status || "offline";
      const uptime = p.uptime_sec ? _formatUptime(p.uptime_sec) : "--";
      const pid = p.pid || "--";

      // Determine visual state class
      let stateClass = "";
      if (p.status === "bootstrapping" || p.bootstrapping) {
        stateClass = "anima-item anima-item--loading";
      } else if (p.status === "not_found" || p.status === "stopped") {
        stateClass = "anima-item anima-item--sleeping";
      } else {
        stateClass = "anima-item";
      }

      const tr = document.createElement("tr");
      tr.className = stateClass;
      tr.dataset.anima = p.name;
      tr.style.cursor = "pointer";
      tr.innerHTML = `
        <td style="font-weight:600;">${escapeHtml(p.name)}</td>
        <td>
          <span class="status-dot ${dotClass}" style="display:inline-block;"></span>
          ${escapeHtml(statusLabel)}
        </td>
        <td>${escapeHtml(String(pid))}</td>
        <td>${escapeHtml(uptime)}</td>
        <td>
          <button class="btn-secondary anima-detail-btn" data-name="${escapeHtml(p.name)}" style="font-size:0.8rem; padding:0.25rem 0.5rem;">${t("animas.detail")}</button>
          <button class="btn-primary anima-trigger-btn" data-name="${escapeHtml(p.name)}" style="font-size:0.8rem; padding:0.25rem 0.5rem;">Heartbeat</button>
        </td>
      `;

      tr.addEventListener("click", (e) => {
        if (e.target.classList.contains("anima-trigger-btn")) return;
        _showDetail(p.name);
      });

      tbody.appendChild(tr);
    }

    // Bind trigger buttons
    content.querySelectorAll(".anima-trigger-btn").forEach(btn => {
      btn.addEventListener("click", async (e) => {
        e.stopPropagation();
        const name = btn.dataset.name;
        btn.disabled = true;
        btn.textContent = t("animas.running");
        try {
          await fetch(`/api/animas/${encodeURIComponent(name)}/trigger`, { method: "POST" });
          btn.textContent = t("animas.success");
          setTimeout(() => { btn.textContent = "Heartbeat"; btn.disabled = false; }, 2000);
        } catch {
          btn.textContent = t("animas.failed");
          setTimeout(() => { btn.textContent = "Heartbeat"; btn.disabled = false; }, 2000);
        }
      });
    });

  } catch (err) {
    content.innerHTML = `<div class="loading-placeholder">${t("common.load_failed")}: ${escapeHtml(err.message)}</div>`;
  }
}

// ── Detail View ────────────────────────────

async function _fetchModels() {
  if (_modelsCache) return _modelsCache;
  try {
    const res = await api("/api/system/available-models");
    _modelsCache = res.models || [];
  } catch {
    _modelsCache = [];
  }
  return _modelsCache;
}

async function _fetchTools() {
  if (_toolsCache) return _toolsCache;
  try {
    const res = await api("/api/system/available-tools");
    _toolsCache = res.tools || [];
  } catch {
    _toolsCache = [];
  }
  return _toolsCache;
}

function _editableCard({ id, title, rawContent, renderFn }) {
  // Returns HTML for a card with edit/preview toggle and save button
  const rendered = rawContent ? renderFn(rawContent) : `<span style="color:var(--text-secondary,#666);">${t("animas.not_set")}</span>`;
  return `
    <div class="card" id="${id}_card">
      <div class="card-header" style="display:flex; justify-content:space-between; align-items:center;">
        <span>${escapeHtml(title)}</span>
        <div style="display:flex; gap:0.4rem; align-items:center;">
          <span id="${id}_status" style="font-size:0.75rem; color:var(--text-secondary,#888);"></span>
          <button class="btn-secondary" id="${id}_editBtn" style="font-size:0.75rem; padding:0.15rem 0.5rem;">${t("animas.edit")}</button>
          <button class="btn-primary" id="${id}_saveBtn" style="font-size:0.75rem; padding:0.15rem 0.5rem; display:none;">${t("animas.save")}</button>
        </div>
      </div>
      <div class="card-body" style="position:relative;">
        <div id="${id}_preview" style="max-height:300px; overflow-y:auto;">${rendered}</div>
        <textarea id="${id}_editor" style="display:none; width:100%; min-height:260px; max-height:400px; resize:vertical; font-family:monospace; font-size:0.85rem; padding:0.5rem; border:1px solid var(--border,#ddd); border-radius:4px; background:var(--bg-secondary,#f9f9f9); color:var(--text-primary,#333);">${escapeHtml(rawContent || "")}</textarea>
      </div>
    </div>
  `;
}

function _bindEditableCard({ id, name, field }) {
  const editBtn = document.getElementById(`${id}_editBtn`);
  const saveBtn = document.getElementById(`${id}_saveBtn`);
  const preview = document.getElementById(`${id}_preview`);
  const editor = document.getElementById(`${id}_editor`);
  const status = document.getElementById(`${id}_status`);
  if (!editBtn || !saveBtn || !preview || !editor) return;

  let editing = false;

  editBtn.addEventListener("click", () => {
    editing = !editing;
    if (editing) {
      preview.style.display = "none";
      editor.style.display = "block";
      saveBtn.style.display = "";
      editBtn.textContent = t("animas.preview");
      editor.focus();
    } else {
      preview.innerHTML = editor.value
        ? renderMarkdown(editor.value)
        : `<span style="color:var(--text-secondary,#666);">${t("animas.not_set")}</span>`;
      preview.style.display = "";
      editor.style.display = "none";
      saveBtn.style.display = "none";
      editBtn.textContent = t("animas.edit");
    }
  });

  saveBtn.addEventListener("click", async () => {
    saveBtn.disabled = true;
    saveBtn.textContent = t("animas.saving");
    status.textContent = "";
    try {
      await fetch(`/api/animas/${encodeURIComponent(name)}/${field}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content: editor.value }),
      });
      status.textContent = t("animas.saved");
      status.style.color = "var(--color-success, #28a745)";
      // Update preview
      preview.innerHTML = editor.value
        ? renderMarkdown(editor.value)
        : `<span style="color:var(--text-secondary,#666);">${t("animas.not_set")}</span>`;
      setTimeout(() => { status.textContent = ""; }, 3000);
    } catch {
      status.textContent = t("animas.save_failed");
      status.style.color = "var(--color-danger, #dc3545)";
    }
    saveBtn.disabled = false;
    saveBtn.textContent = t("animas.save");
  });
}

// ── Aliases Card ───────────────────────────

function _aliasesCardHtml(aliases) {
  const tagsHtml = (aliases || []).map((a, i) => `
    <span class="alias-tag" style="display:inline-flex; align-items:center; gap:0.3rem; background:var(--color-primary-light,#e8f0fe); color:var(--color-primary,#0066cc); border-radius:12px; padding:0.2rem 0.6rem; font-size:0.85rem; margin:0.15rem;">
      ${escapeHtml(a)}
      <button class="alias-remove-btn" data-idx="${i}" style="background:none; border:none; cursor:pointer; color:var(--text-secondary,#666); font-size:0.85rem; padding:0; line-height:1;">✕</button>
    </span>
  `).join("");

  return `
    <div class="card" style="margin-bottom: 1.5rem;" id="aliasesCard">
      <div class="card-header" style="display:flex; justify-content:space-between; align-items:center;">
        <span>${t("animas.aliases")}</span>
        <div style="display:flex; gap:0.4rem; align-items:center;">
          <span id="aliasesStatus" style="font-size:0.75rem; color:var(--text-secondary,#888);"></span>
          <button class="btn-primary" id="aliasesSaveBtn" style="font-size:0.75rem; padding:0.15rem 0.5rem;">${t("animas.save")}</button>
        </div>
      </div>
      <div class="card-body">
        <div id="aliasTagList" style="display:flex; flex-wrap:wrap; gap:0.2rem; min-height:2rem; margin-bottom:0.5rem;">
          ${tagsHtml || `<span style="color:var(--text-secondary,#888); font-size:0.85rem;">${t("animas.aliases_empty")}</span>`}
        </div>
        <div style="display:flex; gap:0.4rem; margin-top:0.4rem;">
          <input type="text" id="aliasInput" placeholder="${t("animas.aliases_add_placeholder")}"
            style="flex:1; padding:0.3rem 0.5rem; border:1px solid var(--border,#ddd); border-radius:4px; font-size:0.85rem; background:var(--bg-secondary,#f9f9f9); color:var(--text-primary,#333);">
          <button class="btn-secondary" id="aliasAddBtn" style="font-size:0.85rem; padding:0.3rem 0.6rem;">${t("animas.aliases_add")}</button>
        </div>
        <div style="font-size:0.75rem; color:var(--text-secondary,#888); margin-top:0.4rem;">${t("animas.aliases_hint")}</div>
      </div>
    </div>
  `;
}

function _bindAliasesCard(name, initialAliases) {
  let aliases = [...(initialAliases || [])];
  const tagList = document.getElementById("aliasTagList");
  const input = document.getElementById("aliasInput");
  const addBtn = document.getElementById("aliasAddBtn");
  const saveBtn = document.getElementById("aliasesSaveBtn");
  const status = document.getElementById("aliasesStatus");
  if (!tagList || !input || !addBtn || !saveBtn) return;

  function _renderTags() {
    if (aliases.length === 0) {
      tagList.innerHTML = `<span style="color:var(--text-secondary,#888); font-size:0.85rem;">${t("animas.aliases_empty")}</span>`;
      return;
    }
    tagList.innerHTML = aliases.map((a, i) => `
      <span class="alias-tag" style="display:inline-flex; align-items:center; gap:0.3rem; background:var(--color-primary-light,#e8f0fe); color:var(--color-primary,#0066cc); border-radius:12px; padding:0.2rem 0.6rem; font-size:0.85rem; margin:0.15rem;">
        ${escapeHtml(a)}
        <button class="alias-remove-btn" data-idx="${i}" style="background:none; border:none; cursor:pointer; color:var(--text-secondary,#666); font-size:0.85rem; padding:0; line-height:1;">✕</button>
      </span>
    `).join("");
    tagList.querySelectorAll(".alias-remove-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        const idx = parseInt(btn.dataset.idx, 10);
        aliases.splice(idx, 1);
        _renderTags();
      });
    });
  }

  // Bind remove on initial render
  tagList.querySelectorAll(".alias-remove-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const idx = parseInt(btn.dataset.idx, 10);
      aliases.splice(idx, 1);
      _renderTags();
    });
  });

  function _addAlias() {
    const val = input.value.trim();
    if (!val || aliases.includes(val)) { input.value = ""; return; }
    aliases.push(val);
    input.value = "";
    _renderTags();
  }

  addBtn.addEventListener("click", _addAlias);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); _addAlias(); }
  });

  saveBtn.addEventListener("click", async () => {
    saveBtn.disabled = true;
    saveBtn.textContent = t("animas.saving");
    status.textContent = "";
    try {
      await fetch(`/api/animas/${encodeURIComponent(name)}/aliases`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ aliases }),
      });
      status.textContent = t("animas.saved");
      status.style.color = "var(--color-success, #28a745)";
      setTimeout(() => { status.textContent = ""; }, 3000);
    } catch {
      status.textContent = t("animas.save_failed");
      status.style.color = "var(--color-danger, #dc3545)";
    }
    saveBtn.disabled = false;
    saveBtn.textContent = t("animas.save");
  });
}

// ── Discord Channels Card ──────────────────

function _discordChannelsCardHtml(discordChannels, animaName) {
  if (!discordChannels || discordChannels.length === 0) {
    return `
      <div class="card" style="margin-bottom: 1.5rem;" id="discordChannelsCard">
        <div class="card-header">${t("animas.discord_channels")}</div>
        <div class="card-body">
          <div style="color:var(--text-secondary,#888); font-size:0.85rem;">${t("animas.discord_channels_empty")}</div>
        </div>
      </div>
    `;
  }
  const checkboxes = discordChannels.map(ch => {
    const checked = (ch.members || []).includes(animaName) ? " checked" : "";
    return `
      <label style="display:flex; align-items:center; gap:0.4rem; font-size:0.85rem; cursor:pointer;">
        <input type="checkbox" class="discord-ch-check" data-channel-id="${escapeHtml(ch.id)}"${checked}>
        <span style="color:var(--text-secondary,#888);">#</span>${escapeHtml(ch.name)}
      </label>
    `;
  }).join("");

  return `
    <div class="card" style="margin-bottom: 1.5rem;" id="discordChannelsCard">
      <div class="card-header" style="display:flex; justify-content:space-between; align-items:center;">
        <span>${t("animas.discord_channels")}</span>
        <div style="display:flex; gap:0.4rem; align-items:center;">
          <span id="discordChStatus" style="font-size:0.75rem; color:var(--text-secondary,#888);"></span>
          <button class="btn-primary" id="discordChSaveBtn" style="font-size:0.75rem; padding:0.15rem 0.5rem;">${t("animas.save")}</button>
        </div>
      </div>
      <div class="card-body">
        <div id="discordChList" style="display:grid; grid-template-columns:repeat(auto-fill,minmax(180px,1fr)); gap:0.3rem 1rem;">
          ${checkboxes}
        </div>
        <div style="font-size:0.75rem; color:var(--text-secondary,#888); margin-top:0.5rem;">${t("animas.discord_channels_hint")}</div>
      </div>
    </div>
  `;
}

function _bindDiscordChannelsCard(animaName, discordChannels) {
  const saveBtn = document.getElementById("discordChSaveBtn");
  const status = document.getElementById("discordChStatus");
  if (!saveBtn) return;

  saveBtn.addEventListener("click", async () => {
    saveBtn.disabled = true;
    saveBtn.textContent = t("animas.saving");
    status.textContent = "";

    // Build channel_id → members map from checkboxes
    const updates = {};
    document.querySelectorAll(".discord-ch-check").forEach(cb => {
      const chId = cb.dataset.channelId;
      const ch = discordChannels.find(c => c.id === chId);
      if (!ch) return;
      const members = [...(ch.members || [])];
      const idx = members.indexOf(animaName);
      if (cb.checked && idx === -1) members.push(animaName);
      if (!cb.checked && idx !== -1) members.splice(idx, 1);
      // Only update if changed
      const orig = ch.members || [];
      if (JSON.stringify(members.sort()) !== JSON.stringify([...orig].sort())) {
        updates[chId] = members;
      }
    });

    try {
      const promises = Object.entries(updates).map(([chId, members]) =>
        fetch(`/api/discord/channel-members/${encodeURIComponent(chId)}`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ members }),
        })
      );
      await Promise.all(promises);

      // Update local state
      for (const [chId, members] of Object.entries(updates)) {
        const ch = discordChannels.find(c => c.id === chId);
        if (ch) ch.members = members;
      }

      status.textContent = t("animas.saved");
      status.style.color = "var(--color-success, #28a745)";
      setTimeout(() => { status.textContent = ""; }, 3000);
    } catch {
      status.textContent = t("animas.save_failed");
      status.style.color = "var(--color-danger, #dc3545)";
    }
    saveBtn.disabled = false;
    saveBtn.textContent = t("animas.save");
  });
}

// ── Permissions UI ─────────────────────────

function _toolChecked(toolName, extTools) {
  const allowAll = extTools.allow_all;
  const allow = extTools.allow || [];
  const deny = extTools.deny || [];
  if (deny.includes(toolName)) return false;
  if (allowAll || allow.length === 0) return true;
  return allow.includes(toolName);
}

function _permPathRowHtml(path, access, idx) {
  const isRw = access !== "r";
  return `
    <div style="display:flex; align-items:center; gap:0.4rem; margin-bottom:0.3rem;" data-perm-path-row="${idx}" data-access-value="${isRw ? "rw" : "r"}">
      <input type="text" value="${escapeHtml(path)}" data-perm-path="${idx}"
        style="flex:1; min-width:0; padding:0.3rem 0.5rem; font-family:monospace; font-size:0.85rem; border:1px solid var(--border,#ddd); border-radius:4px; background:var(--bg-secondary,#f9f9f9); color:var(--text-primary,#333);">
      <div style="display:flex; border:1px solid var(--border,#ddd); border-radius:4px; overflow:hidden; flex-shrink:0;">
        <button class="perm-access-btn" data-access="rw"
          style="padding:0.2rem 0.5rem; font-size:0.75rem; border:none; cursor:pointer; transition:background 0.15s;
                 background:${isRw ? "var(--color-primary,#0066cc)" : "var(--bg-secondary,#f9f9f9)"};
                 color:${isRw ? "#fff" : "var(--text-secondary,#666)"};">
          ${t("animas.permissions_access_rw")}
        </button>
        <button class="perm-access-btn" data-access="r"
          style="padding:0.2rem 0.5rem; font-size:0.75rem; border:none; border-left:1px solid var(--border,#ddd); cursor:pointer; transition:background 0.15s;
                 background:${!isRw ? "var(--color-warning,#e8a000)" : "var(--bg-secondary,#f9f9f9)"};
                 color:${!isRw ? "#fff" : "var(--text-secondary,#666)"};">
          🔒 ${t("animas.permissions_access_r")}
        </button>
      </div>
      <button class="btn-secondary perm-remove-path-btn" style="font-size:0.75rem; padding:0.2rem 0.4rem; color:var(--color-danger,#dc3545); flex-shrink:0;">${t("animas.permissions_remove_path")}</button>
    </div>
  `;
}

function _permissionsCardHtml(perm, availableTools) {
  const fileRoots = perm.file_roots || [];
  const fileRootsReadonly = perm.file_roots_readonly || [];
  const cmds = perm.commands || {};
  const extTools = perm.external_tools || {};
  const toolCreation = perm.tool_creation || {};

  // Merge rw + r paths into one unified list with access metadata
  const allPaths = [
    ...fileRoots.map(p => ({ path: p, access: "rw" })),
    ...fileRootsReadonly.map(p => ({ path: p, access: "r" })),
  ];

  const pathRows = allPaths.map((entry, i) => _permPathRowHtml(entry.path, entry.access, i)).join("");

  return `
    <div class="card" style="margin-bottom: 1.5rem;" id="permissionsCard">
      <div class="card-header" style="display:flex; justify-content:space-between; align-items:center;">
        <span>${t("animas.permissions")}</span>
        <div style="display:flex; gap:0.4rem; align-items:center;">
          <span id="permStatus" style="font-size:0.75rem; color:var(--text-secondary,#888);"></span>
          <button class="btn-primary" id="permSaveBtn" style="font-size:0.75rem; padding:0.15rem 0.5rem;">${t("animas.save")}</button>
        </div>
      </div>
      <div class="card-body">
        <!-- file_roots -->
        <div style="margin-bottom:1rem;">
          <label style="font-weight:600; font-size:0.85rem; display:block; margin-bottom:0.4rem;">${t("animas.permissions_file_roots")}</label>
          <div id="permPathList">${pathRows}</div>
          <button class="btn-secondary" id="permAddPathBtn" style="font-size:0.75rem; padding:0.2rem 0.5rem; margin-top:0.3rem;">+ ${t("animas.permissions_add_path")}</button>
        </div>

        <!-- commands -->
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:1rem; margin-bottom:1rem;">
          <fieldset style="border:1px solid var(--border,#ddd); border-radius:4px; padding:0.6rem;">
            <legend style="font-weight:600; font-size:0.85rem; padding:0 0.3rem;">${t("animas.permissions_commands")}</legend>
            <label style="display:flex; align-items:center; gap:0.4rem; font-size:0.85rem; cursor:pointer;">
              <input type="checkbox" id="permCmdAllowAll" ${cmds.allow_all ? "checked" : ""}>
              ${t("animas.permissions_allow_all")}
            </label>
          </fieldset>

          <!-- external_tools -->
          <fieldset style="border:1px solid var(--border,#ddd); border-radius:4px; padding:0.6rem;">
            <legend style="font-weight:600; font-size:0.85rem; padding:0 0.3rem;">${t("animas.permissions_external_tools")}</legend>
            ${availableTools && availableTools.length > 0 ? `
              <label style="display:flex; align-items:center; gap:0.4rem; font-size:0.85rem; cursor:pointer; font-weight:600; margin-bottom:0.4rem;">
                <input type="checkbox" id="permExtAllowAll" ${availableTools.every(tool => _toolChecked(tool, extTools)) ? "checked" : ""}>
                ${t("animas.permissions_allow_all")}
              </label>
              <div id="toolPermList" style="max-height:180px; overflow-y:auto; padding:0.3rem 0.4rem; border:1px solid var(--border,#ddd); border-radius:4px; background:var(--bg-primary,#fff);">
                ${availableTools.map(tool => `
                  <label style="display:flex; align-items:center; gap:0.4rem; font-size:0.82rem; cursor:pointer; padding:0.1rem 0;">
                    <input type="checkbox" class="tool-perm-check" data-tool="${escapeHtml(tool)}" ${_toolChecked(tool, extTools) ? "checked" : ""}>
                    <code style="font-size:0.8rem;">${escapeHtml(tool)}</code>
                  </label>
                `).join("")}
              </div>
              <div style="display:flex; gap:0.4rem; margin-top:0.3rem;">
                <button class="btn-secondary" id="toolPermSelectAll" style="font-size:0.73rem; padding:0.15rem 0.4rem;">${t("animas.permissions_tools_select_all")}</button>
                <button class="btn-secondary" id="toolPermDeselectAll" style="font-size:0.73rem; padding:0.15rem 0.4rem;">${t("animas.permissions_tools_deselect_all")}</button>
              </div>
            ` : `
              <label style="display:flex; align-items:center; gap:0.4rem; font-size:0.85rem; cursor:pointer;">
                <input type="checkbox" id="permExtAllowAll" ${extTools.allow_all ? "checked" : ""}>
                ${t("animas.permissions_allow_all")}
              </label>
            `}
          </fieldset>
        </div>

        <!-- tool_creation -->
        <fieldset style="border:1px solid var(--border,#ddd); border-radius:4px; padding:0.6rem; margin-bottom:0.5rem;">
          <legend style="font-weight:600; font-size:0.85rem; padding:0 0.3rem;">${t("animas.permissions_tool_creation")}</legend>
          <div style="display:flex; gap:1.5rem;">
            <label style="display:flex; align-items:center; gap:0.4rem; font-size:0.85rem; cursor:pointer;">
              <input type="checkbox" id="permToolPersonal" ${toolCreation.personal ? "checked" : ""}>
              ${t("animas.permissions_personal")}
            </label>
            <label style="display:flex; align-items:center; gap:0.4rem; font-size:0.85rem; cursor:pointer;">
              <input type="checkbox" id="permToolShared" ${toolCreation.shared ? "checked" : ""}>
              ${t("animas.permissions_shared")}
            </label>
          </div>
        </fieldset>

        <div style="font-size:0.75rem; color:var(--text-secondary,#888);">${t("animas.restart_notice")}</div>
      </div>
    </div>
  `;
}

function _bindPermissionsCard(name, perm, availableTools) {
  const saveBtn = document.getElementById("permSaveBtn");
  const status = document.getElementById("permStatus");
  const addBtn = document.getElementById("permAddPathBtn");
  const pathList = document.getElementById("permPathList");
  if (!saveBtn || !pathList) return;

  // Helper: bind access toggle buttons within a row
  function _bindAccessButtons(row) {
    row.querySelectorAll(".perm-access-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        const access = btn.dataset.access;
        // Store on the row element itself
        row.dataset.accessValue = access;
        // Update button styles in this row
        row.querySelectorAll(".perm-access-btn").forEach(b => {
          const active = b.dataset.access === access;
          const bRw = b.dataset.access === "rw";
          if (active) {
            b.style.background = bRw ? "var(--color-primary,#0066cc)" : "var(--color-warning,#e8a000)";
            b.style.color = "#fff";
          } else {
            b.style.background = "var(--bg-secondary,#f9f9f9)";
            b.style.color = "var(--text-secondary,#666)";
          }
        });
      });
    });
  }

  // Bind existing rows
  pathList.querySelectorAll("[data-perm-path-row]").forEach(row => {
    row.querySelector(".perm-remove-path-btn")?.addEventListener("click", () => row.remove());
    _bindAccessButtons(row);
  });

  // Add path button — new rows default to read-write
  addBtn?.addEventListener("click", () => {
    const idx = Date.now(); // unique idx
    const div = document.createElement("div");
    div.innerHTML = _permPathRowHtml("", "rw", idx);
    const row = div.firstElementChild;
    pathList.appendChild(row);
    row.querySelector(".perm-remove-path-btn")?.addEventListener("click", () => row.remove());
    _bindAccessButtons(row);
    row.querySelector("input")?.focus();
  });

  // Tool checkboxes: "allow all" toggle + select/deselect all buttons
  if (availableTools && availableTools.length > 0) {
    const allowAllChk = document.getElementById("permExtAllowAll");
    const toolList = document.getElementById("toolPermList");
    const selectAllBtn = document.getElementById("toolPermSelectAll");
    const deselectAllBtn = document.getElementById("toolPermDeselectAll");

    function _updateAllowAllState() {
      if (!allowAllChk || !toolList) return;
      const checks = toolList.querySelectorAll(".tool-perm-check");
      const allChecked = Array.from(checks).every(c => c.checked);
      allowAllChk.checked = allChecked;
    }

    allowAllChk?.addEventListener("change", () => {
      const checked = allowAllChk.checked;
      toolList?.querySelectorAll(".tool-perm-check").forEach(c => { c.checked = checked; });
    });

    toolList?.querySelectorAll(".tool-perm-check").forEach(c => {
      c.addEventListener("change", _updateAllowAllState);
    });

    selectAllBtn?.addEventListener("click", () => {
      toolList?.querySelectorAll(".tool-perm-check").forEach(c => { c.checked = true; });
      if (allowAllChk) allowAllChk.checked = true;
    });

    deselectAllBtn?.addEventListener("click", () => {
      toolList?.querySelectorAll(".tool-perm-check").forEach(c => { c.checked = false; });
      if (allowAllChk) allowAllChk.checked = false;
    });
  }

  // Save
  saveBtn.addEventListener("click", async () => {
    // Collect paths split by access level
    const rwPaths = [];
    const roPaths = [];

    pathList.querySelectorAll("[data-perm-path-row]").forEach(row => {
      const inp = row.querySelector("input[data-perm-path]");
      const v = inp?.value.trim();
      if (!v) return;
      // Read access value from data attribute (set by toggle buttons)
      const access = row.dataset.accessValue ?? "rw";
      if (access === "r") {
        roPaths.push(v);
      } else {
        rwPaths.push(v);
      }
    });

    const updated = {
      version: perm.version || 1,
      file_roots: rwPaths,
      file_roots_readonly: roPaths,
      commands: {
        allow_all: document.getElementById("permCmdAllowAll")?.checked ?? true,
        allow: perm.commands?.allow || [],
        deny: perm.commands?.deny || [],
      },
      external_tools: (() => {
        if (availableTools && availableTools.length > 0) {
          const toolList = document.getElementById("toolPermList");
          const checks = toolList ? Array.from(toolList.querySelectorAll(".tool-perm-check")) : [];
          const checkedTools = checks.filter(c => c.checked).map(c => c.dataset.tool);
          const allChecked = checkedTools.length === availableTools.length;
          return {
            allow_all: allChecked,
            allow: allChecked ? [] : checkedTools,
            deny: [],
          };
        }
        return {
          allow_all: document.getElementById("permExtAllowAll")?.checked ?? true,
          allow: perm.external_tools?.allow || [],
          deny: perm.external_tools?.deny || [],
        };
      })(),
      tool_creation: {
        personal: document.getElementById("permToolPersonal")?.checked ?? true,
        shared: document.getElementById("permToolShared")?.checked ?? false,
      },
    };

    saveBtn.disabled = true;
    saveBtn.textContent = t("animas.saving");
    status.textContent = "";

    try {
      await fetch(`/api/animas/${encodeURIComponent(name)}/permissions`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(updated),
      });
      // Update local ref so future saves keep changes
      Object.assign(perm, updated);
      status.textContent = t("animas.saved");
      status.style.color = "var(--color-success, #28a745)";
      setTimeout(() => { status.textContent = ""; }, 3000);
    } catch {
      status.textContent = t("animas.save_failed");
      status.style.color = "var(--color-danger, #dc3545)";
    }
    saveBtn.disabled = false;
    saveBtn.textContent = t("animas.save");
  });
}

// ── Show Detail ───────────────────────────

async function _showDetail(name) {
  if (!_container) return;
  _viewMode = "detail";
  _selectedName = name;

  _container.innerHTML = `
    <div class="page-header" style="display:flex; align-items:center; gap:1rem;">
      <button class="btn-secondary" id="animasBackBtn" style="font-size:0.85rem;">&larr; ${t("animas.back")}</button>
      <h2>${escapeHtml(name)}</h2>
    </div>
    <div id="animasDetailContent">
      <div class="loading-placeholder">${t("common.loading")}</div>
    </div>
  `;

  document.getElementById("animasBackBtn").addEventListener("click", () => {
    _viewMode = "list";
    _selectedName = null;
    _renderList();
  });

  const content = document.getElementById("animasDetailContent");
  if (!content) return;

  try {
    const [detail, models, availableTools] = await Promise.all([
      api(`/api/animas/${encodeURIComponent(name)}`),
      _fetchModels(),
      _fetchTools(),
    ]);

    // Try optional endpoints
    let animaConfig = null;
    let memoryStats = null;
    let permissions = {};
    let aliases = [];
    try { animaConfig = await api(`/api/animas/${encodeURIComponent(name)}/config`); } catch { /* 404 ok */ }
    try { memoryStats = await api(`/api/animas/${encodeURIComponent(name)}/memory/stats`); } catch { /* 404 ok */ }
    try { permissions = await api(`/api/animas/${encodeURIComponent(name)}/permissions`); } catch { /* 404 ok */ }
    try { const ar = await api(`/api/animas/${encodeURIComponent(name)}/aliases`); aliases = ar.aliases || []; } catch { /* 404 ok */ }
    let discordChannels = [];
    try { const dc = await api("/api/discord/channels"); discordChannels = dc.channels || []; } catch { /* not configured */ }

    let html = '<div class="card-grid" style="grid-template-columns: 1fr 1fr; margin-bottom: 1.5rem;">';

    // Identity card (editable)
    html += _editableCard({
      id: "anima_identity",
      title: t("animas.identity"),
      rawContent: detail.identity || "",
      renderFn: renderMarkdown,
    });

    // Injection card (editable)
    html += _editableCard({
      id: "anima_injection",
      title: t("animas.injection"),
      rawContent: detail.injection || "",
      renderFn: renderMarkdown,
    });

    html += "</div>";

    // State + Pending
    html += '<div class="card-grid" style="grid-template-columns: 1fr 1fr; margin-bottom: 1.5rem;">';

    html += `
      <div class="card">
        <div class="card-header">${t("animas.state_current")}</div>
        <div class="card-body">
          <pre style="white-space:pre-wrap; word-break:break-word; margin:0;">${escapeHtml(
            detail.state ? (typeof detail.state === "string" ? detail.state : JSON.stringify(detail.state, null, 2)) : t("animas.no_state")
          )}</pre>
        </div>
      </div>
    `;

    html += `
      <div class="card">
        <div class="card-header">${t("animas.pending")}</div>
        <div class="card-body">
          <pre style="white-space:pre-wrap; word-break:break-word; margin:0;">${escapeHtml(
            detail.pending ? (typeof detail.pending === "string" ? detail.pending : JSON.stringify(detail.pending, null, 2)) : t("animas.no_pending")
          )}</pre>
        </div>
      </div>
    `;

    html += "</div>";

    // Memory stats
    const epCount = detail.episode_files?.length ?? _extractStatsCount(memoryStats?.episodes);
    const knCount = detail.knowledge_files?.length ?? _extractStatsCount(memoryStats?.knowledge);
    const prCount = detail.procedure_files?.length ?? _extractStatsCount(memoryStats?.procedures);

    html += `
      <div class="card-grid" style="grid-template-columns: repeat(3, 1fr); margin-bottom: 1.5rem;">
        <div class="stat-card">
          <div class="stat-label">${t("chat.memory_episodes")}</div>
          <div class="stat-value">${epCount}</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">${t("chat.memory_knowledge")}</div>
          <div class="stat-value">${knCount}</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">${t("chat.memory_procedures")}</div>
          <div class="stat-value">${prCount}</div>
        </div>
      </div>
    `;

    // Model config (with combobox)
    const currentModel = animaConfig?.model || "";
    const currentCredential = animaConfig?.config?.credential || "";
    html += `
      <div class="card" style="margin-bottom: 1.5rem;">
        <div class="card-header" style="display:flex; justify-content:space-between; align-items:center;">
          <span>${t("animas.model_config")}</span>
          <span id="modelChangeStatus" style="font-size:0.75rem; color:var(--text-secondary,#888);"></span>
        </div>
        <div class="card-body">
          <div style="display:flex; align-items:center; gap:0.75rem; margin-bottom:0.75rem; flex-wrap:wrap;">
            <label style="font-weight:600; font-size:0.9rem;">${t("animas.model_select")}:</label>
            <select id="modelSelect" style="flex:1; min-width:200px; padding:0.4rem 0.5rem; border:1px solid var(--border,#ddd); border-radius:4px; font-size:0.85rem; background:var(--bg-secondary,#fff); color:var(--text-primary,#333);">
              ${models.map(m => {
                const selected = m.id === currentModel ? " selected" : "";
                return `<option value="${escapeHtml(m.id)}" data-credential="${escapeHtml(m.credential)}"${selected}>${escapeHtml(m.label)}</option>`;
              }).join("")}
              ${currentModel && !models.find(m => m.id === currentModel) ? `<option value="${escapeHtml(currentModel)}" selected>${escapeHtml(currentModel)} (current)</option>` : ""}
            </select>
            <button class="btn-primary" id="modelChangeBtn" style="font-size:0.85rem; padding:0.4rem 0.75rem;">${t("animas.model_change")}</button>
          </div>
          <div style="font-size:0.75rem; color:var(--text-secondary,#888); margin-bottom:0.75rem;">${t("animas.restart_notice")}</div>
          ${animaConfig ? `<details style="margin-top:0.5rem;"><summary style="cursor:pointer; font-size:0.85rem; color:var(--text-secondary,#666);">JSON</summary><pre style="white-space:pre-wrap; margin:0.5rem 0 0; font-size:0.8rem;">${escapeHtml(JSON.stringify(animaConfig, null, 2))}</pre></details>` : ""}
        </div>
      </div>
    `;

    // Aliases card
    html += _aliasesCardHtml(aliases);

    // Discord channels card
    if (discordChannels.length > 0) {
      html += _discordChannelsCardHtml(discordChannels, name);
    }

    // Permissions card
    html += _permissionsCardHtml(permissions, availableTools);

    // Action buttons
    html += `
      <div style="display:flex; gap:0.75rem;">
        <button class="btn-primary" id="animaDetailTrigger">${t("animas.heartbeat_trigger")}</button>
      </div>
    `;

    content.innerHTML = html;

    // Bind editable cards
    _bindEditableCard({ id: "anima_identity", name, field: "identity" });
    _bindEditableCard({ id: "anima_injection", name, field: "injection" });

    // Bind aliases card
    _bindAliasesCard(name, aliases);

    // Bind Discord channels card
    if (discordChannels.length > 0) {
      _bindDiscordChannelsCard(name, discordChannels);
    }

    // Bind permissions card
    _bindPermissionsCard(name, permissions, availableTools);

    // Bind model change button
    document.getElementById("modelChangeBtn")?.addEventListener("click", async () => {
      const select = document.getElementById("modelSelect");
      const btn = document.getElementById("modelChangeBtn");
      const status = document.getElementById("modelChangeStatus");
      if (!select || !btn) return;

      const selectedOption = select.options[select.selectedIndex];
      const model = select.value;
      const credential = selectedOption?.dataset?.credential || "";

      btn.disabled = true;
      btn.textContent = t("animas.saving");
      status.textContent = "";

      try {
        await fetch(`/api/animas/${encodeURIComponent(name)}/model`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ model, credential }),
        });
        status.textContent = t("animas.model_changed");
        status.style.color = "var(--color-success, #28a745)";
        setTimeout(() => { status.textContent = ""; }, 5000);
      } catch {
        status.textContent = t("animas.model_change_failed");
        status.style.color = "var(--color-danger, #dc3545)";
      }
      btn.disabled = false;
      btn.textContent = t("animas.model_change");
    });

    // Bind trigger button
    document.getElementById("animaDetailTrigger")?.addEventListener("click", async (e) => {
      const btn = e.target;
      btn.disabled = true;
      btn.textContent = t("animas.running");
      try {
        await fetch(`/api/animas/${encodeURIComponent(name)}/trigger`, { method: "POST" });
        btn.textContent = t("animas.success");
        setTimeout(() => { btn.textContent = t("animas.heartbeat_trigger"); btn.disabled = false; }, 2000);
      } catch {
        btn.textContent = t("animas.failed");
        setTimeout(() => { btn.textContent = t("animas.heartbeat_trigger"); btn.disabled = false; }, 2000);
      }
    });

  } catch (err) {
    content.innerHTML = `<div class="loading-placeholder">${t("animas.detail_load_failed")}: ${escapeHtml(err.message)}</div>`;
  }
}

// ── Helpers ────────────────────────────────

function _formatUptime(seconds) {
  if (!seconds || seconds < 0) return "--";
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  if (h > 0) return t("animas.uptime_hm", { h, m });
  return t("animas.uptime_m", { m });
}
