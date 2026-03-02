// ── Asset Management ────────────────────────
import { api } from "../modules/api.js";
import { escapeHtml } from "../modules/state.js";
import { t } from "/shared/i18n.js";
import { isRealisticMode } from "../modules/avatar-resolver.js";

let _container = null;
let _animas = [];
let _selectedAnima = null;
let _metadata = null;
let _previewBackupId = null;
let _previewGenerated = false;
let _rebuildInProgress = false;
let _previewHistory = [];
let _currentPreviewIdx = -1;

const EXPRESSION_LIST = ["neutral", "smile", "laugh", "troubled", "surprised", "thinking", "embarrassed"];

// ── Render / Destroy ────────────────────────

export async function render(container) {
  _container = container;
  _selectedAnima = null;
  _metadata = null;
  _previewBackupId = null;
  _previewGenerated = false;
  _rebuildInProgress = false;
  _previewHistory = [];
  _currentPreviewIdx = -1;

  _installWsHandler();

  container.innerHTML = `
    <div class="page-header">
      <h2>${t("assets.page_title")}</h2>
      <p>${t("assets.page_subtitle")}</p>
    </div>

    <div class="assets-anima-selector" id="assetsAnimaSelector">
      <div class="loading-placeholder">${t("assets.loading_list")}</div>
    </div>

    <div id="assetsGalleryContent">
      <div class="loading-placeholder">${t("assets.select_anima")}</div>
    </div>
  `;

  await _loadAnimaList();
}

export function destroy() {
  _removeWsHandler();
  _container = null;
  _animas = [];
  _selectedAnima = null;
  _metadata = null;
  _previewBackupId = null;
  _previewGenerated = false;
  _rebuildInProgress = false;
  _previewHistory = [];
  _currentPreviewIdx = -1;
}

// ── WebSocket Handler ───────────────────────

function _installWsHandler() {
  window.__assetsWsHandler = _handleWsEvent;
}

function _removeWsHandler() {
  delete window.__assetsWsHandler;
}

function _handleWsEvent(eventType, data) {
  switch (eventType) {
    case "anima.remake_preview_ready":
      _onPreviewReady(data);
      break;
    case "anima.remake_progress":
      _onRemakeProgress(data);
      break;
    case "anima.remake_complete":
      _onRemakeComplete(data);
      break;
  }
}

// ── Anima Selector ─────────────────────────

async function _loadAnimaList() {
  const selector = document.getElementById("assetsAnimaSelector");
  if (!selector) return;

  try {
    _animas = await api("/api/animas");

    if (_animas.length === 0) {
      selector.innerHTML = `<div class="loading-placeholder">${t("animas.not_registered")}</div>`;
      return;
    }

    selector.innerHTML = "";
    for (const p of _animas) {
      const btn = document.createElement("button");
      btn.className = "assets-anima-btn";
      btn.dataset.name = p.name;
      btn.textContent = p.name;
      btn.addEventListener("click", () => _selectAnima(p.name));
      selector.appendChild(btn);
    }
  } catch (err) {
    selector.innerHTML = `<div class="loading-placeholder">${t("assets.fetch_failed")}: ${escapeHtml(err.message)}</div>`;
  }
}

function _selectAnima(name) {
  _selectedAnima = name;
  _metadata = null;
  _previewBackupId = null;
  _previewGenerated = false;

  const selector = document.getElementById("assetsAnimaSelector");
  if (selector) {
    selector.querySelectorAll(".assets-anima-btn").forEach(btn => {
      btn.classList.toggle("assets-anima-btn--active", btn.dataset.name === name);
    });
  }

  _loadGallery();
}

// ── Asset Gallery ───────────────────────────

async function _loadGallery() {
  const content = document.getElementById("assetsGalleryContent");
  if (!content || !_selectedAnima) return;

  content.innerHTML = `<div class="loading-placeholder">${t("assets.loading")}</div>`;

  const enc = encodeURIComponent(_selectedAnima);

  try {
    _metadata = await api(`/api/animas/${enc}/assets/metadata`);
  } catch (err) {
    content.innerHTML = `<div class="loading-placeholder">${t("assets.fetch_failed")}: ${escapeHtml(err.message)}</div>`;
    return;
  }

  const animeAssets = _metadata.assets || {};
  const realisticAssets = _metadata.assets_realistic || {};
  const animations = _metadata.animations || {};
  const modelCount = [animeAssets.model_chibi, animeAssets.model_rigged].filter(Boolean).length;
  const animCount = Object.keys(animations).length;

  const isRealistic = isRealisticMode();
  const primaryLabel = isRealistic ? t("assets.realistic_section") : t("assets.anime_section");
  const secondaryLabel = isRealistic ? t("assets.anime_section") : t("assets.realistic_section");

  let html = "";

  // Primary section (open)
  html += `<h3 style="margin-bottom:0.5rem;">${escapeHtml(primaryLabel)}</h3>`;
  html += '<div class="assets-gallery">';
  if (isRealistic) {
    html += _renderThumbnail("Fullbody", realisticAssets.avatar_fullbody_realistic);
    html += _renderThumbnail("Bustup", realisticAssets.avatar_bustup_realistic);
  } else {
    html += _renderThumbnail("Fullbody", animeAssets.avatar_fullbody);
    html += _renderThumbnail("Bustup", animeAssets.avatar_bustup);
    html += _renderThumbnail("Chibi", animeAssets.avatar_chibi);
    html += `
      <div class="assets-thumb-card">
        <div class="assets-thumb-placeholder">
          <span class="assets-badge-icon">3D</span>
        </div>
        <div class="assets-thumb-label">${t("assets.model_3d")}</div>
        <span class="assets-badge">${modelCount}</span>
      </div>
    `;
    html += `
      <div class="assets-thumb-card">
        <div class="assets-thumb-placeholder">
          <span class="assets-badge-icon">Anim</span>
        </div>
        <div class="assets-thumb-label">${t("assets.animation")}</div>
        <span class="assets-badge">${animCount}</span>
      </div>
    `;
  }
  html += "</div>";
  html += _renderExpressionGrid(isRealistic ? "realistic" : "anime");

  // Secondary section (collapsible)
  const secondaryStyle = isRealistic ? "anime" : "realistic";
  html += `<details style="margin-top:1.5rem;">`;
  html += `<summary style="cursor:pointer;font-weight:600;font-size:0.9rem;color:var(--aw-color-text-secondary,#555);">${escapeHtml(secondaryLabel)}</summary>`;
  html += '<div class="assets-gallery" style="margin-top:0.5rem;">';
  if (isRealistic) {
    html += _renderThumbnailSmall("Fullbody", animeAssets.avatar_fullbody);
    html += _renderThumbnailSmall("Bustup", animeAssets.avatar_bustup);
    html += _renderThumbnailSmall("Chibi", animeAssets.avatar_chibi);
  } else {
    html += _renderThumbnailSmall("Fullbody", realisticAssets.avatar_fullbody_realistic);
    html += _renderThumbnailSmall("Bustup", realisticAssets.avatar_bustup_realistic);
  }
  html += "</div>";
  html += _renderExpressionGrid(secondaryStyle);
  html += "</details>";

  // Remake button
  html += `
    <div style="margin-top: 1.5rem;">
      <button class="btn-primary" id="assetsRemakeBtn">Remake Assets</button>
    </div>
  `;

  content.innerHTML = html;

  document.getElementById("assetsRemakeBtn")?.addEventListener("click", () => {
    _openRemakeModal();
  });

  _bindExpressionButtons();
}

function _renderThumbnail(label, assetInfo) {
  if (assetInfo && assetInfo.url) {
    return `
      <div class="assets-thumb-card">
        <img class="assets-thumb-img" src="${escapeHtml(assetInfo.url)}" alt="${escapeHtml(label)}" loading="lazy">
        <div class="assets-thumb-label">${escapeHtml(label)}</div>
      </div>
    `;
  }
  return `
    <div class="assets-thumb-card">
      <div class="assets-thumb-placeholder">
        <span class="assets-thumb-empty">No Image</span>
      </div>
      <div class="assets-thumb-label">${escapeHtml(label)}</div>
    </div>
  `;
}

function _renderThumbnailSmall(label, assetInfo) {
  const badge = assetInfo && assetInfo.url ? t("assets.exists") : t("assets.not_generated");
  const badgeClass = assetInfo && assetInfo.url ? "assets-badge--exists" : "assets-badge--missing";
  if (assetInfo && assetInfo.url) {
    return `
      <div class="assets-thumb-card assets-thumb-card--small">
        <img class="assets-thumb-img" src="${escapeHtml(assetInfo.url)}" alt="${escapeHtml(label)}" loading="lazy">
        <div class="assets-thumb-label">${escapeHtml(label)} <span class="${badgeClass}">${escapeHtml(badge)}</span></div>
      </div>
    `;
  }
  return `
    <div class="assets-thumb-card assets-thumb-card--small">
      <div class="assets-thumb-placeholder">
        <span class="assets-thumb-empty">—</span>
      </div>
      <div class="assets-thumb-label">${escapeHtml(label)} <span class="${badgeClass}">${escapeHtml(badge)}</span></div>
    </div>
  `;
}

// ── Expression Grid ─────────────────────────

function _renderExpressionGrid(style) {
  if (!_metadata) return "";

  const expressions = style === "realistic"
    ? (_metadata.expressions_realistic || {})
    : (_metadata.expressions || {});

  let html = `<div class="assets-expression-section" data-style="${escapeHtml(style)}" style="margin-top:1.5rem;">`;
  html += `<div class="assets-expression-header">`;
  html += `<h3 style="margin:0;">${t("assets.expressions_title")}</h3>`;
  html += `<button class="btn-secondary btn-sm assets-regen-all-expr-btn" data-style="${escapeHtml(style)}">${t("assets.regen_all_expressions")}</button>`;
  html += `</div>`;
  html += `<div class="assets-expression-grid">`;

  for (const emotion of EXPRESSION_LIST) {
    if (emotion === "neutral") continue;
    const info = expressions[emotion];
    const label = t(`assets.expr_${emotion}`) || emotion;
    const imgHtml = info?.url
      ? `<img class="assets-expression-img" src="${escapeHtml(info.url)}?t=${Date.now()}" alt="${escapeHtml(label)}" loading="lazy">`
      : `<div class="assets-expression-placeholder">${t("assets.not_generated")}</div>`;

    html += `
      <div class="assets-expression-card" data-emotion="${escapeHtml(emotion)}" data-style="${escapeHtml(style)}">
        ${imgHtml}
        <div class="assets-expression-card-footer">
          <span class="assets-expression-label">${escapeHtml(label)}</span>
          <button class="assets-expression-regen-btn" data-emotion="${escapeHtml(emotion)}" data-style="${escapeHtml(style)}" title="${t("assets.regen_expression")}">↻</button>
        </div>
      </div>
    `;
  }

  html += `</div></div>`;
  return html;
}

function _bindExpressionButtons() {
  document.querySelectorAll(".assets-expression-regen-btn").forEach(btn => {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      _regenerateExpression(btn.dataset.emotion, btn.dataset.style);
    });
  });

  document.querySelectorAll(".assets-regen-all-expr-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      _regenerateAllExpressions(btn.dataset.style);
    });
  });
}

async function _regenerateExpression(emotion, style) {
  if (!_selectedAnima) return;
  const enc = encodeURIComponent(_selectedAnima);
  const card = document.querySelector(`.assets-expression-card[data-emotion="${CSS.escape(emotion)}"][data-style="${CSS.escape(style)}"]`);
  const btn = card?.querySelector(".assets-expression-regen-btn");

  if (btn) { btn.disabled = true; btn.textContent = "…"; }
  if (card) {
    const target = card.querySelector(".assets-expression-img") || card.querySelector(".assets-expression-placeholder");
    if (target) target.style.opacity = "0.4";
  }

  try {
    const result = await api(`/api/animas/${enc}/assets/generate-expression`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        expression: emotion,
        image_style: style,
      }),
    });

    if (result.url && card) {
      const existingImg = card.querySelector(".assets-expression-img");
      const placeholder = card.querySelector(".assets-expression-placeholder");
      if (existingImg) {
        existingImg.src = result.url + "?t=" + Date.now();
        existingImg.style.opacity = "";
      } else if (placeholder) {
        const img = document.createElement("img");
        img.className = "assets-expression-img";
        img.src = result.url + "?t=" + Date.now();
        img.alt = emotion;
        placeholder.replaceWith(img);
      }
    }
  } catch (err) {
    if (card) {
      const target = card.querySelector(".assets-expression-img") || card.querySelector(".assets-expression-placeholder");
      if (target) target.style.opacity = "";
    }
  } finally {
    if (btn) { btn.disabled = false; btn.textContent = "↻"; }
  }
}

async function _regenerateAllExpressions(style) {
  const allBtn = document.querySelector(`.assets-regen-all-expr-btn[data-style="${CSS.escape(style)}"]`);
  if (allBtn) { allBtn.disabled = true; allBtn.textContent = t("assets.preview_generating"); }

  for (const emotion of EXPRESSION_LIST) {
    if (emotion === "neutral") continue;
    await _regenerateExpression(emotion, style);
  }

  if (allBtn) { allBtn.disabled = false; allBtn.textContent = t("assets.regen_all_expressions"); }
}

// ── Remake Modal ────────────────────────────

function _currentImageStyle() {
  return isRealisticMode() ? "realistic" : "anime";
}

function _openRemakeModal() {
  if (!_selectedAnima || !_metadata) return;

  _closeRemakeModal(false);

  _previewHistory = [];
  _currentPreviewIdx = -1;
  _previewBackupId = null;
  _previewGenerated = false;

  const imgStyle = _currentImageStyle();
  const isReal = imgStyle === "realistic";
  const animeAssets = _metadata.assets || {};
  const realAssets = _metadata.assets_realistic || {};
  const currentFullbodyUrl = isReal
    ? (realAssets.avatar_fullbody_realistic?.url || "")
    : (animeAssets.avatar_fullbody?.url || "");

  // Build style-from options: "None" first, then other animas
  let styleOptions = `<option value="">${t("assets.generate_from_scratch")}</option>`;
  for (const p of _animas) {
    if (p.name !== _selectedAnima) {
      styleOptions += `<option value="${escapeHtml(p.name)}">${escapeHtml(p.name)}</option>`;
    }
  }

  const overlay = document.createElement("div");
  overlay.id = "assetsRemakeOverlay";
  overlay.className = "assets-modal-overlay";
  overlay.innerHTML = `
    <div class="assets-modal">
      <div class="assets-modal-header">
        <h3>Remake Assets - ${escapeHtml(_selectedAnima)} <span style="font-size:0.75em;color:var(--aw-color-text-secondary,#888);">(${isReal ? "Realistic" : "Anime"})</span></h3>
        <button class="assets-modal-close" id="assetsModalCloseBtn">&times;</button>
      </div>

      <div class="assets-modal-body">
        <div class="assets-modal-columns">
          <!-- Left: Current image -->
          <div class="assets-modal-col">
            <div class="assets-modal-col-label">${t("assets.current")}</div>
            ${currentFullbodyUrl
              ? `<img class="assets-modal-preview-img" src="${escapeHtml(currentFullbodyUrl)}" alt="Current fullbody">`
              : `<div class="assets-modal-preview-placeholder">${t("assets.no_fullbody")}</div>`
            }
          </div>

          <!-- Right: Preview image -->
          <div class="assets-modal-col">
            <div class="assets-modal-col-label">${t("assets.preview")}</div>
            <div id="assetsPreviewContainer" class="assets-modal-preview-placeholder">
              ${t("assets.preview_pending")}
            </div>
            <div id="assetsPreviewNav" class="assets-preview-nav" style="display:none;">
              <button class="btn-secondary btn-sm" id="assetsPrevBtn" disabled>&lt; Prev</button>
              <span id="assetsPreviewCounter" class="assets-preview-counter">0 / 0</span>
              <button class="btn-secondary btn-sm" id="assetsNextBtn" disabled>Next &gt;</button>
            </div>
          </div>
        </div>

        <!-- Controls -->
        <div class="assets-modal-controls">
          <div class="assets-modal-control-row">
            <label class="assets-modal-label" for="assetsImageStyle">Image Style:</label>
            <select id="assetsImageStyle" class="assets-modal-select">
              <option value="realistic" ${imgStyle === "realistic" ? "selected" : ""}>${t("assets.realistic_section")}</option>
              <option value="anime" ${imgStyle === "anime" ? "selected" : ""}>${t("assets.anime_section")}</option>
            </select>
          </div>
          <div class="assets-modal-control-row">
            <label class="assets-modal-label" for="assetsStyleFrom">Style From:</label>
            <select id="assetsStyleFrom" class="assets-modal-select">
              ${styleOptions}
            </select>
          </div>

          <div class="assets-modal-control-row">
            <label class="assets-modal-label" for="assetsVibeStrength">
              Vibe Strength: <span id="assetsVibeStrengthVal">0.6</span>
            </label>
            <input type="range" id="assetsVibeStrength" min="0" max="1" step="0.05" value="0.6" class="assets-modal-range">
          </div>

          <div class="assets-modal-control-row">
            <label class="assets-modal-label" for="assetsInfoExtract">
              Info Extract: <span id="assetsInfoExtractVal">0.8</span>
            </label>
            <input type="range" id="assetsInfoExtract" min="0" max="1" step="0.05" value="0.8" class="assets-modal-range">
          </div>
        </div>

        <!-- Action buttons: 3-zone layout -->
        <div class="assets-modal-actions">
          <div class="assets-modal-actions-left">
            <button class="btn-primary" id="assetsGeneratePreviewBtn">Generate Preview</button>
            <button class="btn-secondary" id="assetsRetryBtn" style="display:none;">Retry</button>
          </div>
          <div class="assets-modal-actions-right">
            <button class="btn-primary" id="assetsAcceptBtn" style="display:none; background:#16a34a;">Accept &amp; Rebuild All</button>
            <button class="btn-secondary" id="assetsCancelBtn">Cancel</button>
          </div>
        </div>

        <!-- Status text (shown after completion instead of button) -->
        <div id="assetsStatusText" class="assets-status-text" style="display:none;"></div>

        <!-- Progress section (hidden by default) -->
        <div id="assetsProgressSection" class="assets-progress-section" style="display:none;">
          <div class="assets-progress-header">Rebuild Progress</div>
          <div id="assetsProgressList" class="assets-progress-list"></div>
        </div>
      </div>
    </div>
  `;

  document.body.appendChild(overlay);

  requestAnimationFrame(() => overlay.classList.add("active"));

  // Bind slider live updates + style-from change
  const vibeSlider = document.getElementById("assetsVibeStrength");
  const vibeVal = document.getElementById("assetsVibeStrengthVal");
  if (vibeSlider && vibeVal) {
    vibeSlider.addEventListener("input", () => { vibeVal.textContent = vibeSlider.value; });
  }

  const infoSlider = document.getElementById("assetsInfoExtract");
  const infoVal = document.getElementById("assetsInfoExtractVal");
  if (infoSlider && infoVal) {
    infoSlider.addEventListener("input", () => { infoVal.textContent = infoSlider.value; });
  }

  const styleSelect = document.getElementById("assetsStyleFrom");
  if (styleSelect) {
    styleSelect.addEventListener("change", () => _updateVibeSliderState());
    _updateVibeSliderState();
  }

  const imageStyleSelect = document.getElementById("assetsImageStyle");
  const currentCol = overlay.querySelector(".assets-modal-col:first-child");
  const animeUrl = animeAssets.avatar_fullbody?.url || "";
  const realUrl = realAssets.avatar_fullbody_realistic?.url || "";
  if (imageStyleSelect && currentCol) {
    imageStyleSelect.addEventListener("change", () => {
      const style = imageStyleSelect.value;
      const url = style === "realistic" ? realUrl : animeUrl;
      if (url) {
        currentCol.innerHTML = `
          <div class="assets-modal-col-label">${t("assets.current")}</div>
          <img class="assets-modal-preview-img" src="${escapeHtml(url)}" alt="Current fullbody">
        `;
      } else {
        currentCol.innerHTML = `
          <div class="assets-modal-col-label">${t("assets.current")}</div>
          <div class="assets-modal-preview-placeholder">${t("assets.no_fullbody")}</div>
        `;
      }
    });
  }

  // Bind action buttons
  document.getElementById("assetsGeneratePreviewBtn")?.addEventListener("click", _generatePreview);
  document.getElementById("assetsAcceptBtn")?.addEventListener("click", _confirmAcceptAndRebuild);
  document.getElementById("assetsRetryBtn")?.addEventListener("click", _generatePreview);
  document.getElementById("assetsCancelBtn")?.addEventListener("click", () => _cancelRemake());
  document.getElementById("assetsModalCloseBtn")?.addEventListener("click", () => _cancelRemake());

  // Preview navigation
  document.getElementById("assetsPrevBtn")?.addEventListener("click", () => _navigatePreview(-1));
  document.getElementById("assetsNextBtn")?.addEventListener("click", () => _navigatePreview(1));

  overlay.addEventListener("click", (e) => {
    if (e.target === overlay) _cancelRemake();
  });
}

function _updateVibeSliderState() {
  const styleFrom = document.getElementById("assetsStyleFrom")?.value;
  const vibeSlider = document.getElementById("assetsVibeStrength");
  const infoSlider = document.getElementById("assetsInfoExtract");
  const hasStyle = !!styleFrom;

  [vibeSlider, infoSlider].forEach(slider => {
    if (slider) {
      slider.disabled = !hasStyle;
      slider.closest(".assets-modal-control-row")?.classList.toggle("assets-control-disabled", !hasStyle);
    }
  });
}

// ── Preview Generation ──────────────────────

async function _generatePreview() {
  if (!_selectedAnima) return;

  const styleFrom = document.getElementById("assetsStyleFrom")?.value || null;
  const vibeStrength = parseFloat(document.getElementById("assetsVibeStrength")?.value || "0.6");
  const infoExtracted = parseFloat(document.getElementById("assetsInfoExtract")?.value || "0.8");

  const generateBtn = document.getElementById("assetsGeneratePreviewBtn");
  const retryBtn = document.getElementById("assetsRetryBtn");
  const acceptBtn = document.getElementById("assetsAcceptBtn");
  const previewContainer = document.getElementById("assetsPreviewContainer");

  if (generateBtn) { generateBtn.disabled = true; generateBtn.textContent = "Generating..."; }
  if (retryBtn) { retryBtn.disabled = true; }
  if (acceptBtn) { acceptBtn.disabled = true; }

  if (previewContainer) {
    previewContainer.innerHTML = `<div class="assets-spinner"></div><div style="margin-top:0.5rem;">${t("assets.preview_generating")}</div>`;
    previewContainer.className = "assets-modal-preview-placeholder";
  }

  const enc = encodeURIComponent(_selectedAnima);

  const imageStyle = document.getElementById("assetsImageStyle")?.value || _currentImageStyle();
  const requestBody = {
    vibe_strength: vibeStrength,
    vibe_info_extracted: infoExtracted,
    image_style: imageStyle,
  };
  if (styleFrom) requestBody.style_from = styleFrom;
  if (_previewBackupId) requestBody.backup_id = _previewBackupId;

  try {
    const result = await api(`/api/animas/${enc}/assets/remake-preview`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(requestBody),
    });

    _previewBackupId = result.backup_id || _previewBackupId;

    if (result.preview_url) {
      _addPreviewToHistory(result.preview_url, result.preview_file, result.seed_used);
      _showCurrentPreview();
    }

  } catch (err) {
    if (previewContainer) {
      previewContainer.innerHTML = `<div class="assets-error">${t("assets.preview_failed")}: ${escapeHtml(err.message)}</div>`;
      previewContainer.className = "assets-modal-preview-placeholder";
    }
    if (generateBtn) { generateBtn.disabled = false; generateBtn.textContent = "Generate Preview"; }
    if (retryBtn) { retryBtn.disabled = false; }
    if (acceptBtn) { acceptBtn.disabled = false; }
  }
}

function _addPreviewToHistory(url, previewFile, seed) {
  _previewHistory.push({ url, previewFile, seed, index: _previewHistory.length });
  _currentPreviewIdx = _previewHistory.length - 1;
}

function _showCurrentPreview() {
  if (_currentPreviewIdx < 0 || _currentPreviewIdx >= _previewHistory.length) return;

  const entry = _previewHistory[_currentPreviewIdx];
  const previewContainer = document.getElementById("assetsPreviewContainer");
  if (previewContainer) {
    const img = document.createElement("img");
    img.className = "assets-modal-preview-img";
    img.alt = "Preview";
    img.src = entry.url + (entry.url.includes("?") ? "&" : "?") + "t=" + Date.now();
    img.addEventListener("error", () => {
      previewContainer.innerHTML = `<div class="assets-error">${t("assets.preview_load_failed")}</div>`;
      previewContainer.className = "assets-modal-preview-placeholder";
    });
    previewContainer.innerHTML = "";
    previewContainer.className = "";
    previewContainer.appendChild(img);
  }

  _previewGenerated = true;
  _updatePreviewNav();

  const generateBtn = document.getElementById("assetsGeneratePreviewBtn");
  const acceptBtn = document.getElementById("assetsAcceptBtn");
  const retryBtn = document.getElementById("assetsRetryBtn");
  if (generateBtn) { generateBtn.disabled = false; generateBtn.textContent = "Generate Preview"; generateBtn.style.display = "none"; }
  if (acceptBtn) { acceptBtn.style.display = ""; acceptBtn.disabled = false; }
  if (retryBtn) { retryBtn.style.display = ""; retryBtn.disabled = false; }
}

function _updatePreviewNav() {
  const nav = document.getElementById("assetsPreviewNav");
  const prevBtn = document.getElementById("assetsPrevBtn");
  const nextBtn = document.getElementById("assetsNextBtn");
  const counter = document.getElementById("assetsPreviewCounter");

  if (_previewHistory.length <= 1) {
    if (nav) nav.style.display = "none";
    return;
  }

  if (nav) nav.style.display = "";
  if (prevBtn) prevBtn.disabled = _currentPreviewIdx <= 0;
  if (nextBtn) nextBtn.disabled = _currentPreviewIdx >= _previewHistory.length - 1;
  if (counter) counter.textContent = `${_currentPreviewIdx + 1} / ${_previewHistory.length}`;
}

function _navigatePreview(delta) {
  const newIdx = _currentPreviewIdx + delta;
  if (newIdx < 0 || newIdx >= _previewHistory.length) return;
  _currentPreviewIdx = newIdx;
  _showCurrentPreview();
}

function _showModalError(message) {
  const previewContainer = document.getElementById("assetsPreviewContainer");
  if (previewContainer) {
    previewContainer.innerHTML = `<div class="assets-error">${escapeHtml(message)}</div>`;
    previewContainer.className = "assets-modal-preview-placeholder";
  }
}

// ── Accept & Rebuild ────────────────────────

function _confirmAcceptAndRebuild() {
  if (!_selectedAnima || !_previewBackupId) return;
  if (_rebuildInProgress) return;

  // Show confirmation dialog
  const existing = document.getElementById("assetsConfirmDialog");
  if (existing) existing.remove();

  const dialog = document.createElement("div");
  dialog.id = "assetsConfirmDialog";
  dialog.className = "assets-confirm-overlay";
  dialog.innerHTML = `
    <div class="assets-confirm-dialog">
      <p>${t("assets.confirm_rebuild")}</p>
      <div class="assets-confirm-actions">
        <button class="btn-primary" id="assetsConfirmYes" style="background:#16a34a;">OK</button>
        <button class="btn-secondary" id="assetsConfirmNo">Cancel</button>
      </div>
    </div>
  `;
  document.body.appendChild(dialog);
  requestAnimationFrame(() => dialog.classList.add("active"));

  document.getElementById("assetsConfirmYes")?.addEventListener("click", () => {
    dialog.remove();
    _acceptAndRebuild();
  });
  document.getElementById("assetsConfirmNo")?.addEventListener("click", () => {
    dialog.remove();
  });
}

async function _acceptAndRebuild() {
  if (!_selectedAnima || !_previewBackupId) return;

  const acceptBtn = document.getElementById("assetsAcceptBtn");
  const retryBtn = document.getElementById("assetsRetryBtn");
  const cancelBtn = document.getElementById("assetsCancelBtn");
  const generateBtn = document.getElementById("assetsGeneratePreviewBtn");

  if (acceptBtn) { acceptBtn.disabled = true; acceptBtn.textContent = "Rebuilding..."; }
  if (retryBtn) { retryBtn.disabled = true; }
  if (cancelBtn) { cancelBtn.disabled = true; }
  if (generateBtn) { generateBtn.disabled = true; }

  _rebuildInProgress = true;

  const progressSection = document.getElementById("assetsProgressSection");
  if (progressSection) { progressSection.style.display = ""; }

  const enc = encodeURIComponent(_selectedAnima);

  // Use the currently displayed preview file
  const currentEntry = _previewHistory[_currentPreviewIdx];
  const previewFile = currentEntry?.previewFile || null;

  try {
    const result = await api(`/api/animas/${enc}/assets/remake-confirm`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        backup_id: _previewBackupId,
        image_style: document.getElementById("assetsImageStyle")?.value || _currentImageStyle(),
        preview_file: previewFile,
      }),
    });

    if (result.steps) {
      _initProgressBars(result.steps);
    }

  } catch (err) {
    _showModalError(`${t("assets.rebuild_failed")}: ${err.message}`);
    if (acceptBtn) { acceptBtn.disabled = false; acceptBtn.textContent = "Accept & Rebuild All"; }
    if (retryBtn) { retryBtn.disabled = false; }
    if (cancelBtn) { cancelBtn.disabled = false; }
    if (generateBtn) { generateBtn.disabled = false; }
    _rebuildInProgress = false;
  }
}

async function _cancelRemake() {
  if (_rebuildInProgress) return;

  if (_selectedAnima && _previewBackupId) {
    const enc = encodeURIComponent(_selectedAnima);
    try {
      await fetch(`/api/animas/${enc}/assets/remake-preview`, { method: "DELETE" });
    } catch {
      // Ignore cleanup errors
    }
  }

  _closeRemakeModal(true);
}

function _closeRemakeModal(resetState) {
  const overlay = document.getElementById("assetsRemakeOverlay");
  if (overlay) overlay.remove();
  const confirmDialog = document.getElementById("assetsConfirmDialog");
  if (confirmDialog) confirmDialog.remove();

  if (resetState) {
    _previewBackupId = null;
    _previewGenerated = false;
    _rebuildInProgress = false;
    _previewHistory = [];
    _currentPreviewIdx = -1;
  }
}

// ── Progress Tracking ───────────────────────

function _initProgressBars(steps) {
  const list = document.getElementById("assetsProgressList");
  if (!list) return;

  list.innerHTML = "";
  for (const step of steps) {
    const stepName = typeof step === "string" ? step : step.name || step;
    const item = document.createElement("div");
    item.className = "assets-progress-item";
    item.dataset.step = stepName;
    item.innerHTML = `
      <div class="assets-progress-item-label">${escapeHtml(stepName)}</div>
      <div class="assets-progress-bar-track">
        <div class="assets-progress-bar-fill" style="width: 0%;"></div>
      </div>
      <span class="assets-progress-pct">0%</span>
    `;
    list.appendChild(item);
  }
}

function _updateProgressStep(stepName, percent, status) {
  const list = document.getElementById("assetsProgressList");
  if (!list) return;

  let item = list.querySelector(`[data-step="${CSS.escape(stepName)}"]`);

  if (!item) {
    item = document.createElement("div");
    item.className = "assets-progress-item";
    item.dataset.step = stepName;
    item.innerHTML = `
      <div class="assets-progress-item-label">${escapeHtml(stepName)}</div>
      <div class="assets-progress-bar-track">
        <div class="assets-progress-bar-fill" style="width: 0%;"></div>
      </div>
      <span class="assets-progress-pct">0%</span>
    `;
    list.appendChild(item);
  }

  const fill = item.querySelector(".assets-progress-bar-fill");
  const pct = item.querySelector(".assets-progress-pct");

  const clampedPercent = Math.max(0, Math.min(100, percent));
  if (fill) fill.style.width = `${clampedPercent}%`;
  if (pct) pct.textContent = `${Math.round(clampedPercent)}%`;

  if (status === "completed" || clampedPercent >= 100) {
    item.classList.add("assets-progress-item--done");
    if (fill) fill.style.background = "#16a34a";
  } else if (status === "error") {
    item.classList.add("assets-progress-item--error");
    if (fill) fill.style.background = "#ef4444";
  }
}

// ── WebSocket Event Handlers ────────────────

function _onPreviewReady(data) {
  const animaName = data.name || data.anima;
  if (animaName !== _selectedAnima) return;

  const previewUrl = data.preview_url || data.url;
  if (previewUrl) {
    _previewBackupId = data.backup_id || _previewBackupId;
    _addPreviewToHistory(previewUrl, data.preview_file, data.seed_used);
    _showCurrentPreview();
  }
}

function _onRemakeProgress(data) {
  const animaName = data.name || data.anima;
  if (animaName !== _selectedAnima) return;

  const progressSection = document.getElementById("assetsProgressSection");
  if (progressSection) progressSection.style.display = "";

  const stepName = data.step || data.stage || "unknown";
  const percent = data.progress_pct ?? data.percent ?? data.progress ?? 0;
  const status = data.status || (percent >= 100 ? "completed" : "in_progress");

  _updateProgressStep(stepName, percent, status);
}

function _onRemakeComplete(data) {
  const animaName = data.name || data.anima;
  if (animaName !== _selectedAnima) return;

  _rebuildInProgress = false;

  const acceptBtn = document.getElementById("assetsAcceptBtn");
  const retryBtn = document.getElementById("assetsRetryBtn");
  const generateBtn = document.getElementById("assetsGeneratePreviewBtn");
  const cancelBtn = document.getElementById("assetsCancelBtn");
  const statusText = document.getElementById("assetsStatusText");

  if (data.success !== false) {
    // Hide action buttons, show only Close
    if (acceptBtn) acceptBtn.style.display = "none";
    if (retryBtn) retryBtn.style.display = "none";
    if (generateBtn) generateBtn.style.display = "none";
    if (cancelBtn) { cancelBtn.disabled = false; cancelBtn.textContent = "Close"; }

    // Show completion status as text
    if (statusText) {
      statusText.style.display = "";
      statusText.textContent = t("assets.rebuild_complete") || "All assets rebuilt successfully.";
    }

    const progressList = document.getElementById("assetsProgressList");
    if (progressList) {
      const msg = document.createElement("div");
      msg.className = "assets-progress-complete";
      msg.textContent = "All assets rebuilt successfully.";
      progressList.appendChild(msg);
    }

    setTimeout(() => {
      _loadGallery();
    }, 1500);
  } else {
    const errorMsg = data.error || "Rebuild failed";
    _showModalError(errorMsg);
    if (acceptBtn) { acceptBtn.disabled = false; acceptBtn.textContent = "Accept & Rebuild All"; }
    if (retryBtn) { retryBtn.disabled = false; }
    if (generateBtn) { generateBtn.disabled = false; }
    if (cancelBtn) { cancelBtn.disabled = false; }
  }
}
