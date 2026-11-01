// ── Image Input Module ──────────────────────────────────
// Shared image input handling for chat UIs.
// Supports: Ctrl+V paste, drag & drop, file picker button.

import { t } from "/shared/i18n.js";
import { basePath } from "/shared/base-path.js";

const MAX_IMAGE_SIZE = 5 * 1024 * 1024; // 5MB per image
const MAX_DIMENSION = 1568; // Max pixel dimension (Anthropic recommendation)
const SUPPORTED_TYPES = new Set(["image/jpeg", "image/png", "image/gif", "image/webp"]);
const MAX_FILE_SIZE = 25 * 1024 * 1024;
const MAX_FILE_COUNT = 10;
const MAX_TOTAL_FILE_SIZE = 50 * 1024 * 1024;

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[ch]);
}

const CHAT_FILE_EXTENSION_KINDS = {
  png: "image", jpg: "image", jpeg: "image", gif: "image", webp: "image",
  mp3: "audio", wav: "audio", m4a: "audio", aac: "audio", ogg: "audio", opus: "audio", flac: "audio",
  mp4: "video", mov: "video", webm: "video", mkv: "video", avi: "video",
  pdf: "pdf",
  md: "markdown", markdown: "markdown",
  csv: "csv",
  txt: "text", log: "text", json: "text", yaml: "text", yml: "text", toml: "text",
  ini: "text", py: "text", js: "text", ts: "text", sh: "text", sql: "text", xml: "text",
  xlsx: "xlsx", xlsm: "xlsx",
  docx: "docx",
  pptx: "pptx",
};

const CHAT_FILE_MIME_KINDS = {
  "application/pdf": "pdf",
  "text/markdown": "markdown",
  "text/x-markdown": "markdown",
  "text/csv": "csv",
  "application/csv": "csv",
  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
  "application/vnd.ms-excel.sheet.macroenabled.12": "xlsx",
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
  "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
  "application/json": "text",
  "application/xml": "text",
  "application/yaml": "text",
};

/**
 * Classify a chat attachment for rendering and preview.
 * The media type wins when it identifies a known kind; otherwise the extension is used.
 * SVG and HTML stay "other" so they are never opened as images or text.
 */
export function getChatFileKind(file = {}) {
  const name = String(file.name || "").toLowerCase();
  const mediaType = String(file.media_type || file.type || "").toLowerCase().split(";")[0].trim();
  const ext = name.includes(".") ? name.slice(name.lastIndexOf(".") + 1) : "";

  if (mediaType && mediaType !== "application/octet-stream") {
    if (mediaType === "image/svg+xml" || mediaType === "text/html" || mediaType === "application/xhtml+xml") {
      return "other";
    }
    if (mediaType.startsWith("audio/")) return "audio";
    if (mediaType.startsWith("video/")) return "video";
    if (mediaType.startsWith("image/")) return "image";
    if (CHAT_FILE_MIME_KINDS[mediaType]) return CHAT_FILE_MIME_KINDS[mediaType];
    if (mediaType.startsWith("text/")) return "text";
  }

  return CHAT_FILE_EXTENSION_KINDS[ext] || "other";
}

export function getChatFileIcon(file = {}) {
  const name = String(file.name || "").toLowerCase();
  const type = String(file.media_type || file.type || "").toLowerCase();
  const ext = name.includes(".") ? name.slice(name.lastIndexOf(".") + 1) : "";
  if (type === "application/pdf" || ext === "pdf") return "📕";
  if (["csv", "xls", "xlsx", "ods"].includes(ext) || /spreadsheet|excel|csv/.test(type)) return "📊";
  if (["doc", "docx", "odt", "rtf", "txt", "md", "ppt", "pptx", "odp"].includes(ext) || /wordprocessing|msword|powerpoint|presentation|text\//.test(type)) return "📄";
  if (type.startsWith("image/") || ["png", "jpg", "jpeg", "gif", "webp", "svg"].includes(ext)) return "🖼️";
  if (type.startsWith("audio/") || ["mp3", "wav", "m4a", "aac", "ogg", "flac"].includes(ext)) return "🎵";
  if (type.startsWith("video/") || ["mp4", "mov", "webm", "mkv", "avi"].includes(ext)) return "🎬";
  if (["zip", "rar", "7z", "tar", "gz", "bz2", "xz"].includes(ext) || /zip|compressed|archive/.test(type)) return "🗜️";
  return "📎";
}

export function formatChatFileSize(bytes) {
  const size = Math.max(0, Number(bytes) || 0);
  if (size < 1024) return t("chat.attach.size_bytes", { size: String(Math.floor(size)) });
  if (size < 1024 * 1024) return t("chat.attach.size_kb", { size: (size / 1024).toFixed(1).replace(/\.0$/, "") });
  return t("chat.attach.size_mb", { size: (size / (1024 * 1024)).toFixed(1).replace(/\.0$/, "") });
}

/** Return a localized limit key when a batch would exceed file attachment limits. */
export function validateChatFileBatch(files = [], existingFiles = []) {
  const all = [...existingFiles, ...files];
  if (all.some((file) => Number(file.size) === 0)) return "chat.attach.empty_file";
  if (all.length > MAX_FILE_COUNT) return "chat.attach.limit_count";
  if (all.some((file) => Number(file.size) > MAX_FILE_SIZE)) return "chat.attach.limit_file_size";
  if (all.reduce((total, file) => total + (Number(file.size) || 0), 0) > MAX_TOTAL_FILE_SIZE) {
    return "chat.attach.limit_total_size";
  }
  return null;
}

/**
 * Create image input manager for a chat container.
 *
 * @param {object} options
 * @param {HTMLElement} options.container - Parent container for drop events
 * @param {HTMLElement} options.inputArea - Chat input area element (for paste events)
 * @param {HTMLElement} options.previewContainer - Element to render thumbnails in
 * @param {function(): void} [options.onImagesChanged] - Callback when images array changes
 * @returns {object} Manager with getPendingImages(), clearImages(), getImageCount(), addFiles()
 */
export function createImageInput({ container, inputArea, previewContainer, onImagesChanged, allowFiles = false }) {
  const pendingImages = []; // Array of { data: base64String, media_type: string, dataUrl: string }
  const pendingFiles = []; // Array of { name, media_type, size, data, loading }
  let attachmentError = "";

  // ── File Processing Pipeline ──────────────────────

  function processImageFile(file) {
    if (!SUPPORTED_TYPES.has(file.type)) return;
    if (file.size > MAX_IMAGE_SIZE) {
      console.warn(`Image too large: ${(file.size / 1024 / 1024).toFixed(1)}MB (max 5MB)`);
      return;
    }

    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement("canvas");
      let { width, height } = img;

      // Resize if needed (preserve aspect ratio)
      if (width > MAX_DIMENSION || height > MAX_DIMENSION) {
        if (width > height) {
          height = Math.round(height * (MAX_DIMENSION / width));
          width = MAX_DIMENSION;
        } else {
          width = Math.round(width * (MAX_DIMENSION / height));
          height = MAX_DIMENSION;
        }
      }

      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(img, 0, 0, width, height);

      // Convert to base64
      const outputType = file.type === "image/png" ? "image/png" : "image/jpeg";
      const quality = outputType === "image/jpeg" ? 0.85 : undefined;
      const dataUrl = canvas.toDataURL(outputType, quality);
      const base64Data = dataUrl.split(",")[1];

      pendingImages.push({
        data: base64Data,
        media_type: outputType,
        dataUrl, // Keep for preview display
      });

      renderPreviews();
      onImagesChanged?.();
      URL.revokeObjectURL(img.src);
    };
    img.src = URL.createObjectURL(file);
  }

  function processImageFiles(files) {
    attachmentError = "";
    for (const file of files) {
      if (SUPPORTED_TYPES.has(file.type)) {
        processImageFile(file);
      } else if (allowFiles) {
        processGenericFile(file);
      }
    }
  }

  function processGenericFile(file) {
    const validationError = validateChatFileBatch([file], pendingFiles);
    if (validationError) {
      attachmentError = validationError;
      renderPreviews();
      onImagesChanged?.();
      return;
    }

    const attachment = {
      name: file.name || "file",
      media_type: file.type || "application/octet-stream",
      size: file.size,
      data: "",
      loading: true,
    };
    pendingFiles.push(attachment);
    renderPreviews();
    onImagesChanged?.();

    const reader = new FileReader();
    reader.onload = () => {
      const dataUrl = String(reader.result || "");
      attachment.data = dataUrl.split(",", 2)[1] || "";
      attachment.loading = false;
      renderPreviews();
      onImagesChanged?.();
    };
    reader.onerror = () => {
      const index = pendingFiles.indexOf(attachment);
      if (index >= 0) pendingFiles.splice(index, 1);
      attachmentError = "chat.attach.read_failed";
      renderPreviews();
      onImagesChanged?.();
    };
    reader.readAsDataURL(file);
  }

  // ── Preview Rendering ─────────────────────────────

  function renderPreviews() {
    if (!previewContainer) return;

    if (pendingImages.length === 0 && pendingFiles.length === 0 && !attachmentError) {
      previewContainer.style.display = "none";
      previewContainer.innerHTML = "";
      return;
    }

    previewContainer.style.display = "flex";
    const imageHtml = pendingImages.map((img, i) => `
      <div class="image-preview-item" data-index="${i}">
        <img src="${img.dataUrl}" alt="Preview ${i + 1}" />
        <button class="image-preview-remove" data-index="${i}" title="${t("assets.delete")}" aria-label="${t("chat.attach.remove_image")}">&times;</button>
      </div>
    `).join("");
    const fileHtml = pendingFiles.map((file, i) => `
      <div class="attachment-preview-file" title="${escapeHtml(file.name)}">
        <span class="attachment-file-icon" aria-hidden="true">${getChatFileIcon(file)}</span>
        <span class="attachment-file-name">${escapeHtml(file.name)}</span>
        <span class="attachment-file-size">${escapeHtml(formatChatFileSize(file.size))}</span>
        <button class="image-preview-remove attachment-preview-remove" data-file-index="${i}" type="button" title="${t("chat.attach.remove_file")}" aria-label="${t("chat.attach.remove_file")}">&times;</button>
      </div>
    `).join("");
    const errorHtml = attachmentError
      ? `<div class="attachment-preview-error" role="alert">${escapeHtml(t(attachmentError))}</div>`
      : "";
    previewContainer.innerHTML = imageHtml + fileHtml + errorHtml;

    // Bind remove buttons
    previewContainer.querySelectorAll(".image-preview-remove").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (btn.dataset.fileIndex !== undefined) {
          pendingFiles.splice(parseInt(btn.dataset.fileIndex, 10), 1);
        } else {
          pendingImages.splice(parseInt(btn.dataset.index, 10), 1);
        }
        renderPreviews();
        onImagesChanged?.();
      });
    });
  }

  // ── Event Listeners ───────────────────────────────

  // Ctrl+V paste
  inputArea.addEventListener("paste", (e) => {
    const clipboard = e.clipboardData;
    if (!clipboard) return;
    const itemFiles = Array.from(clipboard.items || []).map((item) => item.getAsFile()).filter(Boolean);
    const files = itemFiles.length ? itemFiles : Array.from(clipboard.files || []);
    if (!files.length) return;
    e.preventDefault();
    processImageFiles(files);
  });

  // Drag & drop on container
  container.addEventListener("dragover", (e) => {
    e.preventDefault();
    container.classList.add("image-drag-over");
  });

  container.addEventListener("dragleave", (e) => {
    // Only remove class when leaving the container entirely
    if (!container.contains(e.relatedTarget)) {
      container.classList.remove("image-drag-over");
    }
  });

  container.addEventListener("drop", (e) => {
    e.preventDefault();
    container.classList.remove("image-drag-over");
    if (e.dataTransfer?.files) {
      processImageFiles(e.dataTransfer.files);
    }
  });

  // ── Public API ────────────────────────────────────

  return {
    /** Get pending images for sending (without dataUrl preview field). */
    getPendingImages() {
      return pendingImages.map(({ data, media_type }) => ({ data, media_type }));
    },

    /** Get pending images with dataUrl for display in chat history. */
    getDisplayImages() {
      return pendingImages.map(({ data, media_type, dataUrl }) => ({ data, media_type, dataUrl }));
    },

    /** Clear all pending images. */
    clearImages() {
      pendingImages.length = 0;
      pendingFiles.length = 0;
      attachmentError = "";
      renderPreviews();
      onImagesChanged?.();
    },

    /** Get current image count. */
    getImageCount() {
      return pendingImages.length;
    },

    /** Get pending file attachments for the chat API. */
    getPendingFiles() {
      return pendingFiles.filter((file) => !file.loading).map(({ name, media_type, data }) => ({ name, media_type, data }));
    },

    /** Get file metadata for an optimistic chat bubble. */
    getDisplayFiles() {
      return pendingFiles.map(({ name, media_type, size }) => ({ name, media_type, size }));
    },

    /** Get file attachment count. */
    getFileCount() {
      return pendingFiles.length;
    },

    /** True while selected files are being read. */
    isProcessingFiles() {
      return pendingFiles.some((file) => file.loading);
    },

    /** Show a localized validation message for a rejected attachment. */
    setAttachmentError(key) {
      attachmentError = key || "";
      renderPreviews();
    },

    /** Restore queued base64 files after a user selects that queue item to edit. */
    restoreFiles(files = [], displayFiles = []) {
      for (const [index, file] of files.entries()) {
        const size = Number(displayFiles[index]?.size) || Math.floor((file.data?.length || 0) * 3 / 4);
        const restored = { ...file, size };
        const validationError = validateChatFileBatch([restored], pendingFiles);
        if (validationError) {
          attachmentError = validationError;
          renderPreviews();
          onImagesChanged?.();
          return false;
        }
        pendingFiles.push({ ...restored, loading: false });
      }
      attachmentError = "";
      renderPreviews();
      onImagesChanged?.();
      return true;
    },

    /** Programmatically add files (for file picker button). */
    addFiles(files) {
      processImageFiles(files);
    },
  };
}

// ── Lightbox ────────────────────────────────────────

/** Open a lightbox when clicking a chat image. Attach once to document. */
let _lightboxInitialized = false;

export function initLightbox() {
  if (_lightboxInitialized) return;
  _lightboxInitialized = true;

  document.addEventListener("click", (e) => {
    const img = e.target.closest(".chat-attached-image");
    if (!img) return;

    const overlay = document.createElement("div");
    overlay.className = "image-lightbox";
    overlay.innerHTML = `<img src="${img.src}" />`;
    overlay.addEventListener("click", () => overlay.remove());
    document.body.appendChild(overlay);
  });
}

// ── Helper: Render images HTML for a chat bubble ────

function _resolveArtifactSrc(img, animaName) {
  if (!img) return "";
  if (img.dataUrl) return img.dataUrl;
  if (img.data && img.media_type) {
    return `data:${img.media_type};base64,${img.data}`;
  }
  if (img.path && animaName) {
    if (img.path.startsWith("assets/")) {
      const filename = img.path.slice("assets/".length);
      return `${basePath}/api/animas/${encodeURIComponent(animaName)}/assets/${encodeURIComponent(filename)}`;
    }
    if (img.path.startsWith("attachments/")) {
      const filename = img.path.slice("attachments/".length);
      return `${basePath}/api/animas/${encodeURIComponent(animaName)}/attachments/${encodeURIComponent(filename)}`;
    }
  }
  if (img.url) {
    return `${basePath}/api/media/proxy?url=${encodeURIComponent(img.url)}`;
  }
  return "";
}

/**
 * Build HTML string for images inside a chat bubble.
 * @param {Array} images - Array of image-like objects
 * @param {Object} [options]
 * @param {string} [options.animaName] - Required for assets/attachments paths
 * @returns {string} HTML string (empty if no images)
 */
export function renderChatImages(images, options = {}) {
  if (!images || images.length === 0) return "";
  const animaName = options.animaName || "";
  let html = '<div class="chat-images">';
  let count = 0;
  for (const img of images) {
    const src = _resolveArtifactSrc(img, animaName);
    if (!src) continue;
    count += 1;
    html += `<img src="${src}" class="chat-attached-image" loading="lazy" alt="Attached image" onerror="this.onerror=null;this.classList.add('chat-attached-image-error');this.alt='Image unavailable';" />`;
  }
  html += '</div>';
  return count > 0 ? html : "";
}

const PREVIEW_KINDS = new Set(["pdf", "markdown", "csv", "text", "xlsx", "docx", "pptx"]);
export const CHAT_FILE_DOWNLOAD_ICON = '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" '
  + 'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
  + '<path d="M12 4v11"/><path d="m7 10 5 5 5-5"/><path d="M5 20h14"/></svg>';

/** Short type label for the coloured tile on a file card (extension, at most 4 letters). */
export function chatFileBadge(file = {}, kind = getChatFileKind(file)) {
  const name = String(file.name || "");
  const ext = name.includes(".") ? name.slice(name.lastIndexOf(".") + 1).toUpperCase() : "";
  if (ext && ext.length <= 4 && /^[A-Z0-9]+$/.test(ext)) return ext;
  return { markdown: "MD", text: "TXT", audio: "AUDIO", video: "VIDEO" }[kind] || "FILE";
}

/** Build file cards, media players, or legacy chips for a chat bubble. */
export function renderChatFiles(files, options = {}) {
  if (!files || files.length === 0) return "";
  const animaName = options.animaName || "";
  const items = files.map((file) => {
    const name = String(file.name || "file");
    const path = String(file.path || "");
    const storedName = path.startsWith("attachments/") ? path.slice("attachments/".length) : "";
    const href = animaName && storedName
      ? `${basePath}/api/animas/${encodeURIComponent(animaName)}/attachments/${encodeURIComponent(storedName)}`
      : "";
    const label = `${getChatFileIcon(file)} <span class="chat-file-chip-name">${escapeHtml(name)}</span>`
      + `<span class="chat-file-chip-size">${escapeHtml(formatChatFileSize(file.size))}</span>`;

    // Optimistic bubbles have no path yet; keep the compact local chip.
    if (!href) return `<div class="chat-file-chip" title="${escapeHtml(name)}">${label}</div>`;

    const kind = getChatFileKind(file);
    const safeHref = escapeHtml(href);
    const inlineSrc = escapeHtml(`${href}?inline=1`);
    const safeName = escapeHtml(name);
    const size = escapeHtml(formatChatFileSize(file.size));
    const downloadLabel = escapeHtml(t("chat.file_viewer.download"));
    const downloadLink = `<a class="chat-file-card-download" href="${safeHref}" download="${safeName}" `
      + `title="${downloadLabel}" aria-label="${downloadLabel}">${CHAT_FILE_DOWNLOAD_ICON}</a>`;
    const info = `<span class="chat-file-card-badge">${escapeHtml(chatFileBadge(file, kind))}</span>`
      + `<span class="chat-file-card-text"><span class="chat-file-card-name">${safeName}</span>`
      + `<span class="chat-file-card-size">${size}</span></span>`;

    if (kind === "image") {
      return `<div class="chat-file-image"><img class="chat-attached-image" src="${inlineSrc}" `
        + `loading="lazy" alt="${safeName}"></div>`;
    }

    if (kind === "audio" || kind === "video") {
      const media = kind === "audio"
        ? `<audio controls preload="metadata" src="${inlineSrc}"></audio>`
        : `<video controls playsinline preload="metadata" src="${inlineSrc}"></video>`;
      return `<div class="chat-file-media chat-file-kind-${kind}">`
        + `<div class="chat-file-card chat-file-card--media"><span class="chat-file-card-open">${info}</span>${downloadLink}</div>`
        + `${media}</div>`;
    }

    if (PREVIEW_KINDS.has(kind)) {
      const previewLabel = escapeHtml(t("chat.file_viewer.preview"));
      const attrs = `data-chat-file-preview data-anima="${escapeHtml(animaName)}"`
        + ` data-stored-name="${escapeHtml(storedName)}" data-name="${safeName}"`
        + ` data-size="${escapeHtml(String(Number(file.size) || 0))}" data-kind="${escapeHtml(kind)}"`;
      // The download link sits beside the open button: interactive content inside a <button>
      // is invalid HTML and Firefox never delivers its clicks.
      return `<div class="chat-file-card chat-file-kind-${kind}">`
        + `<button type="button" class="chat-file-card-open" ${attrs} aria-label="${previewLabel} ${safeName}">`
        + `${info}</button>${downloadLink}</div>`;
    }

    return `<a class="chat-file-chip" href="${safeHref}" download="${safeName}" title="${safeName}">${label}</a>`;
  }).join("");
  return `<div class="chat-files">${items}</div>`;
}
