// ── Image & File Input Module ───────────────────────────
// Shared image + document input handling for chat UIs.
// Supports: Ctrl+V paste, drag & drop, file picker button.
// Documents: PDF, CSV

import { t } from "/shared/i18n.js";
import { basePath } from "/shared/base-path.js";

const MAX_IMAGE_SIZE = 5 * 1024 * 1024; // 5MB per image
const MAX_FILE_SIZE = 15 * 1024 * 1024; // 15MB per document
const MAX_DIMENSION = 1568; // Max pixel dimension (Anthropic recommendation)
const SUPPORTED_TYPES = new Set(["image/jpeg", "image/png", "image/gif", "image/webp"]);
const DOCUMENT_TYPES = new Map([
  ["pdf", "application/pdf"],
  ["csv", "text/csv"],
]);

/**
 * Create image + file input manager for a chat container.
 *
 * @param {object} options
 * @param {HTMLElement} options.container - Parent container for drop events
 * @param {HTMLElement} options.inputArea - Chat input area element (for paste events)
 * @param {HTMLElement} options.previewContainer - Element to render thumbnails in
 * @param {function(): void} [options.onImagesChanged] - Callback when images/files array changes
 * @returns {object} Manager with getPendingImages(), getPendingFiles(), clearImages(), etc.
 */
export function createImageInput({ container, inputArea, previewContainer, onImagesChanged }) {
  const pendingImages = []; // Array of { data: base64String, media_type: string, dataUrl: string }
  const pendingFiles = []; // Array of { data: base64String, media_type: string, name: string }
  let processingCount = 0;
  let status = null;
  let rejectedSinceLastSubmit = false;

  function setStatus(kind, message) {
    status = { kind, message };
    renderPreviews();
  }

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

      rejectedSinceLastSubmit = false;
      status = { kind: "success", message: t("chat.image_ready", { count: pendingImages.length }) };
      renderPreviews();
      onImagesChanged?.();
      URL.revokeObjectURL(img.src);
    };
    img.src = URL.createObjectURL(file);
  }

  function processFiles(files) {
    for (const file of files) {
      const extension = (file?.name || "").split(".").pop()?.toLowerCase();
      if (DOCUMENT_TYPES.has(extension)) processDocumentFile(file, extension);
      else processImageFile(file);
    }
  }

  function processDocumentFile(file, extension) {
    if (file.size > MAX_FILE_SIZE) {
      setStatus("error", t("chat.file_too_large_client", {
        size: (file.size / 1024 / 1024).toFixed(1),
      }));
      return;
    }
    const mediaType = DOCUMENT_TYPES.get(extension);
    processingCount += 1;
    setStatus("info", t("chat.file_processing"));
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const result = String(reader.result || "");
        const base64Data = result.split(",")[1];
        if (!base64Data) throw new Error(t("chat.file_read_failed"));
        pendingFiles.push({ data: base64Data, media_type: mediaType, name: file.name });
        rejectedSinceLastSubmit = false;
        status = { kind: "success", message: t("chat.file_ready", { count: pendingFiles.length }) };
        onImagesChanged?.();
      } catch (error) {
        status = { kind: "error", message: error?.message || t("chat.file_read_failed") };
      } finally {
        processingCount -= 1;
        renderPreviews();
      }
    };
    reader.onerror = () => {
      processingCount -= 1;
      setStatus("error", t("chat.file_read_failed"));
    };
    reader.readAsDataURL(file);
  }

  // ── Preview Rendering ─────────────────────────────

  function renderPreviews() {
    if (!previewContainer) return;

    if (pendingImages.length === 0 && pendingFiles.length === 0 && !status && processingCount === 0) {
      previewContainer.style.display = "none";
      previewContainer.innerHTML = "";
      return;
    }

    previewContainer.style.display = "flex";

    let html = "";

    // Image previews
    html += pendingImages.map((img, i) => `
      <div class="image-preview-item" data-index="${i}">
        <img src="${img.dataUrl}" alt="Preview ${i + 1}" />
        <button class="image-preview-remove" data-index="${i}" data-kind="image" title="${t("assets.delete")}">&times;</button>
      </div>
    `).join("");

    // File previews
    html += pendingFiles.map((file, i) => {
      const ext = (file.name || "").split(".").pop()?.toUpperCase() || "FILE";
      const safeName = escapeHtml(file.name || t("chat.file_attachment"));
      return `
        <div class="image-preview-item file-preview-item" data-index="${i}">
          <div class="file-preview-icon"><b>${ext}</b></div>
          <div class="file-preview-name">${safeName}</div>
          <button class="image-preview-remove" data-index="${i}" data-kind="file" title="${t("assets.delete")}">&times;</button>
        </div>
      `;
    }).join("");

    // Status line
    if (status) {
      const cls = status.kind === "error" ? "image-status-error" : status.kind === "info" ? "image-status-info" : "image-status-success";
      html += `<div class="image-status ${cls}">${escapeHtml(status.message)}</div>`;
    }

    previewContainer.innerHTML = html;

    // Bind remove buttons
    previewContainer.querySelectorAll(".image-preview-remove").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();
        const idx = parseInt(btn.dataset.index, 10);
        const kind = btn.dataset.kind;
        if (kind === "file") {
          pendingFiles.splice(idx, 1);
        } else {
          pendingImages.splice(idx, 1);
        }
        status = null;
        renderPreviews();
        onImagesChanged?.();
      });
    });
  }

  // ── Event Listeners ───────────────────────────────

  // Ctrl+V paste
  inputArea.addEventListener("paste", (e) => {
    const items = e.clipboardData?.items;
    if (!items) return;
    for (const item of items) {
      if (item.type.startsWith("image/")) {
        e.preventDefault();
        processImageFile(item.getAsFile());
      }
    }
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
      processFiles(e.dataTransfer.files);
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

    /** Get pending document files for sending. */
    getPendingFiles() {
      return pendingFiles.map(({ data, media_type, name }) => ({ data, media_type, name }));
    },

    /** Get pending document files for display (without base64 data). */
    getDisplayFiles() {
      return pendingFiles.map(({ media_type, name }) => ({ media_type, name }));
    },

    /** Clear all pending images and files. */
    clearImages() {
      pendingImages.length = 0;
      pendingFiles.length = 0;
      status = null;
      rejectedSinceLastSubmit = false;
      renderPreviews();
    },

    /** Get current image count. */
    getImageCount() {
      return pendingImages.length;
    },

    /** Get current document file count. */
    getFileCount() {
      return pendingFiles.length;
    },

    /** Show an error message in the preview area. */
    showError(message) {
      setStatus("error", message);
    },

    /** True while selected files are still being decoded or converted. */
    isProcessing() {
      return processingCount > 0;
    },

    /** Block sending while processing, or once after every selected item was rejected. */
    prepareForSubmit() {
      if (processingCount > 0) {
        setStatus("info", t("chat.image_wait_for_processing") || "Processing...");
        return false;
      }
      if (rejectedSinceLastSubmit && pendingImages.length === 0 && pendingFiles.length === 0) {
        rejectedSinceLastSubmit = false;
        setStatus("error", t("chat.image_send_without_attachment") || "No attachments");
        return false;
      }
      return true;
    },

    /** Programmatically add files (for file picker button). */
    addFiles(files) {
      processFiles(files);
    },
  };
}

function escapeHtml(value) {
  const div = document.createElement("div");
  div.textContent = value || "";
  return div.innerHTML;
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
