// ── Image Input Module ──────────────────────────────────
// Shared image input handling for chat UIs.
// Supports: Ctrl+V paste, drag & drop, file picker button.

const MAX_IMAGE_SIZE = 5 * 1024 * 1024; // 5MB per image
const MAX_DIMENSION = 1568; // Max pixel dimension (Anthropic recommendation)
const SUPPORTED_TYPES = new Set(["image/jpeg", "image/png", "image/gif", "image/webp"]);

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
export function createImageInput({ container, inputArea, previewContainer, onImagesChanged }) {
  const pendingImages = []; // Array of { data: base64String, media_type: string, dataUrl: string }

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
    for (const file of files) {
      if (file.type.startsWith("image/")) {
        processImageFile(file);
      }
    }
  }

  // ── Preview Rendering ─────────────────────────────

  function renderPreviews() {
    if (!previewContainer) return;

    if (pendingImages.length === 0) {
      previewContainer.style.display = "none";
      previewContainer.innerHTML = "";
      return;
    }

    previewContainer.style.display = "flex";
    previewContainer.innerHTML = pendingImages.map((img, i) => `
      <div class="image-preview-item" data-index="${i}">
        <img src="${img.dataUrl}" alt="Preview ${i + 1}" />
        <button class="image-preview-remove" data-index="${i}" title="削除">&times;</button>
      </div>
    `).join("");

    // Bind remove buttons
    previewContainer.querySelectorAll(".image-preview-remove").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();
        const idx = parseInt(btn.dataset.index, 10);
        pendingImages.splice(idx, 1);
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
      renderPreviews();
    },

    /** Get current image count. */
    getImageCount() {
      return pendingImages.length;
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

/**
 * Build HTML string for images inside a chat bubble.
 * @param {Array} images - Array of { data, media_type, dataUrl }
 * @returns {string} HTML string (empty if no images)
 */
export function renderChatImages(images) {
  if (!images || images.length === 0) return "";
  let html = '<div class="chat-images">';
  for (const img of images) {
    const src = img.dataUrl || `data:${img.media_type};base64,${img.data}`;
    html += `<img src="${src}" class="chat-attached-image" />`;
  }
  html += '</div>';
  return html;
}
