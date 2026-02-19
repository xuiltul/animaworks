// ── Anima Birth Reveal Animation ──────────────────────
// Full-screen "gacha reveal" effect when a Anima's avatar is generated.

const FALLBACK_TIMEOUT = 5000;

/**
 * Preload an image by URL.
 * Resolves on both success and error so the reveal always proceeds.
 * @param {string} url
 * @returns {Promise<boolean>} true if loaded successfully
 */
export function preloadImage(url) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => resolve(true);
    img.onerror = () => resolve(false);
    img.src = url;
  });
}

/**
 * Play the reveal animation for a newly created Anima.
 * @param {{ name: string, avatarUrl?: string }} opts
 * @returns {Promise<void>} resolves when animation completes
 */
export async function playReveal({ name, avatarUrl }) {
  const overlay = document.getElementById("wsRevealOverlay");
  const avatar = document.getElementById("wsRevealAvatar");
  const text = document.getElementById("wsRevealText");
  if (!overlay || !avatar || !text) return;

  // Preload avatar image
  if (avatarUrl) {
    const loaded = await preloadImage(avatarUrl);
    if (loaded) {
      avatar.src = avatarUrl;
      avatar.style.display = "";
    } else {
      avatar.style.display = "none";
    }
  } else {
    avatar.style.display = "none";
  }

  text.textContent = `${name}さんが生まれました`;

  // Restore will-change hints before animation
  const flash = overlay.querySelector(".ws-reveal-flash");
  const content = overlay.querySelector(".ws-reveal-content");
  if (flash) flash.style.willChange = "";
  if (content) content.style.willChange = "";

  // Start animation
  overlay.classList.add("active");
  overlay.setAttribute("aria-hidden", "false");

  // Wait for content animation to finish
  return new Promise((resolve) => {
    let resolved = false;

    const cleanup = () => {
      if (resolved) return;
      resolved = true;
      if (content) content.removeEventListener("animationend", onEnd);
      overlay.classList.remove("active");
      overlay.setAttribute("aria-hidden", "true");
      // Clean up will-change to free GPU layers
      if (flash) flash.style.willChange = "auto";
      if (content) content.style.willChange = "auto";
      resolve();
    };

    const onEnd = () => cleanup();

    if (content) {
      content.addEventListener("animationend", onEnd);
    }

    // Fallback timeout in case animationend doesn't fire
    setTimeout(cleanup, FALLBACK_TIMEOUT);
  });
}
