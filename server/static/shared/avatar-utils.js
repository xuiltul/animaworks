/**
 * Shared avatar utility functions.
 * Generates deterministic colors from Anima names for initial-letter avatars.
 */

/**
 * Generate a consistent HSL color from an Anima name using a hash.
 * @param {string} name
 * @returns {string} CSS hsl() value
 */
export function animaHashColor(name) {
  let hash = 0;
  for (let i = 0; i < name.length; i++) {
    hash = name.charCodeAt(i) + ((hash << 5) - hash);
  }
  return `hsl(${Math.abs(hash) % 360}, 45%, 45%)`;
}
