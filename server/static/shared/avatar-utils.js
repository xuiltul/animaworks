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

/** Curated palette for company accent colors (distinguishable, works on light/dark). */
const COMPANY_PALETTE = [
  "#2563eb", // blue
  "#d97706", // amber
  "#059669", // emerald
  "#7c3aed", // violet
  "#dc2626", // red
  "#0891b2", // cyan
  "#db2777", // pink
  "#65a30d", // lime
];

/**
 * Deterministic accent color for a company name.
 * @param {string|null|undefined} company
 * @returns {string} CSS color, or "" when no company
 */
export function companyColor(company) {
  if (!company) return "";
  let hash = 0;
  for (let i = 0; i < company.length; i++) {
    hash = company.charCodeAt(i) + ((hash << 5) - hash);
  }
  return COMPANY_PALETTE[Math.abs(hash) % COMPANY_PALETTE.length];
}

/**
 * Shorten a model id for compact UI display.
 * e.g. "anthropic.claude-opus-4-8" → "claude-opus-4-8", "openai/gpt-4o" → "gpt-4o"
 * @param {string|null|undefined} model
 * @returns {string}
 */
export function shortModel(model) {
  if (!model) return "";
  return String(model)
    .replace(/^(openai|google|vertex_ai|azure|ollama|bedrock)\//, "")
    .replace(/^jp\.anthropic\./, "")
    .replace(/^anthropic\./, "");
}
