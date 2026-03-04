// ── Org Hierarchy Utilities ──────────────────────
// Helpers for filtering WebSocket events by organisational hierarchy.

/**
 * Build a Set of all descendants (direct + transitive subordinates) of
 * the given anima, using the flat animas array from /api/animas.
 *
 * @param {string} animaName - The supervisor anima name
 * @param {Array<{name:string, supervisor:string|null}>} animas - Flat list
 * @returns {Set<string>} Names of all descendant animas
 */
export function getDescendants(animaName, animas) {
  const bySuper = new Map();
  for (const a of animas) {
    if (!a.supervisor) continue;
    const list = bySuper.get(a.supervisor);
    if (list) list.push(a.name);
    else bySuper.set(a.supervisor, [a.name]);
  }

  const result = new Set();
  const queue = bySuper.get(animaName) || [];
  let i = 0;
  while (i < queue.length) {
    const name = queue[i++];
    if (result.has(name)) continue;
    result.add(name);
    const children = bySuper.get(name);
    if (children) queue.push(...children);
  }
  return result;
}
