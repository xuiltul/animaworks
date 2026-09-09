// ── Model Catalog / Model Picker Helpers ──────────────────────
// Shared helpers used by both the main chat page (pages/chat/) and the
// workspace UI (workspace/modules/) to build the chat model dropdown.
//
// Example I/O (populateModelSelect):
//
//   const catalog = await fetchModelCatalog();   // => { models, groups, ok }
//   populateModelSelect(selectEl, catalog, "c:codex/gpt-5.6-sol");
//
//   - selectEl  : an existing <select> (e.g. data-chat-id="chatPageModel")
//   - catalog   : normalized { models:[...], groups:[...], ok:boolean }
//   - selectedId: currently selected model id, or "" for the anima default.
//
// Behavior:
//   - The first option is always the empty anima-default option
//     (value "", label set by the caller via the `t()` key).
//   - Remaining options are grouped into <optgroup> elements. Group order
//     follows catalog.groups; unknown groups are appended alphabetically.
//   - Each option carries a `title` with the model id (+ note when present).
//   - If selectedId is not present in the catalog, an "(unavailable)"
//     group is appended carrying that value, still selected.
//   - If the fetch failed (ok === false), only the default option is
//     shown, select.disabled is set, and the error is put on `title`.
//
// All construction uses DOM APIs (createDocumentFragment / new Option /
// createElement("optgroup")) — never innerHTML — so external model IDs
// (Ollama / CLI names) cannot break out of attributes.

import { api } from "../../modules/api.js";

let _catalogPromise = null;

/**
 * Fetch the available-models catalog.
 *
 * The result is memoized in this module; pass { force: true } to bypass the
 * cache and refetch (e.g. to invalidate after the catalog is refreshed).
 *
 * Never throws. Returns a normalized object:
 *   { models: [...], groups: [...], ok: boolean }
 * On failure `ok` is false and `models`/`groups` fall back to [].
 */
export async function fetchModelCatalog({ force = false } = {}) {
  if (!force && _catalogPromise) return _catalogPromise;
  const p = (async () => {
    let data = null;
    try {
      data = await api(force ? "/api/system/available-models?refresh=1" : "/api/system/available-models");
    } catch (err) {
      return { models: [], groups: [], ok: false, error: err?.message || "fetch failed" };
    }
    const models = Array.isArray(data?.models) ? data.models : [];
    const groups = Array.isArray(data?.groups) ? data.groups : [];
    return { models, groups, ok: true };
  })();
  _catalogPromise = p;
  try {
    await p;
  } catch {
    /* p never rejects (errors swallowed above) */
  }
  return _catalogPromise;
}

/**
 * Resolve the group label for a model entry.
 * Prefers the server-provided `group`, falls back to `credential`, and
 * finally to "Other" (via the caller-provided fallback label).
 *
 * @param {object|null} m      a model entry
 * @param {string}      fallback  label used when no group is derivable
 * @returns {string}
 */
function _groupLabel(m, fallback) {
  if (m && typeof m.group === "string" && m.group) return m.group;
  if (m && typeof m.credential === "string" && m.credential) return m.credential;
  return fallback || "Other";
}

/**
 * Normalize a models list: ensure every entry is a plain object with an id.
 * @param {Array|null} models
 * @returns {Array} list of { id, label, note, group, ... } entries
 */
function _normalizeModels(models) {
  if (!Array.isArray(models)) return [];
  return models
    .filter(m => m && typeof m === "object" && typeof m.id === "string" && m.id)
    .map(m => ({
      id: m.id,
      label: typeof m.label === "string" && m.label ? m.label : m.id,
      note: typeof m.note === "string" && m.note ? m.note : "",
      group: typeof m.group === "string" && m.group ? m.group : "",
      credential: typeof m.credential === "string" && m.credential ? m.credential : "",
    }));
}

/**
 * Populate a <select> with the model catalog, grouped by <optgroup>.
 *
 * @param {HTMLSelectElement} selectEl  the target <select>
 * @param {object|Array|null} catalog   normalized { models, groups, ok }
 *                                      (return of fetchModelCatalog), or a
 *                                      plain array of model entries
 * @param {string}            selectedId  currently selected model id ("" =
 *                                       anima default)
 * @param {object}            opts      optional overrides
 *   opts.defaultLabel   — text for the empty default option
 *   opts.otherLabel     — label for the "(unavailable)" group / fallback
 *   opts.errorTitle     — optional error string to put on title when failed
 */
export function populateModelSelect(selectEl, catalog, selectedId, opts = {}) {
  if (!selectEl) return;

  let list = null;
  let groups = null;
  let ok = true;
  if (catalog && typeof catalog === "object" && Array.isArray(catalog.models)) {
    list = catalog.models;
    groups = Array.isArray(catalog.groups) ? catalog.groups : [];
    ok = catalog.ok !== false;
  } else if (Array.isArray(catalog)) {
    list = catalog;
    groups = null;
  }

  const models = _normalizeModels(list);
  const defaultLabel = opts.defaultLabel || "";
  const otherLabel = opts.otherLabel || "Other";
  const errorTitle = opts.errorTitle || "";

  const fragment = document.createDocumentFragment();

  // Empty anima-default option first.
  const defaultOpt = new Option(defaultLabel, "");
  defaultOpt.disabled = false;
  fragment.append(defaultOpt);

  // Fetch failure → only the default option; disable the select.
  if (!ok || !catalog || (Array.isArray(catalog) && catalog.length === 0 && models.length === 0 && selectedId === "")) {
    selectEl.replaceChildren(fragment);
    selectEl.disabled = true;
    selectEl.title = errorTitle || "model catalog unavailable";
    return;
  }
  selectEl.disabled = false;

  // Group models. Order known groups first (per catalog.groups), then
  // unknown groups alphabetically.
  const groupEntries = new Map();
  for (const m of models) {
    const label = _groupLabel(m, otherLabel);
    if (!groupEntries.has(label)) groupEntries.set(label, []);
    groupEntries.get(label).push(m);
  }

  const knownGroupOrder = Array.isArray(groups) ? groups.filter(g => groupEntries.has(g)) : [];
  const knownSet = new Set(knownGroupOrder);
  const unknownGroups = [...groupEntries.keys()].filter(g => !knownSet.has(g)).sort();

  const orderedGroups = [...knownGroupOrder, ...unknownGroups];

  const requestedSelected =
    typeof selectedId === "string" && selectedId !== "" ? selectedId : null;

  // Does the requested selection exist anywhere?
  const selectionExists = requestedSelected === null
    || models.some(m => m.id === requestedSelected);

  for (const groupLabel of orderedGroups) {
    const entries = groupEntries.get(groupLabel) || [];
    const optgroup = document.createElement("optgroup");
    optgroup.label = groupLabel;
    for (const m of entries) {
      const opt = new Option(m.label, m.id);
      opt.title = m.note ? `${m.id}\n${m.note}` : m.id;
      if (requestedSelected !== null && m.id === requestedSelected) opt.selected = true;
      optgroup.append(opt);
    }
    fragment.append(optgroup);
  }

  // If the requested selection does not exist in the catalog, add an
  // "(unavailable)" group carrying that value, still selected — never
  // silently fall back to the anima default.
  if (requestedSelected !== null && !selectionExists) {
    const optgroup = document.createElement("optgroup");
    optgroup.label = `${otherLabel} (unavailable)`;
    const opt = new Option(`${requestedSelected} (unavailable)`, requestedSelected);
    opt.title = requestedSelected;
    opt.selected = true;
    optgroup.append(opt);
    fragment.append(optgroup);
  }

  selectEl.replaceChildren(fragment);
  if (requestedSelected !== null) selectEl.value = requestedSelected;
}

/**
 * Build a short display label for a model id (e.g. for buttons).
 *
 * @param {object|Array|null} catalogOrModels  normalized catalog or model list
 * @param {string|null}       id               model id
 * @returns {string} short label, falling back to the id, then ""
 */
export function modelShortLabel(catalogOrModels, id) {
  let list = null;
  if (catalogOrModels && typeof catalogOrModels === "object" && Array.isArray(catalogOrModels.models)) {
    list = catalogOrModels.models;
  } else if (Array.isArray(catalogOrModels)) {
    list = catalogOrModels;
  }
  if (!id) return "";
  if (!Array.isArray(list)) return id;
  const found = list.find(m => m && m.id === id);
  return found?.label || id;
}