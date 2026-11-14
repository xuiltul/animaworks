// ── Hub Page (thin entry) ─────────────────────
// The external-AI screen lives in ./hub/. This file only wires the router's
// render/destroy lifecycle to a hub page instance.
import { createHubPage } from "./hub/index.js";

let _page = null;

export async function render(container, options = {}) {
  if (_page) _page.destroy();
  _page = createHubPage(container, options);
  await _page.render();
}

export function destroy() {
  if (!_page) return;
  _page.destroy();
  _page = null;
}
