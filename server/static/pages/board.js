// ── Board Page (thin entry) ──────────────────
// The board view lives in ./board/. This file only wires the router's
// render/destroy lifecycle to a board page instance.
import { createBoardPage } from "./board/index.js";

let _page = null;

export function render(container, options = {}) {
  if (_page) _page.destroy();
  _page = createBoardPage(container, options);
  _page.render();
}

export function destroy() {
  if (!_page) return;
  _page.destroy();
  _page = null;
}
