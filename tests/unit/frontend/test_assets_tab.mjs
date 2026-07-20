/**
 * Unit tests for Anima assets tab (pages/anima-tabs/assets.js).
 *
 * Run with: node --test tests/unit/frontend/test_assets_tab.mjs
 */

import { describe, it, beforeEach } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const __dirname = dirname(fileURLToPath(import.meta.url));
const STATIC = resolve(__dirname, "../../../server/static");

// ── Minimal DOM ──────────────────────────────

class MockEl {
  constructor(tag = "div") {
    this.tagName = String(tag).toUpperCase();
    this.className = "";
    this.children = [];
    this.attributes = {};
    this.dataset = {};
    this.style = {};
    this._listeners = {};
    this.parentNode = null;
    this._innerHTML = "";
    this.textContent = "";
    this.id = "";
    this.value = "";
    this.disabled = false;
    this.src = "";
    this.alt = "";
    this.type = "";
    this.accept = "";
    this.files = null;
    this.checked = false;
    const self = this;
    this.classList = {
      toggle(cls, force) {
        const parts = new Set(self.className.split(/\s+/).filter(Boolean));
        if (force === true) parts.add(cls);
        else if (force === false) parts.delete(cls);
        else if (parts.has(cls)) parts.delete(cls);
        else parts.add(cls);
        self.className = [...parts].join(" ");
      },
      contains(cls) {
        return self.className.split(/\s+/).includes(cls);
      },
      add(cls) {
        const parts = new Set(self.className.split(/\s+/).filter(Boolean));
        parts.add(cls);
        self.className = [...parts].join(" ");
      },
      remove(cls) {
        const parts = new Set(self.className.split(/\s+/).filter(Boolean));
        parts.delete(cls);
        self.className = [...parts].join(" ");
      },
    };
  }

  set innerHTML(html) {
    this._innerHTML = String(html ?? "");
    this.children = [];
    // Register id hosts for querySelector / getElementById
    const idRe = /id="([^"]+)"/g;
    let idM;
    while ((idM = idRe.exec(this._innerHTML)) !== null) {
      const el = new MockEl("div");
      el.id = idM[1];
      el.parentNode = this;
      this.children.push(el);
      _byId.set(el.id, el);
    }
  }

  get innerHTML() {
    return this._innerHTML;
  }

  setAttribute(name, value) {
    this.attributes[name] = String(value);
    if (name === "data-tab") this.dataset.tab = String(value);
    if (name.startsWith("data-")) {
      const key = name.slice(5).replace(/-([a-z])/g, (_, c) => c.toUpperCase());
      this.dataset[key] = String(value);
    }
  }

  getAttribute(name) {
    return this.attributes[name] ?? null;
  }

  appendChild(child) {
    child.parentNode = this;
    this.children.push(child);
    if (child.id) _byId.set(child.id, child);
    return child;
  }

  remove() {
    if (this.parentNode) {
      const idx = this.parentNode.children.indexOf(this);
      if (idx >= 0) this.parentNode.children.splice(idx, 1);
      this.parentNode = null;
    }
    if (this.id) _byId.delete(this.id);
  }

  querySelectorAll(sel) {
    if (sel.startsWith(".")) {
      const cls = sel.slice(1).split(/[\s.>#[]/)[0];
      const out = [];
      if (this.className.split(/\s+/).includes(cls)) out.push(this);
      for (const c of this.children) out.push(...c.querySelectorAll(sel));
      return out;
    }
    if (sel.startsWith("#")) {
      const id = sel.slice(1);
      if (this.id === id) return [this];
      const out = [];
      for (const c of this.children) out.push(...c.querySelectorAll(sel));
      return out;
    }
    return [];
  }

  querySelector(sel) {
    return this.querySelectorAll(sel)[0] || null;
  }

  addEventListener(type, fn) {
    if (!this._listeners[type]) this._listeners[type] = [];
    this._listeners[type].push(fn);
  }

  closest() {
    return null;
  }
}

const _byId = new Map();

globalThis.document = {
  createElement(tag) {
    return new MockEl(tag);
  },
  getElementById(id) {
    return _byId.get(id) || null;
  },
  querySelector() {
    return null;
  },
  querySelectorAll() {
    return [];
  },
  head: new MockEl("head"),
  body: new MockEl("body"),
};

globalThis.window = globalThis;
globalThis.CSS = { escape: (s) => String(s).replace(/"/g, '\\"') };
globalThis.requestAnimationFrame = (fn) => {
  fn();
  return 0;
};

// ── Load assets tab with stubs ──

function loadAssetsModule({ apiImpl, realistic = false } = {}) {
  const path = resolve(STATIC, "pages/anima-tabs/assets.js");
  let source = readFileSync(path, "utf8");
  source = source.replace(/^import\s+.+;?\s*$/gm, "");

  const apiCalls = [];
  globalThis.__assetsApiCalls = apiCalls;
  globalThis.__assetsApi =
    apiImpl ||
    (async (endpoint, opts) => {
      apiCalls.push({ endpoint, opts });
      if (endpoint === "/api/animas") {
        return [{ name: "sakura" }, { name: "yuki" }];
      }
      if (endpoint.includes("/assets/metadata")) {
        return {
          assets: {
            avatar_fullbody: { url: "/api/animas/sakura/assets/fullbody.png" },
            avatar_bustup: { url: "/api/animas/sakura/assets/bustup.png" },
            avatar_icon: null,
            avatar_chibi: null,
          },
          assets_realistic: {},
          animations: {},
          expressions: {
            smile: { url: "/api/animas/sakura/assets/smile.png" },
          },
          expressions_realistic: {},
        };
      }
      return {};
    });

  globalThis.__isRealisticMode = () => realistic;

  const preamble = `
    const api = (...args) => globalThis.__assetsApi(...args);
    const escapeHtml = (s) => String(s ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
    const t = (k) => k;
    const basePath = "";
    const isRealisticMode = () => globalThis.__isRealisticMode();
  `;

  const url =
    "data:text/javascript;base64," +
    Buffer.from(preamble + "\n" + source, "utf8").toString("base64");
  return import(url + "#assets-" + Math.random());
}

describe("assets tab module (source contract)", () => {
  const source = readFileSync(
    resolve(STATIC, "pages/anima-tabs/assets.js"),
    "utf8",
  );

  it("exports render/destroy and accepts animaName", () => {
    assert.match(source, /export async function render\(/);
    assert.match(source, /export function destroy\(/);
    assert.match(source, /animaName/);
  });

  it("does not render standalone Anima selector UI", () => {
    assert.doesNotMatch(source, /assetsAnimaSelector/);
    assert.doesNotMatch(source, /assets-anima-selector/);
    assert.doesNotMatch(source, /assets-anima-btn/);
  });

  it("installs and removes WebSocket handler", () => {
    assert.match(source, /__assetsWsHandler/);
    assert.match(source, /_removeWsHandler/);
    assert.match(source, /_forceCloseModals|_closeRemakeModal/);
  });
});

describe("assets tab render/API/destroy", () => {
  beforeEach(() => {
    _byId.clear();
    globalThis.__assetsApiCalls = [];
    delete globalThis.__assetsWsHandler;
    document.body = new MockEl("body");
  });

  it("fetches metadata for the provided animaName and renders gallery", async () => {
    const calls = [];
    const mod = await loadAssetsModule({
      apiImpl: async (endpoint, opts) => {
        calls.push(endpoint);
        if (endpoint === "/api/animas") return [{ name: "sakura" }];
        if (endpoint === "/api/animas/sakura/assets/metadata") {
          return {
            assets: {
              avatar_fullbody: { url: "/fb.png" },
              avatar_bustup: { url: "/bu.png" },
              avatar_icon: null,
              avatar_chibi: null,
            },
            assets_realistic: {},
            animations: {},
            expressions: {},
            expressions_realistic: {},
          };
        }
        return {};
      },
    });

    const container = new MockEl("div");
    await mod.render(container, { animaName: "sakura" });

    assert.ok(
      calls.some((c) => c === "/api/animas/sakura/assets/metadata"),
      `expected metadata fetch for sakura, got: ${JSON.stringify(calls)}`,
    );

    const content = document.getElementById("assetsGalleryContent");
    assert.ok(content, "assetsGalleryContent should exist");
    assert.match(content.innerHTML, /assets-gallery|Fullbody|assetsRemakeBtn/);
    assert.ok(typeof globalThis.__assetsWsHandler === "function");

    mod.destroy();
  });

  it("destroy removes WS handler and force-closes remake modal", async () => {
    const mod = await loadAssetsModule();
    const container = new MockEl("div");
    await mod.render(container, { animaName: "sakura" });

    assert.equal(typeof globalThis.__assetsWsHandler, "function");

    // Simulate an open remake overlay (as if remake was in progress)
    const overlay = new MockEl("div");
    overlay.id = "assetsRemakeOverlay";
    overlay.className = "assets-modal-overlay";
    document.body.appendChild(overlay);
    _byId.set("assetsRemakeOverlay", overlay);

    const confirm = new MockEl("div");
    confirm.id = "assetsConfirmDialog";
    document.body.appendChild(confirm);
    _byId.set("assetsConfirmDialog", confirm);

    mod.destroy();

    assert.equal(globalThis.__assetsWsHandler, undefined);
    assert.equal(document.getElementById("assetsRemakeOverlay"), null);
    assert.equal(document.getElementById("assetsConfirmDialog"), null);
  });

  it("destroy is safe when remake was never opened", async () => {
    const mod = await loadAssetsModule();
    const container = new MockEl("div");
    await mod.render(container, { animaName: "yuki" });
    mod.destroy();
    mod.destroy(); // idempotent
    assert.equal(globalThis.__assetsWsHandler, undefined);
  });
});

describe("animas page wires assets tab", () => {
  it("includes assets in _DETAIL_TABS", () => {
    const source = readFileSync(resolve(STATIC, "pages/animas.js"), "utf8");
    assert.match(source, /id:\s*"assets"/);
    assert.match(source, /animas\.tab_assets/);
  });
});
