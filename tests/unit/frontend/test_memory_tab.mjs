/**
 * Unit tests for Anima memory tab (pages/anima-tabs/memory.js).
 *
 * Run with: node --test tests/unit/frontend/test_memory_tab.mjs
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
    // Rebuild children for simple button/tab markup + id hosts
    this.children = [];
    const re = /<button\b([^>]*)>([\s\S]*?)<\/button>/gi;
    let m;
    while ((m = re.exec(this._innerHTML)) !== null) {
      const attrs = m[1];
      const btn = new MockEl("button");
      const classMatch = attrs.match(/class="([^"]*)"/);
      if (classMatch) btn.className = classMatch[1];
      const dataTab = attrs.match(/data-tab="([^"]*)"/);
      if (dataTab) {
        btn.dataset.tab = dataTab[1];
        btn.attributes["data-tab"] = dataTab[1];
      }
      const aria = attrs.match(/aria-selected="([^"]*)"/);
      if (aria) btn.attributes["aria-selected"] = aria[1];
      btn.textContent = m[2];
      btn.parentNode = this;
      this.children.push(btn);
    }
    // id="..." host divs for querySelector
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
      // Also match buttons from innerHTML parse
      for (const c of this.children) {
        if (c.className && c.className.split(/\s+/).includes(cls) && !out.includes(c)) {
          out.push(c);
        }
      }
      return out;
    }
    if (sel.startsWith("#")) {
      const id = sel.slice(1);
      if (this.id === id) return [this];
      const out = [];
      for (const c of this.children) out.push(...c.querySelectorAll(sel));
      return out;
    }
    // attribute selector e.g. .page-tab[data-tab="episodes"]
    const attrM = sel.match(/\.([a-zA-Z0-9_-]+)\[data-tab="([^"]+)"\]/);
    if (attrM) {
      return this.querySelectorAll("." + attrM[1]).filter(
        (el) => el.dataset.tab === attrM[2],
      );
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

  click() {
    for (const fn of this._listeners.click || []) fn({ target: this });
  }
}

const _byId = new Map();
const _createdScripts = [];

globalThis.document = {
  createElement(tag) {
    const el = new MockEl(tag);
    if (tag === "script") _createdScripts.push(el);
    return el;
  },
  getElementById(id) {
    return _byId.get(id) || null;
  },
  querySelector() {
    return null;
  },
  head: new MockEl("head"),
  body: new MockEl("body"),
};

globalThis.window = globalThis;

// ── Load page-tabs + memory tab with stubs ──

function loadPageTabs() {
  const path = resolve(STATIC, "shared/page-tabs.js");
  const source = readFileSync(path, "utf8");
  const url =
    "data:text/javascript;base64," + Buffer.from(source, "utf8").toString("base64");
  return import(url + "#page-tabs-" + Math.random());
}

async function loadMemoryModule({ apiImpl } = {}) {
  const pageTabs = await loadPageTabs();
  const path = resolve(STATIC, "pages/anima-tabs/memory.js");
  let source = readFileSync(path, "utf8");
  source = source.replace(/^import\s+.+;?\s*$/gm, "");

  const apiCalls = [];
  globalThis.__memoryApiCalls = apiCalls;
  globalThis.__memoryApi =
    apiImpl ||
    (async (endpoint) => {
      apiCalls.push(endpoint);
      if (endpoint.endsWith("/memory/stats")) {
        return { episodes: 2, knowledge: 1, procedures: 0 };
      }
      if (endpoint.includes("/episodes") && !endpoint.includes("calendar")) {
        return { files: ["2026-07-01", "2026-07-02"] };
      }
      return { files: [] };
    });

  const preamble = `
    const createPageTabs = ${pageTabs.createPageTabs.toString()};
    function _escapeAttr(str) {
      return String(str ?? "")
        .replace(/&/g, "&amp;")
        .replace(/"/g, "&quot;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
    }
    // rebind createPageTabs's internal _escapeAttr via eval of module body —
    // page-tabs is already a real function with its own _escapeAttr closure from data URL.
    const api = (...args) => globalThis.__memoryApi(...args);
    const escapeAttr = (s) => String(s ?? "").replace(/&/g,"&amp;").replace(/"/g,"&quot;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
    const escapeHtml = escapeAttr;
    const renderMarkdown = (s) => "<p>" + String(s ?? "") + "</p>";
    const renderSafeMarkdown = renderMarkdown;
    const basePath = "";
    const getLocale = () => "en";
    const t = (k) => k;
  `;

  // Use the already-imported createPageTabs from pageTabs module instead of toString
  const preamble2 = `
    const createPageTabs = globalThis.__createPageTabs;
    const api = (...args) => globalThis.__memoryApi(...args);
    const escapeAttr = (s) => String(s ?? "").replace(/&/g,"&amp;").replace(/"/g,"&quot;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
    const escapeHtml = escapeAttr;
    const renderMarkdown = (s) => "<p>" + String(s ?? "") + "</p>";
    const renderSafeMarkdown = renderMarkdown;
    const basePath = "";
    const getLocale = () => "en";
    const t = (k) => k;
  `;

  globalThis.__createPageTabs = pageTabs.createPageTabs;

  const url =
    "data:text/javascript;base64," +
    Buffer.from(preamble2 + "\n" + source, "utf8").toString("base64");
  return import(url + "#memory-" + Math.random());
}

describe("memory tab module (source contract)", () => {
  const source = readFileSync(
    resolve(STATIC, "pages/anima-tabs/memory.js"),
    "utf8",
  );

  it("exports render/destroy and accepts animaName", () => {
    assert.match(source, /export function render\(/);
    assert.match(source, /export function destroy\(/);
    assert.match(source, /animaName/);
  });

  it("does not render standalone Anima select dropdown", () => {
    assert.doesNotMatch(source, /memoryAnimaSelect/);
    assert.doesNotMatch(source, /memory-page-selector/);
  });

  it("uses createPageTabs for sub-tabs and keeps all five kinds", () => {
    assert.match(source, /createPageTabs/);
    assert.match(source, /episodes/);
    assert.match(source, /knowledge/);
    assert.match(source, /procedures/);
    assert.match(source, /graph/);
    assert.match(source, /calendar/);
  });

  it("destroy stops graph simulation and clears state", () => {
    assert.match(source, /_stopGraphSimulation/);
    assert.match(source, /_viewRequestId\s*\+=\s*1/);
    assert.match(source, /_subTabs/);
  });
});

describe("memory tab render/API/destroy", () => {
  beforeEach(() => {
    _byId.clear();
    globalThis.__memoryApiCalls = [];
  });

  it("renders five sub-tabs via createPageTabs", async () => {
    const mod = await loadMemoryModule();
    const container = new MockEl("div");
    // Register host id when render sets innerHTML
    const origSet = Object.getOwnPropertyDescriptor(
      Object.getPrototypeOf(container),
      "innerHTML",
    )?.set;

    mod.render(container, { animaName: "sakura" });

    // createPageTabs appends a .page-tabs element as a child of #memorySubTabsHost
    const host = container.querySelector("#memorySubTabsHost") || container.children.find((c) => c.id === "memorySubTabsHost");
    assert.ok(host, "memorySubTabsHost should exist");

    // page-tabs is appended to host
    const tabBar = host.children.find((c) => c.className.includes("page-tabs"));
    assert.ok(tabBar, "page-tabs bar should be created");
    assert.ok(tabBar.className.includes("memory-page-tabs"));

    const tabs = tabBar.querySelectorAll(".page-tab");
    assert.equal(tabs.length, 5);
    const ids = tabs.map((b) => b.dataset.tab);
    assert.deepEqual(ids, ["episodes", "knowledge", "procedures", "graph", "calendar"]);

    mod.destroy();
  });

  it("calls memory APIs with the provided animaName", async () => {
    const calls = [];
    const mod = await loadMemoryModule({
      apiImpl: async (endpoint) => {
        calls.push(endpoint);
        if (endpoint.includes("/memory/stats")) {
          return { episodes: 3, knowledge: 1, procedures: 2 };
        }
        if (endpoint.endsWith("/episodes")) {
          return { files: ["a.md"] };
        }
        return { files: [] };
      },
    });

    const container = new MockEl("div");
    // Provide memoryMainContent for loaders
    const content = new MockEl("div");
    content.id = "memoryMainContent";
    _byId.set("memoryMainContent", content);

    mod.render(container, { animaName: "yuki" });

    // Allow microtasks from async loaders
    await new Promise((r) => setTimeout(r, 30));

    assert.ok(
      calls.some((c) => c.includes("/api/animas/yuki/")),
      `expected yuki API calls, got: ${JSON.stringify(calls)}`,
    );
    assert.ok(
      calls.some((c) => c === "/api/animas/yuki/memory/stats" || c === "/api/animas/yuki/episodes"),
      `expected stats or episodes for yuki, got: ${JSON.stringify(calls)}`,
    );

    mod.destroy();
  });

  it("destroy invalidates in-flight requests and clears container ref", async () => {
    let resolveApi;
    const pending = new Promise((r) => {
      resolveApi = r;
    });

    const mod = await loadMemoryModule({
      apiImpl: async (endpoint) => {
        if (endpoint.includes("/memory/stats")) {
          return { episodes: 0, knowledge: 0, procedures: 0 };
        }
        await pending;
        return { files: ["late.md"] };
      },
    });

    const container = new MockEl("div");
    const content = new MockEl("div");
    content.id = "memoryMainContent";
    _byId.set("memoryMainContent", content);

    mod.render(container, { animaName: "sakura" });
    mod.destroy();

    // Resolve late; content should not be rewritten by stale request
    const before = content.innerHTML;
    resolveApi();
    await new Promise((r) => setTimeout(r, 20));
    // After destroy, _container is null so stale responses bail out
    // (innerHTML may stay as loading or empty — must not show late.md)
    assert.doesNotMatch(content.innerHTML, /late\.md/);

    // Second render still works after destroy
    const container2 = new MockEl("div");
    const content2 = new MockEl("div");
    content2.id = "memoryMainContent";
    _byId.set("memoryMainContent", content2);
    mod.render(container2, { animaName: "sakura" });
    assert.ok(container2.innerHTML.includes("memorySubTabsHost") || container2.children.length > 0);
    mod.destroy();
  });
});

describe("animas page wires memory tab", () => {
  it("includes memory in _DETAIL_TABS and exports buildAnimaDetailHash", () => {
    const source = readFileSync(resolve(STATIC, "pages/animas.js"), "utf8");
    assert.match(source, /id:\s*"memory"/);
    assert.match(source, /animas\.tab_memory/);
    assert.match(source, /export function buildAnimaDetailHash/);
    assert.match(source, /animasSwitcher/);
    assert.match(source, /fetchAnimasList/);
  });
});
