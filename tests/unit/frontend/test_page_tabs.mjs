/**
 * Unit tests for server/static/shared/page-tabs.js
 *
 * Run with: node --test tests/unit/frontend/test_page_tabs.mjs
 */

import { describe, it, beforeEach } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const __dirname = dirname(fileURLToPath(import.meta.url));

// ── Minimal DOM mock ─────────────────────────

class MockElement {
  constructor(tag = "div") {
    this.tagName = tag.toUpperCase();
    this.className = "";
    this.children = [];
    this.attributes = {};
    this.dataset = {};
    this._listeners = {};
    this.parentNode = null;
    this._innerHTML = "";
  }

  set innerHTML(html) {
    this._innerHTML = String(html ?? "");
    this.children = [];
    // Parse simple <button ... data-tab="x" class="...">label</button> stubs
    const re =
      /<button\b([^>]*)>([\s\S]*?)<\/button>/gi;
    let m;
    while ((m = re.exec(this._innerHTML)) !== null) {
      const attrs = m[1];
      const btn = new MockElement("button");
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
  }

  get innerHTML() {
    return this._innerHTML;
  }

  setAttribute(name, value) {
    this.attributes[name] = String(value);
    if (name === "data-tab") this.dataset.tab = String(value);
  }

  getAttribute(name) {
    return this.attributes[name] ?? null;
  }

  appendChild(child) {
    child.parentNode = this;
    this.children.push(child);
    return child;
  }

  remove() {
    if (this.parentNode) {
      const idx = this.parentNode.children.indexOf(this);
      if (idx >= 0) this.parentNode.children.splice(idx, 1);
      this.parentNode = null;
    }
  }

  querySelectorAll(selector) {
    if (selector === ".page-tab") {
      return this._allDescendants().filter(
        (el) =>
          el.className.split(/\s+/).includes("page-tab") ||
          el.tagName === "BUTTON",
      );
    }
    return [];
  }

  querySelector(selector) {
    return this.querySelectorAll(selector)[0] || null;
  }

  addEventListener(type, fn) {
    if (!this._listeners[type]) this._listeners[type] = [];
    this._listeners[type].push(fn);
  }

  click() {
    for (const fn of this._listeners.click || []) fn({ target: this });
  }

  classList = {
    _el: null,
    toggle(cls, force) {
      const el = this._el;
      const parts = new Set(el.className.split(/\s+/).filter(Boolean));
      if (force === true) parts.add(cls);
      else if (force === false) parts.delete(cls);
      else if (parts.has(cls)) parts.delete(cls);
      else parts.add(cls);
      el.className = [...parts].join(" ");
    },
    contains(cls) {
      return this._el.className.split(/\s+/).includes(cls);
    },
  };

  _allDescendants() {
    const out = [];
    const walk = (node) => {
      for (const c of node.children) {
        out.push(c);
        walk(c);
      }
    };
    walk(this);
    return out;
  }
}

// Bind classList to element
const _orig = MockElement;
function makeEl(tag) {
  const el = new MockElement(tag);
  el.classList._el = el;
  // Also ensure child buttons get bound classList
  const origInner = Object.getOwnPropertyDescriptor(MockElement.prototype, "innerHTML");
  return el;
}

// Patch children created via innerHTML to have working classList
const _setInner = Object.getOwnPropertyDescriptor(MockElement.prototype, "innerHTML").set;
Object.defineProperty(MockElement.prototype, "innerHTML", {
  set(html) {
    _setInner.call(this, html);
    for (const c of this.children) {
      c.classList = {
        _el: c,
        toggle(cls, force) {
          const parts = new Set(c.className.split(/\s+/).filter(Boolean));
          if (force === true) parts.add(cls);
          else if (force === false) parts.delete(cls);
          else if (parts.has(cls)) parts.delete(cls);
          else parts.add(cls);
          c.className = [...parts].join(" ");
        },
        contains(cls) {
          return c.className.split(/\s+/).includes(cls);
        },
      };
    }
  },
  get() {
    return this._innerHTML;
  },
});

globalThis.document = {
  createElement(tag) {
    const el = new MockElement(tag);
    el.classList._el = el;
    return el;
  },
};

// Import module under test
const { createPageTabs } = await import(
  resolve(__dirname, "../../../server/static/shared/page-tabs.js")
);

describe("createPageTabs", () => {
  let container;

  beforeEach(() => {
    container = new MockElement("div");
    container.classList._el = container;
  });

  it("renders a button for each tab with page-tabs / page-tab classes", () => {
    const tabs = createPageTabs({
      tabs: [
        { id: "overview", label: "Overview" },
        { id: "process", label: "Process" },
      ],
      container,
      activeId: "overview",
    });

    assert.equal(tabs.el.className, "page-tabs");
    assert.equal(container.children.includes(tabs.el), true);

    const buttons = tabs.el.querySelectorAll(".page-tab");
    assert.equal(buttons.length, 2);
    assert.equal(buttons[0].dataset.tab, "overview");
    assert.equal(buttons[1].dataset.tab, "process");
    assert.ok(buttons[0].className.includes("active"));
    assert.ok(!buttons[1].className.includes("active"));
    assert.equal(tabs.getActive(), "overview");
  });

  it("defaults activeId to the first tab when omitted", () => {
    const tabs = createPageTabs({
      tabs: [
        { id: "a", label: "A" },
        { id: "b", label: "B" },
      ],
      container,
    });
    assert.equal(tabs.getActive(), "a");
  });

  it("switches active class and fires onChange on click", () => {
    const changes = [];
    const tabs = createPageTabs({
      tabs: [
        { id: "overview", label: "Overview" },
        { id: "process", label: "Process" },
      ],
      container,
      activeId: "overview",
      onChange: (id) => changes.push(id),
    });

    const buttons = tabs.el.querySelectorAll(".page-tab");
    buttons[1].click();

    assert.equal(tabs.getActive(), "process");
    assert.ok(buttons[1].className.includes("active"));
    assert.ok(!buttons[0].className.includes("active"));
    assert.deepEqual(changes, ["process"]);
  });

  it("setActive updates active state without firing onChange", () => {
    const changes = [];
    const tabs = createPageTabs({
      tabs: [
        { id: "overview", label: "Overview" },
        { id: "process", label: "Process" },
      ],
      container,
      onChange: (id) => changes.push(id),
    });

    tabs.setActive("process");
    assert.equal(tabs.getActive(), "process");
    assert.deepEqual(changes, []);
  });

  it("does not fire onChange when clicking the already-active tab", () => {
    const changes = [];
    const tabs = createPageTabs({
      tabs: [
        { id: "overview", label: "Overview" },
        { id: "process", label: "Process" },
      ],
      container,
      activeId: "overview",
      onChange: (id) => changes.push(id),
    });

    tabs.el.querySelectorAll(".page-tab")[0].click();
    assert.deepEqual(changes, []);
  });

  it("destroy removes the tab bar from the container", () => {
    const tabs = createPageTabs({
      tabs: [{ id: "overview", label: "Overview" }],
      container,
    });
    assert.equal(container.children.length, 1);
    tabs.destroy();
    assert.equal(container.children.length, 0);
  });

  it("throws when tabs is empty", () => {
    assert.throws(
      () => createPageTabs({ tabs: [], container }),
      /non-empty/,
    );
  });
});
