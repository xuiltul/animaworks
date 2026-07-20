/**
 * Unit tests for router redirect table and anima subPath parsing.
 *
 * Run with: node --test tests/unit/frontend/test_router_redirects.mjs
 *
 * router.js imports i18n via an absolute path ("/shared/..."), which Node
 * cannot resolve, so we load the module source with that import stubbed out.
 */

import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROUTER_PATH = resolve(
  __dirname,
  "../../../server/static/modules/router.js",
);

const source = readFileSync(ROUTER_PATH, "utf8")
  .replace(
    /^import\s+\{\s*applyTranslations,\s*t\s*\}\s+from\s+["'][^"']+["'];?\s*$/m,
    `export function applyTranslations() {}\nexport function t(k) { return k; }`,
  );

const moduleUrl =
  "data:text/javascript;base64," + Buffer.from(source, "utf8").toString("base64");

const {
  REDIRECTS,
  resolveRedirect,
  resolveRouteMatch,
  parseAnimaSubPath,
} = await import(moduleUrl);

describe("REDIRECTS table", () => {
  it("maps /processes to #/animas", () => {
    assert.equal(REDIRECTS["/processes"], "#/animas");
  });

  it("maps /server to #/ (dashboard)", () => {
    assert.equal(REDIRECTS["/server"], "#/");
  });

  it("maps /setup to #/settings (legacy)", () => {
    assert.equal(REDIRECTS["/setup"], "#/settings");
  });

  it("maps /memory to #/animas", () => {
    assert.equal(REDIRECTS["/memory"], "#/animas");
  });

  it("maps /assets to #/animas", () => {
    assert.equal(REDIRECTS["/assets"], "#/animas");
  });
});

describe("resolveRedirect", () => {
  it("redirects #/processes path to #/animas", () => {
    assert.equal(resolveRedirect("/processes"), "#/animas");
  });

  it("redirects nested /processes/* paths to #/animas", () => {
    assert.equal(resolveRedirect("/processes/anything"), "#/animas");
  });

  it("redirects /server and nested paths to #/", () => {
    assert.equal(resolveRedirect("/server"), "#/");
    assert.equal(resolveRedirect("/server/anything"), "#/");
  });

  it("redirects /setup and nested paths to #/settings", () => {
    assert.equal(resolveRedirect("/setup"), "#/settings");
    assert.equal(resolveRedirect("/setup/foo"), "#/settings");
  });

  it("redirects /memory and nested paths to #/animas", () => {
    assert.equal(resolveRedirect("/memory"), "#/animas");
    assert.equal(resolveRedirect("/memory/anything"), "#/animas");
  });

  it("redirects /assets and nested paths to #/animas", () => {
    assert.equal(resolveRedirect("/assets"), "#/animas");
    assert.equal(resolveRedirect("/assets/anything"), "#/animas");
  });

  it("returns null for non-redirect paths", () => {
    assert.equal(resolveRedirect("/animas"), null);
    assert.equal(resolveRedirect("/chat"), null);
    assert.equal(resolveRedirect("/animas/sakura/process"), null);
  });
});

describe("resolveRouteMatch (subPath decomposition)", () => {
  const keys = ["/", "/chat", "/animas", "/settings", "/logs"];

  it("exact match yields empty subPath", () => {
    const m = resolveRouteMatch("/animas", keys);
    assert.deepEqual(m, { route: "/animas", subPath: "", navPath: "/animas" });
  });

  it("decomposes #/animas/<name> into subPath name", () => {
    const m = resolveRouteMatch("/animas/sakura", keys);
    assert.equal(m.route, "/animas");
    assert.equal(m.subPath, "sakura");
    assert.equal(m.navPath, "/animas");
  });

  it("decomposes #/animas/<name>/<tab> keeping slash in subPath", () => {
    const m = resolveRouteMatch("/animas/sakura/process", keys);
    assert.equal(m.route, "/animas");
    assert.equal(m.subPath, "sakura/process");
    assert.equal(m.navPath, "/animas");
  });

  it("decodes URI-encoded name segments", () => {
    const m = resolveRouteMatch("/animas/" + encodeURIComponent("さくら") + "/overview", keys);
    assert.equal(m.subPath, "さくら/overview");
  });

  it("picks longest matching prefix", () => {
    const keys2 = ["/a", "/a/b", "/animas"];
    const m = resolveRouteMatch("/a/b/c", keys2);
    assert.equal(m.route, "/a/b");
    assert.equal(m.subPath, "c");
  });

  it("returns null when no route matches", () => {
    assert.equal(resolveRouteMatch("/unknown", keys), null);
  });
});

describe("parseAnimaSubPath", () => {
  it("returns null name and overview for empty subPath", () => {
    assert.deepEqual(parseAnimaSubPath(""), { name: null, tab: "overview" });
    assert.deepEqual(parseAnimaSubPath(null), { name: null, tab: "overview" });
    assert.deepEqual(parseAnimaSubPath(undefined), { name: null, tab: "overview" });
  });

  it("defaults tab to overview when only name is present", () => {
    assert.deepEqual(parseAnimaSubPath("sakura"), {
      name: "sakura",
      tab: "overview",
    });
  });

  it("parses name and tab from two segments", () => {
    assert.deepEqual(parseAnimaSubPath("sakura/process"), {
      name: "sakura",
      tab: "process",
    });
    assert.deepEqual(parseAnimaSubPath("sakura/overview"), {
      name: "sakura",
      tab: "overview",
    });
  });
});
