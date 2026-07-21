/**
 * Unit tests for dashboard redesign helpers in server/static/pages/home.js
 *
 * Run with: node --test tests/unit/frontend/test_home_dashboard.mjs
 */

import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const __dirname = dirname(fileURLToPath(import.meta.url));
const STATIC = resolve(__dirname, "../../../server/static");
const HOME_PATH = resolve(STATIC, "pages/home.js");
const I18N_DIR = resolve(STATIC, "i18n");

function loadHomeHelpers() {
  const path = HOME_PATH;
  let source = readFileSync(path, "utf8");
  source = source.replace(/^import\s+.+;?\s*$/gm, "");

  const preamble = `
    const t = (k, params = {}) => {
      // Simulate i18n: keep key, apply {param} replacements when present in template,
      // otherwise append values so callers can assert counts appear in labels.
      let text = String(k);
      const entries = Object.entries(params || {});
      if (entries.length === 0) return text;
      let replaced = false;
      for (const [pk, pv] of entries) {
        const token = "{" + pk + "}";
        if (text.includes(token)) {
          text = text.replaceAll(token, String(pv));
          replaced = true;
        }
      }
      if (!replaced) {
        text = text + " " + entries.map(([, pv]) => String(pv)).join(" ");
      }
      return text;
    };
    const basePath = "";
    const api = async () => ({});
    const escapeHtml = (s) =>
      String(s ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
    const escapeAttr = escapeHtml;
    const timeStr = (v) => (v ? "12:00" : "--:--");
    const statusClass = () => "";
    const animaHashColor = () => "#000";
    const companyColor = () => "#000";
    const getIcon = () => "";
    const getDisplaySummary = () => "";
    const bustupCandidates = () => [];
    const resolveCachedAvatar = async () => null;
  `;

  const url =
    "data:text/javascript;base64," +
    Buffer.from(preamble + "\n" + source, "utf8").toString("base64");
  return import(url + "#" + Math.random());
}

function loadI18n(locale) {
  return JSON.parse(readFileSync(resolve(I18N_DIR, `${locale}.json`), "utf8"));
}

describe("attentionSummaryChips", () => {
  it("builds blocked/pending/in_progress/external chips with counts", async () => {
    const home = await loadHomeHelpers();
    const chips = home.attentionSummaryChips(
      {
        pending: 2,
        in_progress: 5,
        blocked: 23,
        delegated: 0,
        failed_review: 0,
        snoozed: 0,
        suppressed: 5277,
        total_active: 23,
      },
      7,
    );
    assert.equal(chips.length, 4);
    assert.equal(chips[0].key, "blocked");
    assert.equal(chips[0].count, 23);
    assert.equal(chips[0].emphasis, true);
    assert.equal(chips[0].href, "#/task-board");
    assert.match(chips[0].label, /23/);

    assert.equal(chips[1].key, "pending");
    assert.equal(chips[1].count, 2);
    assert.equal(chips[1].emphasis, false);

    assert.equal(chips[2].key, "in_progress");
    assert.equal(chips[2].count, 5);

    assert.equal(chips[3].key, "external");
    assert.equal(chips[3].count, 7);
    assert.equal(chips[3].href, "#homeExternalTasksCard");
  });

  it("treats missing summary as zeros and no blocked emphasis", async () => {
    const home = await loadHomeHelpers();
    const chips = home.attentionSummaryChips(null, 0);
    assert.equal(chips.every((c) => c.count === 0), true);
    assert.equal(chips[0].emphasis, false);
  });

  it("attentionChipsHtml marks blocked chips as danger and escapes labels", async () => {
    const home = await loadHomeHelpers();
    const html = home.attentionChipsHtml([
      {
        key: "blocked",
        label: "Blocked <x>",
        count: 1,
        href: "#/task-board",
        emphasis: true,
      },
      {
        key: "external",
        label: "External 0",
        count: 0,
        href: "#homeExternalTasksCard",
        emphasis: false,
      },
    ]);
    assert.match(html, /home-chip--danger/);
    assert.match(html, /Blocked &lt;x&gt;/);
    assert.match(html, /data-attention="blocked"/);
    assert.match(html, /href="#\/task-board"/);
  });
});

describe("systemStatusBarModel", () => {
  it("marks bar ok only when reachable and scheduler running", async () => {
    const home = await loadHomeHelpers();
    const ok = home.systemStatusBarModel({
      reachable: true,
      schedulerRunning: true,
      animaCount: 13,
      processCount: 13,
      jobCount: 97,
      wsCount: 3,
    });
    assert.equal(ok.ok, true);
    assert.match(ok.animaLabel, /13/);
    assert.match(ok.processLabel, /13/);
    assert.match(ok.jobLabel, /97/);
    assert.match(ok.wsLabel, /3/);

    const bad = home.systemStatusBarModel({
      reachable: false,
      schedulerRunning: false,
      animaCount: 0,
      processCount: 0,
      jobCount: 0,
      wsCount: 0,
    });
    assert.equal(bad.ok, false);
  });
});

describe("home.js render template structure", () => {
  it("places org hero first, includes attention summary, drops quick links", () => {
    const source = readFileSync(HOME_PATH, "utf8");
    // Extract render template body
    const m = source.match(
      /export function render\(container\)\s*\{[\s\S]*?container\.innerHTML\s*=\s*`([\s\S]*?)`;/,
    );
    assert.ok(m, "render() template literal found");
    const html = m[1];

    // Org tree must appear before usage panel and activity
    const orgIdx = html.indexOf('id="homeOrgTree"');
    const usageIdx = html.indexOf('id="homeUsagePanel"');
    const activityIdx = html.indexOf('id="homeActivityTimeline"');
    const attentionIdx = html.indexOf('id="homeAttentionChips"');
    const extIdx = html.indexOf('id="homeExternalTasksCard"');

    assert.ok(orgIdx >= 0, "homeOrgTree present");
    assert.ok(attentionIdx >= 0, "homeAttentionChips present");
    assert.ok(usageIdx >= 0, "homeUsagePanel present");
    assert.ok(activityIdx >= 0, "homeActivityTimeline present");
    assert.ok(extIdx >= 0, "homeExternalTasksCard present");

    assert.ok(orgIdx < attentionIdx, "org chart before attention");
    assert.ok(attentionIdx < usageIdx, "attention before usage");
    assert.ok(usageIdx < activityIdx, "usage before activity");
    assert.ok(activityIdx < extIdx, "activity before external tasks");

    // First card-like block should be org hero
    const firstCard = html.search(/class="card[^"]*"/);
    assert.ok(firstCard >= 0);
    const firstCardSlice = html.slice(firstCard, firstCard + 80);
    assert.match(firstCardSlice, /home-org-hero/);

    // Quick links removed
    assert.doesNotMatch(html, /home\.quick_links/);
    assert.doesNotMatch(html, /home\.link_workspace/);
  });

  it("exports pure helpers used by the redesign", async () => {
    const home = await loadHomeHelpers();
    assert.equal(typeof home.attentionSummaryChips, "function");
    assert.equal(typeof home.attentionChipsHtml, "function");
    assert.equal(typeof home.systemStatusBarModel, "function");
    assert.equal(typeof home.summarizeServerStatus, "function");
    assert.equal(typeof home.serverStatusTableHtml, "function");
    assert.equal(typeof home.extractSchedulerJobs, "function");
  });
});

describe("i18n keys for dashboard redesign", () => {
  const required = [
    "home.attention_blocked",
    "home.attention_external",
    "home.attention_in_progress",
    "home.attention_pending",
    "home.attention_title",
    "home.status_bar_animas",
    "home.status_bar_connections",
    "home.status_bar_jobs",
    "home.status_bar_processes",
  ];

  it("ja/en/ko are valid JSON and include new attention/status keys", () => {
    const ja = loadI18n("ja");
    const en = loadI18n("en");
    const ko = loadI18n("ko");
    for (const k of required) {
      assert.ok(typeof ja[k] === "string" && ja[k].length > 0, `ja missing ${k}`);
      assert.ok(typeof en[k] === "string" && en[k].length > 0, `en missing ${k}`);
      assert.ok(typeof ko[k] === "string" && ko[k].length > 0, `ko missing ${k}`);
    }
    // New home.* keys must be present in all three locales with identical names
    const homeKeys = (obj) => Object.keys(obj).filter((k) => k.startsWith("home.attention_") || k.startsWith("home.status_bar_")).sort();
    assert.deepEqual(homeKeys(en), homeKeys(ja));
    assert.deepEqual(homeKeys(ko), homeKeys(ja));
  });
});
