// ── Anima detail tab: Schedule ──────────────
// Per-anima scheduler jobs (absorbed from pages/server-page.js).

import { api } from "../../modules/api.js";
import { escapeHtml, timeStr } from "../../modules/state.js";
import { t } from "/shared/i18n.js";

let _refreshInterval = null;
let _animaName = null;

/**
 * Normalize scheduler API payload into a flat job list.
 * Accepts either `{ jobs: [...] }` or `{ system_jobs, anima_jobs }`.
 * @param {object|null|undefined} data
 * @returns {object[]}
 */
export function extractSchedulerJobs(data) {
  if (!data || typeof data !== "object") return [];
  if (Array.isArray(data.jobs)) return data.jobs;
  return [
    ...(Array.isArray(data.system_jobs) ? data.system_jobs : []),
    ...(Array.isArray(data.anima_jobs) ? data.anima_jobs : []),
  ];
}

/**
 * Keep jobs belonging to the given anima only.
 * @param {object[]} jobs
 * @param {string} animaName
 * @returns {object[]}
 */
export function filterJobsForAnima(jobs, animaName) {
  if (!Array.isArray(jobs) || !animaName) return [];
  return jobs.filter((j) => j && j.anima === animaName);
}

/**
 * Build HTML for the jobs table (or empty placeholder).
 * @param {object[]} jobs
 * @returns {string}
 */
export function jobsTableHtml(jobs) {
  if (!jobs.length) {
    return `<div class="loading-placeholder">${t("server.no_jobs")}</div>`;
  }
  return `
    <table class="data-table">
      <thead>
        <tr>
          <th>${t("server.job_name")}</th>
          <th>${t("server.job_schedule")}</th>
          <th>${t("server.job_last_run")}</th>
          <th>${t("server.job_next_run")}</th>
        </tr>
      </thead>
      <tbody>
        ${jobs
          .map(
            (j) => `
          <tr>
            <td style="font-weight:500;">${escapeHtml(j.name || j.id || "--")}</td>
            <td><code>${escapeHtml(j.schedule || j.trigger || "--")}</code></td>
            <td>${escapeHtml(j.last_run ? timeStr(j.last_run) : "--")}</td>
            <td>${escapeHtml(j.next_run ? timeStr(j.next_run) : "--")}</td>
          </tr>
        `,
          )
          .join("")}
      </tbody>
    </table>
  `;
}

/**
 * @param {HTMLElement} container
 * @param {{ animaName: string }} opts
 */
export function render(container, { animaName } = {}) {
  _animaName = animaName;

  container.innerHTML = `
    <div id="animaScheduleTabContent">
      <div class="loading-placeholder">${t("common.loading")}</div>
    </div>
  `;

  _load();
  _refreshInterval = setInterval(_load, 30000);
}

export function destroy() {
  if (_refreshInterval) {
    clearInterval(_refreshInterval);
    _refreshInterval = null;
  }
  _animaName = null;
}

async function _load() {
  const content = document.getElementById("animaScheduleTabContent");
  if (!content || !_animaName) return;

  try {
    const data = await api("/api/system/scheduler");
    const jobs = filterJobsForAnima(extractSchedulerJobs(data), _animaName);
    content.innerHTML = jobsTableHtml(jobs);
  } catch (err) {
    content.innerHTML = `<div class="loading-placeholder">${t("server.api_unimplemented")}: ${escapeHtml(err.message)}</div>`;
  }
}
