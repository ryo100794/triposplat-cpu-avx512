"use strict";

const state = { overview: null, experiments: [], artifacts: [], documents: [], activity: [], filter: "all", selected: null };

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const escapeHtml = (value) => String(value ?? "").replace(/[&<>'"]/g, (char) => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[char]));

function bytes(value) {
  if (value === null || value === undefined) return "--";
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  let amount = Number(value); let unit = 0;
  while (amount >= 1024 && unit < units.length - 1) { amount /= 1024; unit += 1; }
  return `${amount >= 10 || unit === 0 ? amount.toFixed(0) : amount.toFixed(2)} ${units[unit]}`;
}

function duration(value) {
  if (value === null || value === undefined) return "--";
  const seconds = Number(value);
  if (seconds < 60) return `${seconds.toFixed(1)} 秒`;
  const minutes = Math.floor(seconds / 60);
  const rest = Math.round(seconds % 60);
  return `${minutes}分 ${String(rest).padStart(2, "0")}秒`;
}

function dateTime(value) {
  if (!value) return "--";
  return new Intl.DateTimeFormat("ja-JP", { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", timeZone: "UTC" }).format(new Date(value)) + " UTC";
}

function metricValue(metric) {
  if (!metric) return "--";
  if (metric.unit === "percent") return `${metric.value.toFixed(2)}%`;
  if (metric.unit === "seconds") return duration(metric.value);
  if (metric.unit === "bytes") return bytes(metric.value);
  if (metric.unit === "rmse") return Number(metric.value).toExponential(2);
  return `${metric.value}${metric.unit ? ` ${metric.unit}` : ""}`;
}

async function request(path) {
  const response = await fetch(path, { headers: { Accept: "application/json" }, cache: "no-store" });
  if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
  return response.json();
}

function setConnection(ok) {
  $("#connection-dot").className = `connection-dot ${ok ? "ok" : "error"}`;
  $("#connection-label").textContent = ok ? "PostgreSQL 接続中" : "データベース切断";
}

function toast(message) {
  const node = $("#toast"); node.textContent = message; node.classList.add("show");
  window.setTimeout(() => node.classList.remove("show"), 2200);
}

function renderOverview() {
  const data = state.overview; const project = data.project || {};
  $("#project-title").textContent = project.title || "TripoSplat CPU Lab";
  $("#project-phase").textContent = project.phase || "PROJECT";
  $("#project-status").textContent = (project.status || "unknown").toUpperCase();
  $("#project-objective").textContent = project.objective || "--";
  $("#project-progress-label").textContent = `${project.progress || 0}% 完了`;
  $("#project-progress-value").textContent = `${project.progress || 0}%`;
  $("#project-progress-ring").style.setProperty("--progress", `${(project.progress || 0) * 3.6}deg`);
  $("#updated-at").textContent = `更新 ${dateTime(project.updated_at)}`;
  $("#metric-experiments").textContent = data.counts?.experiments ?? "--";
  $("#metric-passed").textContent = `${data.counts?.passed ?? 0} 件 採用`;
  $("#metric-baseline").textContent = duration(data.settings?.baseline_s20_seconds || 2747.27);
  $("#metric-speedup").textContent = `${Number(data.settings?.best_process_speedup_percent || 3.2).toFixed(2)}%`;
  $("#metric-memory").textContent = bytes(data.settings?.candidate_peak_rss_bytes || 4254904320);

  const milestones = data.milestones || [];
  $("#milestone-count").textContent = `${milestones.filter((item) => item.status === "complete").length} / ${milestones.length} 完了`;
  $("#pipeline-track").innerHTML = milestones.map((item) => `
    <article class="pipeline-item ${escapeHtml(item.status)}">
      <strong>${escapeHtml(item.title)}</strong><span>${escapeHtml(item.stage)} · ${item.progress}%</span>
      <div class="mini-progress"><i style="width:${Math.max(0, Math.min(100, item.progress))}%"></i></div>
    </article>`).join("");

  const chart = data.settings?.benchmark_chart || [];
  $("#benchmark-chart").innerHTML = chart.map((item) => `
    <div class="benchmark-row ${escapeHtml(item.tone || "")}">
      <div class="benchmark-label"><strong>${escapeHtml(item.label)}</strong><span>${escapeHtml(item.detail)}</span></div>
      <div class="bar-track"><div class="bar-value" style="width:${Math.max(2, Math.min(100, item.bar_percent))}%"></div></div>
      <span class="benchmark-delta">${escapeHtml(item.delta)}</span>
    </div>`).join("");

  const gates = data.settings?.admission_gates || [];
  $("#gate-list").innerHTML = gates.map((item, index) => `
    <div class="gate-item ${item.done ? "done" : ""}"><span class="gate-marker">${item.done ? "✓" : index + 1}</span>
    <div><strong>${escapeHtml(item.title)}</strong><span>${escapeHtml(item.detail)}</span></div></div>`).join("");

  const system = data.system || {};
  $("#system-grid").innerHTML = [
    ["HOST", system.host || "--"], ["CPU", system.cpu_model || "--"],
    ["利用可能メモリ", bytes(system.memory_available_bytes)],
    ["Workspace空き", bytes(system.workspace_available_bytes)]
  ].map(([label, value]) => `<div class="system-cell"><span>${escapeHtml(label)}</span><strong title="${escapeHtml(value)}">${escapeHtml(value)}</strong></div>`).join("");
}

function renderExperimentDetail(experiment) {
  if (!experiment) { $("#experiment-detail").innerHTML = '<p class="empty-state">実験を選択してください</p>'; return; }
  $("#experiment-detail").innerHTML = `
    <span class="state-chip ${escapeHtml(experiment.status)}">${escapeHtml(experiment.status)}</span>
    <h3>${escapeHtml(experiment.title)}</h3><p class="variant">${escapeHtml(experiment.variant)} · ${escapeHtml(experiment.hardware)}</p>
    <div class="detail-metrics">${(experiment.metrics || []).map((metric) => `<div class="detail-metric"><span>${escapeHtml(metric.label)}</span><strong>${escapeHtml(metricValue(metric))}</strong></div>`).join("")}</div>
    <p class="detail-note">${escapeHtml(experiment.notes || "記録なし")}</p>`;
}

function renderExperiments() {
  const list = state.filter === "all" ? state.experiments : state.experiments.filter((item) => item.status === state.filter);
  $("#experiment-rows").innerHTML = list.map((item) => `
    <tr data-slug="${escapeHtml(item.slug)}" class="${state.selected === item.slug ? "selected" : ""}" tabindex="0">
      <td class="table-title"><strong>${escapeHtml(item.title)}</strong><small>${escapeHtml(item.variant)}</small></td>
      <td><span class="state-chip ${escapeHtml(item.status)}">${escapeHtml(item.status)}</span></td>
      <td>${item.steps ?? "--"}</td><td>${duration(item.duration_sec)}</td><td>${bytes(item.peak_rss_bytes)}</td>
      <td>${escapeHtml(item.quality_status)}</td></tr>`).join("");
  const select = (slug) => { state.selected = slug; renderExperiments(); renderExperimentDetail(state.experiments.find((item) => item.slug === slug)); };
  $$("#experiment-rows tr").forEach((row) => {
    row.addEventListener("click", () => select(row.dataset.slug));
    row.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") select(row.dataset.slug); });
  });
  if (!state.selected && list[0]) { state.selected = list[0].slug; renderExperimentDetail(list[0]); }
}

function renderArtifacts(query = "") {
  const term = query.trim().toLowerCase();
  const artifacts = state.artifacts.filter((item) => [item.title, item.kind, item.location, item.storage].join(" ").toLowerCase().includes(term));
  const documents = state.documents.filter((item) => [item.title, item.category, item.path, item.summary].join(" ").toLowerCase().includes(term));
  $("#artifact-count").textContent = `${artifacts.length} 件`;
  $("#document-count").textContent = `${documents.length} 件`;
  $("#artifact-list").innerHTML = artifacts.map((item) => `
    <article class="artifact-row"><div class="artifact-main"><strong>${escapeHtml(item.title)}</strong><small>${escapeHtml(item.kind)} · ${escapeHtml(item.storage)}</small></div>
    <span class="state-chip ${escapeHtml(item.state)}">${escapeHtml(item.state)}</span><span>${bytes(item.size_bytes)}</span>
    <span class="artifact-path">${escapeHtml(item.location)}</span></article>`).join("") || '<p class="empty-state">該当する成果物はありません</p>';
  $("#document-grid").innerHTML = documents.map((item) => `
    <article class="document-item"><span class="state-chip">${escapeHtml(item.category)}</span><strong>${escapeHtml(item.title)}</strong>
    <p>${escapeHtml(item.summary)}</p><span class="document-path">${escapeHtml(item.path)}</span></article>`).join("") || '<p class="empty-state">該当する文書はありません</p>';
}

function renderActivity() {
  $("#activity-timeline").innerHTML = state.activity.map((item) => `
    <article class="timeline-item"><span class="timeline-marker"></span><div class="timeline-content"><strong>${escapeHtml(item.title)}</strong><p>${escapeHtml(item.detail)}</p></div><time class="timeline-time">${dateTime(item.occurred_at)}</time></article>`).join("");
}

async function loadData(showToast = false) {
  const button = $("#refresh-button"); button.classList.add("loading");
  try {
    const [overview, experiments, artifacts, activity] = await Promise.all([
      request("/api/overview"), request("/api/experiments"), request("/api/artifacts"), request("/api/activity")
    ]);
    state.overview = overview; state.experiments = experiments.experiments; state.artifacts = artifacts.artifacts;
    state.documents = artifacts.documents; state.activity = activity.activity;
    renderOverview(); renderExperiments(); renderArtifacts($("#artifact-search").value); renderActivity(); setConnection(true);
    if (showToast) toast("最新データに更新しました");
  } catch (error) { console.error(error); setConnection(false); toast("データを取得できませんでした"); }
  finally { button.classList.remove("loading"); }
}

$$('.nav-button').forEach((button) => button.addEventListener("click", () => {
  $$(".nav-button").forEach((item) => item.classList.toggle("active", item === button));
  $$(".view").forEach((view) => view.classList.toggle("active", view.id === `view-${button.dataset.view}`));
  history.replaceState(null, "", `#${button.dataset.view}`);
}));

$$('.filter-button').forEach((button) => button.addEventListener("click", () => {
  $$(".filter-button").forEach((item) => item.classList.toggle("active", item === button)); state.filter = button.dataset.filter; state.selected = null; renderExperiments();
}));

$("#artifact-search").addEventListener("input", (event) => renderArtifacts(event.target.value));
$("#refresh-button").addEventListener("click", () => loadData(true));

const initialView = location.hash.slice(1);
if (["experiments", "artifacts", "activity"].includes(initialView)) $(`.nav-button[data-view="${initialView}"]`).click();
loadData();
window.setInterval(() => loadData(false), 60000);
