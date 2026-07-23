"use strict";

const state = { overview: null, experiments: [], artifacts: [], documents: [], filter: "all", selected: null };
const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const escapeHtml = (value) => String(value ?? "").replace(/[&<>'"]/g, (char) => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[char]));

function bytes(value) {
  if (value === null || value === undefined) return "--";
  const units = ["B", "KiB", "MiB", "GiB", "TiB"]; let amount = Number(value); let unit = 0;
  while (amount >= 1024 && unit < units.length - 1) { amount /= 1024; unit += 1; }
  return `${amount >= 10 || unit === 0 ? amount.toFixed(0) : amount.toFixed(2)} ${units[unit]}`;
}
function duration(value) {
  if (value === null || value === undefined) return "--";
  const seconds = Number(value); if (seconds < 60) return `${seconds.toFixed(1)} 秒`;
  const minutes = Math.floor(seconds / 60); return `${minutes}分 ${String(Math.round(seconds % 60)).padStart(2, "0")}秒`;
}
function dateTime(value) { return value ? new Intl.DateTimeFormat("ja-JP", {year:"numeric",month:"2-digit",day:"2-digit",hour:"2-digit",minute:"2-digit",timeZone:"UTC"}).format(new Date(value)) + " UTC" : "--"; }
function shortDate(value) { return new Intl.DateTimeFormat("ja-JP", {month:"2-digit",day:"2-digit",timeZone:"UTC"}).format(new Date(value)); }
function metricValue(metric) {
  if (!metric) return "--"; if (metric.unit === "percent") return `${metric.value.toFixed(2)}%`;
  if (metric.unit === "seconds") return duration(metric.value); if (metric.unit === "bytes") return bytes(metric.value);
  if (metric.unit === "rmse") return Number(metric.value).toExponential(2); return `${metric.value}${metric.unit ? ` ${metric.unit}` : ""}`;
}
async function request(path) { const response = await fetch(path, {headers:{Accept:"application/json"},cache:"no-store"}); if (!response.ok) throw new Error(`HTTP ${response.status}`); return response.json(); }
function setConnection(ok) { $("#connection-dot").className = `connection-dot ${ok ? "ok" : "error"}`; $("#connection-label").textContent = ok ? "データ接続中" : "データ接続エラー"; }
function toast(message) { const node=$("#toast"); node.textContent=message; node.classList.add("show"); setTimeout(()=>node.classList.remove("show"),2200); }

function timelineMetric(item) {
  if (item.metric_unit === "seconds") return duration(item.metric_value);
  if (item.metric_unit === "percent") return `${item.metric_value.toFixed(2)}%`;
  return `${Number(item.metric_value).toLocaleString("ja-JP")} ${item.metric_unit}`;
}

function renderOverview() {
  const data=state.overview; const project=data.project||{}; const timeline=data.timeline||[];
  $("#project-title").textContent=project.title||"TripoSplat CPU Lab"; $("#project-phase").textContent=project.phase||"PROJECT";
  $("#project-status").textContent=(project.status||"unknown").toUpperCase(); $("#project-objective").textContent=project.objective||"--";
  $("#project-progress-label").textContent=`${project.progress||0}% 完了`; $("#project-progress-value").textContent=`${project.progress||0}%`;
  $("#project-progress-ring").style.setProperty("--progress",`${(project.progress||0)*3.6}deg`); $("#updated-at").textContent=`更新 ${dateTime(project.updated_at)}`;
  const original=timeline.find((item)=>item.slug==="cpu-original-baseline")?.metric_value||10856.388;
  const current=timeline.find((item)=>item.slug==="nf24-low-resource")?.metric_value||2747.27;
  const speedup=original/current; $("#metric-original").textContent=duration(original); $("#metric-current").textContent=duration(current);
  $("#metric-cumulative").textContent=`${speedup.toFixed(2)}x`; $("#metric-reduction").textContent=`wall time ${(100*(1-current/original)).toFixed(1)}%削減`;
  $("#metric-quality").textContent=(2.06857e-5).toExponential(2);

  if (!document.querySelector(".nf24-note")) {
    const note=document.createElement("aside"); note.className="nf24-note";
    note.innerHTML='<strong>NF24とは</strong><p>各weightを非線形24-bit（int16 code + int8 residual + scale）で表し、206個のLinear層でGEMM中に復号する低リソース形式です。FP32の4 byte/weightに対してcode payloadは3 byte/weightです。高メモリprofileは量子化前へ戻すのではなく、同じNF24値のQKV/out 56層だけを起動時にFP32常駐させます。</p>';
    document.querySelector(".cumulative-metrics").after(note);
  }

  $("#journey-count").textContent=`${timeline.length} milestones`;
  $("#journey").innerHTML=timeline.map((item)=>`<article class="journey-item ${escapeHtml(item.status)}"><time class="journey-date">${shortDate(item.occurred_at)}</time><span class="journey-phase">${escapeHtml(item.phase)}</span><h3>${escapeHtml(item.title)}</h3><p>${escapeHtml(item.summary)}</p><span class="journey-metric">${escapeHtml(item.metric_label)} · ${escapeHtml(timelineMetric(item))}</span></article>`).join("");
  const milestones=data.milestones||[]; $("#milestone-count").textContent=`${milestones.filter((item)=>item.status==="complete").length} / ${milestones.length} 完了`;
  $("#pipeline-track").innerHTML=milestones.map((item)=>`<article class="pipeline-item ${escapeHtml(item.status)}"><strong>${escapeHtml(item.title)}</strong><span>${escapeHtml(item.stage)} · ${item.progress}%</span><div class="mini-progress"><i style="width:${Math.max(0,Math.min(100,item.progress))}%"></i></div></article>`).join("");
  $("#benchmark-chart").innerHTML=(data.settings?.benchmark_chart||[]).map((item)=>`<div class="benchmark-row ${escapeHtml(item.tone||"")}"><div class="benchmark-label"><strong>${escapeHtml(item.label)}</strong><span>${escapeHtml(item.detail)}</span></div><div class="bar-track"><div class="bar-value" style="width:${Math.max(2,Math.min(100,item.bar_percent))}%"></div></div><span class="benchmark-delta">${escapeHtml(item.delta)}</span></div>`).join("");
  $("#gate-list").innerHTML=(data.settings?.admission_gates||[]).map((item,index)=>`<div class="gate-item ${item.done?"done":""}"><span class="gate-marker">${item.done?"✓":index+1}</span><div><strong>${escapeHtml(item.title)}</strong><span>${escapeHtml(item.detail)}</span></div></div>`).join("");
  const job=data.maintenance?.latest_job||{}; const schedule=data.maintenance?.schedule||{}; const summary=job.result_summary||{};
  $("#maintenance-status").innerHTML=[["状態",job.status||"登録待ち"],["実行間隔",schedule.interval_seconds?`${Math.round(schedule.interval_seconds/60)}分`:"--"],["直近の退避",`${summary.archived_count||0}件`],["退避容量",bytes(summary.archived_bytes||0)]].map(([label,value])=>`<div class="maintenance-cell"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`).join("");
}

function renderExperimentDetail(item) {
  if (!item) { $("#experiment-detail").innerHTML='<p class="empty-state">実験を選択してください</p>'; return; }
  $("#experiment-detail").innerHTML=`<span class="state-chip ${escapeHtml(item.status)}">${escapeHtml(item.status)}</span><h3>${escapeHtml(item.title)}</h3><p class="variant">${escapeHtml(item.variant)} · ${escapeHtml(item.category)}</p><div class="detail-metrics">${(item.metrics||[]).map((metric)=>`<div class="detail-metric"><span>${escapeHtml(metric.label)}</span><strong>${escapeHtml(metricValue(metric))}</strong></div>`).join("")}</div><p class="detail-note">${escapeHtml(item.notes||"記録なし")}</p>`;
}
function renderExperiments() {
  const list=state.filter==="all"?state.experiments:state.experiments.filter((item)=>item.status===state.filter);
  $("#experiment-rows").innerHTML=list.map((item)=>`<tr data-slug="${escapeHtml(item.slug)}" class="${state.selected===item.slug?"selected":""}" tabindex="0"><td class="table-title"><strong>${escapeHtml(item.title)}</strong><small>${escapeHtml(item.variant)}</small></td><td><span class="state-chip ${escapeHtml(item.status)}">${escapeHtml(item.status)}</span></td><td>${item.steps??"--"}</td><td>${duration(item.duration_sec)}</td><td>${bytes(item.peak_rss_bytes)}</td><td>${escapeHtml(item.quality_status)}</td></tr>`).join("");
  const select=(slug)=>{state.selected=slug;renderExperiments();renderExperimentDetail(state.experiments.find((item)=>item.slug===slug));};
  $$("#experiment-rows tr").forEach((row)=>{row.addEventListener("click",()=>select(row.dataset.slug));row.addEventListener("keydown",(event)=>{if(event.key==="Enter")select(row.dataset.slug);});});
  if(!state.selected&&list[0]){state.selected=list[0].slug;renderExperimentDetail(list[0]);}
}

function openPreview(preview,title) { $("#dialog-image").src=preview.url; $("#dialog-image").alt=preview.caption||title; $("#dialog-title").textContent=title; $("#dialog-caption").textContent=preview.caption||"生成結果"; $("#preview-dialog").showModal(); }
function renderArtifacts(query="") {
  const term=query.trim().toLowerCase(); const artifacts=state.artifacts.filter((item)=>[item.title,item.kind,item.storage,item.description].join(" ").toLowerCase().includes(term));
  const documents=state.documents.filter((item)=>[item.title,item.category,item.summary].join(" ").toLowerCase().includes(term));
  const previews=artifacts.flatMap((artifact)=>(artifact.previews||[]).map((preview)=>({artifact,preview})));
  $("#visual-gallery").innerHTML=previews.map(({artifact,preview})=>`<button class="preview-card" type="button" data-preview="${preview.id}"><span class="preview-frame"><img src="${escapeHtml(preview.url)}" alt="${escapeHtml(preview.caption||artifact.title)}" loading="lazy"></span><span class="preview-copy"><strong>${escapeHtml(artifact.title)}</strong><small>${escapeHtml(preview.caption||"生成結果")}</small></span></button>`).join("")||'<p class="empty-state">プレビューを準備中です</p>';
  $$(".preview-card").forEach((button)=>{const found=previews.find(({preview})=>String(preview.id)===button.dataset.preview);button.addEventListener("click",()=>openPreview(found.preview,found.artifact.title));});
  $("#artifact-count").textContent=`${artifacts.length} 件`; $("#document-count").textContent=`${documents.length} 件`;
  $("#artifact-list").innerHTML=artifacts.map((item)=>`<article class="artifact-row-private"><div class="artifact-main"><strong>${escapeHtml(item.title)}</strong><small>${escapeHtml(item.kind)} · ${escapeHtml(item.description)}</small></div><span class="state-chip ${escapeHtml(item.state)}">${escapeHtml(item.state)}</span><span>${bytes(item.size_bytes)}</span><span>${(item.previews||[]).length} preview</span></article>`).join("");
  $("#document-grid").innerHTML=documents.map((item)=>`<article class="document-item"><span class="state-chip">${escapeHtml(item.category)}</span><strong>${escapeHtml(item.title)}</strong><p>${escapeHtml(item.summary)}</p></article>`).join("");
}
async function loadData(showToast=false){const button=$("#refresh-button");button.classList.add("loading");try{const[overview,experiments,artifacts]=await Promise.all([request("/api/overview"),request("/api/experiments"),request("/api/artifacts")]);state.overview=overview;state.experiments=experiments.experiments;state.artifacts=artifacts.artifacts;state.documents=artifacts.documents;renderOverview();renderExperiments();renderArtifacts($("#artifact-search").value);setConnection(true);if(showToast)toast("最新データに更新しました");}catch(error){console.error(error);setConnection(false);toast("データを取得できませんでした");}finally{button.classList.remove("loading");}}
$$('.nav-button').forEach((button)=>button.addEventListener("click",()=>{$$(".nav-button").forEach((item)=>item.classList.toggle("active",item===button));$$(".view").forEach((view)=>view.classList.toggle("active",view.id===`view-${button.dataset.view}`));history.replaceState(null,"",`#${button.dataset.view}`);}));
$$('.filter-button').forEach((button)=>button.addEventListener("click",()=>{$$(".filter-button").forEach((item)=>item.classList.toggle("active",item===button));state.filter=button.dataset.filter;state.selected=null;renderExperiments();}));
$("#artifact-search").addEventListener("input",(event)=>renderArtifacts(event.target.value)); $("#refresh-button").addEventListener("click",()=>loadData(true));
$("#dialog-close").addEventListener("click",()=>$("#preview-dialog").close()); $("#preview-dialog").addEventListener("click",(event)=>{if(event.target===$("#preview-dialog"))$("#preview-dialog").close();});
const initialView=location.hash.slice(1);if(["experiments","artifacts"].includes(initialView))$(`.nav-button[data-view="${initialView}"]`).click();loadData();setInterval(()=>loadData(false),60000);
