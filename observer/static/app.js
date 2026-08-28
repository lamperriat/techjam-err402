const state = {
  overview: null,
  sessions: [],
  metrics: {},
  trace: null,
  selectedId: null,
  turnIndex: 0,
  catalog: null,
  catalogOffset: 0,
  catalogLimit: 30,
  jobs: [],
  selectedJobId: null,
  completedEvaluationIds: new Set(),
  experiments: [],
  documents: [],
  selectedDocumentId: null,
  lab: null,
  apiToken: "",
};

const $ = selector => document.querySelector(selector);
const $$ = selector => [...document.querySelectorAll(selector)];
const esc = value => String(value ?? "").replace(/[&<>'"]/g, char => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
})[char]);
const fmt = (value, digits = 3) => value === null || value === undefined
  ? "—"
  : Number(value).toFixed(digits).replace(/\.0+$/, "").replace(/(\.\d*?)0+$/, "$1");
const pct = value => value === null || value === undefined ? "—" : `${(Number(value) * 100).toFixed(1)}%`;
const scenarioName = value => ({
  buying: "Buying", browsing: "Browsing", intent_override: "Intent Override", boundary: "Boundary",
})[value] || value;
const bytes = value => {
  if (value === null || value === undefined) return "—";
  const units = ["B", "KB", "MB", "GB"];
  let number = Number(value); let index = 0;
  while (number >= 1024 && index < units.length - 1) { number /= 1024; index += 1; }
  return `${number.toFixed(index ? 1 : 0)} ${units[index]}`;
};

async function requestJSON(url, options = {}) {
  const headers = new Headers(options.headers || {});
  if (state.apiToken) headers.set("X-Observer-Token", state.apiToken);
  const response = await fetch(url, { ...options, headers });
  let payload;
  try { payload = await response.json(); } catch { payload = {}; }
  if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
  return payload;
}

function postJSON(url, payload = {}) {
  return requestJSON(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

let toastTimer;
function toast(message, tone = "") {
  const element = $("#toast");
  element.textContent = message;
  element.className = `toast ${tone}`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => element.classList.add("hidden"), 3200);
}

function navigate(page) {
  const knownPages = ["overview", "sessions", "catalog", "runs", "lab", "docs"];
  if (!knownPages.includes(page)) page = "overview";
  $$(".page").forEach(element => element.classList.toggle("active", element.id === `${page}Page`));
  $$(".nav-tab").forEach(button => button.classList.toggle("active", button.dataset.page === page));
  if (window.location.hash !== `#${page}`) history.replaceState(null, "", `#${page}`);
  if (page === "sessions" && !state.selectedId && state.sessions.length) selectSession(state.sessions[0].sample_id);
  if (page === "catalog" && !state.catalog) loadCatalog(true);
  if (page === "lab" && !state.lab) resetLab();
  if (page === "docs" && !state.selectedDocumentId && state.documents.length) selectDocument(state.documents[0].document_id);
}

function renderMetrics() {
  const metrics = state.metrics || {};
  const cards = [
    ["Hit Rate@10", pct(metrics.hit_rate_at_10)],
    ["MRR", fmt(metrics.mrr, 4)],
    ["MTTC", fmt(metrics.mttc, 2)],
    ["Efficiency", fmt(metrics.efficiency, 3)],
    ["TechnicalScore", fmt(metrics.recommended_technical_score, 5)],
  ];
  $("#metrics").innerHTML = cards.map(([label, value]) => `
    <div class="metric-card"><span>${label}</span><strong>${value}</strong></div>`).join("");
}

function renderOverview() {
  const overview = state.overview;
  if (!overview) return;
  const repository = overview.repository || {};
  const rerankMode = overview.runtime?.rerank_mode || "off";
  $("#repoBadge").textContent = `${repository.branch || "no git"} @ ${repository.commit || "—"}${repository.dirty ? " · dirty" : ""} · rerank ${rerankMode}`;
  const runtime = overview.runtime;
  const sourceState = overview.source_state || { restart_required: false, files: {} };
  const agentSource = (sourceState.files || {}).agent || {};
  $("#runtimeFacts").innerHTML = [
    ["Python", runtime.python], ["SQLite", runtime.sqlite], ["初始化", `${fmt(runtime.initialization_seconds, 2)} s`],
    ["Trace schema", runtime.trace_schema], ["Rerank mode", rerankMode], ["网络", runtime.network_required ? "required" : "offline"], ["进程", runtime.executable],
    ["Loaded Agent", agentSource.loaded_sha256 ? `${agentSource.loaded_sha256.slice(0, 12)}…` : "unavailable"],
  ].map(([label, value]) => `<div class="fact"><span>${esc(label)}</span><strong>${esc(value)}</strong></div>`).join("");
  const freshness = $("#codeFreshness");
  freshness.classList.toggle("hidden", !sourceState.restart_required);
  freshness.textContent = sourceState.restart_required
    ? "检测到 Agent/attributes/reranker/slot-ledger/clarification/shadow-analysis/evaluator/generalization 源码或已加载的 catalog/public set 在 Workbench 启动后发生变化。为防止混用新旧代码或数据产生假实验，请先停止并重新双击 Start Observer.vbs；评测、泛化、重放和 Lab 已锁定。"
    : "";
  $$('[data-action="evaluation"], [data-action="generalization"]').forEach(button => { button.disabled = sourceState.restart_required; });
  [$("#refreshTrace"), $("#labReset"), $("#labInput")].forEach(element => {
    if (element) element.disabled = sourceState.restart_required;
  });
  $("#dataHealth").innerHTML = overview.data.map(item => `
    <article class="data-row ${item.exists ? "ok" : "bad"}">
      <span class="health-dot"></span><div><strong>${esc(item.label)}</strong><small>${esc(item.path)}</small></div>
      <div class="data-meta"><b>${item.records ?? "—"}</b><span>${bytes(item.bytes)}</span>${item.sha256 ? `<code>${esc(item.sha256.slice(0, 12))}…</code>` : ""}</div>
    </article>`).join("");
  $("#groundTruthBoundary").textContent = overview.ground_truth_boundary;
  const base = overview.baseline_metrics || {};
  const latest = overview.latest_metrics || {};
  const comparison = [
    ["HR@10", "hit_rate_at_10", pct], ["MRR", "mrr", value => fmt(value, 4)],
    ["MTTC", "mttc", value => fmt(value, 2)], ["Score", "recommended_technical_score", value => fmt(value, 5)],
  ];
  $("#metricComparison").innerHTML = comparison.map(([label, key, formatter]) => {
    const delta = latest[key] == null || base[key] == null ? null : Number(latest[key]) - Number(base[key]);
    const tone = delta === null || Math.abs(delta) < 1e-10 ? "flat" : (key === "mttc" ? (delta < 0 ? "up" : "down") : (delta > 0 ? "up" : "down"));
    return `<div class="comparison-card"><span>${label}</span><div><strong>${formatter(latest[key])}</strong><small>official ${formatter(base[key])}</small></div><em class="${tone}">${delta === null ? "—" : `${delta > 0 ? "+" : ""}${fmt(delta, 5)}`}</em></div>`;
  }).join("");
  $("#pipelineRegistry").innerHTML = overview.pipeline.map((item, index) => `
    <article class="pipeline-card ${esc(item.status.replace(" ", "-"))}">
      <div class="pipeline-number">${String(index + 1).padStart(2, "0")}</div><div><h3>${esc(item.layer)}</h3><p>${esc(item.detail)}</p><small>${esc(item.source || "no runtime module")}</small></div><span class="status-label">${esc(item.status)}${item.mode ? ` · ${esc(item.mode)}` : ""}</span>
    </article>`).join("");
  const index = overview.index;
  $("#indexSummary").innerHTML = [
    [index.engine, `${index.rows} rows`], [index.tokenizer, "tokenizer"], [index.fields.join(" · "), "indexed fields"],
  ].map(([value, label]) => `<div><strong>${esc(value)}</strong><span>${esc(label)}</span></div>`).join("");
}

function filteredSessions() {
  const text = $("#searchInput").value.trim().toLowerCase();
  const scenario = $("#scenarioFilter").value;
  const result = $("#resultFilter").value;
  return state.sessions.filter(session => {
    const textMatch = !text || session.sample_id.toLowerCase().includes(text) || session.target_title.toLowerCase().includes(text);
    const scenarioMatch = scenario === "all" || session.scenario_type === scenario;
    const resultMatch = result === "all" || (result === "hit" ? session.hit === true : session.hit === false);
    return textMatch && scenarioMatch && resultMatch;
  });
}

function renderSessions() {
  const sessions = filteredSessions();
  $("#sessionCount").textContent = `${sessions.length}/${state.sessions.length}`;
  $("#sessionList").innerHTML = sessions.map(session => `
    <button class="session-item ${session.sample_id === state.selectedId ? "active" : ""}" data-id="${esc(session.sample_id)}">
      <div class="session-top"><span>${esc(session.sample_id)}</span><span class="result-dot ${session.hit === true ? "hit" : session.hit === false ? "miss" : ""}"></span></div>
      <div class="session-title">${esc(scenarioName(session.scenario_type))} · ${esc(session.target_title)}</div>
    </button>`).join("") || `<div class="empty-row">没有符合条件的会话</div>`;
  $$(".session-item").forEach(button => button.addEventListener("click", () => selectSession(button.dataset.id)));
}

async function selectSession(sampleId, refresh = false) {
  const requestedId = sampleId;
  state.selectedId = sampleId;
  state.turnIndex = 0;
  renderSessions();
  $("#emptyState").classList.remove("hidden");
  $("#emptyState").innerHTML = `<div class="loader-ring"></div><h2>运行 ${esc(sampleId)}</h2><p>使用 opaque session ID 经过当前 Agent 与公开模拟器。</p>`;
  $("#traceView").classList.add("hidden");
  try {
    const suffix = refresh ? "&refresh=1" : "";
    const trace = await requestJSON(`/api/trace?sample_id=${encodeURIComponent(sampleId)}${suffix}`);
    if (state.selectedId !== requestedId) return;
    state.trace = trace;
    $("#emptyState").classList.add("hidden");
    $("#traceView").classList.remove("hidden");
    renderTrace();
  } catch (error) {
    $("#emptyState").innerHTML = `<h2>Trace 失败</h2><p>${esc(error.message)}</p>`;
  }
}

function renderTrace() {
  const trace = state.trace;
  const result = trace.result;
  $("#traceEyebrow").textContent = `${scenarioName(trace.scenario_type)} / ${trace.difficulty_bucket || "unrated"}`;
  $("#traceTitle").textContent = trace.sample_id;
  $("#traceSubtitle").textContent = `${trace.target.title} · total trace ${fmt(trace.elapsed_ms, 1)} ms`;
  $("#observerNote").textContent = trace.observer_note;
  $("#traceResult").innerHTML = `
    <div class="result-badge">Result <strong>${result.hit ? "HIT" : "MISS"}</strong></div>
    <div class="result-badge">Turn <strong>${result.first_hit_turn ?? "—"}</strong></div>
    <div class="result-badge">Best rank <strong>${result.best_rank ?? "—"}</strong></div>
    <div class="result-badge">Diagnosis <strong>${esc(result.diagnosis)}</strong></div>
    <div class="result-badge">Derived contribution <strong>${fmt(result.technical_contribution, 4)}</strong></div>`;
  renderTurnRail(); renderTurn(); renderTarget();
}

function renderTurnRail() {
  $("#turnRail").innerHTML = state.trace.turns.map((turn, index) => `
    <button class="turn-node ${index === state.turnIndex ? "active" : ""} ${turn.event}" data-index="${index}">
      <strong>Turn ${turn.turn}</strong><span>${turn.hit ? `Hit @${turn.target_top10_rank}` : turn.event === "override_next" ? "Override next" : esc(turn.failure_code)}</span>
    </button>`).join("");
  $$(".turn-node").forEach(button => button.addEventListener("click", () => {
    state.turnIndex = Number(button.dataset.index); renderTurnRail(); renderTurn();
  }));
}

function eventData(turn, layer) {
  return (turn.agent_events || []).find(event => event.layer === layer)?.data || {};
}

function layerCard(index, title, value, meta, tone = "", kind = "actual") {
  return `<article class="layer-card ${tone}"><div class="layer-top"><span class="layer-index">L${index}</span><span class="layer-kind ${kind}">${kind}</span></div><h3>${esc(title)}</h3><div class="layer-value">${value}</div><div class="layer-meta">${esc(meta)}</div></article>`;
}

function renderTurn() {
  const turn = state.trace.turns[state.turnIndex];
  const parse = eventData(turn, "parse");
  const stateEvent = eventData(turn, "state");
  const session = Object.keys(stateEvent).length ? stateEvent : eventData(turn, "session");
  const retrieval = eventData(turn, "retrieval");
  const policy = eventData(turn, "policy");
  const output = eventData(turn, "output");
  const posthocRank = turn.retrieval.posthoc_target_rank;
  const routeCounts = turn.retrieval.route_counts || retrieval.route_counts || {};
  const mode = turn.retrieval.rerank_mode || state.trace.rerank_mode || "off";
  const rerank = retrieval.rerank || {};
  const slotLedger = session.slot_ledger || {};
  const questionShadow = policy.question_shadow || rerank.question_shadow || {};
  const shadowQuestion = questionShadow.selected_attribute || "none";
  const shadowTop = (questionShadow.candidates || [])[0] || {};
  const activeLedger = (slotLedger.active || []).slice(0, 5).map(record => (
    `${record.slot}${Number(record.polarity) < 0 ? "!=" : "="}${record.value}`
  )).join(" · ") || "none";
  const retiredLedger = (slotLedger.records || []).filter(record => record.status !== "active");
  const retiredPreview = retiredLedger.slice(-3).map(record => (
    `${record.slot}=${record.value} (${record.status}@t${record.ended_turn ?? "?"})`
  )).join(" · ") || "none";
  const shadowValues = (shadowTop.value_counts || []).slice(0, 3).map(item => (
    `${item[0]}:${item[1]}`
  )).join(" · ") || "no candidate split";
  const shadowComponents = shadowTop.attribute
    ? `IG ${fmt(shadowTop.information_gain, 3)} · coverage ${fmt(shadowTop.coverage, 3)} · answerability ${fmt(shadowTop.answerability, 3)} · turn cost ${fmt(shadowTop.turn_cost, 3)}`
    : (questionShadow.reason || "no positive-value candidate question");
  const blockedQuestions = (questionShadow.blocked_attributes || []).join(", ") || "none";
  const targetBreakdown = turn.retrieval.target_rerank_breakdown || {};
  const targetEvidence = (targetBreakdown.matched_evidence || []).join(", ") || "no normalized match";
  $("#turnLabel").textContent = `Turn ${turn.turn} / ${state.trace.turns.length}`;
  $("#turnLatency").textContent = `${fmt(turn.elapsed_ms, 1)} ms · tokens ${(turn.usage.prompt_tokens || 0) + (turn.usage.completion_tokens || 0)}`;
  $("#prevTurn").disabled = state.turnIndex === 0;
  $("#nextTurn").disabled = state.turnIndex === state.trace.turns.length - 1;
  const terms = (parse.terms || turn.retrieval.terms || []).length
    ? (parse.terms || turn.retrieval.terms).map(term => `<span class="chip">${esc(term)}</span>`).join(" ")
    : "No searchable terms";
  const disclosed = turn.simulator_disclosed_after.length ? turn.simulator_disclosed_after.map(esc).join(" · ") : "No disclosed constraints";
  const scoreText = turn.hit ? `Hit at rank ${turn.target_top10_rank}` : (!turn.eligible_for_hit && turn.target_top10_rank ? "Target present before override eligibility" : "No scored hit");
  $("#layerFlow").innerHTML = [
    layerCard(1, "Input", esc(turn.user_message), `turn ${turn.turn}`),
    layerCard(2, "Parse", terms, `broad ${parse.fts_expression || "empty"} · strict ${parse.strict_fts_expression || "empty"}`),
    layerCard(3, "Agent state", `${esc(session.memory_mode || "No state event")}<br><b>Active slots:</b> ${esc(activeLedger)}<br><b>Retired:</b> ${esc(retiredPreview)}`, `v${session.version ?? 1} · ${(session.active_terms || []).length} active terms · ${slotLedger.active_count ?? 0}/${slotLedger.record_count ?? 0} ledger slots · ${retiredLedger.length} retired · ${session.override_count || 0} overrides · pending ${esc(session.pending_attribute || "none")}`),
    layerCard(4, "Sparse retrieval + fusion", `${fmt(retrieval.candidate_count ?? turn.retrieval.candidate_count, 0)} candidates<br><b>${esc(retrieval.engine || "SQLite FTS5 BM25 + weighted RRF")}</b>`, `broad ${routeCounts.broad ?? 0} · strict ${routeCounts.strict ?? 0} · fused ${routeCounts.fused ?? 0} · ${fmt(output.elapsed_ms, 2)} ms`, posthocRank ? "success" : ""),
    layerCard(5, "Constraint rerank", `Mode: <b>${esc(mode)}</b> · pool <b>${rerank.pool_size ?? 0}/${rerank.top_n ?? 50}</b><br>Reranked ${routeCounts.reranked ?? routeCounts.fused ?? 0} · Final ${routeCounts.final ?? routeCounts.fused ?? 0}`, `${rerank.scorer_version || "legacy/no scorer"} · ${rerank.attribute_schema_version || "no attribute schema"} · ${retrieval.rerank_affects_output ? "affects output" : "diagnostic only"}`),
    layerCard(6, "Target annotation", `Broad: <b>${turn.retrieval.target_broad_rank ?? "not found"}</b> · Strict: <b>${turn.retrieval.target_strict_rank ?? "not found"}</b><br>Fused: <b>${turn.retrieval.target_fused_rank ?? "not found"}</b> · Reranked: <b>${turn.retrieval.target_reranked_rank ?? turn.retrieval.target_fused_rank ?? "not found"}</b><br>Final: <b>${posthocRank ?? "not found"}</b> · Top 10: <b>${turn.target_top10_rank ?? "no"}</b><br>Rerank total: <b>${fmt(targetBreakdown.total, 4)}</b> · ${esc(targetEvidence)}`, `joined after Agent.respond · actual route ${turn.retrieval.actual_route || "fused"}`, posthocRank ? "" : "alert", "post-hoc"),
    layerCard(7, "Policy", `Actual ask: <b>${esc(policy.ask_attribute ?? turn.ask_attribute ?? "none")}</b><br>Candidate-aware shadow: <b>${esc(shadowQuestion)}</b> · value ${fmt(shadowTop.score, 4)}<br>${esc(shadowComponents)}<br><b>Candidate split:</b> ${esc(shadowValues)}<br>${esc(turn.event)}`, `${esc(policy.reason || "next action")} · blocked ${esc(blockedQuestions)} · shadow does not affect output`),
    layerCard(8, "Score", esc(scoreText), `${turn.failure_code} · ${turn.eligible_for_hit ? "eligible" : "blocked"}`, turn.hit ? "success" : "", "derived"),
  ].join("");
  $("#conversation").innerHTML = `
    <div class="bubble user"><span class="bubble-label">SIMULATED USER</span>${esc(turn.user_message)}</div>
    <div class="bubble agent"><span class="bubble-label">CURRENT AGENT</span>${esc(turn.agent_message || "No message")}</div>
    <div class="policy-line">ask_attribute: <b>${esc(turn.ask_attribute ?? "null")}</b>${turn.next_user_message ? `<br>Next simulator reply: ${esc(turn.next_user_message)}` : ""}${turn.error ? `<br><span class="signal bad-text">${esc(turn.error)}</span>` : ""}<br>Output validation: malformed ${turn.validation.malformed_count} · invalid ${turn.validation.invalid_catalog_count} · duplicate ${turn.validation.duplicate_count}<br>Simulator disclosed (diagnostic): ${disclosed}</div>`;
  $("#agentEvents").textContent = JSON.stringify(turn.agent_events || [], null, 2);
  renderRecommendations(turn, retrieval.top_results || []);
}

function renderTarget() {
  const target = state.trace.target;
  const card = state.trace.intent_card;
  const profile = state.trace.profile || {};
  $("#targetCard").innerHTML = `
    <div class="truth-title">${esc(target.title)}</div><div class="truth-asin">${esc(target.parent_asin)}</div>
    <div class="truth-grid"><div><span>Price</span><strong>${target.price == null ? "—" : `$${esc(target.price)}`}</strong></div><div><span>Rating</span><strong>${target.average_rating ?? "—"}</strong></div><div><span>Store</span><strong>${esc(target.store || "—")}</strong></div><div><span>Scenario</span><strong>${esc(scenarioName(state.trace.scenario_type))}</strong></div></div>
    <div class="debug-section"><b>Derived intent card</b><p>Hard: ${card.hard_constraints.map(esc).join(" · ")}</p><p>Soft: ${card.soft_preferences.map(esc).join(" · ")}</p></div>
    <div class="debug-section"><b>Profile given to reset</b><p>${esc(profile.summary || "No summary")}</p><p>Tags: ${(profile.preference_tags || []).map(esc).join(" · ") || "—"}</p></div>`;
}

function renderRecommendations(turn, actualResults) {
  const evidenceById = Object.fromEntries(actualResults.map(item => [item.parent_asin, item]));
  const mode = turn.retrieval.rerank_mode || state.trace.rerank_mode || "off";
  $("#rankSummary").textContent = `${turn.valid_recommendation_count} valid · rerank ${mode} · post-hoc final target rank ${turn.retrieval.posthoc_target_rank ?? "not found"}`;
  $("#recommendationRows").innerHTML = turn.recommendations.map((product, index) => {
    const evidence = evidenceById[product.parent_asin] || {};
    const breakdown = evidence.rerank || {};
    const score = breakdown.total ?? evidence.fusion_score ?? evidence.bm25_score;
    const ranks = `F${evidence.fused_rank ?? "—"} → R${evidence.reranked_rank ?? "—"} → Final ${evidence.final_rank ?? index + 1}`;
    const components = breakdown.total == null
      ? "rerank not computed"
      : `prior ${fmt(breakdown.rrf_prior, 2)} · cat ${fmt(breakdown.category_consistency, 2)} · slot ${fmt(breakdown.positive_slot_match, 2)} · exact ${fmt(breakdown.exact_feature_match, 2)} · violation ${fmt(breakdown.negative_violation, 2)}`;
    return `<tr class="${product.is_target ? "target-row" : ""}"><td class="rank-number">${index + 1}</td><td class="product-name">${esc(product.title)}</td><td class="asin">${esc(product.parent_asin)}</td><td>${fmt(score, 6)}<small>${esc(ranks)}<br>${esc(components)}</small></td><td>${product.price == null ? "—" : `$${esc(product.price)}`}</td><td>${product.average_rating ?? "—"}</td><td class="signal">${product.is_target ? "TARGET" : ""}</td></tr>`;
  }).join("") || `<tr><td colspan="7" class="empty-row">No valid recommendations</td></tr>`;
}

async function loadCatalog(reset = false) {
  if (reset) state.catalogOffset = 0;
  const query = $("#catalogSearch").value.trim();
  try {
    state.catalog = await requestJSON(`/api/catalog?q=${encodeURIComponent(query)}&offset=${state.catalogOffset}&limit=${state.catalogLimit}`);
    renderCatalog();
  } catch (error) { toast(`目录搜索失败：${error.message}`, "error"); }
}

function renderCatalog() {
  const catalog = state.catalog;
  $("#catalogCount").textContent = `${catalog.total.toLocaleString()} matches`;
  $("#catalogRows").innerHTML = catalog.items.map(product => `
    <tr class="catalog-row" data-asin="${esc(product.parent_asin)}"><td class="product-name">${esc(product.title)}</td><td class="asin">${esc(product.parent_asin)}</td><td>${esc((product.categories || []).slice(-2).join(" / "))}</td><td>${product.price == null ? "—" : `$${esc(product.price)}`}</td><td>${fmt(product.bm25_score, 6)}</td></tr>`).join("") || `<tr><td colspan="5" class="empty-row">没有匹配商品</td></tr>`;
  $$(".catalog-row").forEach(row => row.addEventListener("click", () => selectProduct(row.dataset.asin)));
  const first = catalog.total ? catalog.offset + 1 : 0;
  const last = Math.min(catalog.total, catalog.offset + catalog.items.length);
  $("#catalogPageLabel").textContent = `${first}–${last} / ${catalog.total}`;
  $("#catalogPrev").disabled = catalog.offset === 0;
  $("#catalogNext").disabled = catalog.offset + catalog.limit >= catalog.total;
}

async function selectProduct(parentAsin) {
  try {
    const product = await requestJSON(`/api/product?parent_asin=${encodeURIComponent(parentAsin)}`);
    $("#productDetail").textContent = JSON.stringify(product, null, 2);
  } catch (error) { toast(error.message, "error"); }
}

function jobStatusLabel(job) {
  return ({ queued: "排队", running: "运行中", cancelling: "取消中", cancelled: "已取消", completed: "完成", failed: "失败" })[job.status] || job.status;
}

function renderJobs() {
  const kindLabel = { evaluation: "EVAL", generalization: "ROBUST", tests: "TEST" };
  const html = state.jobs.slice(0, 8).map(job => `
    <button class="job-row ${job.job_id === state.selectedJobId ? "active" : ""}" data-job-id="${esc(job.job_id)}">
      <span class="job-kind">${kindLabel[job.kind] || esc(job.kind)}</span><div><strong>${esc(job.message)}</strong><small>${esc(job.job_id)} · ${job.elapsed_seconds == null ? job.created_at : `${fmt(job.elapsed_seconds, 1)} s`}</small></div><span class="job-status ${job.status}">${jobStatusLabel(job)}</span>
    </button>`).join("") || `<div class="empty-row">尚无运行记录</div>`;
  $("#runJobs").innerHTML = html;
  $("#overviewJobs").innerHTML = html;
  $$(".job-row").forEach(button => button.addEventListener("click", () => { state.selectedJobId = button.dataset.jobId; renderJobs(); renderSelectedJob(); }));
  renderSelectedJob();
}

function renderSelectedJob() {
  const active = state.jobs.find(job => ["queued", "running", "cancelling"].includes(job.status));
  const selected = state.jobs.find(job => job.job_id === state.selectedJobId) || active || state.jobs[0];
  if (!selected) {
    $("#jobProgressLabel").textContent = "—"; $("#jobProgress").style.width = "0%"; $("#jobLogs").textContent = "尚未运行任务。"; return;
  }
  state.selectedJobId = selected.job_id;
  $("#jobProgressLabel").textContent = `${jobStatusLabel(selected)} · ${Math.round((selected.progress || 0) * 100)}%`;
  $("#jobProgress").style.width = `${Math.round((selected.progress || 0) * 100)}%`;
  $("#jobLogs").textContent = (selected.logs || []).join("\n") || selected.message;
  $("#jobLogs").scrollTop = $("#jobLogs").scrollHeight;
}

async function loadJobs() {
  try {
    const payload = await requestJSON("/api/jobs");
    state.jobs = payload.jobs;
    const completed = state.jobs.filter(job => job.kind === "evaluation" && job.status === "completed");
    for (const job of completed) {
      if (!state.completedEvaluationIds.has(job.job_id)) {
        state.completedEvaluationIds.add(job.job_id);
        await reloadResults();
      }
    }
    const completedRobustness = state.jobs.filter(job => job.kind === "generalization" && job.status === "completed");
    for (const job of completedRobustness) {
      if (!state.completedEvaluationIds.has(job.job_id)) {
        state.completedEvaluationIds.add(job.job_id);
        await loadExperiments();
      }
    }
    renderJobs();
  } catch { /* health polling handles disconnects */ }
}

async function startJob(kind) {
  try {
    const job = await postJSON(`/api/jobs/${kind}`);
    state.selectedJobId = job.job_id;
    const label = kind === "evaluation" ? "公开评测" : (kind === "generalization" ? "泛化压力测试" : "单元测试");
    toast(`${label}已启动`, "success");
    navigate("runs");
    await loadJobs();
  } catch (error) { toast(error.message, "error"); }
}

async function cancelActiveJob() {
  const job = state.jobs.find(item => item.job_id === state.selectedJobId && ["queued", "running", "cancelling"].includes(item.status))
    || state.jobs.find(item => ["queued", "running", "cancelling"].includes(item.status));
  if (!job) { toast("没有可取消的活动任务"); return; }
  try { await postJSON(`/api/jobs/${encodeURIComponent(job.job_id)}/cancel`); await loadJobs(); } catch (error) { toast(error.message, "error"); }
}

async function loadExperiments() {
  const payload = await requestJSON("/api/experiments");
  state.experiments = payload.experiments;
  renderExperiments();
}

function renderExperiments() {
  $("#experimentsTable").innerHTML = `<table><thead><tr><th>Experiment</th><th>Created</th><th>HR@10</th><th>MRR</th><th>MTTC</th><th>Score</th><th>Shadow policy</th><th>Default-suite robust HR</th><th>Scenario HR</th></tr></thead><tbody>${state.experiments.map(item => {
    const metrics = item.metrics || {}; const scenarios = metrics.scenario_metrics || {};
    const mode = item.implementation?.rerank_mode || "off";
    const robust = item.robustness?.released_public?.all_suites_robust_hit_rate;
    const shadow = item.shadow_policy_analysis || {};
    const shadowText = shadow.turn_count == null
      ? "—"
      : `${shadow.disagreement_count}/${shadow.turn_count} disagree · ${shadow.shadow_question_turns} selected · ${shadow.blocked_selection_violations} blocked`;
    const scenarioText = Object.entries(scenarios).map(([name, values]) => `${scenarioName(name)} ${pct(values.hit_rate_at_10)}`).join(" · ");
    return `<tr><td class="product-name">${esc(item.label)}<small>rerank ${esc(mode)}</small></td><td>${esc(item.created_at || "reference")}</td><td>${pct(metrics.hit_rate_at_10)}</td><td>${fmt(metrics.mrr, 4)}</td><td>${fmt(metrics.mttc, 2)}</td><td>${fmt(metrics.recommended_technical_score, 5)}</td><td class="small muted">${esc(shadowText)}</td><td>${robust == null ? "—" : pct(robust)}</td><td class="small muted">${esc(scenarioText || "—")}</td></tr>`;
  }).join("")}</tbody></table>`;
}

async function reloadResults() {
  const [overview, sessions] = await Promise.all([requestJSON("/api/overview"), requestJSON("/api/sessions")]);
  state.overview = overview; state.sessions = sessions.sessions; state.metrics = sessions.metrics;
  renderMetrics(); renderOverview(); renderSessions(); await loadExperiments();
}

async function resetLab() {
  try {
    state.lab = await postJSON("/api/lab/reset");
    $("#labHistory").innerHTML = `<div class="system-message">新会话 ${esc(state.lab.session_id.slice(-8))} · rerank ${esc(state.lab.rerank_mode || "off")} · 当前 Agent 会累积对话约束；可用多轮消息观察状态、改写、稀疏融合与属性重排变化。</div>`;
    $("#labEvents").textContent = "等待第一条消息。";
    $("#labInput").focus();
  } catch (error) { toast(error.message, "error"); }
}

async function sendLabMessage(message) {
  if (!state.lab) await resetLab();
  try {
    const entry = await postJSON("/api/lab/respond", { session_id: state.lab.session_id, message });
    const recommendations = entry.recommendations.map((product, index) => `<li><b>${index + 1}</b><span>${esc(product.title)}</span><code>${esc(product.parent_asin)}</code></li>`).join("");
    $("#labHistory").insertAdjacentHTML("beforeend", `<div class="bubble user"><span class="bubble-label">YOU · TURN ${entry.turn}</span>${esc(message)}</div><div class="bubble agent"><span class="bubble-label">AGENT</span>${esc(entry.response.message)}<small>ask_attribute: ${esc(entry.response.ask_attribute ?? "null")}</small></div><ol class="lab-recommendations">${recommendations || "<li>No results</li>"}</ol>`);
    $("#labHistory").scrollTop = $("#labHistory").scrollHeight;
    $("#labEvents").textContent = JSON.stringify(entry.events, null, 2);
  } catch (error) { toast(error.message, "error"); }
}

function renderDocuments() {
  const query = $("#documentSearch").value.trim().toLowerCase();
  const documents = state.documents.filter(item => !query || `${item.title} ${item.path} ${item.group}`.toLowerCase().includes(query));
  $("#documentList").innerHTML = documents.map(item => `
    <button class="document-item ${item.document_id === state.selectedDocumentId ? "active" : ""}" data-document-id="${esc(item.document_id)}"><span>${esc(item.group)}</span><strong>${esc(item.title)}</strong><small>${esc(item.path)} · ${bytes(item.bytes)}</small></button>`).join("") || `<div class="empty-row">没有匹配文档</div>`;
  $$(".document-item").forEach(button => button.addEventListener("click", () => selectDocument(button.dataset.documentId)));
}

async function selectDocument(documentId) {
  try {
    const document = await requestJSON(`/api/document?id=${encodeURIComponent(documentId)}`);
    state.selectedDocumentId = documentId;
    $("#documentGroup").textContent = document.group;
    $("#documentTitle").textContent = document.title;
    $("#documentPath").textContent = document.path;
    $("#documentContent").textContent = document.content;
    renderDocuments();
  } catch (error) { toast(error.message, "error"); }
}

function exportTrace() {
  if (!state.trace) return;
  const blob = new Blob([JSON.stringify(state.trace, null, 2)], { type: "application/json" });
  const link = document.createElement("a"); link.href = URL.createObjectURL(blob); link.download = `${state.trace.sample_id}_trace.json`; link.click(); URL.revokeObjectURL(link.href);
}

async function checkHealth() {
  const status = $("#runtimeStatus");
  try {
    const health = await requestJSON("/api/health");
    status.classList.remove("offline"); status.querySelector("span:last-child").textContent = `${health.branch || "local"} @ ${health.commit || "—"} · rerank ${health.rerank_mode || "off"}`;
    if (state.overview && health.restart_required !== state.overview.source_state?.restart_required) {
      state.overview = await requestJSON("/api/overview");
      renderOverview();
    }
  } catch {
    status.classList.add("offline"); status.querySelector("span:last-child").textContent = "已断开";
  }
}

function bindEvents() {
  $$(".nav-tab").forEach(button => button.addEventListener("click", () => navigate(button.dataset.page)));
  $$('[data-page-jump]').forEach(button => button.addEventListener("click", () => navigate(button.dataset.pageJump)));
  $$('[data-action="evaluation"]').forEach(button => button.addEventListener("click", () => startJob("evaluation")));
  $$('[data-action="generalization"]').forEach(button => button.addEventListener("click", () => startJob("generalization")));
  $$('[data-action="tests"]').forEach(button => button.addEventListener("click", () => startJob("tests")));
  [$("#searchInput"), $("#scenarioFilter"), $("#resultFilter")].forEach(element => element.addEventListener("input", renderSessions));
  $("#prevTurn").addEventListener("click", () => { if (state.turnIndex > 0) { state.turnIndex -= 1; renderTurnRail(); renderTurn(); } });
  $("#nextTurn").addEventListener("click", () => { if (state.turnIndex < state.trace.turns.length - 1) { state.turnIndex += 1; renderTurnRail(); renderTurn(); } });
  $("#refreshTrace").addEventListener("click", () => selectSession(state.selectedId, true));
  $("#exportTrace").addEventListener("click", exportTrace);
  $("#catalogSearchButton").addEventListener("click", () => loadCatalog(true));
  $("#catalogSearch").addEventListener("keydown", event => { if (event.key === "Enter") loadCatalog(true); });
  $("#catalogPrev").addEventListener("click", () => { state.catalogOffset = Math.max(0, state.catalogOffset - state.catalogLimit); loadCatalog(); });
  $("#catalogNext").addEventListener("click", () => { state.catalogOffset += state.catalogLimit; loadCatalog(); });
  $("#cancelJob").addEventListener("click", cancelActiveJob);
  $("#labReset").addEventListener("click", resetLab);
  $("#labForm").addEventListener("submit", event => { event.preventDefault(); const input = $("#labInput"); const message = input.value.trim(); if (message) { input.value = ""; sendLabMessage(message); } });
  $("#documentSearch").addEventListener("input", renderDocuments);
  $("#shutdownButton").addEventListener("click", async () => {
    if (!window.confirm("停止本地 Agent Workbench？再次使用时双击 Start Observer.vbs。")) return;
    await postJSON("/api/shutdown"); toast("Workbench 已停止"); setTimeout(checkHealth, 700);
  });
}

(async function init() {
  bindEvents();
  try {
    const tokenResponse = await fetch("/api/token", { cache: "no-store" });
    const tokenPayload = await tokenResponse.json();
    if (!tokenResponse.ok || !tokenPayload.token) throw new Error(tokenPayload.error || "无法取得本机控制令牌");
    state.apiToken = tokenPayload.token;
    const [overview, sessions, experiments, documents, jobs] = await Promise.all([
      requestJSON("/api/overview"), requestJSON("/api/sessions"), requestJSON("/api/experiments"), requestJSON("/api/documents"), requestJSON("/api/jobs"),
    ]);
    state.overview = overview; state.sessions = sessions.sessions; state.metrics = sessions.metrics;
    state.experiments = experiments.experiments; state.documents = documents.documents; state.jobs = jobs.jobs;
    renderMetrics(); renderOverview(); renderSessions(); renderExperiments(); renderDocuments(); renderJobs(); checkHealth();
    navigate(window.location.hash.slice(1) || "overview");
  } catch (error) {
    $("#overviewPage").innerHTML = `<div class="empty-state panel"><h2>Workbench 启动失败</h2><p>${esc(error.message)}</p></div>`;
  }
  setInterval(loadJobs, 1200);
  setInterval(checkHealth, 3500);
})();
