const state = { sessions: [], metrics: {}, trace: null, selectedId: null, turnIndex: 0 };
const $ = (selector) => document.querySelector(selector);
const esc = (value) => String(value ?? "").replace(/[&<>'"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"})[c]);
const fmt = (value, digits = 3) => value === null || value === undefined ? "—" : Number(value).toFixed(digits).replace(/\.0+$/, "");
const pct = value => value === null || value === undefined ? "—" : `${(Number(value) * 100).toFixed(1)}%`;
const scenarioName = value => ({buying:"Buying", browsing:"Browsing", intent_override:"Intent Override", boundary:"Boundary"})[value] || value;

async function getJSON(url) {
  const response = await fetch(url);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
  return payload;
}

function renderMetrics() {
  const m = state.metrics;
  const cards = [
    ["Hit Rate@10", pct(m.hit_rate_at_10)],
    ["MRR", fmt(m.mrr, 4)],
    ["MTTC", fmt(m.mttc, 2)],
    ["Efficiency", fmt(m.efficiency, 3)],
    ["TechnicalScore", fmt(m.recommended_technical_score, 5)],
  ];
  $("#metrics").innerHTML = cards.map(([label, value]) => `<div class="metric-card"><span>${label}</span><strong>${value}</strong></div>`).join("");
}

function filteredSessions() {
  const text = $("#searchInput").value.trim().toLowerCase();
  const scenario = $("#scenarioFilter").value;
  const result = $("#resultFilter").value;
  return state.sessions.filter(session => {
    const matchesText = !text || session.sample_id.toLowerCase().includes(text) || session.target_title.toLowerCase().includes(text);
    const matchesScenario = scenario === "all" || session.scenario_type === scenario;
    const matchesResult = result === "all" || (result === "hit" ? session.hit === true : session.hit === false);
    return matchesText && matchesScenario && matchesResult;
  });
}

function renderSessions() {
  const sessions = filteredSessions();
  $("#sessionCount").textContent = `${sessions.length}/${state.sessions.length}`;
  $("#sessionList").innerHTML = sessions.map(session => `
    <button class="session-item ${session.sample_id === state.selectedId ? "active" : ""}" data-id="${esc(session.sample_id)}">
      <div class="session-top"><span>${esc(session.sample_id)}</span><span class="result-dot ${session.hit === true ? "hit" : session.hit === false ? "miss" : ""}"></span></div>
      <div class="session-title">${esc(scenarioName(session.scenario_type))} · ${esc(session.target_title)}</div>
    </button>`).join("") || `<div class="empty-row">No matching sessions</div>`;
  document.querySelectorAll(".session-item").forEach(button => button.addEventListener("click", () => selectSession(button.dataset.id)));
}

async function selectSession(sampleId) {
  const requestedId = sampleId;
  state.selectedId = sampleId;
  state.turnIndex = 0;
  renderSessions();
  $("#emptyState").classList.remove("hidden");
  $("#emptyState").innerHTML = `<div class="loader-ring"></div><h2>Tracing ${esc(sampleId)}</h2><p>Running the current Agent through the public simulator.</p>`;
  $("#traceView").classList.add("hidden");
  try {
    const trace = await getJSON(`/api/trace?sample_id=${encodeURIComponent(sampleId)}`);
    if (state.selectedId !== requestedId) return;
    state.trace = trace;
    $("#emptyState").classList.add("hidden");
    $("#traceView").classList.remove("hidden");
    renderTrace();
  } catch (error) {
    $("#emptyState").innerHTML = `<h2>Trace failed</h2><p>${esc(error.message)}</p>`;
  }
}

function renderTrace() {
  const trace = state.trace;
  const result = trace.result;
  $("#traceEyebrow").textContent = `${scenarioName(trace.scenario_type)} / ${trace.difficulty_bucket || "unrated"}`;
  $("#traceTitle").textContent = trace.sample_id;
  $("#traceSubtitle").textContent = trace.target.title;
  $("#observerNote").textContent = trace.observer_note;
  $("#traceResult").innerHTML = `
    <div class="result-badge">Result <strong>${result.hit ? "HIT" : "MISS"}</strong></div>
    <div class="result-badge">Turn <strong>${result.first_hit_turn ?? "—"}</strong></div>
    <div class="result-badge">Best rank <strong>${result.best_rank ?? "—"}</strong></div>
    <div class="result-badge">Contribution <strong>${fmt(result.technical_contribution, 4)}</strong></div>`;
  renderTurnRail();
  renderTurn();
  renderTarget();
}

function renderTurnRail() {
  $("#turnRail").innerHTML = state.trace.turns.map((turn, index) => `
    <button class="turn-node ${index === state.turnIndex ? "active" : ""} ${turn.event}" data-index="${index}">
      <strong>Turn ${turn.turn}</strong><span>${turn.hit ? `Hit @${turn.target_top10_rank}` : turn.event === "override_next" ? "Override next" : "No hit"}</span>
    </button>`).join("");
  document.querySelectorAll(".turn-node").forEach(button => button.addEventListener("click", () => {
    state.turnIndex = Number(button.dataset.index); renderTurnRail(); renderTurn();
  }));
}

function layerCard(index, title, value, meta, tone = "") {
  return `<article class="layer-card ${tone}"><div class="layer-index">L${index}</div><h3>${esc(title)}</h3><div class="layer-value">${value}</div><div class="layer-meta">${esc(meta)}</div></article>`;
}

function renderTurn() {
  const turn = state.trace.turns[state.turnIndex];
  const retrieval = turn.retrieval;
  $("#turnLabel").textContent = `Turn ${turn.turn} / ${state.trace.turns.length}`;
  $("#prevTurn").disabled = state.turnIndex === 0;
  $("#nextTurn").disabled = state.turnIndex === state.trace.turns.length - 1;
  const terms = retrieval.terms.length ? retrieval.terms.map(term => `<span class="chip">${esc(term)}</span>`).join(" ") : "No searchable terms";
  const disclosed = turn.simulator_disclosed_after.length ? turn.simulator_disclosed_after.map(esc).join(" · ") : "No constraints disclosed";
  const rankTone = turn.target_top10_rank ? "success" : retrieval.target_retrieval_rank ? "" : "alert";
  const scoreText = turn.hit ? `Hit at rank ${turn.target_top10_rank}` : !turn.eligible_for_hit && turn.target_top10_rank ? "Target present, override not active" : "No scored hit";
  $("#layerFlow").innerHTML = [
    layerCard(1, "Input", esc(turn.user_message), `turn ${turn.turn}`),
    layerCard(2, "Parse", terms, `${retrieval.terms.length} unique terms`),
    layerCard(3, "Session", esc(disclosed), "simulator disclosure"),
    layerCard(4, "Retrieval", `${fmt(retrieval.candidate_count, 0)} candidates<br>Target rank: <b>${retrieval.target_retrieval_rank ?? "not found"}</b>`, "full BM25 pool", rankTone),
    layerCard(5, "Ranking", `${turn.valid_recommendation_count} valid outputs<br>Target Top 10: <b>${turn.target_top10_rank ?? "no"}</b>`, "ordered results", turn.target_top10_rank ? "success" : ""),
    layerCard(6, "Policy", `Ask: <b>${esc(turn.ask_attribute ?? "none")}</b><br>${esc(turn.event)}`, "next action"),
    layerCard(7, "Score", esc(scoreText), turn.eligible_for_hit ? "eligible" : "blocked", turn.hit ? "success" : ""),
  ].join("");
  $("#conversation").innerHTML = `
    <div class="bubble user"><span class="bubble-label">SIMULATED USER</span>${esc(turn.user_message)}</div>
    <div class="bubble agent"><span class="bubble-label">CURRENT AGENT</span>${esc(turn.agent_message || "No message")}</div>
    <div class="policy-line">ask_attribute: <b>${esc(turn.ask_attribute ?? "null")}</b>${turn.next_user_message ? `<br>Next simulator reply: ${esc(turn.next_user_message)}` : ""}${turn.error ? `<br><span class="signal">${esc(turn.error)}</span>` : ""}</div>`;
  renderRecommendations(turn);
}

function renderTarget() {
  const target = state.trace.target;
  const card = state.trace.intent_card;
  $("#targetCard").innerHTML = `
    <div class="truth-title">${esc(target.title)}</div>
    <div class="truth-asin">${esc(target.parent_asin)}</div>
    <div class="truth-grid">
      <div><span>Price</span><strong>${target.price == null ? "—" : `$${esc(target.price)}`}</strong></div>
      <div><span>Rating</span><strong>${target.average_rating ?? "—"}</strong></div>
      <div><span>Store</span><strong>${esc(target.store || "—")}</strong></div>
      <div><span>Scenario</span><strong>${esc(scenarioName(state.trace.scenario_type))}</strong></div>
    </div>
    <ul class="intent-list"><li><b>Hard:</b> ${card.hard_constraints.map(esc).join(" · ")}</li><li><b>Soft:</b> ${card.soft_preferences.map(esc).join(" · ")}</li></ul>`;
}

function renderRecommendations(turn) {
  $("#rankSummary").textContent = `${turn.valid_recommendation_count} valid · retrieval target rank ${turn.retrieval.target_retrieval_rank ?? "not found"}`;
  $("#recommendationRows").innerHTML = turn.recommendations.map((product, index) => `
    <tr class="${product.is_target ? "target-row" : ""}">
      <td class="rank-number">${index + 1}</td>
      <td class="product-name">${esc(product.title)}</td>
      <td class="asin">${esc(product.parent_asin)}</td>
      <td>${product.price == null ? "—" : `$${esc(product.price)}`}</td>
      <td>${product.average_rating ?? "—"}</td>
      <td class="signal">${product.is_target ? "TARGET" : ""}</td>
    </tr>`).join("") || `<tr><td colspan="6" class="empty-row">No valid recommendations</td></tr>`;
}

$("#prevTurn").addEventListener("click", () => { if (state.turnIndex > 0) { state.turnIndex--; renderTurnRail(); renderTurn(); } });
$("#nextTurn").addEventListener("click", () => { if (state.turnIndex < state.trace.turns.length - 1) { state.turnIndex++; renderTurnRail(); renderTurn(); } });
[$("#searchInput"), $("#scenarioFilter"), $("#resultFilter")].forEach(element => element.addEventListener("input", renderSessions));

(async function init() {
  try {
    const payload = await getJSON("/api/sessions");
    state.sessions = payload.sessions;
    state.metrics = payload.metrics;
    renderMetrics();
    renderSessions();
    const first = state.sessions.find(session => session.sample_id === "public_0001") || state.sessions[0];
    if (first) selectSession(first.sample_id);
  } catch (error) {
    $("#emptyState").innerHTML = `<h2>Observer failed to start</h2><p>${esc(error.message)}</p>`;
  }
})();
