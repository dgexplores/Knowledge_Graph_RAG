const status = document.querySelector("#status");
const statusDot = document.querySelector(".status-dot");
const loadButton = document.querySelector("#load-demo");
const scenarios = document.querySelector("#scenarios");
const askButton = document.querySelector("#ask-button");
const question = document.querySelector("#question");
const form = document.querySelector("#ask-form");
const result = document.querySelector("#result");
const steps = { load: document.querySelector("#step-load"), ask: document.querySelector("#step-ask"), proof: document.querySelector("#step-proof") };

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, char => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#039;", '"': "&quot;" })[char]);
}
function setStatus(message, state = "") { status.textContent = message; statusDot.className = `status-dot ${state}`; }
function setStep(active) {
  const names = ["load", "ask", "proof"];
  names.forEach((name, index) => {
    steps[name].classList.toggle("step-current", name === active);
    steps[name].classList.toggle("step-complete", index < names.indexOf(active));
  });
}
function explainRoute(route) {
  if (route === "graph") return "This needs a relationship or count, so Atlas used an allowlisted graph traversal.";
  if (route === "definition" || route === "vector") return "This is passage-based, so Atlas searched the filing text for supporting context.";
  if (route === "out_of_scope") return "This is outside the filing corpus, so Atlas refused rather than inventing an answer.";
  return "Atlas combined retrieval paths and checked the evidence before responding.";
}
function renderResult(data) {
  const citations = data.citations.length ? `<ul class="citation-list">${data.citations.map(id => `<li>${escapeHtml(id)}</li>`).join("")}</ul>` : "<p class=\"helper\">No citations are shown because no unsupported answer was returned.</p>";
  const evidence = { route: data.route, query_type: data.query_type, generator: data.generator, latency_ms: data.latency_ms, template_params: data.params, graph_records: data.graph_records, retrieved_passages: data.passages, citation_report: data.citation_report };
  result.classList.remove("empty", "loading");
  result.innerHTML = `
    <p class="result-kicker">${data.refused ? "SAFE REFUSAL" : "EVIDENCE-BACKED RESULT"}</p>
    <h3 class="result-title">${data.refused ? "No answer by design." : "Here is what the corpus supports."}</h3>
    <p class="result-answer">${data.refused ? "Atlas did not produce an answer." : escapeHtml(data.answer)}</p>
    ${data.refused ? `<p class="result-reason"><strong>Why:</strong> ${escapeHtml(data.reason)}</p>` : ""}
    <div class="decision-strip" aria-label="How the answer was produced"><div><span>Decision</span><strong>${escapeHtml(data.route.replaceAll("_", " "))}</strong></div><div><span>Query type</span><strong>${escapeHtml(data.query_type || "Not applicable")}</strong></div><div><span>Latency</span><strong>${escapeHtml(String(data.latency_ms))} ms</strong></div></div>
    <p class="helper"><strong>What happened:</strong> ${escapeHtml(explainRoute(data.route))}</p>
    <details open><summary>View source citations (${data.citations.length})</summary>${citations}</details>
    <details><summary>View technical retrieval evidence</summary><pre>${escapeHtml(JSON.stringify(evidence, null, 2))}</pre></details>`;
  result.focus(); setStep("proof");
}
async function checkHealth() {
  try { const response = await fetch("/health"); const data = await response.json();
    if (data.status === "ok") setStatus("Services are ready. Load the local demo corpus to continue.", "ready");
    else setStatus("The retrieval services are not ready. Start Docker Compose, then refresh.", "error");
  } catch { setStatus("Could not reach the API. Start the FastAPI service, then refresh.", "error"); }
}
loadButton.addEventListener("click", async () => {
  loadButton.disabled = true; setStatus("Loading filings and building the graph and vector index…");
  try { const response = await fetch("/demo/load"); if (!response.ok) throw new Error("Ingestion failed"); const data = await response.json();
    scenarios.disabled = false; askButton.disabled = false; setStatus(`Ready: ${data.documents} filings, ${data.chunks} source chunks, and ${data.edges_written} relationships loaded.`, "ready"); setStep("ask"); question.focus();
  } catch { setStatus("Could not load the corpus. Check Neo4j and pgvector, then try again.", "error"); } finally { loadButton.disabled = false; }
});
document.querySelectorAll("[data-question]").forEach(button => button.addEventListener("click", () => { question.value = button.dataset.question; question.focus(); }));
form.addEventListener("submit", async event => {
  event.preventDefault(); const prompt = question.value.trim(); if (!prompt) return;
  askButton.disabled = true; askButton.textContent = "Checking evidence…"; setStatus("Classifying the question and retrieving supporting evidence…");
  result.classList.remove("empty");
  result.classList.add("loading");
  result.innerHTML = "<div class=\"loading-state\" aria-busy=\"true\" aria-label=\"Retrieving evidence\"><p class=\"card-eyebrow\">CHECKING EVIDENCE</p><div class=\"loading-line short\"></div><div class=\"loading-line\"></div><div class=\"loading-line medium\"></div></div>";
  try { const response = await fetch(`/demo/ask?question=${encodeURIComponent(prompt)}`); const data = await response.json(); if (!response.ok) throw new Error(data.detail || "Request failed"); renderResult(data); setStatus("Result ready. Inspect the route and citations in the proof panel.", "ready");
  } catch (error) { result.classList.remove("empty", "loading"); result.innerHTML = `<p class="result-reason"><strong>Request failed:</strong> ${escapeHtml(error.message || "Please try again.")}</p>`; result.focus(); setStatus("The request could not be completed. Check the local services and retry.", "error");
  } finally { askButton.disabled = false; askButton.innerHTML = "Run question <span aria-hidden=\"true\">→</span>"; }
});
checkHealth();
