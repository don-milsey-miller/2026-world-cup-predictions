const form = document.querySelector("#prediction-form");
const teamA = document.querySelector("#team-a");
const teamB = document.querySelector("#team-b");
const tournament = document.querySelector("#tournament");
const matchDate = document.querySelector("#match-date");
const neutral = document.querySelector("#neutral");
const datalist = document.querySelector("#teams");
const results = document.querySelector("#results");

matchDate.value = new Date().toISOString().slice(0, 10);

async function loadTeams() {
  const response = await fetch("/api/teams");
  if (!response.ok) {
    results.innerHTML = `<div class="empty">Model is not trained yet. Run the training script first.</div>`;
    return;
  }
  const data = await response.json();
  datalist.innerHTML = data.teams.map((team) => `<option value="${escapeHtml(team)}"></option>`).join("");
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  results.innerHTML = `<div class="empty">Calculating prediction...</div>`;
  const response = await fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      team_a: teamA.value,
      team_b: teamB.value,
      neutral: neutral.checked,
      match_date: matchDate.value,
      tournament: tournament.value,
    }),
  });
  const data = await response.json();
  if (!response.ok) {
    results.innerHTML = `<div class="empty">${escapeHtml(data.detail || "Prediction failed.")}</div>`;
    return;
  }
  renderPrediction(data);
});

function renderPrediction(data) {
  const title = data.predicted_outcome === "draw" ? "Draw is most likely" : `${data.predicted_winner} leads`;
  results.innerHTML = `
    <h2 class="prediction-title">${escapeHtml(title)}</h2>
    <p class="winner">Winner lean: ${escapeHtml(data.predicted_winner)}</p>
    ${bar(`${teamA.value} win`, data.team_a_win_probability)}
    ${bar("Draw", data.draw_probability)}
    ${bar(`${teamB.value} win`, data.team_b_win_probability)}
    <section class="signals">
      <h2>Top signals</h2>
      <ul>
        ${data.top_feature_signals
          .map((signal) => `<li>${escapeHtml(signal.name)}: ${escapeHtml(String(signal.value))}</li>`)
          .join("")}
      </ul>
    </section>
  `;
}

function bar(label, value) {
  const pct = Math.round(value * 1000) / 10;
  return `
    <div class="bar-row">
      <div class="bar-head"><span>${escapeHtml(label)}</span><span>${pct}%</span></div>
      <div class="bar-track"><div class="bar-fill" style="width:${pct}%"></div></div>
    </div>
  `;
}

function escapeHtml(value) {
  return value.replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;",
  })[char]);
}

loadTeams();
