// ============================================================================
// Vitalis — Patient Risk Analysis dashboard logic
// ============================================================================

const RISK_COLORS = { Low: "#2f7a53", Medium: "#b8792f", High: "#b0402f" };
const GAUGE_CIRCUMFERENCE = 2 * Math.PI * 60; // r=60

// ---- Navigation -------------------------------------------------------
document.querySelectorAll(".nav-item").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".nav-item").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".view").forEach((v) => v.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(`view-${btn.dataset.view}`).classList.add("active");

    if (btn.dataset.view === "records") loadRecords();
    if (btn.dataset.view === "about") loadImportance();
  });
});

// ---- Form submission ----------------------------------------------------
const form = document.getElementById("risk-form");
const submitBtn = document.getElementById("submit-btn");

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  submitBtn.disabled = true;
  submitBtn.textContent = "Analyzing...";

  const fd = new FormData(form);
  const payload = {
    name: fd.get("name") || "Unnamed Patient",
    age: Number(fd.get("age")),
    gender: fd.get("gender"),
    bmi: Number(fd.get("bmi")),
    systolic_bp: Number(fd.get("systolic_bp")),
    diastolic_bp: Number(fd.get("diastolic_bp")),
    glucose: Number(fd.get("glucose")),
    cholesterol: Number(fd.get("cholesterol")),
    heart_rate: Number(fd.get("heart_rate")),
    smoking: fd.get("smoking") ? 1 : 0,
    alcohol_consumption: fd.get("alcohol_consumption") ? 1 : 0,
    physical_activity: fd.get("physical_activity"),
    family_history: fd.get("family_history") ? 1 : 0,
    sleep_hours: Number(fd.get("sleep_hours")),
    stress_level: Number(fd.get("stress_level")),
  };

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    renderResult(data);
    loadStats();
  } catch (err) {
    alert("Prediction failed: " + err.message);
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Run risk analysis";
  }
});

function renderResult(data) {
  document.getElementById("result-empty").style.display = "none";
  const card = document.getElementById("result-card");
  card.classList.remove("hidden");

  // Gauge
  const score = data.risk_score;
  const offset = GAUGE_CIRCUMFERENCE - (score / 100) * GAUGE_CIRCUMFERENCE;
  const fill = document.getElementById("gauge-fill");
  fill.style.stroke = RISK_COLORS[data.risk_level];
  requestAnimationFrame(() => { fill.style.strokeDashoffset = offset; });
  document.getElementById("gauge-score").textContent = score;

  // Pill
  const pill = document.getElementById("risk-pill");
  pill.textContent = `${data.risk_level} risk`;
  pill.className = `risk-pill ${data.risk_level}`;

  // Probability bars
  const probaContainer = document.getElementById("proba-bars");
  probaContainer.innerHTML = "";
  ["Low", "Medium", "High"].forEach((level) => {
    const pct = data.risk_probabilities[level] ?? 0;
    const row = document.createElement("div");
    row.className = "proba-row";
    row.innerHTML = `
      <span>${level}</span>
      <div class="proba-track"><div class="proba-fill ${level}" style="width:${pct}%"></div></div>
      <span>${pct}%</span>
    `;
    probaContainer.appendChild(row);
  });

  // Sub-risks
  document.getElementById("diabetes-value").textContent = data.diabetes_risk + "%";
  document.getElementById("diabetes-bar").style.width = data.diabetes_risk + "%";
  document.getElementById("heart-value").textContent = data.heart_disease_risk + "%";
  document.getElementById("heart-bar").style.width = data.heart_disease_risk + "%";

  // Recommendations
  const list = document.getElementById("recommendations-list");
  list.innerHTML = "";
  data.recommendations.forEach((r) => {
    const li = document.createElement("li");
    li.textContent = r;
    list.appendChild(li);
  });
}

// ---- Records table ----------------------------------------------------
async function loadRecords() {
  const res = await fetch("/api/patients");
  const rows = await res.json();
  const tbody = document.getElementById("records-body");
  const emptyMsg = document.getElementById("records-empty");
  tbody.innerHTML = "";

  if (rows.length === 0) {
    emptyMsg.style.display = "block";
    return;
  }
  emptyMsg.style.display = "none";

  rows.forEach((r) => {
    const tr = document.createElement("tr");
    const date = new Date(r.created_at).toLocaleString();
    tr.innerHTML = `
      <td>${r.name}</td>
      <td>${r.age}</td>
      <td>${r.gender}</td>
      <td>${r.bmi}</td>
      <td>${r.systolic_bp}/${r.diastolic_bp}</td>
      <td>${r.glucose}</td>
      <td><span class="badge ${r.risk_level}">${r.risk_level}</span></td>
      <td>${r.risk_score}</td>
      <td>${r.diabetes_risk}%</td>
      <td>${r.heart_disease_risk}%</td>
      <td>${date}</td>
      <td><button class="del-btn" data-id="${r.id}" title="Delete">✕</button></td>
    `;
    tbody.appendChild(tr);
  });

  document.querySelectorAll(".del-btn").forEach((btn) => {
    btn.addEventListener("click", async () => {
      await fetch(`/api/patients/${btn.dataset.id}`, { method: "DELETE" });
      loadRecords();
      loadStats();
    });
  });
}

// ---- Sidebar stats ------------------------------------------------------
async function loadStats() {
  const res = await fetch("/api/stats");
  const data = await res.json();
  document.getElementById("sidebar-total").textContent = data.total;

  const total = data.total || 1;
  document.getElementById("bar-low").style.width = (data.counts.Low / total * 100) + "%";
  document.getElementById("bar-medium").style.width = (data.counts.Medium / total * 100) + "%";
  document.getElementById("bar-high").style.width = (data.counts.High / total * 100) + "%";
}

// ---- About: feature importance ------------------------------------------
const FEATURE_IMPORTANCE = [
  ["Age", 0.2112],
  ["Systolic BP", 0.1686],
  ["BMI", 0.1248],
  ["Glucose", 0.1106],
  ["Cholesterol", 0.0845],
  ["Diastolic BP", 0.0672],
  ["Smoking", 0.0524],
  ["Family history", 0.0489],
  ["Heart rate", 0.0365],
  ["Sleep hours", 0.0358],
  ["Stress level", 0.0251],
  ["Activity level", 0.018],
  ["Alcohol use", 0.0109],
  ["Gender", 0.0053],
];

function loadImportance() {
  const el = document.getElementById("importance-chart");
  if (el.dataset.loaded) return;
  el.dataset.loaded = "1";
  const max = Math.max(...FEATURE_IMPORTANCE.map((f) => f[1]));
  FEATURE_IMPORTANCE.forEach(([name, val]) => {
    const row = document.createElement("div");
    row.className = "imp-row";
    row.innerHTML = `
      <span>${name}</span>
      <div class="imp-track"><div class="imp-fill" style="width:${(val / max * 100).toFixed(1)}%"></div></div>
      <span class="imp-value">${(val * 100).toFixed(1)}%</span>
    `;
    el.appendChild(row);
  });
}

// Initial load
loadStats();
