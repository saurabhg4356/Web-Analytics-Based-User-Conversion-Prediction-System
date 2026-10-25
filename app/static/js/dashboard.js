/**
 * app/static/js/dashboard.js
 * Interactive Chart.js visualizer and dynamic asynchronous filter controller.
 */

let outcomeChartInstance = null;
let monthlyChartInstance = null;
let visitorChartInstance = null;
let trafficChartInstance = null;
let weekendChartInstance = null;

document.addEventListener("DOMContentLoaded", function () {
  initCharts();
  bindFilterEvents();
});

function initCharts() {
  const chartDataEl = document.getElementById("initialChartData");
  if (!chartDataEl) return;

  try {
    const data = JSON.parse(chartDataEl.textContent);
    renderOutcomeChart(data.outcome_distribution);
    renderMonthlyChart(data.monthly_conversion);
    renderVisitorChart(data.visitor_conversion);
    renderTrafficChart(data.traffic_conversion);
    renderWeekendChart(data.weekend_conversion);
  } catch (err) {
    console.error("Error parsing dashboard initial chart data:", err);
  }
}

function renderOutcomeChart(dist) {
  const ctx = document.getElementById("outcomeChart");
  if (!ctx) return;
  if (outcomeChartInstance) outcomeChartInstance.destroy();

  outcomeChartInstance = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: dist.labels,
      datasets: [{
        data: dist.counts,
        backgroundColor: ["#94a3b8", "#0284c7"],
        borderWidth: 2,
        borderColor: "#ffffff"
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: "bottom", labels: { boxWidth: 12, font: { size: 11 } } },
        tooltip: {
          callbacks: {
            label: function (context) {
              const total = context.dataset.data.reduce((a, b) => a + b, 0);
              const val = context.raw;
              const pct = ((val / total) * 100).toFixed(1);
              return ` ${context.label}: ${val.toLocaleString()} (${pct}%)`;
            }
          }
        }
      },
      cutout: "68%"
    }
  });
}

function renderMonthlyChart(monthly) {
  const ctx = document.getElementById("monthlyChart");
  if (!ctx) return;
  if (monthlyChartInstance) monthlyChartInstance.destroy();

  monthlyChartInstance = new Chart(ctx, {
    type: "line",
    data: {
      labels: monthly.labels,
      datasets: [{
        label: "Conversion Rate (%)",
        data: monthly.conversion_rates,
        borderColor: "#0284c7",
        backgroundColor: "rgba(2, 132, 199, 0.08)",
        fill: true,
        tension: 0.35,
        pointBackgroundColor: "#0284c7",
        pointBorderColor: "#ffffff",
        pointBorderWidth: 2,
        pointRadius: 4,
        pointHoverRadius: 6
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          beginAtZero: true,
          ticks: { callback: (val) => val + "%" },
          grid: { color: "#f1f5f9" }
        },
        x: { grid: { display: false } }
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => ` Conversion Rate: ${ctx.raw}%`
          }
        }
      }
    }
  });
}

function renderVisitorChart(vis) {
  const ctx = document.getElementById("visitorChart");
  if (!ctx) return;
  if (visitorChartInstance) visitorChartInstance.destroy();

  visitorChartInstance = new Chart(ctx, {
    type: "bar",
    data: {
      labels: vis.labels,
      datasets: [{
        label: "Conversion Rate (%)",
        data: vis.conversion_rates,
        backgroundColor: ["#0284c7", "#38bdf8", "#94a3b8"],
        borderRadius: 6
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          beginAtZero: true,
          ticks: { callback: (val) => val + "%" },
          grid: { color: "#f1f5f9" }
        },
        x: { grid: { display: false } }
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => ` Conversion Rate: ${ctx.raw}%`
          }
        }
      }
    }
  });
}

function renderTrafficChart(traffic) {
  const ctx = document.getElementById("trafficChart");
  if (!ctx) return;
  if (trafficChartInstance) trafficChartInstance.destroy();

  trafficChartInstance = new Chart(ctx, {
    type: "bar",
    data: {
      labels: traffic.labels,
      datasets: [{
        label: "Conversion Rate (%)",
        data: traffic.conversion_rates,
        backgroundColor: "#0d9488",
        borderRadius: 4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      indexAxis: "y",
      scales: {
        x: {
          beginAtZero: true,
          ticks: { callback: (val) => val + "%" },
          grid: { color: "#f1f5f9" }
        },
        y: { grid: { display: false } }
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => ` Conversion: ${ctx.raw}%`
          }
        }
      }
    }
  });
}

function renderWeekendChart(weekend) {
  const ctx = document.getElementById("weekendChart");
  if (!ctx) return;
  if (weekendChartInstance) weekendChartInstance.destroy();

  weekendChartInstance = new Chart(ctx, {
    type: "bar",
    data: {
      labels: weekend.labels,
      datasets: [{
        label: "Conversion Rate (%)",
        data: weekend.conversion_rates,
        backgroundColor: ["#475569", "#0284c7"],
        borderRadius: 6
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          beginAtZero: true,
          ticks: { callback: (val) => val + "%" },
          grid: { color: "#f1f5f9" }
        },
        x: { grid: { display: false } }
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => ` Conversion: ${ctx.raw}%`
          }
        }
      }
    }
  });
}

function bindFilterEvents() {
  const filterForm = document.getElementById("dashboardFilterForm");
  if (!filterForm) return;

  const selects = filterForm.querySelectorAll("select");
  selects.forEach((sel) => {
    sel.addEventListener("change", applyFiltersAsync);
  });

  const resetBtn = document.getElementById("resetFiltersBtn");
  if (resetBtn) {
    resetBtn.addEventListener("click", function () {
      selects.forEach((sel) => (sel.value = "all"));
      applyFiltersAsync();
    });
  }
}

async function applyFiltersAsync() {
  const month = document.getElementById("filterMonth")?.value || "all";
  const visitor = document.getElementById("filterVisitor")?.value || "all";
  const traffic = document.getElementById("filterTraffic")?.value || "all";
  const weekend = document.getElementById("filterWeekend")?.value || "all";

  const url = `/api/analytics/summary?month=${encodeURIComponent(month)}&visitor_type=${encodeURIComponent(visitor)}&traffic_type=${encodeURIComponent(traffic)}&weekend=${encodeURIComponent(weekend)}`;

  // Show subtle loading state
  const loadingIndicator = document.getElementById("dashboardLoadingState");
  if (loadingIndicator) loadingIndicator.classList.remove("d-none");

  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP error ${res.status}`);
    const data = await res.json();

    // Update KPI Counters
    updateKpiText("kpiTotalSessions", data.kpis.total_sessions.toLocaleString());
    updateKpiText("kpiConversions", data.kpis.conversions.toLocaleString());
    updateKpiText("kpiConversionRate", data.kpis.conversion_rate.toFixed(2) + "%");
    updateKpiText("kpiNonConversions", data.kpis.non_conversions.toLocaleString());
    updateKpiText("kpiReturningVisitors", data.kpis.returning_visitors.toLocaleString());
    updateKpiText("kpiNewVisitors", data.kpis.new_visitors.toLocaleString());
    updateKpiText("kpiAvgDuration", data.kpis.avg_product_duration_secs.toFixed(0) + "s");

    // Update Charts
    renderOutcomeChart(data.charts.outcome_distribution);
    renderMonthlyChart(data.charts.monthly_conversion);
    renderVisitorChart(data.charts.visitor_conversion);
    renderTrafficChart(data.charts.traffic_conversion);
    renderWeekendChart(data.charts.weekend_conversion);
  } catch (err) {
    console.error("Failed to update dashboard filters:", err);
  } finally {
    if (loadingIndicator) loadingIndicator.classList.add("d-none");
  }
}

function updateKpiText(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}
