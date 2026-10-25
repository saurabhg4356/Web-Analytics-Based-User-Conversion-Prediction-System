/**
 * app/static/js/prediction.js
 * Form automation presets, threshold indicator, and client validation helper.
 */

const PRESETS = {
  high_intent: {
    administrative: 3,
    administrative_duration: 85.0,
    informational: 1,
    informational_duration: 35.0,
    product_related: 32,
    product_related_duration: 1250.0,
    bounce_rates: 0.002,
    exit_rates: 0.015,
    special_day: 0.0,
    month: "Nov",
    operating_systems: "2",
    browser: "2",
    region: "1",
    traffic_type: "2",
    visitor_type: "Returning_Visitor",
    weekend: "False",
    decision_threshold: 0.40,
    session_summary: "High-Intent Comparison Shopper (Cart Review)"
  },
  window_shopper: {
    administrative: 0,
    administrative_duration: 0.0,
    informational: 0,
    informational_duration: 0.0,
    product_related: 4,
    product_related_duration: 65.0,
    bounce_rates: 0.035,
    exit_rates: 0.075,
    special_day: 0.0,
    month: "May",
    operating_systems: "1",
    browser: "1",
    region: "3",
    traffic_type: "3",
    visitor_type: "Returning_Visitor",
    weekend: "False",
    decision_threshold: 0.40,
    session_summary: "Casual Window Shopper (Brief Browse)"
  },
  immediate_bounce: {
    administrative: 0,
    administrative_duration: 0.0,
    informational: 0,
    informational_duration: 0.0,
    product_related: 1,
    product_related_duration: 0.0,
    bounce_rates: 0.20,
    exit_rates: 0.20,
    special_day: 0.0,
    month: "Feb",
    operating_systems: "3",
    browser: "2",
    region: "2",
    traffic_type: "1",
    visitor_type: "Returning_Visitor",
    weekend: "False",
    decision_threshold: 0.40,
    session_summary: "Immediate Bounce (Search Ad Misclick)"
  },
  holiday_deal_hunter: {
    administrative: 2,
    administrative_duration: 48.0,
    informational: 0,
    informational_duration: 0.0,
    product_related: 22,
    product_related_duration: 890.0,
    bounce_rates: 0.005,
    exit_rates: 0.022,
    special_day: 0.6,
    month: "Dec",
    operating_systems: "2",
    browser: "2",
    region: "1",
    traffic_type: "2",
    visitor_type: "New_Visitor",
    weekend: "True",
    decision_threshold: 0.40,
    session_summary: "Holiday Promo Campaign Hunter (Weekend Deal)"
  }
};

document.addEventListener("DOMContentLoaded", function () {
  bindPresetButtons();
  bindThresholdSlider();
  bindFormSubmitLoading();
});

function bindPresetButtons() {
  const presetButtons = document.querySelectorAll("[data-preset]");
  presetButtons.forEach((btn) => {
    btn.addEventListener("click", function () {
      const presetKey = this.getAttribute("data-preset");
      const config = PRESETS[presetKey];
      if (!config) return;

      setFieldValue("administrative", config.administrative);
      setFieldValue("administrative_duration", config.administrative_duration);
      setFieldValue("informational", config.informational);
      setFieldValue("informational_duration", config.informational_duration);
      setFieldValue("product_related", config.product_related);
      setFieldValue("product_related_duration", config.product_related_duration);
      setFieldValue("bounce_rates", config.bounce_rates);
      setFieldValue("exit_rates", config.exit_rates);
      setFieldValue("special_day", config.special_day);
      setFieldValue("month", config.month);
      setFieldValue("operating_systems", config.operating_systems);
      setFieldValue("browser", config.browser);
      setFieldValue("region", config.region);
      setFieldValue("traffic_type", config.traffic_type);
      setFieldValue("visitor_type", config.visitor_type);
      setFieldValue("weekend", config.weekend);
      setFieldValue("decision_threshold", config.decision_threshold);
      setFieldValue("session_summary", config.session_summary);

      updateThresholdDisplay(config.decision_threshold);

      // Flash feedback border
      const formCard = document.getElementById("predictionFormCard");
      if (formCard) {
        formCard.classList.add("border-primary");
        setTimeout(() => formCard.classList.remove("border-primary"), 600);
      }
    });
  });
}

function setFieldValue(name, value) {
  const field = document.querySelector(`[name="${name}"]`);
  if (field) {
    field.value = value;
  }
}

function bindThresholdSlider() {
  const thresholdInput = document.querySelector('[name="decision_threshold"]');
  if (thresholdInput) {
    thresholdInput.addEventListener("input", function () {
      updateThresholdDisplay(this.value);
    });
    updateThresholdDisplay(thresholdInput.value);
  }
}

function updateThresholdDisplay(val) {
  const display = document.getElementById("thresholdDisplayVal");
  if (display) {
    const num = parseFloat(val);
    display.textContent = isNaN(num) ? "40%" : `${(num * 100).toFixed(0)}%`;
  }
}

function bindFormSubmitLoading() {
  const form = document.querySelector("form");
  const submitBtn = document.getElementById("predictSubmitBtn");
  if (form && submitBtn) {
    form.addEventListener("submit", function () {
      submitBtn.disabled = true;
      submitBtn.innerHTML = `<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>Analyzing Session Behavior...`;
    });
  }
}
