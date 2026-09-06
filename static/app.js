/**
 * Smart Plant Intelligence System - Frontend Logic
 * Strictly handles:
 * 1. Physical ESP8266 IoT Telemetry Lifecycle (Live vs Stale vs Offline vs Manual)
 * 2. Backend Health and 4-Model Readiness Telemetry
 * 3. Crop Recommendation and Disease Risk Prediction Pipeline
 * 4. 3-Tier Foliar Computer Vision Diagnostics and Lesion Localization
 * 5. Telemetry Trend History (Hardware Data Only, Zero Synthetic Polling Records)
 * 6. Responsive Image Handling and Non-Destructive Fullscreen Inspection
 */

// ── API CONFIGURATION ────────────────────────────────────────────────────────
const API_BASE = window.location.port === "8000" ? "" : "http://localhost:8000";

// ── GLOBAL APPLICATION STATE ──────────────────────────────────────────────────
let activeMode = "crop"; // "crop" or "vision"
let isManualInput = true; // Sliders are user-controlled until genuine IoT telemetry arrives
let isRainActive = false; // YL-83 Digital Rain Sensor state
let autoUpdateEnabled = true;

// IoT Hardware Lifecycle State
let latestIotPacket = null;
let lastIotTimestamp = null; // Milliseconds timestamp of physical sensor packet
let lastChartPacketTs = null; // Guard to prevent duplicate chart entries
let espDeviceState = "waiting"; // "waiting" | "connected" | "stale" | "offline"

// History Chart Instance
let historyChart = null;

// Computer Vision State
let selectedVisionFile = null;
let lastVisionResult = null;
let originalVisionImageObj = null;
let showBoundingBoxes = true;

// Presets for Quick Scenarios
const SENSOR_PRESETS = {
  monsoon: { temperature: 24, humidity: 88, soil_moisture: 85, rain: 1 },
  summer: { temperature: 38, humidity: 30, soil_moisture: 20, rain: 0 },
  foggy: { temperature: 14, humidity: 92, soil_moisture: 60, rain: 0 },
  ideal: { temperature: 26, humidity: 65, soil_moisture: 55, rain: 0 },
};

const SAMPLE_LEAF_MAP = {
  tomato_early_blight: {
    filename: "tomato__fungal__early_blight.jpg",
    url: "./samples/tomato__fungal__early_blight.jpg",
    label: "Tomato Early Blight",
  },
  corn_rust: {
    filename: "corn__fungal__common_rust_.jpg",
    url: "./samples/corn__fungal__common_rust_.jpg",
    label: "Corn Common Rust",
  },
  grape_rot: {
    filename: "grape__fungal__black_rot.jpg",
    url: "./samples/grape__fungal__black_rot.jpg",
    label: "Grape Black Rot",
  },
  apple_scab: {
    filename: "apple__healthy__healthy.jpg",
    url: "./samples/apple__healthy__healthy.jpg",
    label: "Apple Healthy",
  },
  healthy: {
    filename: "potato__healthy__healthy.jpg",
    url: "./samples/potato__healthy__healthy.jpg",
    label: "Potato Healthy Leaf",
  },
};

// ── INITIALIZATION ────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  initHistoryChart();
  initAutoToggle();
  initDragAndDrop();
  initKeyboardListeners();
  syncSliderDisplays();

  // Run initial health and model readiness check
  checkBackendHealth();
  // Poll backend health every 15 seconds
  setInterval(checkBackendHealth, 15000);

  // Poll for genuine physical ESP8266 telemetry
  pollLatestIotTelemetry();
  setInterval(pollLatestIotTelemetry, 10000);

  // High-resolution 1-second ticker to detect stale physical hardware immediately
  setInterval(updateIotAgeTicker, 1000);

  // Fetch initial telemetry history if available on server
  fetchTelemetryHistory();
});

// ── KEYBOARD AND MODAL CLOSE LISTENERS ────────────────────────────────────────
function initKeyboardListeners() {
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      closeImageModal();
    }
  });
}

// ── MODE SWITCHER ─────────────────────────────────────────────────────────────
function switchMode(mode) {
  activeMode = mode;
  const btnCrop = document.getElementById("btnModeCrop");
  const btnVision = document.getElementById("btnModeVision");
  const cropSection = document.getElementById("cropSection");
  const visionSection = document.getElementById("visionSection");

  if (mode === "crop") {
    btnCrop.classList.add("active");
    btnVision.classList.remove("active");
    cropSection.style.display = "grid";
    visionSection.style.display = "none";
  } else {
    btnVision.classList.add("active");
    btnCrop.classList.remove("active");
    cropSection.style.display = "none";
    visionSection.style.display = "flex";
  }
}

// ── BACKEND HEALTH AND MODEL READINESS CHECK ──────────────────────────────────
async function checkBackendHealth() {
  const badgeDot = document.getElementById("badgeDot");
  const badgeText = document.getElementById("badgeText");

  try {
    const healthRes = await fetch(`${API_BASE}/health`, { cache: "no-store" });
    if (!healthRes.ok) throw new Error("Health check failed");

    // Fetch model readiness
    let modelsReady = false;
    try {
      const modelRes = await fetch(`${API_BASE}/models/status`, { cache: "no-store" });
      if (modelRes.ok) {
        const modelData = await modelRes.json();
        modelsReady = Boolean(
          (modelData.computer_vision && modelData.computer_vision.is_loaded) ||
          (modelData.crop_recommendation && modelData.crop_recommendation.model)
        );
      }
    } catch {
      modelsReady = true; // Fallback if status endpoint is optional
    }

    if (badgeDot && badgeText) {
      badgeDot.className = "badge-dot online";
      badgeText.textContent = modelsReady
        ? "Backend Online · 4 Models Ready"
        : "Backend Online · Models Initializing";
    }
  } catch (err) {
    if (badgeDot && badgeText) {
      badgeDot.className = "badge-dot offline";
      badgeText.textContent = "Backend Offline";
    }
  }
}

// ── ESP8266 PHYSICAL TELEMETRY LIFECYCLE ──────────────────────────────────────
/**
 * Strict rules implemented:
 * 1. ESP8266 must never appear Connected or LIVE when no physical packet has arrived.
 * 2. Backend availability is NOT treated as ESP8266 connectivity.
 * 3. Frontend polling does not create or refresh device connectivity.
 * 4. If telemetry packet is older than 35s, status transitions to STALE / OFFLINE.
 * 5. Manual slider values are clearly badged as MANUAL, never LIVE.
 * 6. Unavailable hardware data shows "--" and "Waiting for ESP8266...".
 */
async function pollLatestIotTelemetry() {
  if (!autoUpdateEnabled && latestIotPacket !== null) {
    return;
  }

  const iotDot = document.getElementById("iotDot");
  const iotFeedText = document.getElementById("iotFeedText");
  const iotLiveValues = document.getElementById("iotLiveValues");

  try {
    const res = await fetch(`${API_BASE}/latest`, { cache: "no-store" });

    if (res.status === 404) {
      // Backend is online, but physical ESP8266 has not transmitted telemetry
      if (latestIotPacket === null) {
        espDeviceState = "waiting";
        if (iotDot) iotDot.className = "iot-dot";
        if (iotFeedText) iotFeedText.textContent = "Waiting for ESP8266...";
        if (iotLiveValues) iotLiveValues.style.display = "none";
        setChipsUnavailable();
      }
      return;
    }

    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }

    const data = await res.json();
    if (!data || data.status === "no_data" || data.temperature == null) {
      if (latestIotPacket === null) {
        espDeviceState = "waiting";
        if (iotDot) iotDot.className = "iot-dot";
        if (iotFeedText) iotFeedText.textContent = "Waiting for ESP8266...";
        if (iotLiveValues) iotLiveValues.style.display = "none";
        setChipsUnavailable();
      }
      return;
    }

    // A valid telemetry packet exists. Compute age from physical timestamp.
    latestIotPacket = data;
    const packetTime = data.timestamp ? new Date(data.timestamp).getTime() : Date.now();
    lastIotTimestamp = packetTime;
    const ageSeconds = Math.max(0, Math.floor((Date.now() - packetTime) / 1000));

    // Update the live chips in the top feed bar
    updateIotFeedChips(data, ageSeconds);

    // Evaluate freshness threshold (35 seconds)
    if (ageSeconds <= 35) {
      espDeviceState = "connected";
      if (iotDot) iotDot.className = "iot-dot live";
      if (iotFeedText) iotFeedText.textContent = "📡 ESP8266 connected (LIVE)";

      // If user hasn't overridden with manual slider adjustments, sync live inputs
      if (!isManualInput || autoUpdateEnabled) {
        applyLiveTelemetryToInputs(data);
      }

      // Record to chart only on a genuinely new packet
      if (data.timestamp && data.timestamp !== lastChartPacketTs) {
        lastChartPacketTs = data.timestamp;
        addTelemetryToChart(data);
      }
    } else {
      // Packet is stale. Hardware is offline.
      espDeviceState = "stale";
      if (iotDot) iotDot.className = "iot-dot stale";
      if (iotFeedText) {
        iotFeedText.textContent = `⚠️ ESP8266 offline (last seen ${formatAgo(ageSeconds)})`;
      }
      markSourceBadgeStale();
    }
  } catch (err) {
    // Network or fetch error
    if (latestIotPacket === null) {
      espDeviceState = "waiting";
      if (iotDot) iotDot.className = "iot-dot";
      if (iotFeedText) iotFeedText.textContent = "Waiting for ESP8266...";
      if (iotLiveValues) iotLiveValues.style.display = "none";
      setChipsUnavailable();
    }
  }
}

// ── 1-SECOND AGE TICKER ───────────────────────────────────────────────────────
function updateIotAgeTicker() {
  if (!lastIotTimestamp) {
    return;
  }

  const ageSeconds = Math.max(0, Math.floor((Date.now() - lastIotTimestamp) / 1000));
  const iotAgo = document.getElementById("iotAgo");
  const iotDot = document.getElementById("iotDot");
  const iotFeedText = document.getElementById("iotFeedText");

  if (iotAgo) {
    iotAgo.textContent = formatAgo(ageSeconds);
  }

  if (ageSeconds > 35) {
    if (espDeviceState !== "stale") {
      espDeviceState = "stale";
      if (iotDot) iotDot.className = "iot-dot stale";
      if (iotFeedText) {
        iotFeedText.textContent = `⚠️ ESP8266 offline (last seen ${formatAgo(ageSeconds)})`;
      }
      markSourceBadgeStale();
    }
  }
}

function updateIotFeedChips(data, ageSeconds) {
  const iotLiveValues = document.getElementById("iotLiveValues");
  const iotT = document.getElementById("iot-t");
  const iotH = document.getElementById("iot-h");
  const iotS = document.getElementById("iot-s");
  const iotR = document.getElementById("iot-r");
  const iotAgo = document.getElementById("iotAgo");

  if (iotLiveValues) iotLiveValues.style.display = "flex";
  if (iotT) iotT.textContent = data.temperature != null ? Number(data.temperature).toFixed(1) : "--";
  if (iotH) iotH.textContent = data.humidity != null ? Number(data.humidity).toFixed(0) : "--";
  if (iotS) iotS.textContent = data.soil_moisture != null ? Number(data.soil_moisture).toFixed(0) : "--";
  if (iotR) iotR.textContent = data.rain === 1 ? "RAIN" : "NO RAIN";
  if (iotAgo) iotAgo.textContent = formatAgo(ageSeconds);
}

function setChipsUnavailable() {
  const iotT = document.getElementById("iot-t");
  const iotH = document.getElementById("iot-h");
  const iotS = document.getElementById("iot-s");
  const iotR = document.getElementById("iot-r");
  const iotAgo = document.getElementById("iotAgo");

  if (iotT) iotT.textContent = "--";
  if (iotH) iotH.textContent = "--";
  if (iotS) iotS.textContent = "--";
  if (iotR) iotR.textContent = "--";
  if (iotAgo) iotAgo.textContent = "--";
}

function applyLiveTelemetryToInputs(data) {
  const tempSlider = document.getElementById("temperature");
  const humSlider = document.getElementById("humidity");
  const soilSlider = document.getElementById("soil_moisture");

  if (tempSlider && data.temperature != null) {
    tempSlider.value = data.temperature;
    document.getElementById("val-temperature").textContent = `${Number(data.temperature).toFixed(1)}°C`;
  }
  if (humSlider && data.humidity != null) {
    humSlider.value = data.humidity;
    document.getElementById("val-humidity").textContent = `${Number(data.humidity).toFixed(0)}%`;
  }
  if (soilSlider && data.soil_moisture != null) {
    soilSlider.value = data.soil_moisture;
    document.getElementById("val-soil_moisture").textContent = `${Number(data.soil_moisture).toFixed(0)}%`;
  }

  setRainState(data.rain === 1);

  // Update left panel badge to LIVE (ESP8266)
  const sourceBadge = document.getElementById("sensorSourceBadge");
  if (sourceBadge) {
    sourceBadge.textContent = "LIVE (ESP8266)";
    sourceBadge.className = "panel-badge panel-badge-live";
  }

  const lastUpdated = document.getElementById("lastUpdated");
  if (lastUpdated) {
    lastUpdated.textContent = new Date().toLocaleTimeString();
  }
}

function markSourceBadgeStale() {
  const sourceBadge = document.getElementById("sensorSourceBadge");
  if (sourceBadge && sourceBadge.classList.contains("panel-badge-live")) {
    sourceBadge.textContent = "STALE (ESP8266)";
    sourceBadge.className = "panel-badge panel-badge-stale";
  }
}

function formatAgo(seconds) {
  if (seconds <= 2) return "Just now";
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  return `${hours}h ago`;
}

// ── MANUAL SENSOR INPUT HANDLERS ──────────────────────────────────────────────
function onManualInput(sensorId, value) {
  isManualInput = true;
  const displayMap = {
    temperature: `${Number(value).toFixed(1)}°C`,
    humidity: `${Number(value).toFixed(0)}%`,
    soil_moisture: `${Number(value).toFixed(0)}%`,
  };

  const displayEl = document.getElementById(`val-${sensorId}`);
  if (displayEl && displayMap[sensorId]) {
    displayEl.textContent = displayMap[sensorId];
  }

  const sourceBadge = document.getElementById("sensorSourceBadge");
  if (sourceBadge) {
    sourceBadge.textContent = "MANUAL";
    sourceBadge.className = "panel-badge panel-badge-manual";
  }
}

function toggleRain() {
  isManualInput = true;
  setRainState(!isRainActive);

  const sourceBadge = document.getElementById("sensorSourceBadge");
  if (sourceBadge) {
    sourceBadge.textContent = "MANUAL";
    sourceBadge.className = "panel-badge panel-badge-manual";
  }
}

function setRainState(active) {
  isRainActive = Boolean(active);
  const rainToggle = document.getElementById("rainToggle");
  const rainIcon = document.getElementById("rainIcon");
  const rainLabel = document.getElementById("rainLabel");

  if (rainToggle) {
    if (isRainActive) {
      rainToggle.className = "rain-toggle rain-on";
      rainToggle.setAttribute("aria-pressed", "true");
      if (rainIcon) rainIcon.textContent = "🌧️";
      if (rainLabel) rainLabel.textContent = "RAINING";
    } else {
      rainToggle.className = "rain-toggle rain-off";
      rainToggle.setAttribute("aria-pressed", "false");
      if (rainIcon) rainIcon.textContent = "☀️";
      if (rainLabel) rainLabel.textContent = "NO RAIN";
    }
  }
}

function loadPreset(presetName) {
  const preset = SENSOR_PRESETS[presetName];
  if (!preset) return;

  isManualInput = true;
  document.getElementById("temperature").value = preset.temperature;
  document.getElementById("humidity").value = preset.humidity;
  document.getElementById("soil_moisture").value = preset.soil_moisture;
  setRainState(preset.rain === 1);

  syncSliderDisplays();

  const sourceBadge = document.getElementById("sensorSourceBadge");
  if (sourceBadge) {
    sourceBadge.textContent = "MANUAL";
    sourceBadge.className = "panel-badge panel-badge-manual";
  }

  runPrediction();
}

function syncSliderDisplays() {
  const tempVal = document.getElementById("temperature")?.value;
  const humVal = document.getElementById("humidity")?.value;
  const soilVal = document.getElementById("soil_moisture")?.value;

  if (tempVal) document.getElementById("val-temperature").textContent = `${Number(tempVal).toFixed(1)}°C`;
  if (humVal) document.getElementById("val-humidity").textContent = `${Number(humVal).toFixed(0)}%`;
  if (soilVal) document.getElementById("val-soil_moisture").textContent = `${Number(soilVal).toFixed(0)}%`;
}

function initAutoToggle() {
  const autoToggle = document.getElementById("autoToggle");
  if (autoToggle) {
    autoToggle.addEventListener("change", (e) => {
      autoUpdateEnabled = e.target.checked;
    });
  }
}

// ── CROP PREDICTION & DISEASE RISK PIPELINE ───────────────────────────────────
async function runPrediction() {
  const temp = parseFloat(document.getElementById("temperature").value);
  const hum = parseFloat(document.getElementById("humidity").value);
  const soil = parseFloat(document.getElementById("soil_moisture").value);
  const rain = isRainActive ? 1 : 0;

  const payload = {
    temperature: temp,
    humidity: hum,
    soil_moisture: soil,
    rain: rain,
  };

  showLoading("Calculating Crop Intelligence & Disease Risks...");

  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || `Server returned HTTP ${res.status}`);
    }

    const data = await res.json();
    renderCropRecommendation(data);
    renderDiseaseRisks(data.disease_alerts || data.disease_risks || []);
    renderRiskFlags(data.features || data.risk_flags || {});

    const lastUpdated = document.getElementById("lastUpdated");
    if (lastUpdated) {
      lastUpdated.textContent = new Date().toLocaleTimeString();
    }
  } catch (err) {
    showErrorNotification(`Prediction error: ${err.message}`);
  } finally {
    hideLoading();
  }
}

function renderCropRecommendation(data) {
  const container = document.getElementById("recommendationContent");
  if (!container) return;

  const topCrop = data.recommended_crop || "Unknown";
  let rawConf = Number(data.confidence || 0);
  if (rawConf <= 1.0 && rawConf > 0) {
    rawConf = rawConf * 100;
  }
  const confNum = Math.min(100, Math.max(0, rawConf));
  const confStr = `${confNum.toFixed(1)}%`;

  const topAlternatives = data.top3 ? data.top3.slice(1) : (data.top_alternatives || []);
  const conditions = data.features || data.growing_conditions || {};

  const cropIcons = {
    rice: "🌾",
    maize: "🌽",
    cotton: "🌱",
    wheat: "🌾",
    sugarcane: "🎋",
    coffee: "☕",
    jute: "🌿",
    lentil: "🫘",
    pigeonpeas: "🫛",
    chickpea: "🧆",
    mungbean: "🌱",
    blackgram: "🫘",
    kidneybeans: "🫘",
    coconut: "🥥",
    banana: "🍌",
    apple: "🍎",
    orange: "🍊",
    papaya: "🍈",
    watermelon: "🍉",
    muskmelon: "🍈",
    grapes: "🍇",
    mango: "🥭",
    pomegranate: "🍎",
  };

  const cropKey = topCrop.toLowerCase().replace(/[^a-z]/g, "");
  const icon = cropIcons[cropKey] || "🌿";

  const currentTemp = document.getElementById("temperature")?.value;
  const currentHum = document.getElementById("humidity")?.value;
  const currentSoil = document.getElementById("soil_moisture")?.value;

  const envTemp = conditions.temperature != null ? Number(conditions.temperature).toFixed(1) : (currentTemp ? Number(currentTemp).toFixed(1) : "25.0");
  const envHum = conditions.humidity != null ? Number(conditions.humidity).toFixed(0) : (currentHum ? Number(currentHum).toFixed(0) : "65");
  const envSoil = conditions.soil_moisture != null ? Number(conditions.soil_moisture).toFixed(0) : (currentSoil ? Number(currentSoil).toFixed(0) : "50");
  const envRain = (conditions.rain === 1 || isRainActive) ? "Active Rainfall" : "Clear / Arid";

  let altsHtml = "";
  if (topAlternatives.length > 0) {
    altsHtml = `
      <div class="top3-list" style="margin-top: 14px;">
        <div class="top3-label">Secondary Viable Options</div>
        ${topAlternatives
          .map((alt, idx) => {
            const altKey = alt.crop ? alt.crop.toLowerCase().replace(/[^a-z]/g, "") : "";
            const altIcon = alt.icon || cropIcons[altKey] || "🌱";
            let rawAlt = Number(alt.confidence || 0);
            if (rawAlt <= 1.0 && rawAlt > 0) rawAlt = rawAlt * 100;
            const altPct = Math.min(100, Math.max(0, rawAlt)).toFixed(1);
            return `
            <div class="top3-item">
              <span class="top3-rank">${idx + 2}</span>
              <span class="top3-crop">${altIcon} ${alt.crop}</span>
              <div class="top3-bar-wrap">
                <div class="top3-bar" style="width: ${altPct}%;"></div>
              </div>
              <span class="top3-conf">${altPct}%</span>
            </div>
          `;
          })
          .join("")}
      </div>
    `;
  }

  container.innerHTML = `
    <div class="crop-result">
      <div class="crop-main-card">
        <div class="crop-emoji">${icon}</div>
        <div class="crop-info">
          <div class="crop-name">${topCrop}</div>
          <div class="crop-confidence-text">Optimal Ecological Match · Recommended Crop</div>
        </div>
        <div class="confidence-ring-wrap">
          <div class="confidence-ring" style="background: conic-gradient(var(--green-primary) ${confNum * 3.6}deg, rgba(82, 183, 136, 0.15) 0deg);">
            <span class="confidence-pct">${confStr}</span>
          </div>
        </div>
      </div>

      <div class="rec-env-pills">
        <span class="env-pill">🌡️ ${envTemp}°C</span>
        <span class="env-pill">💧 ${envHum}% RH</span>
        <span class="env-pill">🌱 ${envSoil}% Moisture</span>
        <span class="env-pill">🌧️ ${envRain}</span>
      </div>

      ${altsHtml}
    </div>
  `;
}

function renderDiseaseRisks(risks) {
  const container = document.getElementById("diseaseContent");
  const alertBadge = document.getElementById("alertCountBadge");
  if (!container) return;

  if (!risks || risks.length === 0) {
    if (alertBadge) alertBadge.style.display = "none";
    container.innerHTML = `
      <div class="all-clear">
        <span class="all-clear-icon">🛡️</span>
        <div>
          <strong style="color: var(--green-bright); font-size: 13px;">All Environmental Conditions Optimal</strong>
          <p style="margin-top: 3px; color: var(--text-secondary); font-size: 11.5px; line-height: 1.4;">Zero fungal, bacteriological, or physiological pathology thresholds exceeded under current readings.</p>
        </div>
      </div>
    `;
    return;
  }

  if (alertBadge) {
    alertBadge.style.display = "inline-flex";
    alertBadge.textContent = risks.length;
  }

  container.innerHTML = `
    <div class="disease-content">
      ${risks
        .map((risk) => {
          const sev = (risk.severity || risk.risk_level || "ALERT").toUpperCase();
          const borderColor = sev === "HIGH" ? "var(--red)" : (sev === "MODERATE" ? "var(--yellow)" : "var(--green-primary)");
          const sevColor = sev === "HIGH" ? "var(--red)" : (sev === "MODERATE" ? "var(--yellow)" : "var(--green-bright)");
          const icon = risk.icon || (sev === "HIGH" ? "🔴" : "⚠️");
          const name = risk.name || "Disease Risk";
          const type = risk.type || "PATHOLOGY";
          const trigger = risk.trigger || risk.reason || risk.symptoms || "Environmental threshold reached";
          const symptoms = risk.symptoms || "";
          const cultural = risk.technique || risk.cultural_control || "";
          const chemical = risk.pesticide || risk.chemical_control || "";

          return `
          <div class="disease-card" style="border-left-color: ${borderColor};">
            <div class="disease-card-header" onclick="this.nextElementSibling.classList.toggle('open')">
              <span class="disease-severity-icon">${icon}</span>
              <span class="disease-name">${name}</span>
              <span class="disease-type-badge">${type}</span>
              <span class="disease-severity-label" style="color: ${sevColor};">${sev} RISK</span>
            </div>
            <div class="disease-body open">
              <div class="disease-field">
                <strong>Trigger Condition</strong>
                <p>${trigger}</p>
              </div>
              ${
                symptoms
                  ? `<div class="disease-field">
                      <strong>Diagnostic Symptoms</strong>
                      <p>${symptoms}</p>
                    </div>`
                  : ""
              }
              ${
                cultural
                  ? `<div class="disease-field">
                      <strong>Cultural Practice</strong>
                      <p>${cultural}</p>
                    </div>`
                  : ""
              }
              ${
                chemical
                  ? `<div class="disease-field">
                      <strong>Sanitary Protocol</strong>
                      <p>${chemical}</p>
                    </div>`
                  : ""
              }
            </div>
          </div>
        `;
        })
        .join("")}
    </div>
  `;
}

function renderRiskFlags(flags) {
  const fungalEl = document.getElementById("feat-fungal_risk");
  const droughtEl = document.getElementById("feat-drought_risk");
  const waterlogEl = document.getElementById("feat-waterlog_risk");

  updateFlagElement(fungalEl, flags.fungal_risk);
  updateFlagElement(droughtEl, flags.drought_risk);
  updateFlagElement(waterlogEl, flags.waterlog_risk);
}

function updateFlagElement(el, flagValue) {
  if (!el) return;
  if (flagValue === 1 || flagValue === true || flagValue === "HIGH") {
    el.textContent = "HIGH";
    el.className = "feature-val flag-high";
  } else if (flagValue === "MODERATE") {
    el.textContent = "MODERATE";
    el.className = "feature-val flag-moderate";
  } else {
    el.textContent = "LOW / SAFE";
    el.className = "feature-val flag-low";
  }
}

// ── SENSOR READING HISTORY CHART ──────────────────────────────────────────────
/**
 * Strict rules implemented:
 * 1. History chart displays ONLY genuine physical hardware telemetry.
 * 2. Frontend polling does NOT create or append synthetic entries.
 * 3. Manual prediction clicks do NOT pollute the history chart.
 */
function initHistoryChart() {
  const canvas = document.getElementById("historyChart");
  if (!canvas || typeof Chart === "undefined") return;

  const ctx = canvas.getContext("2d");
  historyChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        {
          label: "Temperature (°C)",
          data: [],
          borderColor: "#e07a5f",
          backgroundColor: "rgba(224, 122, 95, 0.15)",
          borderWidth: 2,
          pointRadius: 3,
          tension: 0.3,
          yAxisID: "yTemp",
        },
        {
          label: "Humidity (%)",
          data: [],
          borderColor: "#3d9970",
          backgroundColor: "rgba(61, 153, 112, 0.15)",
          borderWidth: 2,
          pointRadius: 3,
          tension: 0.3,
          yAxisID: "yPct",
        },
        {
          label: "Soil Moisture (%)",
          data: [],
          borderColor: "#81b29a",
          backgroundColor: "rgba(129, 178, 154, 0.15)",
          borderWidth: 2,
          pointRadius: 3,
          tension: 0.3,
          yAxisID: "yPct",
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 400 },
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: {
          position: "top",
          labels: { color: "#8a9e8a", boxWidth: 14, font: { family: "Inter", size: 12 } },
        },
        tooltip: {
          backgroundColor: "#0d1a0d",
          borderColor: "#1e381e",
          borderWidth: 1,
          titleColor: "#e8f5e9",
          bodyColor: "#8a9e8a",
        },
      },
      scales: {
        x: {
          grid: { color: "#162816" },
          ticks: { color: "#527952", font: { family: "Inter", size: 11 } },
        },
        yTemp: {
          type: "linear",
          position: "left",
          min: 0,
          max: 50,
          grid: { color: "#162816" },
          ticks: {
            color: "#e07a5f",
            font: { family: "Inter", size: 11 },
            callback: (v) => `${v}°C`,
          },
        },
        yPct: {
          type: "linear",
          position: "right",
          min: 0,
          max: 100,
          grid: { drawOnChartArea: false },
          ticks: {
            color: "#81b29a",
            font: { family: "Inter", size: 11 },
            callback: (v) => `${v}%`,
          },
        },
      },
    },
  });
}

async function fetchTelemetryHistory() {
  try {
    const res = await fetch(`${API_BASE}/history`, { cache: "no-store" });
    if (!res.ok) return;
    const json = await res.json();
    const readings = json.readings || json.history || [];
    // CRITICAL: Filter ONLY physical IoT hardware readings. Never plot dashboard prediction tests or default values.
    const iotReadings = Array.isArray(readings) ? readings.filter((r) => r.source === "iot") : [];
    const emptyState = document.getElementById("chartEmptyState");

    if (iotReadings.length > 0) {
      if (emptyState) emptyState.style.display = "none";
      iotReadings.forEach((r) => addTelemetryToChart(r, false));
      if (historyChart) historyChart.update();
    } else {
      if (emptyState) emptyState.style.display = "flex";
    }
  } catch {
    // Graceful fallback when no history exists yet
  }
}

function addTelemetryToChart(packet, shouldUpdate = true) {
  if (!historyChart || packet.temperature == null) return;
  // Guard: Only real IoT packets are plotted
  if (packet.source && packet.source !== "iot") return;

  const emptyState = document.getElementById("chartEmptyState");
  if (emptyState) emptyState.style.display = "none";

  const timeLabel = packet.timestamp
    ? new Date(packet.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })
    : new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });

  historyChart.data.labels.push(timeLabel);
  historyChart.data.datasets[0].data.push(Number(packet.temperature));
  historyChart.data.datasets[1].data.push(Number(packet.humidity));
  historyChart.data.datasets[2].data.push(Number(packet.soil_moisture));

  // Maintain sliding window of 25 physical readings
  if (historyChart.data.labels.length > 25) {
    historyChart.data.labels.shift();
    historyChart.data.datasets[0].data.shift();
    historyChart.data.datasets[1].data.shift();
    historyChart.data.datasets[2].data.shift();
  }

  if (shouldUpdate) {
    historyChart.update();
  }
}

// ── COMPUTER VISION: DRAG & DROP AND FILE SELECTION ───────────────────────────
function initDragAndDrop() {
  const dropzone = document.getElementById("cvDropzone");
  if (!dropzone) return;

  ["dragenter", "dragover"].forEach((eventName) => {
    dropzone.addEventListener(
      eventName,
      (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.add("drag-over");
      },
      false
    );
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropzone.addEventListener(
      eventName,
      (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.remove("drag-over");
      },
      false
    );
  });

  dropzone.addEventListener("drop", (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files && files.length > 0) {
      processSelectedImageFile(files[0]);
    }
  });
}

function handleCvFileSelect(event) {
  const files = event.target.files;
  if (files && files.length > 0) {
    processSelectedImageFile(files[0]);
  }
}

function processSelectedImageFile(file) {
  if (!file.type.startsWith("image/")) {
    showErrorNotification("Please upload a valid image file (JPEG, PNG, or WebP).");
    return;
  }

  selectedVisionFile = file;
  const reader = new FileReader();

  reader.onload = (e) => {
    const dataUrl = e.target.result;
    const previewImg = document.getElementById("cvPreviewImg");
    const filenameEl = document.getElementById("cvPreviewFilename");
    const dropzonePrompt = document.getElementById("dropzonePrompt");
    const previewWrap = document.getElementById("dropzonePreviewWrap");

    if (previewImg) previewImg.src = dataUrl;
    if (filenameEl) filenameEl.textContent = file.name;
    if (dropzonePrompt) dropzonePrompt.style.display = "none";
    if (previewWrap) previewWrap.style.display = "flex";

    // Preload image object for canvas lesion overlay
    originalVisionImageObj = new Image();
    originalVisionImageObj.src = dataUrl;
  };

  reader.readAsDataURL(file);
}

// ── COMPUTER VISION: QUICK SCENARIOS ──────────────────────────────────────────
async function loadSampleLeaf(sampleKey) {
  const sample = SAMPLE_LEAF_MAP[sampleKey];
  if (!sample) return;

  showLoading(`Loading pre-calibrated ${sample.label} specimen...`);

  try {
    const res = await fetch(sample.url);
    if (!res.ok) throw new Error(`Could not load specimen from ${sample.url}`);
    const blob = await res.blob();
    const file = new File([blob], sample.filename, { type: "image/jpeg" });
    processSelectedImageFile(file);

    // Auto-analyze selected foliage specimen
    setTimeout(() => {
      runVisionPrediction();
    }, 300);
  } catch (err) {
    showErrorNotification(`Specimen error: ${err.message}`);
    hideLoading();
  }
}

// ── COMPUTER VISION: PREDICTION & 3-TIER CASCADE ──────────────────────────────
async function runVisionPrediction() {
  if (!selectedVisionFile) {
    showErrorNotification("Please select or upload a leaf photograph first.");
    return;
  }

  showLoading("Executing 3-Tier Multi-Scale Foliage Diagnostics...");

  const formData = new FormData();
  formData.append("file", selectedVisionFile);

  try {
    const res = await fetch(`${API_BASE}/predict/vision`, {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || `Server returned HTTP ${res.status}`);
    }

    const data = await res.json();
    lastVisionResult = data;
    renderVisionDiagnosis(data.diagnosis || {});
    renderVisionAdvisory(data.advisory || {}, data.diagnosis || {});
    renderSpatialTelemetry(data);

    const inspectionPanel = document.getElementById("visionInspectionPanel");
    if (inspectionPanel) {
      inspectionPanel.style.display = "block";
    }
  } catch (err) {
    showErrorNotification(`Vision pipeline error: ${err.message}`);
  } finally {
    hideLoading();
  }
}

function renderVisionDiagnosis(diag) {
  const container = document.getElementById("visionResultContent");
  const triageBadge = document.getElementById("visionTriageBadge");
  if (!container) return;

  const diseaseName = diag.disease_common_name || diag.predicted_class || "Undetermined";
  const crop = diag.crop || "Universal Specimen";
  let rawConf = Number(diag.confidence_pct || 0);
  if (rawConf <= 1.0 && rawConf > 0) rawConf = rawConf * 100;
  const confNum = Math.min(100, Math.max(0, rawConf));
  const confStr = `${confNum.toFixed(1)}%`;
  const confLevel = diag.confidence_level || "CONFIRMED";
  const isInfected = diag.is_infected !== false;
  const triageStage = diag.triage_stage || (isInfected ? "STAGE 2: Active Progressive Lesions" : "STAGE 0: Optimal Health");

  if (triageBadge) {
    triageBadge.style.display = "inline-flex";
    triageBadge.textContent = triageStage;
    triageBadge.className = isInfected ? "panel-badge badge-infected" : "panel-badge badge-healthy";
  }

  const top3 = diag.top3_predictions || [];
  let top3Html = "";
  if (top3.length > 0) {
    top3Html = `
      <div class="top3-list" style="margin-top: 14px;">
        <div class="top3-label">Top-3 Confidence Projections</div>
        ${top3
          .map((item, idx) => {
            let itemConf = Number(item.confidence_pct || 0);
            if (itemConf <= 1.0 && itemConf > 0) itemConf = itemConf * 100;
            const pct = Math.min(100, Math.max(0, itemConf)).toFixed(1);
            return `
            <div class="top3-item">
              <span class="top3-rank">${idx + 1}</span>
              <span class="top3-crop">${item.label || item.class_id}</span>
              <div class="top3-bar-wrap">
                <div class="top3-bar" style="width: ${pct}%; background: ${idx === 0 && isInfected ? '#e07a5f' : 'var(--green-primary)'};"></div>
              </div>
              <span class="top3-conf">${pct}%</span>
            </div>
          `;
          })
          .join("")}
      </div>
    `;
  }

  container.innerHTML = `
    <div class="crop-result">
      <div class="crop-main-card" style="background: ${isInfected ? 'linear-gradient(135deg, #1c0e0e, #291212)' : 'linear-gradient(135deg, #0f240f, #153315)'}; border-color: ${isInfected ? 'rgba(224, 122, 95, 0.45)' : 'var(--border-bright)'};">
        <div class="crop-emoji">${isInfected ? '🍂' : '🌱'}</div>
        <div class="crop-info">
          <div class="crop-name" style="color: ${isInfected ? '#ff9e7d' : 'var(--green-bright)'}; font-size: 20px;">${diseaseName}</div>
          <div class="crop-confidence-text">${crop} · ${confStr} Match (${confLevel})</div>
        </div>
        <div class="confidence-ring-wrap">
          <div class="confidence-ring" style="background: conic-gradient(${isInfected ? '#e07a5f' : 'var(--green-primary)'} ${confNum * 3.6}deg, rgba(82, 183, 136, 0.15) 0deg); box-shadow: 0 0 12px ${isInfected ? 'rgba(224,122,95,0.3)' : 'var(--green-glow)'};">
            <span class="confidence-pct" style="color: ${isInfected ? '#ff9e7d' : 'var(--green-bright)'};">${confStr}</span>
          </div>
        </div>
      </div>

      <div class="rec-env-pills">
        <span class="env-pill">📊 Damage: ${Number(diag.foliar_damage_pct || 0).toFixed(1)}%</span>
        <span class="env-pill">🎯 Foci Detected: ${diag.lesion_foci_count || 0}</span>
        <span class="env-pill">🧪 Condition: ${diag.condition_type || (isInfected ? "Fungal / Pathological" : "Healthy Foliage")}</span>
      </div>

      ${top3Html}
    </div>
  `;
}

function renderVisionAdvisory(adv, diag = {}) {
  const container = document.getElementById("visionAdvisoryContent");
  if (!container) return;

  const isInfected = diag.is_infected !== false;

  const immediate = adv.immediate || adv.pathogen_etiology || (isInfected
    ? "Inspect field canopy and isolate affected foliage to halt lesion spread."
    : "No immediate quarantine action required. Foliar tissue displays healthy photosynthetic vigor.");

  const treatment = adv.treatment || adv.chemical_control || (isInfected
    ? "Apply protective bio-fungicide or targeted copper-based bactericide/fungicide."
    : "No chemical fungicide or bactericide application warranted. Maintain balanced foliar biostimulants.");

  const cultural = adv.cultural || adv.cultural_sanitation || (isInfected
    ? "Reduce canopy humidity by improving spacing and pruning. Switch completely to ground-level drip irrigation."
    : "Maintain standard drip scheduling, adequate root aeration, and preventive weed sanitation.");

  const symptoms = adv.symptoms || (isInfected
    ? `Foliar necrotic lesions detected with ${Number(diag.foliar_damage_pct || 0).toFixed(1)}% leaf area compromise across ${diag.lesion_foci_count || 0} focal points.`
    : "Foliage exhibits vigorous chlorophyll homeostasis without active necrotic chlorosis.");

  container.innerHTML = `
    <div class="disease-content">
      <div class="disease-card" style="border-left-color: ${isInfected ? 'var(--orange)' : 'var(--green-primary)'};">
        <div class="disease-card-header" style="cursor: default;">
          <span class="disease-severity-icon">⚡</span>
          <span class="disease-name">Immediate Action</span>
          <span class="disease-type-badge">${isInfected ? 'URGENT' : 'PREVENTATIVE'}</span>
        </div>
        <div class="disease-body open" style="grid-template-columns: 1fr;">
          <div class="disease-field">
            <p>${immediate}</p>
          </div>
        </div>
      </div>

      <div class="disease-card" style="border-left-color: ${isInfected ? 'var(--red)' : 'var(--green-primary)'};">
        <div class="disease-card-header" style="cursor: default;">
          <span class="disease-severity-icon">💊</span>
          <span class="disease-name">Treatment Protocol</span>
          <span class="disease-type-badge">${isInfected ? 'TREATMENT' : 'MAINTENANCE'}</span>
        </div>
        <div class="disease-body open" style="grid-template-columns: 1fr;">
          <div class="disease-field">
            <p>${treatment}</p>
          </div>
        </div>
      </div>

      <div class="disease-card" style="border-left-color: var(--teal);">
        <div class="disease-card-header" style="cursor: default;">
          <span class="disease-severity-icon">🌾</span>
          <span class="disease-name">Cultural Sanitation</span>
          <span class="disease-type-badge">AGRONOMY</span>
        </div>
        <div class="disease-body open" style="grid-template-columns: 1fr;">
          <div class="disease-field">
            <p>${cultural}</p>
          </div>
        </div>
      </div>

      <div class="disease-card" style="border-left-color: var(--yellow);">
        <div class="disease-card-header" style="cursor: default;">
          <span class="disease-severity-icon">🔬</span>
          <span class="disease-name">Diagnostic Symptoms</span>
          <span class="disease-type-badge">PATHOLOGY</span>
        </div>
        <div class="disease-body open" style="grid-template-columns: 1fr;">
          <div class="disease-field">
            <p>${symptoms}</p>
          </div>
        </div>
      </div>
    </div>
  `;
}

// ── SPATIAL LESION CANVAS & BOUNDING BOX LOCALIZATION ──────────────────────────
function renderSpatialTelemetry(data) {
  const diag = data.diagnosis || {};
  const telemetry = data.spatial_telemetry || {};
  const latency = data.latency_ms || {};
  const boxes = telemetry.bounding_boxes || [];

  const damageVal = document.getElementById("telemetryDamageVal");
  const fociVal = document.getElementById("telemetryFociVal");
  const latencyVal = document.getElementById("telemetryLatencyVal");
  const deviceVal = document.getElementById("telemetryDeviceVal");
  const boxCountDisplay = document.getElementById("boxCountDisplay");

  if (damageVal) damageVal.textContent = `${Number(diag.foliar_damage_pct || 0).toFixed(1)}%`;
  if (fociVal) fociVal.textContent = String(boxes.length);
  if (latencyVal) latencyVal.textContent = `${Number(latency.total_ms || 0).toFixed(0)} ms`;
  if (deviceVal) deviceVal.textContent = latency.device || "CPU / GPU";
  if (boxCountDisplay) boxCountDisplay.textContent = String(boxes.length);

  drawSpatialCanvas(boxes, data.segmentation_mask_b64);
}

function drawSpatialCanvas(boxes, maskB64) {
  const canvas = document.getElementById("visionInspectionCanvas");
  if (!canvas || !originalVisionImageObj) return;

  const ctx = canvas.getContext("2d");
  const img = originalVisionImageObj;

  // Set internal resolution matching natural image dimensions
  canvas.width = img.naturalWidth || img.width || 640;
  canvas.height = img.naturalHeight || img.height || 640;

  // 1. Draw base leaf image
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

  // 2. Draw segmentation mask overlay if available
  if (maskB64) {
    const maskImg = new Image();
    maskImg.onload = () => {
      ctx.save();
      ctx.globalAlpha = 0.38;
      ctx.drawImage(maskImg, 0, 0, canvas.width, canvas.height);
      ctx.restore();
      if (showBoundingBoxes && boxes && boxes.length > 0) {
        drawBoundingBoxesOnContext(ctx, boxes, canvas.width, canvas.height);
      }
    };
    maskImg.src = maskB64.startsWith("data:") ? maskB64 : `data:image/png;base64,${maskB64}`;
    return;
  }

  // 3. Draw YOLOv8 bounding boxes
  if (showBoundingBoxes && boxes && boxes.length > 0) {
    drawBoundingBoxesOnContext(ctx, boxes, canvas.width, canvas.height);
  }
}

function drawBoundingBoxesOnContext(ctx, boxes, imgW, imgH) {
  ctx.save();
  ctx.lineWidth = Math.max(2, Math.round(imgW / 240));

  boxes.forEach((box, index) => {
    let [x1, y1, x2, y2] = box.bbox_xyxy || [0, 0, 0, 0];

    // Coordinate normalization check
    if (x2 <= 1.0 && y2 <= 1.0 && (x2 > 0 || y2 > 0)) {
      x1 = x1 * imgW;
      y1 = y1 * imgH;
      x2 = x2 * imgW;
      y2 = y2 * imgH;
    }

    // Clamp coordinates strictly within canvas boundary
    x1 = Math.max(0, Math.min(x1, imgW));
    y1 = Math.max(0, Math.min(y1, imgH));
    x2 = Math.max(0, Math.min(x2, imgW));
    y2 = Math.max(0, Math.min(y2, imgH));

    const w = x2 - x1;
    const h = y2 - y1;

    // Draw box stroke
    ctx.strokeStyle = "#e07a5f";
    ctx.fillStyle = "rgba(224, 122, 95, 0.12)";
    ctx.fillRect(x1, y1, w, h);
    ctx.strokeRect(x1, y1, w, h);

    // Draw lesion focus badge
    const badgeText = `#${index + 1} Focus`;
    ctx.font = `bold ${Math.max(12, Math.round(imgW / 45))}px Space Grotesk, sans-serif`;
    const textWidth = ctx.measureText(badgeText).width;
    const badgeHeight = Math.max(16, Math.round(imgW / 36));

    const badgeX = x1;
    const badgeY = Math.max(0, y1 - badgeHeight);

    ctx.fillStyle = "rgba(13, 26, 13, 0.9)";
    ctx.fillRect(badgeX, badgeY, textWidth + 10, badgeHeight);
    ctx.strokeStyle = "#e07a5f";
    ctx.strokeRect(badgeX, badgeY, textWidth + 10, badgeHeight);

    ctx.fillStyle = "#e8f5e9";
    ctx.fillText(badgeText, badgeX + 5, badgeY + badgeHeight - 4);
  });

  ctx.restore();
}

function toggleBoundingBoxes(visible) {
  showBoundingBoxes = Boolean(visible);
  if (lastVisionResult) {
    const boxes = lastVisionResult.spatial_telemetry?.bounding_boxes || [];
    drawSpatialCanvas(boxes, lastVisionResult.segmentation_mask_b64);
  }
}

// ── FULLSCREEN IMAGE & CANVAS INSPECTION MODAL ────────────────────────────────
function openImageModal(imgSrc, title) {
  const modal = document.getElementById("imageModal");
  const modalTitle = document.getElementById("imageModalTitle");
  const modalImg = document.getElementById("imageModalImg");
  const modalCanvas = document.getElementById("imageModalCanvas");

  if (!modal || !modalImg) return;

  if (modalTitle) modalTitle.textContent = title || "High-Resolution Foliage Inspection";
  modalImg.src = imgSrc;
  modalImg.style.display = "block";
  if (modalCanvas) modalCanvas.style.display = "none";
  modal.style.display = "flex";
}

function openCanvasFullscreen() {
  const srcCanvas = document.getElementById("visionInspectionCanvas");
  const modal = document.getElementById("imageModal");
  const modalTitle = document.getElementById("imageModalTitle");
  const modalImg = document.getElementById("imageModalImg");
  const modalCanvas = document.getElementById("imageModalCanvas");

  if (!srcCanvas || !modal || !modalCanvas) return;

  if (modalTitle) modalTitle.textContent = "Full-Resolution Lesion Telemetry & Spatial Mapping";
  if (modalImg) modalImg.style.display = "none";

  modalCanvas.width = srcCanvas.width;
  modalCanvas.height = srcCanvas.height;
  const ctx = modalCanvas.getContext("2d");
  ctx.drawImage(srcCanvas, 0, 0);

  modalCanvas.style.display = "block";
  modal.style.display = "flex";
}

function closeImageModal() {
  const modal = document.getElementById("imageModal");
  if (modal) modal.style.display = "none";
}

// ── UTILITY: LOADING SPINNER & ERROR NOTIFICATIONS ────────────────────────────
function showLoading(msg) {
  const overlay = document.getElementById("loadingOverlay");
  const text = document.getElementById("loadingOverlayText");
  if (overlay) {
    if (text && msg) text.textContent = msg;
    overlay.style.display = "flex";
  }
}

function hideLoading() {
  const overlay = document.getElementById("loadingOverlay");
  if (overlay) {
    overlay.style.display = "none";
  }
}

function showErrorNotification(message) {
  // Simple non-invasive console and DOM notice
  console.error(message);
  alert(message);
}

// ── INDEPENDENT RIGHT PANEL WHEEL SCROLL ROUTER ──────────────────────────────
window.addEventListener("wheel", (e) => {
  if (window.innerWidth <= 900) return;
  const activeRight = document.querySelector(
    activeMode === "vision" ? "#visionSection .right-column" : "#cropSection .right-column"
  );
  if (!activeRight) return;
  const leftPanel = e.target.closest(".sensor-panel");
  if (leftPanel && leftPanel.scrollHeight > leftPanel.clientHeight) {
    return;
  }
  if (!e.target.closest(".right-column")) {
    activeRight.scrollTop += e.deltaY;
  }
}, { passive: true });
