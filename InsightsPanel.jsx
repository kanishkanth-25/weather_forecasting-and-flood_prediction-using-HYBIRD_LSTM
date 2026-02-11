import React from "react";
import "./insights.css";

function classifyFlood(avgRain, maxFloodRisk) {
  if (maxFloodRisk >= 0.7 || avgRain > 40) {
    return {
      level: "high",
      label: "High",
      percent: 90,
      en: "High flood risk in the next few days. Avoid low-lying and river-side areas.",
      ta: "அடுத்த நாட்களில் அதிக மழை/வெள்ள அபாயம் உள்ளது. தாழ்வான பகுதிகளில் செல்வதை தவிர்க்கவும்."
    };
  } else if (maxFloodRisk >= 0.4 || avgRain > 20) {
    return {
      level: "medium",
      label: "Medium",
      percent: 65,
      en: "Some chance of local waterlogging and minor flooding in low-lying streets.",
      ta: "சில தாழ்வான பகுதிகளில் நீர்நிலைத்தல் / சிறிய அளவில் வெள்ளம் ஏற்படும் வாய்ப்பு உள்ளது."
    };
  } else {
    return {
      level: "low",
      label: "Low",
      percent: 35,
      en: "Overall flood risk is low for this period.",
      ta: "இந்த காலத்தில் வெள்ள அபாயம் குறைவாக உள்ளது."
    };
  }
}

function classifyHeat(avgTemp) {
  if (avgTemp >= 35) {
    return {
      level: "high",
      label: "Very Hot",
      percent: 90,
      en: "Very hot conditions. High heat stress for outdoor workers.",
      ta: "மிக அதிக வெப்பம். வெளியில் வேலை செய்பவர்கள் நீர் அதிகமாக குடித்து, நேரடி வெயிலை தவிர்க்க வேண்டும்."
    };
  } else if (avgTemp >= 30) {
    return {
      level: "medium",
      label: "Warm",
      percent: 65,
      en: "Warm to hot weather. Some discomfort during afternoon time.",
      ta: "சூடான வானிலை. மதியம் நேரத்தில் சிறிது தொந்தரவு உணரப்படலாம்."
    };
  } else {
    return {
      level: "low",
      label: "Pleasant / Cool",
      percent: 40,
      en: "Pleasant to cool weather conditions.",
      ta: "சராசரி அல்லது சற்றே குளிர்ந்த வானிலை."
    };
  }
}

function classifyWind(avgWind) {
  if (avgWind >= 12) {
    return {
      level: "high",
      label: "Strong",
      percent: 85,
      en: "Strong winds expected. Secure loose items and be careful near trees and hoardings.",
      ta: "வலுவான காற்று வீசும். தளர்ந்த பொருட்களை உறுதியாக கட்டி, மரங்கள் / ஹோர்டிங் அருகில் கவனமாக இருக்கவும்."
    };
  } else if (avgWind >= 6) {
    return {
      level: "medium",
      label: "Breezy",
      percent: 60,
      en: "Moderate breeze. May affect light objects and two-wheelers.",
      ta: "மிதமான காற்று. இலகு பொருட்கள் மற்றும் இருசக்கர வாகனங்களுக்கு சற்றே பாதிப்பு இருக்கலாம்."
    };
  } else {
    return {
      level: "low",
      label: "Calm",
      percent: 35,
      en: "Winds will be mostly calm or light.",
      ta: "காற்று வேகம் குறைவு அல்லது மிதம்தான்."
    };
  }
}

export default function InsightsPanel({ predictions }) {
  if (!predictions || predictions.length === 0) return null;

  const n = predictions.length;

  const avgTemp =
    predictions.reduce((s, p) => s + (p.temp_c ?? 0), 0) / n;
  const avgRain =
    predictions.reduce((s, p) => s + (p.rain_mm ?? 0), 0) / n;
  const totalRain =
    predictions.reduce((s, p) => s + (p.rain_mm ?? 0), 0);
  const avgHumidity =
    predictions.reduce((s, p) => s + (p.humidity ?? 0), 0) / n;
  const avgWind =
    predictions.reduce((s, p) => s + (p.wind_speed ?? 0), 0) / n;
  const maxFloodRisk = Math.max(
    ...predictions.map((p) => p.flood_risk ?? 0),
    0
  );

  const flood = classifyFlood(avgRain, maxFloodRisk);
  const heat = classifyHeat(avgTemp);
  const wind = classifyWind(avgWind);

  return (
    <div className="insights-card">
      <h2>Weather Insights for This Location</h2>
      <p className="insights-sub">
        Based on the next <b>{n}</b> days of LSTM forecast for this point in Tamil Nadu.
      </p>

      {/* Flood Risk Row */}
      <div className="insight-row">
        <div className="insight-label">
          🌧️ Flood Risk{" "}
          <span className={`badge badge-${flood.level}`}>
            {flood.label}
          </span>
        </div>
        <div className="meter">
          <div
            className={`meter-fill level-${flood.level}`}
            style={{ width: `${flood.percent}%` }}
          />
        </div>
        <div className="insight-text">
          <div>{flood.en}</div>
          <div className="ta">{flood.ta}</div>
        </div>
      </div>

      {/* Heat / Temperature Row */}
      <div className="insight-row">
        <div className="insight-label">
          🌡️ Temperature ({avgTemp.toFixed(1)} °C avg){" "}
          <span className={`badge badge-${heat.level}`}>
            {heat.label}
          </span>
        </div>
        <div className="meter">
          <div
            className={`meter-fill level-${heat.level}`}
            style={{ width: `${heat.percent}%` }}
          />
        </div>
        <div className="insight-text">
          <div>{heat.en}</div>
          <div className="ta">{heat.ta}</div>
        </div>
      </div>

      {/* Wind Row */}
      <div className="insight-row">
        <div className="insight-label">
          💨 Wind ({avgWind.toFixed(1)} m/s avg){" "}
          <span className={`badge badge-${wind.level}`}>
            {wind.label}
          </span>
        </div>
        <div className="meter">
          <div
            className={`meter-fill level-${wind.level}`}
            style={{ width: `${wind.percent}%` }}
          />
        </div>
        <div className="insight-text">
          <div>{wind.en}</div>
          <div className="ta">{wind.ta}</div>
        </div>
      </div>

      {/* Summary for farmers / public */}
      <div className="insight-row summary">
        <div className="insight-label">🌾 Summary for Farmers</div>
        <div className="insight-text">
          <div>
            • Total rain in this period: <b>{totalRain.toFixed(1)} mm</b> <br />
            • Average humidity: <b>{avgHumidity.toFixed(1)} %</b>
          </div>
          <div>
            {avgRain > 20 ? (
              <>
                For heavy rain days, protect crops with covers, ensure field
                drainage and avoid storing grains on the floor.
                <div className="ta">
                  அதிக மழை நாள்களில் பயிர்களை போர்வை / தர்பாய் கொண்டு
                  மூடி, வயல்வெளியில் வடிகால் வழிகளை சுத்தப்படுத்த வேண்டும்.
                </div>
              </>
            ) : (
              <>
                Rainfall is mostly light to moderate. Good time for irrigation
                planning and fertilizer application.
                <div className="ta">
                  மழை அளவு குறைவு முதல் மிதமாக உள்ளது. பாசனம் மற்றும் உர
                  பயன்பாட்டை திட்டமிட நல்ல காலம்.
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
