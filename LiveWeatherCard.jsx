import React from "react";

export default function LiveWeatherCard({ live }) {
  const getIcon = (desc) => {
    desc = desc?.toLowerCase();
    if (desc.includes("rain")) return "🌧️";
    if (desc.includes("cloud")) return "☁️";
    if (desc.includes("storm")) return "⛈️";
    return "☀️";
  };

  return (
    <div className="card live-card">
      <h3>Live Weather</h3>

      {live ? (
        <>
          <div className="live-icon">{getIcon(live.description)}</div>
          <div className="big">{live.temp}°C</div>
          <div>{live.description}</div>
          <div>Humidity: {live.humidity}%</div>
          <div>Wind: {live.wind} m/s</div>
        </>
      ) : (
        <div className="muted">Click map + Predict</div>
      )}
    </div>
  );
}
