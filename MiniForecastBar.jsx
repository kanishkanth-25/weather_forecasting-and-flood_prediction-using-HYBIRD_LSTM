import React from "react";

export default function MiniForecastBar({ data }) {
  if (!data || data.length === 0) return null;

  return (
    <div className="forecast-bar card">
      <h3>7-Day Summary</h3>
      <div className="forecast-row">
        {data.map((d, i) => (
          <div className="forecast-item" key={i}>
            <div className="f-date">{d.date}</div>
            <div className="f-temp">{d.temp}°C</div>
            <div className="f-sub">
              🌧 {d.rain} mm &nbsp;·&nbsp; 💧 {d.humidity}% 
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
