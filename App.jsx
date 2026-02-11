import React, { useState, useEffect } from "react";
import axios from "axios";
import { MapContainer, TileLayer, Marker, useMapEvents } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import L from "leaflet";
import "./styles.css";

// UI Components
import WeatherChart from "./components/WeatherChart";
import MiniForecastBar from "./components/MiniForecastBar";
import LiveWeatherCard from "./components/LiveWeatherCard";
import AIAssistant from "./components/AIAssistant"; // UPDATED ✔
import InsightsPanel from "./components/InsightsPanel";

// Backend URL
const API = "http://localhost:8000";

// Fix Leaflet Marker Icons
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl:
    "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon-2x.png",
  iconUrl:
    "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon.png",
  shadowUrl:
    "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png",
});

// Click event marker
function ClickMarker({ position, setPosition }) {
  useMapEvents({
    click(e) {
      setPosition([e.latlng.lat, e.latlng.lng]);
    },
  });
  return position ? <Marker position={position} /> : null;
}

export default function App() {
  const [position, setPosition] = useState([11.1271, 78.6569]); // Tamil Nadu Center
  const [days, setDays] = useState(7);
  const [loading, setLoading] = useState(false);
  const [predictions, setPredictions] = useState([]);
  const [live, setLive] = useState(null);
  const [message, setMessage] = useState("");
  const [dark, setDark] = useState(false);

  // Backend health check
  useEffect(() => {
    axios.get(API + "/health")
      .catch(() => setMessage("Backend Offline ❌ Start FastAPI"));
  }, []);

  // LIVE weather fetch
  const fetchLive = async (lat, lon) => {
    try {
      const r = await axios.get(`${API}/live-weather?lat=${lat}&lon=${lon}`);
      setLive(r.data);
    } catch {
      setLive(null);
    }
  };

  // Predict button action
  const onPredict = async () => {
    setLoading(true);
    setMessage("Predicting...");

    try {
      await fetchLive(position[0], position[1]);
      const r = await axios.get(
        `${API}/predict-seq?days=${days}&lat=${position[0]}&lon=${position[1]}`
      );
      setPredictions(r.data.predictions);
      setMessage("Prediction Ready ✔");
    } catch (err) {
      console.error(err);
      setMessage("Prediction failed ❌");
    }

    setLoading(false);
  };

  // Chart data mapping
  const chartData = predictions.map((p) => ({
    date: p.date,
    temp: p.temp_c,
    rain: p.rain_mm,
    humidity: p.humidity,
    wind: p.wind_speed,
  }));

  return (
    <div className={dark ? "page dark" : "page"}>
      {/* HEADER */}
      <header className="header">
        <h1>🌦 Tamil Nadu Weather Forecast (LSTM)</h1>
        <div className="hint">Click anywhere on the map 🌍 to get predictions</div>

        {/* Dark mode toggle */}
        <button className="dark-btn" onClick={() => setDark(!dark)}>
          {dark ? "🌞" : "🌙"}
        </button>
      </header>

      {/* MAIN CONTENT */}
      <div className="content">
        {/* LEFT PANEL */}
                {/* LEFT PANEL */}
        <div className="left">
          <MapContainer
            center={position}
            zoom={7}
            style={{ height: "560px", borderRadius: 12 }}
          >
            <TileLayer
              attribution="© OpenStreetMap contributors"
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            />
            <ClickMarker position={position} setPosition={setPosition} />
          </MapContainer>

          <div className="controls">
            <label>
              Days:{" "}
              <input
                type="number"
                min="1"
                max="14"
                value={days}
                onChange={(e) => setDays(Number(e.target.value))}
              />
            </label>

            <button
              className="btn"
              onClick={onPredict}
              disabled={loading}
            >
              {loading ? <div className="loader"></div> : "Predict"}
            </button>

            <div className="message">{message}</div>
          </div>

          {/* NEW INSIGHTS PANEL */}
          <InsightsPanel predictions={predictions} />
        </div>


        {/* RIGHT PANEL */}
        <div className="right">
          <LiveWeatherCard live={live} />
          <MiniForecastBar data={chartData} />
          <WeatherChart data={chartData} />

          {/* TABLE */}
          <div className="card">
            <h3>Predictions</h3>

            <table className="pred-table">
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Temp (°C)</th>
                  <th>Rain (mm)</th>
                  <th>Humidity (%)</th>
                  <th>Wind (m/s)</th>
                  <th>Flood Risk</th>
                </tr>
              </thead>

              <tbody>
                {predictions.length === 0 ? (
                  <tr>
                    <td colSpan="4" className="muted">
                      No data yet ❌ Click Predict
                    </td>
                  </tr>
                ) : (
                  predictions.map((p, i) => (
                    <tr key={i}>
                      <td>{p.date}</td>
                      <td>{p.temp_c}</td>
                      <td>{p.rain_mm}</td>
                      <td>{p.humidity}</td>
                      <td>{p.wind_speed}</td>
                      <td>{p.flood_risk}</td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* AI Weather Assistant */}
      <AIAssistant lat={position[0]} lon={position[1]} live={live} />
    </div>
  );
}
