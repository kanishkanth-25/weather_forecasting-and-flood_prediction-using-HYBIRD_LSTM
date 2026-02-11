import React from "react";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, Legend, ResponsiveContainer
} from "recharts";

export default function WeatherChart({ data }) {
  if (!data || data.length === 0) return null;

  return (
    <div className="card">
      <h3>Temperature, Rainfall, Humidity & Wind Trend</h3>

      <ResponsiveContainer width="100%" height={260}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="date" />
          <YAxis />
          <Tooltip />
          <Legend />

          <Line type="monotone" dataKey="temp" stroke="#ef4444" strokeWidth={2} name="Temp (°C)" />
          <Line type="monotone" dataKey="rain" stroke="#3b82f6" strokeWidth={2} name="Rain (mm)" />
          <Line type="monotone" dataKey="humidity" stroke="#10b981" strokeWidth={2} name="Humidity (%)" />
          <Line type="monotone" dataKey="wind" stroke="#8b5cf6" strokeWidth={2} name="Wind (m/s)" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
