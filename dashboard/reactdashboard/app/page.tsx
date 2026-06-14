"use client"

import React, { useEffect, useState } from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";

type TelemetryPayload = {
  timestamp: string;
  raw: { ax: number[]; ay: number[]; az: number[] }; //getting raw data as array
  features: {
    rms: { x: number; y: number; z: number };
    band_power: { low: number; mid: number; high: number };
    band_level: string;
  };
  fft: { frequencies: number[]; magnitudes: number[] };
  inference: {
    prediction: string;
    confidence: number;
    scores: Record<string, number>;
  };
  status: {
    fan_state: string;
    sensor_health: string;
    connection: string;
  };
};

export default function Home() {
  const [connected, setConnected] = useState(false);
  const [data, setData] = useState<TelemetryPayload | null>(null);

  useEffect(() => {
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    const host = window.location.hostname || "localhost";
    const port = "8000";
    const wsUrl = `${protocol}://${host}:${port}/ws/dashboard`;

    const ws = new WebSocket(wsUrl);
    ws.addEventListener("open", () => setConnected(true));
    ws.addEventListener("close", () => setConnected(false));
    ws.addEventListener("message", (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        if (msg.type === "snapshot" && msg.data) {
          setData(msg.data);
        }
      } catch (e) {
        console.error("ws parse error", e);
      }
    });

    return () => {
      try {
        ws.close();
      } catch {}
    };
  }, []);

  if (!data) {
    return (
      <div className="p-8 text-center">
        <h1 className="text-2xl font-bold">IoT Predictive Maintenance Dashboard</h1>
        <p className="mt-4 text-gray-500">Connecting...</p>
      </div>
    );
  }


  const getConfidenceColor = (confidence: number) => {
    if (confidence > 0.7) return "text-red-600";
    if (confidence > 0.4) return "text-yellow-600";
    return "text-green-600";
  };

  
  const getStatusColor = (status: string) => {
    if (status === "OK" || status === "Running" || status === "Connected") return "bg-green-100 text-green-800";
    if (status === "Warning") return "bg-yellow-100 text-yellow-800";
    return "bg-red-100 text-red-800";
  };

  // Prepare FFT data for Recharts
  const fftChartData = data.fft.frequencies.map((freq, idx) => ({
    frequency: freq,
    magnitude: data.fft.magnitudes[idx] || 0,
  }));

  // Prepare class scores for Recharts
  const classScoreData = Object.entries(data.inference.scores).map(([className, score]) => ({
    name: className,
    score: parseFloat((score * 100).toFixed(1)),
  }));

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 to-slate-800 text-white p-8">
      {/* Header */}
      <header className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold">🔧 IoT Predictive Maintenance</h1>
          <p className="text-sm text-gray-400">Live Sensor & Inference Dashboard</p>
        </div>
        <div className="flex gap-4">
          <div className={`px-4 py-2 rounded-lg font-semibold ${connected ? "bg-green-600" : "bg-red-600"}`}>
            {connected ? "🟢 Connected" : "🔴 Disconnected"}
          </div>
          <div className="text-sm text-gray-400">{data.timestamp}</div>
        </div>
      </header>

      {/* Status Cards */}
      <section className="grid grid-cols-3 gap-4 mb-8">
        <div className="bg-slate-700 rounded-lg p-4">
          <div className="text-sm text-gray-400">Fan State</div>
          <div className={`text-xl font-bold mt-2 ${getStatusColor(data.status.fan_state).split(" ")[0]}`}>
            {data.status.fan_state}
          </div>
        </div>
        <div className="bg-slate-700 rounded-lg p-4">
          <div className="text-sm text-gray-400">Sensor Health</div>
          <div className={`text-xl font-bold mt-2 px-2 py-1 rounded inline-block ${getStatusColor(data.status.sensor_health)}`}>
            {data.status.sensor_health}
          </div>
        </div>
        <div className="bg-slate-700 rounded-lg p-4">
          <div className="text-sm text-gray-400">Connection</div>
          <div className={`text-xl font-bold mt-2 px-2 py-1 rounded inline-block ${getStatusColor(data.status.connection)}`}>
            {data.status.connection}
          </div>
        </div>
      </section>

      {/* Raw Acceleration Data */}
      <section className="bg-slate-700 rounded-lg p-6 mb-8">
        <h2 className="text-2xl font-bold mb-4">📊 Raw Acceleration</h2>
        <div className="grid grid-cols-3 gap-4">
          <div>
            <div className="text-sm text-gray-400">Ax (X-axis)</div>
            <div className="text-lg font-mono bg-slate-800 rounded p-2 mt-2">
              {data.raw.ax.map((v, i) => (
                <div key={i}>{v.toFixed(3)} g</div>
              ))}
            </div>
          </div>
          <div>
            <div className="text-sm text-gray-400">Ay (Y-axis)</div>
            <div className="text-lg font-mono bg-slate-800 rounded p-2 mt-2">
              {data.raw.ay.map((v, i) => (
                <div key={i}>{v.toFixed(3)} g</div>
              ))}
            </div>
          </div>
          <div>
            <div className="text-sm text-gray-400">Az (Z-axis)</div>
            <div className="text-lg font-mono bg-slate-800 rounded p-2 mt-2">
              {data.raw.az.map((v, i) => (
                <div key={i}>{v.toFixed(3)} g</div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="grid grid-cols-2 gap-8 mb-8">
        {/* RMS & Band Power */}
        <div className="bg-slate-700 rounded-lg p-6">
          <h2 className="text-2xl font-bold mb-4">⚡ Features</h2>
          <div className="space-y-4">
            <div className="bg-slate-800 rounded p-4">
              <div className="text-sm text-gray-400">RMS (Root Mean Square)</div>
              <div className="grid grid-cols-3 gap-4 mt-2">
                <div>
                  <span className="text-xs">X:</span>
                  <div className="text-lg font-mono">{data.features.rms.x.toFixed(4)}</div>
                </div>
                <div>
                  <span className="text-xs">Y:</span>
                  <div className="text-lg font-mono">{data.features.rms.y.toFixed(4)}</div>
                </div>
                <div>
                  <span className="text-xs">Z:</span>
                  <div className="text-lg font-mono">{data.features.rms.z.toFixed(4)}</div>
                </div>
              </div>
            </div>
            <div className="bg-slate-800 rounded p-4">
              <div className="text-sm text-gray-400">Band Power</div>
              <div className="grid grid-cols-3 gap-4 mt-2">
                <div>
                  <span className="text-xs">Low:</span>
                  <div className="text-lg font-mono">{(data.features.band_power.low * 100).toFixed(1)}%</div>
                </div>
                <div>
                  <span className="text-xs">Mid:</span>
                  <div className="text-lg font-mono">{(data.features.band_power.mid * 100).toFixed(1)}%</div>
                </div>
                <div>
                  <span className="text-xs">High:</span>
                  <div className="text-lg font-mono">{(data.features.band_power.high * 100).toFixed(1)}%</div>
                </div>
              </div>
            </div>
            <div className="bg-slate-800 rounded p-4">
              <div className="text-sm text-gray-400">Band Level</div>
              <div className="text-lg font-bold mt-2 capitalize">{data.features.band_level}</div>
            </div>
          </div>
        </div>

        {/* FFT Data with Recharts */}
        <div className="bg-slate-700 rounded-lg p-6">
          <h2 className="text-2xl font-bold mb-4">🌊 FFT Spectrum</h2>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart
              data={fftChartData}
              margin={{ top: 20, right: 30, left: 0, bottom: 60 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#475569" />
              <XAxis 
                dataKey="frequency" 
                stroke="#94a3b8"
                angle={-45}
                textAnchor="end"
                height={100}
              />
              <YAxis 
                stroke="#94a3b8"
              />
              <Tooltip 
                contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569', borderRadius: '8px', color: '#e2e8f0' }}
                formatter={(value) => (typeof value === 'number' ? value.toFixed(4) : value)}
              />
              <Bar dataKey="magnitude" fill="#3b82f6" isAnimationActive={true} radius={[8, 8, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </section>

      {/* Inference Results */}
      <section className="bg-slate-700 rounded-lg p-6">
        <h2 className="text-2xl font-bold mb-4">🤖 AI Inference Results</h2>
        <div className="grid grid-cols-2 gap-8">
          {/* Prediction Card */}
          <div className="bg-slate-800 rounded p-6">
            <div className="text-sm text-gray-400 mb-2">Prediction</div>
            <div className={`text-3xl font-bold ${getConfidenceColor(data.inference.confidence)}`}>
              {data.inference.prediction}
            </div>
            <div className={`text-lg font-semibold mt-2 ${getConfidenceColor(data.inference.confidence)}`}>
              Confidence: {(data.inference.confidence * 100).toFixed(1)}%
            </div>
            <div className="mt-4 bg-slate-900 rounded h-3 overflow-hidden">
              <div
                className={`h-full transition-all ${
                  data.inference.confidence > 0.7
                    ? "bg-red-500"
                    : data.inference.confidence > 0.4
                    ? "bg-yellow-500"
                    : "bg-green-500"
                }`}
                style={{ width: `${data.inference.confidence * 100}%` }}
              />
            </div>
          </div>

          {/* Class Scores with Recharts */}
          <div className="bg-slate-800 rounded p-6">
            <div className="text-sm text-gray-400 mb-4">Class Scores</div>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart
                data={classScoreData}
                margin={{ top: 20, right: 30, left: 0, bottom: 60 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis 
                  dataKey="name" 
                  stroke="#94a3b8" 
                  angle={-45}
                  textAnchor="end"
                  height={100}
                />
                <YAxis stroke="#94a3b8" />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569', borderRadius: '8px', color: '#e2e8f0' }}
                  formatter={(value) => `${typeof value === 'number' ? value.toFixed(1) : value}%`}
                />
                <Bar dataKey="score" fill="#06b6d4" isAnimationActive={true} radius={[8, 8, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </section>
    </div>
  );
}
