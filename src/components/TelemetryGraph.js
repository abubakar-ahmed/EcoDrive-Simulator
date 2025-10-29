import React, { useState, useEffect } from 'react';
import config from '../config';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import './TelemetryGraph.css';

const TelemetryGraph = ({ track }) => {
  const [telemetryData, setTelemetryData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadTelemetryData();
  }, [track]);

  const loadTelemetryData = async () => {
    try {
      setLoading(true);
      setError(null);

      const response = await fetch(config.getApiUrl(config.endpoints.telemetry), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          duration: 120, // 2 minutes
          points: 200
        })
      });

      const result = await response.json();
      
      if (result.success) {
        setTelemetryData(result);
      } else {
        throw new Error(result.error || 'Failed to load telemetry data');
      }
    } catch (err) {
      console.error('Error loading telemetry data:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const formatTime = (time) => {
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  };

  const formatSpeed = (speed) => {
    return `${Math.round(speed)} km/h`;
  };

  const formatPercentage = (value) => {
    return `${Math.round(value)}%`;
  };

  if (loading) {
    return (
      <div className="telemetry-graph loading">
        <div className="loading-spinner"></div>
        <p>Loading telemetry data...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="telemetry-graph error">
        <p>Error loading telemetry: {error}</p>
        <button onClick={loadTelemetryData} className="retry-button">
          Retry
        </button>
      </div>
    );
  }

  if (!telemetryData || !telemetryData.ai_telemetry) {
    return (
      <div className="telemetry-graph no-data">
        <p>No telemetry data available</p>
        <button onClick={loadTelemetryData} className="retry-button">
          Load Data
        </button>
      </div>
    );
  }

  // Prepare data for the chart
  const chartData = telemetryData.ai_telemetry.map((point, index) => ({
    time: formatTime(point.time),
    timeValue: point.time,
    aiSpeed: point.speed,
    aiThrottle: point.throttle,
    aiBrake: point.brake,
    aiSteering: point.steering,
    aiEnergy: point.energy || 0,
    aiCo2Saved: point.co2_saved || 0,
    hamiltonSpeed: telemetryData.hamilton_telemetry[index]?.speed || 0,
    hamiltonThrottle: telemetryData.hamilton_telemetry[index]?.throttle || 0,
    hamiltonBrake: telemetryData.hamilton_telemetry[index]?.brake || 0,
    hamiltonSteering: telemetryData.hamilton_telemetry[index]?.steering || 0,
    hamiltonEnergy: telemetryData.hamilton_telemetry[index]?.energy || 0
  }));

  return (
    <div className="telemetry-graph">
      <div className="telemetry-header">
        <h3>Live Telemetry Analysis</h3>
        <div className="telemetry-controls">
          <button onClick={loadTelemetryData} className="refresh-button">
            Refresh Data
          </button>
        </div>
      </div>

      <div className="telemetry-charts">
        {/* Speed Comparison */}
        <div className="chart-container">
          <h4>Speed Comparison</h4>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
              <XAxis 
                dataKey="time" 
                stroke="rgba(255,255,255,0.7)"
                fontSize={12}
              />
              <YAxis 
                stroke="rgba(255,255,255,0.7)"
                fontSize={12}
                label={{ value: 'km/h', angle: -90, position: 'insideLeft' }}
              />
              <Tooltip 
                formatter={(value, name) => [formatSpeed(value), name]}
                labelFormatter={(label) => `Time: ${label}`}
                contentStyle={{
                  backgroundColor: 'rgba(0,0,0,0.8)',
                  border: '1px solid rgba(255,255,255,0.2)',
                  borderRadius: '8px',
                  color: '#ffffff'
                }}
              />
              <Legend />
              <Line 
                type="monotone" 
                dataKey="aiSpeed" 
                stroke="#00ff88" 
                strokeWidth={2}
                name="AI Speed"
                dot={false}
              />
              {telemetryData.hamilton_telemetry.length > 0 && (
                <Line 
                  type="monotone" 
                  dataKey="hamiltonSpeed" 
                  stroke="#ff6b35" 
                  strokeWidth={2}
                  name="Hamilton Speed"
                  dot={false}
                />
              )}
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Throttle & Brake */}
        <div className="chart-container">
          <h4>Throttle & Brake Input</h4>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
              <XAxis 
                dataKey="time" 
                stroke="rgba(255,255,255,0.7)"
                fontSize={12}
              />
              <YAxis 
                stroke="rgba(255,255,255,0.7)"
                fontSize={12}
                label={{ value: '%', angle: -90, position: 'insideLeft' }}
              />
              <Tooltip 
                formatter={(value, name) => [formatPercentage(value), name]}
                labelFormatter={(label) => `Time: ${label}`}
                contentStyle={{
                  backgroundColor: 'rgba(0,0,0,0.8)',
                  border: '1px solid rgba(255,255,255,0.2)',
                  borderRadius: '8px',
                  color: '#ffffff'
                }}
              />
              <Legend />
              <Line 
                type="monotone" 
                dataKey="aiThrottle" 
                stroke="#00ff88" 
                strokeWidth={2}
                name="AI Throttle"
                dot={false}
              />
              <Line 
                type="monotone" 
                dataKey="aiBrake" 
                stroke="#ff4444" 
                strokeWidth={2}
                name="AI Brake"
                dot={false}
              />
              {telemetryData.hamilton_telemetry.length > 0 && (
                <>
                  <Line 
                    type="monotone" 
                    dataKey="hamiltonThrottle" 
                    stroke="#ff6b35" 
                    strokeWidth={2}
                    strokeDasharray="5 5"
                    name="Hamilton Throttle"
                    dot={false}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="hamiltonBrake" 
                    stroke="#ff9999" 
                    strokeWidth={2}
                    strokeDasharray="5 5"
                    name="Hamilton Brake"
                    dot={false}
                  />
                </>
              )}
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Steering */}
        <div className="chart-container">
          <h4>Steering Input</h4>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
              <XAxis 
                dataKey="time" 
                stroke="rgba(255,255,255,0.7)"
                fontSize={12}
              />
              <YAxis 
                stroke="rgba(255,255,255,0.7)"
                fontSize={12}
                label={{ value: '%', angle: -90, position: 'insideLeft' }}
              />
              <Tooltip 
                formatter={(value, name) => [formatPercentage(value), name]}
                labelFormatter={(label) => `Time: ${label}`}
                contentStyle={{
                  backgroundColor: 'rgba(0,0,0,0.8)',
                  border: '1px solid rgba(255,255,255,0.2)',
                  borderRadius: '8px',
                  color: '#ffffff'
                }}
              />
              <Legend />
              <Line 
                type="monotone" 
                dataKey="aiSteering" 
                stroke="#00ff88" 
                strokeWidth={2}
                name="AI Steering"
                dot={false}
              />
              {telemetryData.hamilton_telemetry.length > 0 && (
                <Line 
                  type="monotone" 
                  dataKey="hamiltonSteering" 
                  stroke="#ff6b35" 
                  strokeWidth={2}
                  strokeDasharray="5 5"
                  name="Hamilton Steering"
                  dot={false}
                />
              )}
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Energy & CO₂ */}
        <div className="chart-container">
          <h4>Energy Consumption & CO₂ Savings</h4>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
              <XAxis 
                dataKey="time" 
                stroke="rgba(255,255,255,0.7)"
                fontSize={12}
              />
              <YAxis 
                stroke="rgba(255,255,255,0.7)"
                fontSize={12}
              />
              <Tooltip 
                formatter={(value, name) => {
                  if (name.includes('CO₂')) {
                    return [`${value.toFixed(3)} kg`, name];
                  }
                  return [`${value.toFixed(2)} kWh`, name];
                }}
                labelFormatter={(label) => `Time: ${label}`}
                contentStyle={{
                  backgroundColor: 'rgba(0,0,0,0.8)',
                  border: '1px solid rgba(255,255,255,0.2)',
                  borderRadius: '8px',
                  color: '#ffffff'
                }}
              />
              <Legend />
              <Line 
                type="monotone" 
                dataKey="aiEnergy" 
                stroke="#00ff88" 
                strokeWidth={2}
                name="AI Energy"
                dot={false}
              />
              <Line 
                type="monotone" 
                dataKey="aiCo2Saved" 
                stroke="#88ff00" 
                strokeWidth={2}
                name="AI CO₂ Saved"
                dot={false}
              />
              {telemetryData.hamilton_telemetry.length > 0 && (
                <Line 
                  type="monotone" 
                  dataKey="hamiltonEnergy" 
                  stroke="#ff6b35" 
                  strokeWidth={2}
                  strokeDasharray="5 5"
                  name="Hamilton Energy"
                  dot={false}
                />
              )}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="telemetry-summary">
        <div className="summary-stats">
          <div className="stat-item">
            <span className="stat-label">Total Points:</span>
            <span className="stat-value">{telemetryData.points}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Duration:</span>
            <span className="stat-value">{formatTime(telemetryData.duration)}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Data Source:</span>
            <span className="stat-value">
              {telemetryData.hamilton_telemetry.length > 0 ? 'Real Simulation' : 'Mock Data'}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TelemetryGraph;
