import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { ArrowLeft, BarChart3, TrendingUp, Zap, Leaf, Gauge, Crown, Award, Clock, Target, Flame, Wind } from 'lucide-react';
import TelemetryGraph from '../components/TelemetryGraph';
import config from '../config';
import './ResultsComparison.css';

const ResultsComparison = () => {
  const [comparisonData, setComparisonData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [animatedValues, setAnimatedValues] = useState({
    aiSpeed: 0,
    hamiltonSpeed: 0,
    co2Saved: 0,
    efficiency: 0
  });

  useEffect(() => {
    loadComparisonData();
  }, []);

  // Animate values when data loads
  useEffect(() => {
    if (comparisonData) {
      const animateValue = (key, targetValue, duration = 2000) => {
        const startValue = 0;
        const startTime = Date.now();
        
        const animate = () => {
          const elapsed = Date.now() - startTime;
          const progress = Math.min(elapsed / duration, 1);
          const currentValue = startValue + (targetValue - startValue) * progress;
          
          setAnimatedValues(prev => ({ ...prev, [key]: currentValue }));
          
          if (progress < 1) {
            requestAnimationFrame(animate);
          }
        };
        
        requestAnimationFrame(animate);
      };

      // Animate each value with slight delays
      setTimeout(() => animateValue('aiSpeed', comparisonData.ai.average_speed), 100);
      setTimeout(() => animateValue('hamiltonSpeed', comparisonData.hamilton.average_speed), 300);
      setTimeout(() => animateValue('co2Saved', comparisonData.comparison.co2_saved || 0), 500);
      setTimeout(() => animateValue('efficiency', comparisonData.ai.efficiency_gain), 700);
    }
  }, [comparisonData]);

  const loadComparisonData = async () => {
    try {
      setLoading(true);
      
      // First check if we have stored simulation results
      const storedResults = localStorage.getItem('simulationResults');
      if (storedResults) {
        const results = JSON.parse(storedResults);
        setComparisonData(results);
        setLoading(false);
        return;
      }
      
      // Otherwise, get track from simulation config or default to Silverstone
      const simConfig = JSON.parse(localStorage.getItem('simulationConfig') || '{}');
      const track = simConfig.track || 'Silverstone';
      
      const response = await fetch(config.getApiUrl(config.endpoints.comparison(track)));
      const result = await response.json();
      
      if (result.success) {
        setComparisonData(result.data);
      } else {
        throw new Error(result.error || 'Failed to load comparison data');
      }
    } catch (err) {
      console.error('Error loading comparison data:', err);
      setError(err.message);
      // Fallback data
      setComparisonData(getFallbackData());
    } finally {
      setLoading(false);
    }
  };

  const getFallbackData = () => ({
    track: 'Silverstone',
    hamilton: {
      driver_name: 'Lewis Hamilton',
      lap_time: 87.5,
      lap_time_formatted: '1:27.500',
      energy_consumption: 2.8,
      average_speed: 185.5,
      data_source: 'FastF1 API',
      is_real_data: true
    },
    ai: {
      model: 'PPO Eco-Drive',
      lap_time: 88.2,
      lap_time_formatted: '1:28.200',
      energy_consumption: 2.4,
      average_speed: 184.2,
      efficiency_gain: 14.3
    },
    comparison: {
      time_difference: 0.7,
      energy_saved: 0.4,
      ai_faster: false,
      ai_more_efficient: true
    }
  });

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = (seconds % 60).toFixed(3);
    return `${mins}:${secs.padStart(6, '0')}`;
  };

  if (loading) {
    return (
      <div className="results-comparison">
        <div className="container">
          <div className="loading-container">
            <div className="spinner"></div>
            <p>Loading Lewis Hamilton vs AI comparison...</p>
          </div>
        </div>
      </div>
    );
  }

  if (error && !comparisonData) {
    return (
      <div className="results-comparison">
        <div className="container">
          <div className="error-container">
            <h2>Error Loading Data</h2>
            <p>{error}</p>
            <button onClick={loadComparisonData} className="retry-button">
              Retry
            </button>
          </div>
        </div>
      </div>
    );
  }

  const metrics = [
    {
      icon: Gauge,
      title: 'Lap Time',
      aiValue: comparisonData.ai.lap_time_formatted,
      hamiltonValue: comparisonData.hamilton.lap_time_formatted,
      improvement: comparisonData.comparison.ai_faster ? 
        `-${Math.abs(comparisonData.comparison.time_difference).toFixed(3)}s` : 
        `+${comparisonData.comparison.time_difference.toFixed(3)}s`,
      color: comparisonData.comparison.ai_faster ? '#00ff88' : '#ff6b35',
      better: comparisonData.comparison.ai_faster ? 'AI' : 'Hamilton'
    },
    {
      icon: Zap,
      title: 'Energy Consumption',
      aiValue: `${comparisonData.ai.energy_consumption} kWh`,
      hamiltonValue: `${comparisonData.hamilton.energy_consumption} kWh`,
      improvement: `-${comparisonData.comparison.energy_saved.toFixed(1)} kWh`,
      color: '#00ff88',
      better: 'AI'
    },
    {
      icon: Leaf,
      title: 'CO₂ Savings',
      aiValue: `${comparisonData.comparison.co2_saved_formatted || '0.000 kg'}`,
      hamiltonValue: '0.000 kg',
      improvement: `+${comparisonData.comparison.co2_saved_formatted || '0.000 kg'}`,
      color: '#00ff88',
      better: 'AI'
    },
    {
      icon: TrendingUp,
      title: 'Average Speed',
      aiValue: `${comparisonData.ai.average_speed.toFixed(1)} km/h`,
      hamiltonValue: `${comparisonData.hamilton.average_speed.toFixed(1)} km/h`,
      improvement: comparisonData.ai.average_speed > comparisonData.hamilton.average_speed ? 
        `+${(comparisonData.ai.average_speed - comparisonData.hamilton.average_speed).toFixed(1)} km/h` :
        `${(comparisonData.ai.average_speed - comparisonData.hamilton.average_speed).toFixed(1)} km/h`,
      color: comparisonData.ai.average_speed > comparisonData.hamilton.average_speed ? '#00ff88' : '#ff6b35',
      better: comparisonData.ai.average_speed > comparisonData.hamilton.average_speed ? 'AI' : 'Hamilton'
    }
  ];

  return (
    <div className="results-comparison">
      <div className="container">
        <div className="page-header">
          <Link to="/dashboard" className="back-button">
            <ArrowLeft size={20} />
            Back to Dashboard
          </Link>
          <h1>AI vs Lewis Hamilton</h1>
          <p>Compare AI eco-driving performance against 7-time World Champion Lewis Hamilton</p>
        </div>

        <div className="driver-profiles">
          <div className="driver-card hamilton">
            <div className="driver-header">
              <Crown size={24} className="crown-icon" />
              <h3>Lewis Hamilton</h3>
              <span className="driver-title">7x World Champion</span>
            </div>
            <div className="driver-info">
              <p><strong>Team:</strong> Mercedes</p>
              <p><strong>Data Source:</strong> {comparisonData.hamilton.data_source}</p>
              <p><strong>Track:</strong> {comparisonData.track}</p>
              {comparisonData.hamilton.is_real_data && (
                <span className="real-data-badge">Real F1 Data</span>
              )}
            </div>
          </div>

          <div className="driver-card ai">
            <div className="driver-header">
              <Award size={24} className="award-icon" />
              <h3>AI Eco-Driver</h3>
              <span className="driver-title">{comparisonData.ai.model}</span>
            </div>
            <div className="driver-info">
              <p><strong>Algorithm:</strong> PPO Reinforcement Learning</p>
              <p><strong>Focus:</strong> Energy Efficiency</p>
              <p><strong>Training:</strong> Multi-track simulation</p>
              <span className="ai-badge">AI Powered</span>
            </div>
          </div>
        </div>

        {/* Visual Performance Dashboard */}
        <div className="visual-dashboard">
          <div className="dashboard-card">
            <h2>Speed Comparison</h2>
            <div className="speedometer-container">
              <div className="speedometer">
                <div className="speedometer-ai">
                  <div className="speed-label">AI Speed</div>
                  <div className="speed-value">{Math.round(animatedValues.aiSpeed)}</div>
                  <div className="speed-unit">km/h</div>
                </div>
                <div className="speedometer-hamilton">
                  <div className="speed-label">Hamilton Speed</div>
                  <div className="speed-value">{Math.round(animatedValues.hamiltonSpeed)}</div>
                  <div className="speed-unit">km/h</div>
                </div>
              </div>
            </div>
          </div>

          <div className="dashboard-card">
            <h2>Environmental Impact</h2>
            <div className="environmental-impact">
              <div className="co2-visualization">
                <div className="co2-icon">
                  <Leaf size={48} />
                </div>
                <div className="co2-stats">
                  <div className="co2-saved">
                    <span className="co2-value">{animatedValues.co2Saved.toFixed(3)}</span>
                    <span className="co2-unit">kg CO₂</span>
                  </div>
                  <div className="co2-label">Saved per lap</div>
                </div>
              </div>
              <div className="efficiency-bar">
                <div className="efficiency-label">Efficiency Gain</div>
                <div className="progress-bar">
                  <div 
                    className="progress-fill" 
                    style={{ width: `${Math.min(animatedValues.efficiency, 100)}%` }}
                  ></div>
                </div>
                <div className="efficiency-value">{animatedValues.efficiency.toFixed(1)}%</div>
              </div>
            </div>
          </div>

          <div className="dashboard-card">
            <h2>Performance Summary</h2>
            <div className="performance-summary">
              <div className="summary-item">
                <Clock size={24} />
                <div className="summary-content">
                  <div className="summary-label">Lap Time</div>
                  <div className="summary-value">
                    {comparisonData.comparison.ai_faster ? 'AI Faster' : 'Hamilton Faster'}
                  </div>
                  <div className="summary-detail">
                    {Math.abs(comparisonData.comparison.time_difference).toFixed(3)}s difference
                  </div>
                </div>
              </div>
              <div className="summary-item">
                <Target size={24} />
                <div className="summary-content">
                  <div className="summary-label">Energy Efficiency</div>
                  <div className="summary-value">AI Superior</div>
                  <div className="summary-detail">
                    {comparisonData.comparison.energy_saved.toFixed(1)} kWh saved
                  </div>
                </div>
              </div>
              <div className="summary-item">
                <Wind size={24} />
                <div className="summary-content">
                  <div className="summary-label">Environmental</div>
                  <div className="summary-value">Eco-Friendly</div>
                  <div className="summary-detail">
                    {comparisonData.comparison.co2_saved_formatted || '0.000 kg'} CO₂ saved
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="results-content">
          <div className="comparison-card">
            <h2>Performance Metrics</h2>
            <div className="metrics-grid">
              {metrics.map((metric, index) => (
                <div key={index} className="metric-item">
                  <div className="metric-header">
                    <metric.icon size={20} style={{ color: metric.color }} />
                    <h3>{metric.title}</h3>
                  </div>
                  <div className="metric-values">
                    <div className="value-group">
                      <div className="value-label">AI</div>
                      <div className="ai-value">{metric.aiValue}</div>
                    </div>
                    <div className="improvement" style={{ color: metric.color }}>
                      {metric.improvement}
                    </div>
                    <div className="value-group">
                      <div className="value-label">Hamilton</div>
                      <div className="hamilton-value">{metric.hamiltonValue}</div>
                    </div>
                  </div>
                  <div className="winner-indicator">
                    <span className={`winner-badge ${metric.better.toLowerCase()}`}>
                      {metric.better} Wins
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="chart-section">
            <h2>Performance Analysis</h2>
            <div className="analysis-content">
              <div className="analysis-summary">
                <h4>Key Insights:</h4>
                <ul>
                  <li>
                    <strong>Energy Efficiency:</strong> AI achieves {comparisonData.ai.efficiency_gain}% better energy efficiency
                  </li>
                  <li>
                    <strong>Lap Time:</strong> {comparisonData.comparison.ai_faster ? 'AI is faster' : 'Hamilton is faster'} by {Math.abs(comparisonData.comparison.time_difference).toFixed(3)}s
                  </li>
                  <li>
                    <strong>Energy Savings:</strong> AI saves {comparisonData.comparison.energy_saved.toFixed(1)} kWh per lap
                  </li>
                  <li>
                    <strong>Data Source:</strong> Hamilton's data from {comparisonData.hamilton.data_source}
                  </li>
                </ul>
              </div>
            </div>
          </div>
        </div>

        {/* Telemetry Analysis */}
        <TelemetryGraph track={comparisonData.track} />
      </div>
    </div>
  );
};

export default ResultsComparison; 