import React, { useState, useEffect, useRef } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { 
  Play, 
  Pause, 
  RotateCcw, 
  ArrowLeft,
  Zap,
  Gauge,
  Leaf,
  Clock,
  MapPin,
  Car,
  AlertCircle,
  Square,
} from 'lucide-react';
import config from '../config';
import './SimulationDashboard.css';

const SimulationDashboard = () => {
  const navigate = useNavigate();
  const [isRunning, setIsRunning] = useState(false);
  const [currentLap, setCurrentLap] = useState(0);
  const [totalLaps, setTotalLaps] = useState(3);
  const [simulationTime, setSimulationTime] = useState(0);
  const [simulationId, setSimulationId] = useState(null);
  const [webSimulationAvailable, setWebSimulationAvailable] = useState(false);
  const [simulationFrame, setSimulationFrame] = useState(null);
  const [simulationError, setSimulationError] = useState(null);
  const [fallbackMode, setFallbackMode] = useState(false);
  const eventSourceRef = useRef(null);
  
  // Basic simulation data
  const [simulationData, setSimulationData] = useState({
    ai: {
      lapTime: 0,
      energyConsumption: 0,
      co2Saved: 0,
      speed: 0,
      throttle: 0,
      brake: 0,
    },
    human: {
      lapTime: 0,
      energyConsumption: 0,
      co2Saved: 0,
      speed: 0,
      throttle: 0,
      brake: 0,
    }
  });

  useEffect(() => {
    // Load simulation config
    const config = JSON.parse(localStorage.getItem('simulationConfig') || '{}');
    if (!config.track) {
      navigate('/simulator');
      return;
    }

    // Check if web simulation is available
    checkWebSimulationAvailability();
  }, [navigate]);

  const checkWebSimulationAvailability = async () => {
    try {
      const response = await fetch(config.getApiUrl(config.endpoints.simulation.status));
      if (response.ok) {
        setWebSimulationAvailable(true);
      }
    } catch (error) {
      setWebSimulationAvailable(false);
    }
  };

  const startSimulation = async () => {
    try {
      const simConfig = JSON.parse(localStorage.getItem('simulationConfig') || '{}');
      const savedSimulationId = localStorage.getItem('simulationId');
      
      // Store config for use in completion handler
      window.simulationConfig = simConfig;
      
      if (savedSimulationId) {
        // Use existing simulation ID
        setSimulationId(savedSimulationId);
        setIsRunning(true);
        setSimulationError(null);
        
        // Start streaming if web simulation is available
        if (webSimulationAvailable) {
          startFrameStreaming();
        }
      } else {
        // Start new simulation
        const response = await fetch(config.getApiUrl(config.endpoints.simulation.start), {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            track: simConfig.track,
            model_version: simConfig.model_version || 'ppo',
            user: 'web_user'
          })
        });

        const result = await response.json();
        
        if (result.simulation_id) {
          setSimulationId(result.simulation_id);
          localStorage.setItem('simulationId', result.simulation_id);
          setIsRunning(true);
          setSimulationError(null);
          
          // If web simulation is available, start streaming frames
          if (result.web_simulation && webSimulationAvailable) {
            startFrameStreaming();
          }
        } else {
          const errorMsg = result.error || 'Failed to start simulation';
          console.error('Simulation start failed:', errorMsg);
          setSimulationError(errorMsg);
          
          // If F1Tenth is not available, start mock simulation
          if (errorMsg.includes('F1Tenth')) {
            setFallbackMode(true);
            startMockSimulation();
          }
        }
      }
    } catch (error) {
      console.error('Failed to start simulation:', error);
      setSimulationError('Failed to start simulation: ' + error.message);
      
      // Fallback to mock simulation
      setFallbackMode(true);
      startMockSimulation();
    }
  };

  const startMockSimulation = () => {
    setIsRunning(true);
    setSimulationError(null);
    
    // Simulate a lap over 90 seconds
    let progress = 0;
    const interval = setInterval(() => {
      progress += 0.01; // 1% every 100ms
      const lapTime = progress * 90; // 90 second lap
      
      setSimulationTime(lapTime);
      setCurrentLap(Math.floor(progress * 10));
      
      // Update simulation data
      setSimulationData(prev => ({
        ...prev,
        ai: {
          lapTime: lapTime,
          energyConsumption: lapTime * 0.03,
          co2Saved: lapTime * 0.015,
          speed: 120 + Math.sin(progress * Math.PI) * 40,
          throttle: 60 + Math.sin(progress * Math.PI * 2) * 20,
          brake: 20 + Math.sin(progress * Math.PI * 3) * 15,
        }
      }));
      
      // Add mock frame data
      const mockFrame = {
        step: Math.floor(progress * 1000),
        time: lapTime,
        speed: 120 + Math.sin(progress * Math.PI) * 40,
        position: [progress * 100, Math.sin(progress * Math.PI) * 50],
        progress: progress
      };
      setSimulationFrame(JSON.stringify(mockFrame));
      
      if (progress >= 1.0) {
        clearInterval(interval);
        setIsRunning(false);
        
        // Get config
        const config = window.simulationConfig || JSON.parse(localStorage.getItem('simulationConfig') || '{}');
        
        // Store mock results in the format expected by ResultsComparison
        const mockResults = {
          track: config.track || 'Silverstone',
          hamilton: {
            driver_name: 'Lewis Hamilton',
            lap_time: 89.2,
            lap_time_formatted: "1:29.200",
            energy_consumption: 2.8,
            average_speed: 185.5,
            data_source: 'Mock Data',
            is_real_data: false
          },
          ai: {
            model: config.model_version?.toUpperCase() || 'PPO',
            lap_time: 87.5,
            lap_time_formatted: "1:27.500",
            energy_consumption: 2.4,
            average_speed: 188.2,
            efficiency_gain: 14.3
          },
          comparison: {
            time_difference: -1.7,
            energy_saved: 0.4,
            co2_saved: 0.000,
            co2_saved_formatted: "0.000 kg",
            ai_faster: true,
            ai_more_efficient: true
          }
        };
        localStorage.setItem('simulationResults', JSON.stringify(mockResults));
      }
    }, 100);
    
    // Store interval reference for cleanup
    eventSourceRef.current = { close: () => clearInterval(interval) };
  };

  const startFrameStreaming = () => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }

    eventSourceRef.current = new EventSource(config.getApiUrl(config.endpoints.simulation.stream));
    
    eventSourceRef.current.onopen = () => {
      // Connection opened
    };
    
    eventSourceRef.current.onerror = (error) => {
      console.error('EventSource error:', error);
    };
    
    eventSourceRef.current.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        // Update simulation data with live values
        setSimulationData(prev => {
          const newAIData = {
            ...prev.ai,
            lapTime: data.lap_time || 0,
            energyConsumption: data.energy_consumption || 0,
            co2Saved: data.co2_saved || 0,
            speed: data.current_speed || 0,
            throttle: data.current_throttle || 0,
            brake: data.current_brake || 0,
            progress: data.progress || 0,
          };
          
          return {
            ...prev,
            ai: newAIData,
            human: {
              ...prev.human,
            lapTime: data.hamilton_data?.lap_time || 0,
            energyConsumption: 0,
            co2Saved: 0,
            speed: 0,
            throttle: 0,
            brake: 0,
          }
          };
        });
        
        // Update simulation time and progress
        setSimulationTime(data.lap_time || 0);
        setCurrentLap(Math.floor((data.progress || 0) * 10));
        
        // Update video frame (only check current_frame now, no more video_frames array)
        if (data.current_frame) {
          setSimulationFrame(data.current_frame);
        }
        
        // Check if simulation completed
        if (data.status === 'completed') {
          setIsRunning(false);
          
          // Store results for comparison page
          if (data.comparison_data) {
            // Get config from window (set when simulation starts)
            const config = window.simulationConfig || JSON.parse(localStorage.getItem('simulationConfig') || '{}');
            
            // Transform comparison_data to match ResultsComparison format
            const transformedData = {
              track: config.track || 'Silverstone',
              hamilton: {
                driver_name: data.comparison_data.hamilton_data?.driver_name || 'Lewis Hamilton',
                lap_time: data.comparison_data.hamilton_lap_time,
                lap_time_formatted: data.comparison_data.hamilton_lap_time_formatted,
                energy_consumption: data.comparison_data.hamilton_data?.energy_consumption || 2.8,
                average_speed: data.comparison_data.hamilton_data?.average_speed || 185.5,
                data_source: data.comparison_data.hamilton_data?.data_source || 'FastF1 API',
                is_real_data: data.comparison_data.hamilton_data?.is_real_data !== false
              },
              ai: {
                model: config.model_version?.toUpperCase() || 'PPO',
                lap_time: data.comparison_data.ai_lap_time,
                lap_time_formatted: data.comparison_data.ai_lap_time_formatted,
                energy_consumption: data.energy_consumption || 2.4,
                average_speed: data.average_speed || data.current_speed || 200.0,
                efficiency_gain: data.comparison_data.hamilton_data?.energy_consumption ? 
                  ((data.comparison_data.hamilton_data.energy_consumption - (data.energy_consumption || 2.4)) / data.comparison_data.hamilton_data.energy_consumption * 100).toFixed(1) : 14.3
              },
              comparison: {
                time_difference: data.comparison_data.time_difference,
                energy_saved: (data.comparison_data.hamilton_data?.energy_consumption || 2.8) - (data.energy_consumption || 2.4),
                co2_saved: data.comparison_data.co2_saved || 0,
                co2_saved_formatted: data.comparison_data.co2_saved_formatted || '0.000 kg',
                ai_faster: data.comparison_data.ai_faster,
                ai_more_efficient: true
              }
            };
            
            localStorage.setItem('simulationResults', JSON.stringify(transformedData));
          }
        } else if (data.status === 'error') {
          setIsRunning(false);
          setSimulationError(data.error_message || 'Simulation error occurred');
        }
      } catch (error) {
        console.error('Error parsing stream data:', error);
      }
    };

    eventSourceRef.current.onerror = (error) => {
      console.error('EventSource error:', error);
      setSimulationError('Connection to simulation lost');
    };
  };

  const pauseSimulation = async () => {
    setIsRunning(false);
    // TODO: Implement pause functionality
  };

  const stopSimulation = async () => {
    try {
      // Stop web simulation if available
      if (webSimulationAvailable) {
        try {
          const response = await fetch(config.getApiUrl(config.endpoints.simulation.stop), {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            }
          });
          
          if (!response.ok) {
            console.error('Failed to stop web simulation');
          }
        } catch (error) {
          console.error('Error stopping web simulation:', error);
        }
      }
      
      // Close EventSource connection
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
      
      // Update UI state
      setIsRunning(false);
      setSimulationError(null);
      
    } catch (error) {
      console.error('Error stopping simulation:', error);
      setIsRunning(false);
    }
  };

  const resetSimulation = async () => {
    try {
      // Stop current simulation first
      await stopSimulation();
      
      // Clear all simulation data
      setIsRunning(false);
      setSimulationTime(0);
      setCurrentLap(0);
      setSimulationId(null);
      setSimulationFrame(null);
      setSimulationError(null);
      setFallbackMode(false);
      
      // Reset simulation data
      setSimulationData({
        ai: {
          lapTime: 0,
          energyConsumption: 0,
          co2Saved: 0,
          speed: 0,
          throttle: 0,
          brake: 0,
        },
        human: {
          lapTime: 0,
          energyConsumption: 0,
          co2Saved: 0,
          speed: 0,
          throttle: 0,
          brake: 0,
        }
      });
      
      // Clear localStorage
      localStorage.removeItem('simulationId');
      localStorage.removeItem('simulationResults');
      localStorage.removeItem('simulationConfig');
      
    } catch (error) {
      console.error('Error resetting simulation:', error);
    }
  };

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = (seconds % 60).toFixed(1);
    return `${mins}:${secs.padStart(4, '0')}`;
  };
  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, []);

  return (
    <div className="simulation-dashboard">
      <div className="dashboard-container">
        {/* Header */}
        <div className="dashboard-header">
          <div className="header-left">
            <Link to="/simulator" className="back-button">
              <ArrowLeft size={20} />
              Back to Setup
            </Link>
            <div className="simulation-info">
              <h1>Live Simulation</h1>
              <div className="simulation-meta">
                <span><MapPin size={16} /> {JSON.parse(localStorage.getItem('simulationConfig') || '{}').track || 'Unknown Track'}</span>
                <span><Car size={16} /> {JSON.parse(localStorage.getItem('simulationConfig') || '{}').model_version || 'PPO'} Model</span>
                <span><Clock size={16} /> {formatTime(simulationTime)}</span>
                {webSimulationAvailable && (
                  <span className="web-simulation-badge">Web Simulation</span>
                )}
                {fallbackMode && (
                  <span className="fallback-badge">Mock Simulation</span>
                )}
                {simulationError && (
                  <span className="error-badge">Error: {simulationError}</span>
                )}
                {!isRunning && !simulationError && (
                  <span className="status-badge">Ready to Start</span>
                )}
                {isRunning && (
                  <span className="status-badge running">Running...</span>
                )}
              </div>
            </div>
          </div>
          
          <div className="header-controls">
            <button 
              className={`control-btn ${isRunning ? 'pause' : 'play'}`}
              onClick={isRunning ? pauseSimulation : startSimulation}
            >
              {isRunning ? <Pause size={20} /> : <Play size={20} />}
              {isRunning ? 'Pause' : 'Start'}
            </button>
            {isRunning && (
              <button className="control-btn stop" onClick={stopSimulation}>
                <Square size={20} />
                Stop
              </button>
            )}
            <button className="control-btn reset" onClick={resetSimulation}>
              <RotateCcw size={20} />
              Reset
            </button>
          </div>
        </div>

        {/* Error Display */}
        {simulationError && (
          <div className="error-alert">
            <AlertCircle size={20} />
            <span>{simulationError}</span>
          </div>
        )}

        {/* Progress Bar */}
        <div className="progress-section">
          <div className="progress-bar">
            <div 
              className="progress-fill" 
              style={{ width: `${simulationData.ai.progress * 100}%` }}
            ></div>
          </div>
          <div className="progress-info">
            <span>Lap Progress: {Math.round(simulationData.ai.progress * 100)}%</span>
            <span>{formatTime(simulationTime)}</span>
          </div>
        </div>

        {/* Main Content */}
        <div className="dashboard-content">
          {/* Video Panel */}
          <div className="video-panel">
            <div className="video-container">
              <h3>Live Simulation</h3>
              <div className="video-frame">
                {simulationFrame ? (
                  <img 
                    src={simulationFrame} 
                    alt="Simulation" 
                    className="simulation-image"
                  />
                ) : (
                  <div className="video-placeholder">
                    <Car size={48} />
                    <p>{isRunning ? 'Simulation Running...' : 'No simulation active'}</p>
                    {fallbackMode && (
                      <small>Running in mock simulation mode</small>
                    )}
                    {!webSimulationAvailable && !fallbackMode && (
                      <small>Web simulation not available - using mock data</small>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Metrics Panel */}
          <div className="metrics-panel">
            <div className="metric-card">
              <div className="metric-header">
                <Zap size={20} />
                <span>AI Performance</span>
              </div>
              <div className="metric-value">{formatTime(simulationData.ai.lapTime)}</div>
              <div className="metric-label">Lap Time</div>
            </div>

            <div className="metric-card">
              <div className="metric-header">
                <Gauge size={20} />
                <span>F1 Speed</span>
              </div>
              <div className="metric-value">{simulationData.ai.speed.toFixed(1)}</div>
              <div className="metric-label">km/h</div>
            </div>

            <div className="metric-card">
              <div className="metric-header">
                <Leaf size={20} />
                <span>CO₂ Savings</span>
              </div>
              <div className="metric-value">{simulationData.ai.co2Saved.toFixed(3)}</div>
              <div className="metric-label">kg CO₂ Saved</div>
            </div>
          </div>
          
          {/* Completion Message */}
          {!isRunning && !simulationError && localStorage.getItem('simulationResults') && (
            <div className="completion-message">
              <div className="completion-content">
                <h3>Simulation Complete!</h3>
                <p>View detailed comparison with Lewis Hamilton</p>
                <Link to="/results" className="view-results-btn">
                  View Results
                </Link>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default SimulationDashboard;