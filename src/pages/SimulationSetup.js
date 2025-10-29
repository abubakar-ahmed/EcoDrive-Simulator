import React, { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { 
  MapPin, 
  Car, 
  Cloud, 
  Settings, 
  Play, 
  ArrowLeft,
  Zap,
  Leaf,
} from 'lucide-react';
import config from '../config';
import './SimulationSetup.css';

const SimulationSetup = () => {
  const navigate = useNavigate();
  const [selectedTrack, setSelectedTrack] = useState('');
  const [modelVersion, setModelVersion] = useState('ppo');
  const [tracks, setTracks] = useState([]);
  const [models, setModels] = useState({});
  const [loading, setLoading] = useState(true);
  const [starting, setStarting] = useState(false);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      setLoading(true);
      // Load tracks and models in parallel
      const [tracksResponse, modelsResponse] = await Promise.all([
        fetch(config.getApiUrl(config.endpoints.tracks)),
        fetch(config.getApiUrl(config.endpoints.models))
      ]);
      
      const tracksData = await tracksResponse.json();
      const modelsData = await modelsResponse.json();
      
      setTracks(tracksData);
      setModels(modelsData);
    } catch (error) {
      console.error('Failed to load data:', error);
      // Fallback data
      setTracks(['Catalunya', 'Spielberg', 'Silverstone', 'Monza', 'Spa']);
      setModels({
        'ppo': { name: 'PPO Multi-Track', description: 'Proximal Policy Optimization' },
        'sac': { name: 'SAC Single-Track', description: 'Soft Actor-Critic' }
      });
    } finally {
      setLoading(false);
    }
  };

  const handleStartSimulation = async () => {
    if (!selectedTrack) {
      alert('Please select a track');
      return;
    }

    setStarting(true);

    const simConfig = {
      track: selectedTrack,
      model_version: modelVersion,
      user: 'web_user'
    };

    localStorage.setItem('simulationConfig', JSON.stringify(simConfig));
    
    // Start the simulation immediately
    try {
      const response = await fetch(config.getApiUrl(config.endpoints.simulation.start), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(simConfig)
      });

      const result = await response.json();
      
      if (result.simulation_id) {
        localStorage.setItem('simulationId', result.simulation_id);
        localStorage.setItem('simulationStatus', result.status || 'started');
        navigate('/dashboard');
      } else {
        alert('Failed to start simulation: ' + (result.error || 'Unknown error'));
      }
    } catch (error) {
      console.error('Failed to start simulation:', error);
      alert('Failed to start simulation: ' + error.message);
    } finally {
      setStarting(false);
    }
  };

  return (
    <div className="simulation-setup">
      <div className="container">
        <div className="setup-header">
          <Link to="/" className="back-button">
            <ArrowLeft size={20} />
            Back to Home
          </Link>
          <h1 className="setup-title">Simulation Setup</h1>
          <p className="setup-description">Select a track and AI model to run a one-lap simulation</p>
        </div>

        <div className="setup-content">
          {/* Selection Summary */}
          {(selectedTrack || modelVersion) && (
            <div className="selection-summary">
              <h3>Current Selection</h3>
              <div className="summary-items">
                {selectedTrack && (
                  <div className="summary-item">
                    <MapPin size={16} />
                    <span>Track: <strong>{selectedTrack}</strong></span>
                  </div>
                )}
                {modelVersion && models[modelVersion] && (
                  <div className="summary-item">
                    <Settings size={16} />
                    <span>Model: <strong>{models[modelVersion].name} ({modelVersion.toUpperCase()})</strong></span>
                  </div>
                )}
              </div>
            </div>
          )}

          <div className="setup-section">
            <div className="section-header">
              <MapPin size={24} />
              <h2>Track Selection</h2>
            </div>
            <div className="track-grid">
              {tracks.map(track => (
                <button
                  key={track.name}
                  className={`track-card ${selectedTrack === track.name ? 'selected' : ''}`}
                  onClick={() => setSelectedTrack(track.name)}
                >
                  <div className="track-image">
                    {track.has_image ? (
                      <img 
                        src={track.image_path.startsWith('http') ? track.image_path : `${config.API_BASE_URL}${track.image_path}`}
                        alt={track.display_name}
                        className="track-map-image"
                        onError={(e) => {
                          e.target.style.display = 'none';
                          e.target.nextSibling.style.display = 'flex';
                        }}
                      />
                    ) : null}
                    <div className="track-placeholder" style={{ display: track.has_image ? 'none' : 'flex' }}>
                      <MapPin size={32} />
                    </div>
                  </div>
                  <div className="track-info">
                    <h3>{track.display_name}</h3>
                    <div className="track-country">F1 Racing Circuit</div>
                    <div className="track-details">
                      <span>Centerline</span>
                      <span>{track.has_raceline ? 'Raceline' : 'No Raceline'}</span>
                    </div>
                  </div>
                </button>
              ))}
            </div>
          </div>


          <div className="setup-section">
            <div className="section-header">
              <Settings size={24} />
              <h2>AI Model</h2>
            </div>
            
            {/* Selected Model Display */}
            {modelVersion && models[modelVersion] && (
              <div className="selected-model-display">
                <div className="selected-model-card">
                  <div className="selected-model-header">
                    <Settings size={20} />
                    <span>Selected Model</span>
                  </div>
                  <div className="selected-model-info">
                    <h3>{models[modelVersion].name}</h3>
                    <div className="model-badge">{modelVersion.toUpperCase()}</div>
                    <p>{models[modelVersion].description}</p>
                    {models[modelVersion].trained_tracks && (
                      <div className="trained-tracks">
                        <small>Trained on: {models[modelVersion].trained_tracks.slice(0, 3).join(', ')}{models[modelVersion].trained_tracks.length > 3 ? '...' : ''}</small>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}
            
            <div className="model-grid">
              {Object.entries(models).map(([key, model]) => (
                <button
                  key={key}
                  className={`model-card ${modelVersion === key ? 'selected' : ''}`}
                  onClick={() => setModelVersion(key)}
                >
                  <span>{key.toUpperCase()}</span>
                  <p>{model.name}</p>
                  <small>{model.description}</small>
                  {model.trained_tracks && (
                    <div className="model-tracks">
                      <small>Trained on: {model.trained_tracks.slice(0, 3).join(', ')}{model.trained_tracks.length > 3 ? '...' : ''}</small>
                    </div>
                  )}
                  <div className="model-status">
                    <small>{modelVersion === key ? 'Selected' : 'Ready for simulation'}</small>
                  </div>
                </button>
              ))}
            </div>
          </div>

          <div className="setup-actions">
            <button 
              className="start-simulation"
              onClick={handleStartSimulation}
              disabled={!selectedTrack || !modelVersion || starting || loading}
            >
              <Play size={20} />
              {starting ? 'Starting...' : `Start ${modelVersion ? modelVersion.toUpperCase() : 'AI'} Simulation`}
            </button>
            {loading && (
              <div className="loading-indicator">
                <small>Loading tracks and models...</small>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default SimulationSetup;