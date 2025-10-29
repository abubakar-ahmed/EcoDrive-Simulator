# EcoDrive Simulator - One-Lap Simulation System

## Overview

The EcoDrive Simulator now supports one-lap simulations using trained PPO and SAC reinforcement learning models, with real-time streaming and Lewis Hamilton comparison functionality.

## Features

### ✅ Implemented Features

1. **Model Support**
   - PPO (Proximal Policy Optimization) - Multi-track trained
   - SAC (Soft Actor-Critic) - Single-track trained
   - Automatic fallback to random actions if model loading fails

2. **One-Lap Simulation**
   - Automatic stopping after one complete lap
   - Real-time progress tracking
   - Live telemetry data streaming

3. **Lewis Hamilton Comparison**
   - Real-time lap time comparison using FastF1 API
   - Fallback data when FastF1 is unavailable
   - Performance metrics analysis

4. **Real-Time Streaming**
   - Server-Sent Events (SSE) for live data
   - Video frame streaming (when available)
   - Progress updates every 100ms

5. **Frontend Integration**
   - Updated SimulationSetup page
   - Enhanced SimulationDashboard with streaming
   - ResultsComparison with stored results

## Backend Architecture

### Core Components

1. **web_simulation.py** - Main simulation runner
   - `SimulationRunner` - Manages simulation lifecycle
   - `ModelLoader` - Handles PPO/SAC model loading
   - `SimulationConfig` - Configuration data structure
   - `SimulationData` - Real-time simulation data

2. **app.py** - Flask API endpoints
   - `/api/simulation/start` - Start new simulation
   - `/api/web-simulation/status` - Get simulation status
   - `/api/web-simulation/stream` - Real-time data stream
   - `/api/models/available` - List available models
   - `/api/tracks` - List available tracks

3. **fastf1_service.py** - Hamilton data integration
   - Real-time F1 data fetching
   - Fallback data for offline mode
   - Performance comparison metrics

### Model Paths

- **PPO Model**: `backend/models/multi_track_ppo_2025-10-12_02-58-46/final_model.zip`
- **SAC Model**: `backend/models/2025-10-08_21-27-56/model.zip`

## API Endpoints

### Simulation Management

```http
POST /api/simulation/start
Content-Type: application/json

{
  "track": "Catalunya",
  "model_version": "ppo",
  "user": "web_user"
}
```

**Response:**
```json
{
  "success": true,
  "simulation_id": "sim_20250101_120000_1234",
  "status": "started",
  "message": "Simulation started with PPO model on Catalunya"
}
```

### Real-Time Streaming

```http
GET /api/web-simulation/stream
Accept: text/event-stream
```

**Stream Format:**
```
data: {"status": "running", "lap_time": 45.2, "progress": 0.3, "current_speed": 120.5}

data: {"status": "completed", "lap_time": 87.5, "comparison_data": {...}}
```

### Model Information

```http
GET /api/models/available
```

**Response:**
```json
{
  "ppo": {
    "name": "PPO Multi-Track",
    "description": "Proximal Policy Optimization trained on multiple tracks",
    "trained_tracks": ["Catalunya", "Spielberg", "Silverstone", ...]
  },
  "sac": {
    "name": "SAC Single-Track", 
    "description": "Soft Actor-Critic trained on Catalunya track",
    "trained_tracks": ["Catalunya"]
  }
}
```

## Frontend Integration

### SimulationSetup.js
- Dynamic model loading from API
- Track selection with real-time data
- Immediate simulation start
- Loading states and error handling

### SimulationDashboard.js
- Real-time data streaming via EventSource
- Live progress updates
- Automatic completion detection
- Results storage for comparison

### ResultsComparison.js
- Uses stored simulation results
- Fallback to API comparison data
- Real-time Hamilton vs AI metrics

## Usage Flow

1. **Setup Phase**
   - User selects track and model
   - Frontend loads available options from API
   - Simulation starts immediately upon selection

2. **Simulation Phase**
   - Backend loads selected model (PPO/SAC)
   - One-lap simulation runs automatically
   - Real-time data streams to frontend
   - Progress updates every 100ms

3. **Completion Phase**
   - Simulation stops after one lap
   - Hamilton comparison data fetched
   - Results stored for comparison page
   - Frontend navigates to results

## Error Handling

### Model Loading Failures
- Automatic fallback to random actions
- Graceful degradation without crashes
- User notification of fallback mode

### F1Tenth Environment Issues
- Mock environment when dependencies unavailable
- Fallback simulation data
- Clear error messages to user

### Network Issues
- Retry mechanisms for API calls
- Offline fallback data
- Connection status indicators

## Configuration

### Environment Variables
- `FASTF1_CACHE_DIR` - FastF1 cache directory
- `MODEL_BASE_PATH` - Base path for model files
- `SIMULATION_TIMEOUT` - Maximum simulation time

### Model Configuration
```python
model_paths = {
    'ppo': 'backend/models/multi_track_ppo_2025-10-12_02-58-46/final_model.zip',
    'sac': 'backend/models/2025-10-08_21-27-56/model.zip'
}
```

## Testing

### Backend Testing
```bash
cd backend
python -c "import web_simulation; print('✅ Module loaded successfully')"
python -c "import app; print('✅ Flask app loaded successfully')"
```

### Frontend Testing
```bash
npm start
# Navigate to http://localhost:3000/simulator
# Select track and model
# Start simulation
# Monitor real-time updates
```

## Dependencies

### Backend
- Flask 2.3.3
- Flask-CORS 4.0.0
- stable-baselines3 2.3.0+
- gymnasium 0.29.1
- fastf1 (optional)
- torch 2.6.0+

### Frontend
- React 18.2.0
- react-router-dom 6.3.0
- axios 0.27.2
- lucide-react 0.263.1

## Performance Considerations

- **Streaming Frequency**: 10 FPS (100ms intervals)
- **Model Loading**: Cached after first load
- **Memory Usage**: Automatic cleanup on completion
- **Network**: Efficient SSE streaming
- **Fallbacks**: Multiple layers of error handling

## Future Enhancements

1. **Multi-Lap Support**: Extend to multiple laps
2. **Custom Models**: User-uploaded model support
3. **Advanced Telemetry**: Detailed sensor data
4. **Video Recording**: Full simulation recording
5. **Performance Analytics**: Detailed performance metrics

## Troubleshooting

### Common Issues

1. **Model Loading Fails**
   - Check model file paths
   - Verify stable-baselines3 installation
   - Check file permissions

2. **F1Tenth Environment Issues**
   - Install missing dependencies
   - Check Python path configuration
   - Use fallback mode

3. **Streaming Connection Lost**
   - Check network connectivity
   - Verify backend is running
   - Check browser EventSource support

### Debug Mode

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review error logs in browser console
3. Verify backend logs for detailed errors
4. Test with fallback data mode
