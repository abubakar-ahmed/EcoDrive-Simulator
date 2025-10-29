# Lewis Hamilton FastF1 Integration Setup

## Overview
The EcoDrive Simulator now integrates with the FastF1 API to fetch real Lewis Hamilton lap data for comparison with AI performance.

## Installation

### 1. Install FastF1 Package
```bash
cd "frontend and backend/backend"
pip install fastf1==3.1.8
```

### 2. Install Additional Dependencies
```bash
pip install -r requirements.txt
```

### 3. Start the Backend Server
```bash
python app.py
```

## Features

### Real Lewis Hamilton Data
- **Data Source**: FastF1 API (official F1 timing data)
- **Driver**: Lewis Hamilton (HAM)
- **Tracks**: All major F1 circuits (Silverstone, Monza, Spa, etc.)
- **Sessions**: Qualifying (Q), Race (R), Practice sessions
- **Years**: 2018-2023 seasons

### API Endpoints

#### Get Hamilton Data for Track
```
GET /api/hamilton/{track}
```
Returns Lewis Hamilton's performance data for a specific track.

#### Get AI vs Hamilton Comparison
```
GET /api/comparison/{track}
```
Returns detailed comparison between AI and Lewis Hamilton performance.

#### Get Detailed Lap Data
```
POST /api/hamilton/lap-data
{
  "track": "Silverstone",
  "year": 2023,
  "session": "Q"
}
```

### Data Structure

#### Hamilton Data
```json
{
  "driver_name": "Lewis Hamilton",
  "driver_code": "HAM",
  "team": "Mercedes",
  "lap_time": 87.5,
  "lap_time_formatted": "1:27.500",
  "average_speed": 185.5,
  "max_speed": 320.0,
  "energy_consumption": 2.8,
  "data_source": "FastF1 API",
  "is_real_data": true
}
```

#### Comparison Data
```json
{
  "track": "Silverstone",
  "hamilton": { ... },
  "ai": {
    "model": "PPO Eco-Drive",
    "lap_time": 88.2,
    "energy_consumption": 2.4,
    "efficiency_gain": 14.3
  },
  "comparison": {
    "time_difference": 0.7,
    "energy_saved": 0.4,
    "ai_faster": false,
    "ai_more_efficient": true
  }
}
```

## Usage

### Frontend Integration
The ResultsComparison component automatically:
1. Fetches Lewis Hamilton data from the backend
2. Displays driver profiles with real F1 data badges
3. Shows detailed performance metrics comparison
4. Provides fallback data if FastF1 API is unavailable

### Track Mapping
The system maps track names to FastF1 format:
- `Silverstone` → `Silverstone`
- `Monza` → `Monza`
- `Spa` → `Spa`
- `Catalunya` → `Barcelona`
- `YasMarina` → `Abu Dhabi`
- And many more...

## Fallback System
If FastF1 API data is not available, the system provides realistic fallback data based on Lewis Hamilton's historical performance.

## Caching
FastF1 data is cached locally to improve performance and reduce API calls. Cache directory: `./fastf1_cache`

## Error Handling
- Graceful fallback to estimated data
- Error messages displayed to users
- Retry functionality for failed requests

## Performance Metrics
- **Lap Time**: Direct comparison of fastest lap times
- **Energy Consumption**: Estimated energy usage per lap
- **Efficiency Score**: AI's efficiency improvement percentage
- **Average Speed**: Speed comparison across the lap

## Future Enhancements
- Real-time telemetry data visualization
- Historical performance trends
- Multiple driver comparisons
- Detailed sector analysis
