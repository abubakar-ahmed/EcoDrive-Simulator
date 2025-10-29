import fastf1
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from typing import Dict, List, Optional, Tuple

class FastF1Service:
    def __init__(self, cache_dir: str = './fastf1_cache'):
        """Initialize FastF1 service with caching enabled"""
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        fastf1.Cache.enable_cache(cache_dir)
        
        # Lewis Hamilton's driver code
        self.hamilton_code = 'HAM'
        
        # Available tracks mapping
        self.track_mapping = {
            'Silverstone': 'Silverstone',
            'Monza': 'Monza',
            'Spa': 'Spa',
            'Catalunya': 'Barcelona',
            'Monaco': 'Monaco',
            'YasMarina': 'Abu Dhabi',
            'Austin': 'Austin',
            'Melbourne': 'Melbourne',
            'Montreal': 'Montreal',
            'Shanghai': 'Shanghai',
            'Sakhir': 'Bahrain',
            'Spielberg': 'Austria',
            'Hockenheim': 'Hockenheim',
            'Hungaroring': 'Hungary',
            'Suzuka': 'Japan',
            'Interlagos': 'Brazil',
            'Mexico City': 'Mexico',
            'Sochi': 'Russia',
            'Sepang': 'Malaysia',
            'Zandvoort': 'Netherlands'
        }
    
    def get_hamilton_lap_data(self, year: int, track: str, session: str = 'Q') -> Dict:
        """
        Get Lewis Hamilton's lap data for a specific track and session
        
        Args:
            year: F1 season year
            track: Track name (mapped to FastF1 format)
            session: Session type ('Q' for qualifying, 'R' for race, 'FP1', 'FP2', 'FP3')
        
        Returns:
            Dictionary containing lap data and telemetry
        """
        try:
            # Map track name to FastF1 format
            fastf1_track = self.track_mapping.get(track, track)
            
            # Get session
            session_obj = fastf1.get_session(year, fastf1_track, session)
            session_obj.load()
            
            # Get Hamilton's laps
            hamilton_laps = session_obj.laps.pick_driver(self.hamilton_code)
            
            if hamilton_laps.empty:
                return self._get_fallback_data(track)
            
            # Get fastest lap
            fastest_lap = hamilton_laps.pick_fastest()
            
            # Get telemetry data
            telemetry = fastest_lap.get_car_data().add_distance()
            
            # Calculate metrics
            lap_time_seconds = fastest_lap['LapTime'].total_seconds()
            avg_speed = telemetry['Speed'].mean()
            max_speed = telemetry['Speed'].max()
            
            # Calculate energy efficiency metrics (simplified)
            throttle_usage = telemetry['Throttle'].mean()
            brake_usage = telemetry['Brake'].mean()
            
            # Estimate energy consumption based on speed and throttle patterns
            energy_consumption = self._calculate_energy_consumption(telemetry)
            
            return {
                'success': True,
                'driver': 'Lewis Hamilton',
                'track': track,
                'year': year,
                'session': session,
                'lap_time': lap_time_seconds,
                'lap_time_formatted': str(fastest_lap['LapTime']),
                'average_speed': float(avg_speed),
                'max_speed': float(max_speed),
                'throttle_usage': float(throttle_usage),
                'brake_usage': float(brake_usage),
                'energy_consumption': energy_consumption,
                'telemetry_points': len(telemetry),
                'data_source': 'FastF1 API'
            }
            
        except Exception as e:
            print(f"Error fetching Hamilton data: {str(e)}")
            return self._get_fallback_data(track)
    
    def _calculate_energy_consumption(self, telemetry: pd.DataFrame) -> float:
        """Calculate estimated energy consumption from telemetry data"""
        # Simplified energy calculation based on speed and throttle
        speed_factor = telemetry['Speed'].mean() / 100  # Normalize speed
        throttle_factor = telemetry['Throttle'].mean() / 100  # Normalize throttle
        
        # Base energy consumption (kWh)
        base_energy = 2.5
        
        # Adjust based on driving style
        energy_multiplier = 1 + (speed_factor * 0.3) + (throttle_factor * 0.2)
        
        return round(base_energy * energy_multiplier, 2)
    
    def _get_fallback_data(self, track: str) -> Dict:
        """Provide fallback data when FastF1 data is not available"""
        # Realistic Lewis Hamilton lap times for different tracks (in seconds)
        fallback_times = {
            'Silverstone': 87.5,
            'Monza': 78.2,
            'Spa': 103.8,
            'Catalunya': 78.1,
            'Monaco': 70.1,
            'YasMarina': 103.2,
            'Austin': 93.4,
            'Melbourne': 78.2,
            'Montreal': 70.9,
            'Shanghai': 91.2,
            'Sakhir': 87.4,
            'Spielberg': 63.0,
            'Hockenheim': 78.1,
            'Hungaroring': 70.2,
            'Suzuka': 89.3,
            'Interlagos': 70.4,
            'Mexico City': 78.1,
            'Sochi': 93.2,
            'Sepang': 91.6,
            'Zandvoort': 70.1
        }
        
        lap_time = fallback_times.get(track, 80.0)
        
        return {
            'success': False,
            'driver': 'Lewis Hamilton',
            'track': track,
            'year': 2023,
            'session': 'Q',
            'lap_time': lap_time,
            'lap_time_formatted': f"{int(lap_time//60)}:{lap_time%60:06.3f}",
            'average_speed': 185.5,
            'max_speed': 320.0,
            'throttle_usage': 65.2,
            'brake_usage': 25.8,
            'energy_consumption': 2.8,
            'telemetry_points': 0,
            'data_source': 'Fallback Data',
            'note': 'Using estimated data - FastF1 API data not available'
        }
    
    def get_hamilton_comparison_data(self, track: str) -> Dict:
        """
        Get Lewis Hamilton's data formatted for comparison with AI
        
        Args:
            track: Track name
        
        Returns:
            Dictionary with Hamilton's performance metrics
        """
        # Try to get recent data (2023 season)
        hamilton_data = self.get_hamilton_lap_data(2023, track, 'Q')
        
        # If 2023 data not available, try 2022
        if not hamilton_data['success']:
            hamilton_data = self.get_hamilton_lap_data(2022, track, 'Q')
        
        # Format for comparison
        return {
            'driver_name': 'Lewis Hamilton',
            'driver_code': 'HAM',
            'team': 'Mercedes',
            'lap_time': hamilton_data['lap_time'],
            'lap_time_formatted': hamilton_data['lap_time_formatted'],
            'average_speed': hamilton_data['average_speed'],
            'max_speed': hamilton_data['max_speed'],
            'energy_consumption': hamilton_data['energy_consumption'],
            'throttle_efficiency': 100 - hamilton_data['throttle_usage'],
            'brake_efficiency': 100 - hamilton_data['brake_usage'],
            'data_source': hamilton_data['data_source'],
            'track': track,
            'year': hamilton_data['year'],
            'session': hamilton_data['session'],
            'is_real_data': hamilton_data['success']
        }
    
    def get_available_seasons(self) -> List[int]:
        """Get list of available F1 seasons"""
        return list(range(2018, 2024))  # FastF1 supports 2018-2023
    
    def get_available_tracks(self, year: int) -> List[str]:
        """Get list of available tracks for a specific year"""
        try:
            schedule = fastf1.get_event_schedule(year)
            return schedule['Location'].tolist()
        except:
            return list(self.track_mapping.keys())

# Global instance
fastf1_service = FastF1Service()
