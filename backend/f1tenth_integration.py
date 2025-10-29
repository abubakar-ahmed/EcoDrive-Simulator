import os
import sys
from typing import Dict, Any, Optional

class F1TenthSimulationManager:
    """Basic F1Tenth simulation manager"""
    
    def __init__(self):
        self.active_simulations = {}
        self.available_models = {
            'ppo': {
                'type': 'PPO',
                'status': 'available'
            },
            'sac': {
                'type': 'SAC',
                'status': 'available'
            }
        }
        self.available_tracks = [
            'Catalunya', 'IMS', 'Silverstone', 'Monza', 'Spa', 'YasMarina',
            'Austin', 'BrandsHatch', 'Budapest', 'Hockenheim', 'Melbourne',
            'Mexico City', 'Montreal', 'MoscowRaceway', 'Nuerburgring',
            'Oschersleben', 'Sakhir', 'SaoPaulo', 'Sepang', 'Shanghai',
            'Sochi', 'Spielberg', 'Zandvoort'
        ]
    
    def get_available_models(self) -> Dict[str, Any]:
        """Get list of available models"""
        return self.available_models
    
    def get_available_tracks(self) -> list:
        """Get list of available tracks"""
        return self.available_tracks
    
    def start_simulation(self, simulation_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Start a new simulation"""
        try:
            # Validate config
            required_params = ['track', 'model_type', 'driving_mode']
            if not all(param in config for param in required_params):
                return {'error': 'Missing required parameters'}
            
            # Check if model is available
            model_info = self.available_models.get(config['model_type'])
            if not model_info or model_info['status'] != 'available':
                return {'error': f'Model {config["model_type"]} not available'}
            
            self.active_simulations[simulation_id] = {
                'status': 'running',
                'config': config,
                'data': []
            }
            
            return {'status': 'started', 'simulation_id': simulation_id}
            
        except Exception as e:
            return {'error': f'Failed to start simulation: {str(e)}'}
    
    def get_simulation_data(self, simulation_id: str) -> Dict[str, Any]:
        """Get current simulation data"""
        if simulation_id not in self.active_simulations:
            return {'error': 'Simulation not found', 'simulation_id': simulation_id}
        
        return self.active_simulations[simulation_id]
    
    def stop_simulation(self, simulation_id: str) -> Dict[str, Any]:
        """Stop a running simulation"""
        if simulation_id not in self.active_simulations:
            return {'error': 'Simulation not found'}
        
        self.active_simulations[simulation_id]['status'] = 'stopped'
        return {'status': 'stopped', 'simulation_id': simulation_id}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and statistics"""
        return {
            'active_simulations': len([s for s in self.active_simulations.values() if s['status'] == 'running']),
            'total_simulations': len(self.active_simulations),
            'available_models': len([m for m in self.available_models.values() if m['status'] == 'available']),
            'available_tracks': len(self.available_tracks)
        }

# Global simulation manager instance
simulation_manager = F1TenthSimulationManager()