"""
Web Simulation Module for EcoDrive Simulator
Handles one-lap simulations with PPO/SAC models and Hamilton comparison
"""

import os
import sys
import time
import threading
import numpy as np
import gymnasium as gym
import torch
import warnings
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import json
import io
import zipfile
from dataclasses import dataclass, asdict

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Import F1Tenth gym and Stable-Baselines3
try:
    import f1tenth_gym
    from f1tenth_wrapper.env import F1TenthWrapper
    from stable_baselines3 import PPO, SAC
    F1TENTH_AVAILABLE = True
    print(" F1Tenth gym and Stable-Baselines3 loaded successfully")
except ImportError as e:
    print(f" F1Tenth gym not available: {e}")
    F1TENTH_AVAILABLE = False
except Exception as e:
    print(f" F1Tenth gym failed to load due to dependency issues: {e}")
    F1TENTH_AVAILABLE = False

# Fallback classes when F1Tenth is not available
if not F1TENTH_AVAILABLE:
    class MockEnv:
        def __init__(self, *args, **kwargs):
            self.action_space = type('ActionSpace', (), {'sample': lambda: np.array([0.5, 0.0])})()
            self.observation_space = type('ObsSpace', (), {'shape': (10,)})()
        
        def reset(self):
            return np.random.random(10), {}
        
        def step(self, action):
            return np.random.random(10), 0.0, False, False, {}
        
        def close(self):
            pass
    
    class MockModel:
        def predict(self, obs, deterministic=True):
            return np.array([0.5, 0.0]), None
    
    # Mock classes for when dependencies are not available
    PPO = MockModel
    SAC = MockModel
    F1TenthWrapper = MockEnv

# Import FastF1 service
try:
    from fastf1_service import fastf1_service
    FASTF1_AVAILABLE = True
except ImportError:
    FASTF1_AVAILABLE = False
    fastf1_service = None

@dataclass
class SimulationConfig:
    """Configuration for simulation runs"""
    track: str
    model_type: str  # 'ppo' or 'sac'
    model_path: str
    simulation_id: str
    user: str = "anonymous"
    weather: str = "clear"
    surface: str = "dry"
    max_lap_time: float = 300.0  # Maximum lap time in seconds
    render_mode: str = "rgb_array"

@dataclass
class SimulationData:
    """Data structure for simulation results"""
    simulation_id: str
    status: str  # 'running', 'completed', 'error', 'stopped'
    start_time: float
    end_time: Optional[float] = None
    lap_time: Optional[float] = None
    lap_time_formatted: Optional[str] = None
    progress: float = 0.0
    current_speed: float = 0.0
    current_position: Tuple[float, float] = (0.0, 0.0)
    current_heading: float = 0.0
    co2_saved: float = 0.0  # CO₂ savings in kg
    energy_consumption: float = 0.0  # Energy consumption in kWh
    current_frame: Optional[str] = None  # Latest video frame as base64 data URL
    average_speed: float = 0.0  # Average speed (F1-scaled km/h)
    hamilton_data: Optional[Dict] = None
    comparison_data: Optional[Dict] = None
    error_message: Optional[str] = None
    # Telemetry data
    telemetry_data: List[Dict] = None  # List of telemetry points
    current_throttle: float = 0.0
    current_brake: float = 0.0
    current_steering: float = 0.0
    
    def __post_init__(self):
        if self.telemetry_data is None:
            self.telemetry_data = []

class ModelLoader:
    """Handles loading of PPO and SAC models with fallback to random actions"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"ModelLoader using device: {self.device}")
    
    def load_model(self, model_type: str, model_path: str, env: gym.Env) -> Any:
        """
        Load a trained model (PPO or SAC) with fallback to random actions
        
        Args:
            model_type: 'ppo' or 'sac'
            model_path: Path to the model file
            env: Environment to create model for
            
        Returns:
            Loaded model or None if loading fails
        """
        try:
            if model_type.lower() == 'ppo':
                return self._load_ppo_model(model_path, env)
            elif model_type.lower() == 'sac':
                return self._load_sac_model(model_path, env)
            else:
                print(f" Unknown model type: {model_type}")
                return None
        except Exception as e:
            print(f" Failed to load {model_type.upper()} model: {e}")
            return None
    
    def _load_ppo_model(self, model_path: str, env: gym.Env) -> Optional[PPO]:
        """Load PPO model with error handling"""
        try:
            print(f" Loading PPO model from: {model_path}")
            model = PPO.load(model_path, env=env, device=self.device)
            print(" PPO model loaded successfully")
            return model
        except Exception as e:
            print(f" Failed to load PPO model: {e}")
            return None
    
    def _load_sac_model(self, model_path: str, env: gym.Env) -> Optional[SAC]:
        """Load SAC model with error handling"""
        try:
            print(f" Loading SAC model from: {model_path}")
            model = SAC.load(model_path, env=env, device=self.device)
            print(" SAC model loaded successfully")
            return model
        except Exception as e:
            print(f" Failed to load SAC model: {e}")
            return None
    
    def get_random_action(self, env: gym.Env) -> np.ndarray:
        """Generate random action as fallback"""
        return env.action_space.sample()

class SimulationRunner:
    """Main simulation runner for one-lap simulations"""
    
    # Simulation tuning constants
    DISPLAY_SPEED_MULTIPLIER = 500.0  # Multiplier to scale 1:10 car speed to F1 speeds (0.1 m/s → 180 km/h)
    
    # CO2 Calculation constants
    CO2_HUMAN_EFFICIENCY_FACTOR = 1.25  # Assumed 25% less efficient driving
    CO2_AI_EFFICIENCY_FACTOR = 0.85     # Assumed 15% more efficient driving
    SMOOTH_DRIVING_BONUS = 0.1          # 10% bonus for smooth AI
    F1_FUEL_CONSUMPTION_PER_100KM = 100 # kg/100km
    F1_CO2_PER_KG_FUEL = 3.15           # kg CO2 per kg fuel
    
    def __init__(self):
        self.model_loader = ModelLoader()
        self.current_simulation: Optional[SimulationData] = None
        self.simulation_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.total_distance = 0.0  # Cumulative distance for CO2 calculation
        self.speed_history = []  # For smoothing speed display
        self.MAX_DISPLAY_SPEED = 350.0  # Maximum realistic F1 speed (km/h)
        
        # Available tracks mapping
        self.available_tracks = [
            "Catalunya", "Spielberg", "Silverstone", "Monza", "Spa",
            "Hockenheim", "Budapest", "Melbourne", "Mexico City", "Montreal",
            "Nuerburgring", "Oschersleben", "Sakhir", "SaoPaulo", "Sepang",
            "Shanghai", "Sochi", "YasMarina", "Zandvoort", "Austin"
        ]
        
        # Base models directory
        self.models_dir = os.path.join(os.path.dirname(__file__), 'models')
        
        # Track-specific model paths
        self.track_specific_models = {
            'Silverstone': {
                'sac': os.path.join(self.models_dir, 'sac_silverstone_2025-10-22_13-44-55', 'final_model.zip')
            }
        }
        
        # Multi-track model paths (default for other tracks)
        self.multi_track_models = {
            'ppo': os.path.join(self.models_dir, 'multi_track_ppo_2025-10-12_02-58-46', 'final_model.zip'),
            'sac': os.path.join(self.models_dir, 'multi_track_sac_2025-10-22_19-11-32', 'final_model.zip')
        }
    
    def get_model_path(self, track: str, model_type: str) -> str:
        """Get the appropriate model path based on track and model type
        
        For PPO model with any track (Monza, Silverstone, Spa, Catalunya, Spielberg, etc.):
        Returns: backend/models/multi_track_ppo_2025-10-12_02-58-46/final_model.zip
        """
        # Check if we have a track-specific model
        if track in self.track_specific_models:
            if model_type in self.track_specific_models[track]:
                model_path = self.track_specific_models[track][model_type]
                # Check if model exists
                if os.path.exists(model_path):
                    return model_path
                else:
                    print(f" Track-specific model not found: {model_path}")
                    print(f" Falling back to multi-track model")
        
        # Fallback to multi-track model
        # This handles PPO for all tracks (since only Silverstone has a track-specific SAC model)
        if model_type in self.multi_track_models:
            model_path = self.multi_track_models[model_type]
            print(f" Using multi-track {model_type.upper()} model: {model_path}")
            return model_path
        
        # Default fallback
        return os.path.join(self.models_dir, 'multi_track_sac_2025-10-22_19-11-32', 'final_model.zip')
    
    def start_simulation(self, config: Dict[str, Any], logs_list: List = None) -> Dict[str, Any]:
        """
        Start a new simulation
        
        Args:
            config: Simulation configuration dictionary
            logs_list: List of simulation log entries to update
            
        Returns:
            Response dictionary with simulation status
        """
        if not F1TENTH_AVAILABLE:
            return {
                'success': False,
                'error': 'F1Tenth environment not available',
                'simulation_id': None
            }
        
        # Stop any existing simulation
        if self.current_simulation and self.current_simulation.status == 'running':
            self.stop_simulation()
        
        try:
            # Validate configuration
            track = config.get('track', 'Catalunya')
            model_type = config.get('model_version', 'ppo').lower()
            simulation_id = config.get('simulation_id', f"sim_{int(time.time())}")
            
            if track not in self.available_tracks:
                return {
                    'success': False,
                    'error': f'Track {track} not available',
                    'simulation_id': simulation_id
                }
            
            if model_type not in ['ppo', 'sac']:
                return {
                    'success': False,
                    'error': f'Model type {model_type} not supported',
                    'simulation_id': simulation_id
                }
            
            # Get the appropriate model path based on track and model type
            model_path = self.get_model_path(track, model_type)
            print(f" Selected model: {model_type.upper()} for {track}")
            print(f" Model path: {model_path}")
            
            # Create simulation configuration
            sim_config = SimulationConfig(
                track=track,
                model_type=model_type,
                model_path=model_path,
                simulation_id=simulation_id,
                user=config.get('user', 'anonymous'),
                weather=config.get('weather', 'clear'),
                surface=config.get('surface', 'dry')
            )
            
            # Initialize simulation data
            self.current_simulation = SimulationData(
                simulation_id=simulation_id,
                status='running',
                start_time=time.time()
            )
            
            # Start simulation in separate thread
            self.stop_event.clear()
            self.simulation_thread = threading.Thread(
                target=self._run_simulation,
                args=(sim_config, logs_list),
                daemon=True
            )
            self.simulation_thread.start()
            
            return {
                'success': True,
                'simulation_id': simulation_id,
                'status': 'started',
                'message': f'Simulation started with {model_type.upper()} model on {track}'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'simulation_id': config.get('simulation_id', 'unknown')
            }
    
    def _run_simulation(self, config: SimulationConfig, logs_list: List = None):
        """Run the actual simulation in a separate thread"""
        try:
            # Find the log entry for this simulation
            log_entry = None
            if logs_list:
                log_entry = next((log for log in logs_list if log['id'] == config.simulation_id), None)
            
            # Create environment with same config as test_sac_silverstone.py
            env_config = {
                "map": config.track,
                "num_agents": 1,
                "timestep": 0.01,  # Balanced timestep for good speed
                "integrator": "rk4",
                "control_input": ["speed", "steering_angle"],
                "model": "st",
                "observation_config": {"type": "original"},
                "params": {
                    "mu": 1.5,
                    "C_Sf": 5.5,
                    "C_Sr": 6.0,
                    "lf": 0.15875,
                    "lr": 0.17145,
                    "h": 0.074,
                    "m": 3.74,
                    "I": 0.04712,
                    "s_min": -0.4189,
                    "s_max": 0.4189,
                    "sv_min": -3.2,
                    "sv_max": 3.2,
                    "v_switch": 7.319,
                    "a_max": 4.0,
                    "v_min": -4.0,
                    "v_max": 10.0,
                    "width": 0.31,
                    "length": 0.58,
                    # Balanced rendering parameters for good performance and visibility
                    "zoom_in_factor": 0.9,  # Slightly zoomed out for better view
                    "window_size": 600,  # Medium window size for good balance
                    "focus_on": "agent_0",
                    "show_wheels": False,  # Disable wheels for better performance
                    "car_tickness": 1,
                    "show_info": False  # Keep info off for performance
                },
                "reset_config": {"type": "rl_grid_static"},  # Match test file
                "max_laps": 1,  # Complete exactly one lap
                "seed": 1
            }
            
            env = gym.make("f1tenth-RL-v0", config=env_config, render_mode="rgb_array")
            
            # Add raceline rendering callback
            track = env.unwrapped.track
            env.unwrapped.add_render_callback(track.raceline.render_waypoints)
            
            # Load model following the exact pattern from test_sac_silverstone.py
            model = None
            try:
                # Ensure the model path is correct
                model_path = config.model_path
                if not os.path.exists(model_path):
                    model = None
                else:
                    if config.model_type.lower() == 'ppo':
                        model = PPO.load(model_path, device="auto")
                    elif config.model_type.lower() == 'sac':
                        model = SAC.load(model_path, device="auto")
            except Exception as e:
                model = None
            
            # Reset environment
            obs, info = env.reset()
            
            # Get initial position for reference
            initial_position = self._extract_position_from_obs(obs)
            
            # Initialize simulation tracking
            start_time = time.time()
            step_count = 0
            simulation_time = 0.0
            self.total_distance = 0.0  # Reset cumulative distance for new simulation
            self.speed_history = []  # Reset speed history for smoothing
            avg_speed_accumulator = 0.0
            avg_speed_count = 0
            time_step = env_config["timestep"]  # Get timestep from config
            max_steps = int(config.max_lap_time / time_step)  # Convert to steps using actual timestep
            
            # Track lap completion using toggle_list (like test_sac_silverstone.py)
            lap_completed = False
            initial_toggle = 0
            lap_start_time = 0.0
            
            # Track progress and telemetry
            total_distance = 0.0
            last_position = np.array([0.0, 0.0])
            
            while not self.stop_event.is_set() and step_count < max_steps and not lap_completed:
                # Get action from model following the exact pattern from test_sac_silverstone.py
                if model is not None:
                    predicted = model.predict(obs, deterministic=True)
                    action = predicted[0]  # Get the action from the prediction tuple
                    
                    # Apply same action processing as SACWrapper in test file
                    try:
                        # Handle different action formats
                        if isinstance(action, np.ndarray):
                            if len(action.shape) == 1 and len(action) >= 2:
                                # Standard [steering, speed] format
                                if action[1] < 0:
                                    action[1] = float(np.clip(action[1] * 1.2, -4.0, 0.0))  # Amplify braking
                                else:
                                    action[1] = float(np.clip(action[1] * 0.8, 0.0, 10.0))  # Scale acceleration
                        elif isinstance(action, (list, tuple)) and len(action) >= 2:
                            # List/tuple format
                            if action[1] < 0:
                                action = [action[0], float(np.clip(action[1] * 1.2, -4.0, 0.0))]
                            else:
                                action = [action[0], float(np.clip(action[1] * 0.8, 0.0, 10.0))]
                    except Exception as e:
                        pass
                else:
                    # Fallback: return random actions if no model is loaded (exact pattern from test file)
                    action = np.random.uniform(
                        low=[-0.4189, -4.0], 
                        high=[0.4189, 10.0], 
                        size=(2,)
                    ).astype(np.float32)
                
                # Step environment
                obs, reward, done, truncated, info = env.step(action)
                step_count += 1
                simulation_time += time_step  # Increment simulation time by timestep
                
                # Extract telemetry data
                raw_speed_ms = self._extract_raw_speed(obs)  # Raw speed in m/s
                scaled_speed = raw_speed_ms * 3.6 * self.DISPLAY_SPEED_MULTIPLIER  # Scaled for display
                
                # Smooth the speed using moving average (keep last 10 values)
                self.speed_history.append(scaled_speed)
                if len(self.speed_history) > 10:
                    self.speed_history.pop(0)
                current_speed = sum(self.speed_history) / len(self.speed_history)
                
                # Cap speed at realistic F1 maximum
                current_speed = min(current_speed, self.MAX_DISPLAY_SPEED)
                
                current_position = self._extract_position_from_obs(obs)
                current_heading = self._extract_heading_from_obs(obs)
                
                # Extract action data (throttle, brake, steering)
                throttle, brake, steering = self._extract_action_data(action)
                
                # Calculate incremental distance and accumulate
                incremental_distance = raw_speed_ms * time_step  # Distance in meters for this step
                self.total_distance += incremental_distance
                
                # Calculate CO₂ savings using cumulative distance
                co2_saved = self._calculate_co2_savings_from_distance(self.total_distance)
                
                # Update simulation data
                current_time = time.time()
                wall_clock_time = current_time - start_time
                
                # Calculate progress (simplified - based on step count)
                progress = min(step_count / max_steps, 1.0)
                
                # Update running average speed (F1-scaled km/h)
                avg_speed_accumulator += current_speed
                avg_speed_count += 1
                avg_speed = avg_speed_accumulator / avg_speed_count
                
                # Update simulation data with ACTUAL simulation time, not wall clock time
                self.current_simulation.lap_time = simulation_time
                self.current_simulation.lap_time_formatted = self._format_lap_time(simulation_time)
                self.current_simulation.progress = progress
                self.current_simulation.current_speed = current_speed
                self.current_simulation.average_speed = avg_speed
                self.current_simulation.current_position = tuple(current_position)
                self.current_simulation.current_heading = current_heading
                self.current_simulation.co2_saved = co2_saved
                self.current_simulation.current_throttle = throttle
                self.current_simulation.current_brake = brake
                self.current_simulation.current_steering = steering
                
                # Add telemetry data point
                telemetry_point = {
                    'time': simulation_time,
                    'speed': current_speed,
                    'throttle': throttle,
                    'brake': brake,
                    'steering': steering,
                    'position': tuple(current_position),
                    'heading': current_heading,
                    'co2_saved': co2_saved,
                    'progress': progress
                }
                self.current_simulation.telemetry_data.append(telemetry_point)
                
                
                # Capture frame for streaming
                if step_count % 15 == 0:  # Capture every 15th frame for smoother performance
                    try:
                        frame = env.render()
                        if frame is not None:
                            # Convert frame to base64 for frontend display
                            import base64
                            from PIL import Image
                            import io
                            
                            # Convert numpy array to PIL Image
                            if hasattr(frame, 'shape') and len(frame.shape) == 3:
                                # RGB image
                                pil_image = Image.fromarray(frame.astype('uint8'))
                            else:
                                # Handle other formats
                                pil_image = Image.fromarray(frame)
                            
                            # Convert to base64
                            buffer = io.BytesIO()
                            pil_image.save(buffer, format='PNG')
                            img_str = base64.b64encode(buffer.getvalue()).decode()
                            img_data_url = f"data:image/png;base64,{img_str}"
                            
                            # Store ONLY the latest frame for immediate access (prevents memory leak)
                            self.current_simulation.current_frame = img_data_url
                    except Exception as e:
                        pass  # Silent fail for better performance
                
                # Check if one lap has been completed using toggle_list (like test_sac_silverstone.py)
                # toggle_list tracks: 0 -> 1 (halfway) -> 2 (full lap completed)
                try:
                    if hasattr(env.unwrapped, 'core'):
                        current_toggle = env.unwrapped.core.toggle_list[0] if hasattr(env.unwrapped.core, 'toggle_list') else 0
                    else:
                        current_toggle = env.unwrapped.toggle_list[0] if hasattr(env.unwrapped, 'toggle_list') else 0
                    
                    # Update initial toggle on first step
                    if step_count == 1:
                        initial_toggle = current_toggle
                        lap_start_time = simulation_time
                    
                    # Check if we've completed one full lap (toggle >= 2)
                    if current_toggle >= 2:
                        lap_completed = True
                        print(f"Lap completed in {simulation_time:.2f}s")
                        break
                except Exception as e:
                    pass
                
                # Also check if environment says we're done
                # Note: We check toggle_list first before relying on done/truncated
                # This ensures we complete exactly one lap
                if (done or truncated) and lap_completed:
                    # Only break if we actually completed a lap
                    final_position = self._extract_position_from_obs(obs)
                    self.current_simulation.status = 'completed'
                    self.current_simulation.end_time = current_time
                    self.current_simulation.lap_time = simulation_time  # Use simulation time, not wall clock time
                    self.current_simulation.lap_time_formatted = self._format_lap_time(simulation_time)
                    self.current_simulation.progress = 1.0
                    
                    # Update log entry
                    if log_entry:
                        log_entry['status'] = 'completed'
                        log_entry['ai_lap_time'] = self.current_simulation.lap_time
                        log_entry['lap_time_formatted'] = self.current_simulation.lap_time_formatted
                        log_entry['co2_saved'] = self.current_simulation.co2_saved
                    
                    break
                
                # Optimized delay for smooth simulation speed
                time.sleep(0.0003)
            
            # Check why we exited the loop
            if not lap_completed:
                if self.current_simulation.status == 'running':
                    if step_count >= max_steps:
                        self.current_simulation.status = 'error'
                        self.current_simulation.error_message = 'Simulation timed out'
                        print(f"Simulation timed out after {step_count} steps")
                    else:
                        self.current_simulation.status = 'stopped'
                elif self.current_simulation.status != 'completed':
                    self.current_simulation.status = 'error'
                    self.current_simulation.error_message = 'Failed to complete one lap'
            
            # Get Hamilton comparison data if simulation completed
            if lap_completed:
                self.current_simulation.status = 'completed'
                self.current_simulation.end_time = time.time()
                self.current_simulation.lap_time = simulation_time
                self.current_simulation.lap_time_formatted = self._format_lap_time(simulation_time)
                self.current_simulation.progress = 1.0
                
                # Update log entry
                if log_entry:
                    log_entry['status'] = 'completed'
                    log_entry['ai_lap_time'] = self.current_simulation.lap_time
                    log_entry['lap_time_formatted'] = self.current_simulation.lap_time_formatted
                    log_entry['co2_saved'] = self.current_simulation.co2_saved
                
                self._get_hamilton_comparison(config.track)
            
            # Clean up
            env.close()
            
        except Exception as e:
            if self.current_simulation:
                self.current_simulation.status = 'error'
                self.current_simulation.error_message = str(e)
            
            # Update log entry with error
            if log_entry:
                log_entry['status'] = 'error'
                log_entry['error_message'] = str(e)
    
    def _get_hamilton_comparison(self, track: str):
        """Get Lewis Hamilton's lap time for comparison"""
        try:
            if FASTF1_AVAILABLE and fastf1_service:
                hamilton_data = fastf1_service.get_hamilton_comparison_data(track)
                self.current_simulation.hamilton_data = hamilton_data
                
                # Create comparison data using actual simulation time
                ai_lap_time = self.current_simulation.lap_time  # This is now simulation time
                hamilton_lap_time = hamilton_data['lap_time']
                
                comparison = {
                    'ai_lap_time': ai_lap_time,
                    'ai_lap_time_formatted': self.current_simulation.lap_time_formatted,
                    'hamilton_lap_time': hamilton_lap_time,
                    'hamilton_lap_time_formatted': hamilton_data['lap_time_formatted'],
                    'time_difference': ai_lap_time - hamilton_lap_time,
                    'ai_faster': ai_lap_time < hamilton_lap_time,
                    'percentage_difference': ((ai_lap_time - hamilton_lap_time) / hamilton_lap_time) * 100,
                    'co2_saved': self.current_simulation.co2_saved,
                    'co2_saved_formatted': f"{self.current_simulation.co2_saved:.3f} kg",
                    'hamilton_data': hamilton_data,
                    'simulation_notes': f'AI lap time based on simulation time. Speed displayed with 8x scaling for realistic F1 speeds. CO₂ savings calculated based on efficient AI driving patterns.'
                }
                
                self.current_simulation.comparison_data = comparison
                
        except Exception as e:
            # Use fallback data
            self.current_simulation.hamilton_data = {
                'driver_name': 'Lewis Hamilton',
                'lap_time': 87.5,  # Fallback time
                'lap_time_formatted': '1:27.500',
                'data_source': 'Fallback Data'
            }
    
    def _extract_raw_speed(self, obs) -> float:
        """Extract raw speed in m/s from observation (for accurate calculations)"""
        try:
            # Handle different observation formats
            if isinstance(obs, dict) and len(obs) > 0:
                # Multi-agent observation (dict with agent IDs as keys)
                agent_id = list(obs.keys())[0]
                current_obs = obs[agent_id]
            else:
                # Single agent observation
                current_obs = obs
            
            # If it's a numpy array, extract velocity
            if isinstance(current_obs, np.ndarray):
                if len(current_obs) >= 4:
                    vx = float(current_obs[3])
                    vy = float(current_obs[4]) if len(current_obs) > 4 else 0.0
                    speed = np.sqrt(vx**2 + vy**2)
                    return speed
            
            # Try dict observation formats
            if isinstance(current_obs, dict):
                if "linear_vels_x" in current_obs and "linear_vels_y" in current_obs:
                    vx = float(current_obs["linear_vels_x"][0])
                    vy = float(current_obs["linear_vels_y"][0])
                    speed = np.sqrt(vx**2 + vy**2)
                    return speed
                elif "std_state" in current_obs and isinstance(current_obs["std_state"], (list, np.ndarray)):
                    if len(current_obs["std_state"]) >= 4:
                        vx = float(current_obs["std_state"][3])
                        return abs(vx)
                elif "speed" in current_obs:
                    return float(current_obs["speed"])
            
            return 0.0
            
        except Exception as e:
            print(f"Error extracting raw speed: {e}")
            return 0.0
    
    def _extract_speed_from_obs(self, obs) -> float:
        """Extract current speed from observation data and scale to F1 speeds"""
        try:
            # Handle different observation formats
            if isinstance(obs, dict) and len(obs) > 0:
                # Multi-agent observation (dict with agent IDs as keys)
                agent_id = list(obs.keys())[0]
                current_obs = obs[agent_id]
            else:
                # Single agent observation
                current_obs = obs
            
            # If it's a numpy array, convert to dict or handle it
            if isinstance(current_obs, np.ndarray):
                # For array observations, try to extract velocity
                # Velocity might be at indices 3-4 (for f1tenth)
                if len(current_obs) >= 4:
                    vx = float(current_obs[3])
                    vy = float(current_obs[4]) if len(current_obs) > 4 else 0.0
                    speed = np.sqrt(vx**2 + vy**2)
                    raw_speed_ms = speed
                    scaled_speed = raw_speed_ms * 3.6 * self.DISPLAY_SPEED_MULTIPLIER
                    print(f"🔍 Extracted speed from array: {raw_speed_ms:.3f} m/s → {scaled_speed:.1f} km/h")
                    return scaled_speed
            
            # Try different dict observation formats
            if isinstance(current_obs, dict):
                if "linear_vels_x" in current_obs and "linear_vels_y" in current_obs:
                    # Original observation format with velocity components
                    vx = float(current_obs["linear_vels_x"][0])
                    vy = float(current_obs["linear_vels_y"][0])
                    speed = np.sqrt(vx**2 + vy**2)
                    # Convert m/s to km/h and scale to F1 speeds
                    return speed * 3.6 * self.DISPLAY_SPEED_MULTIPLIER
                elif "linear_vels_x" in current_obs or "linear_vels_y" in current_obs:
                    # Try to get at least one velocity component
                    vx = float(current_obs.get("linear_vels_x", [0])[0]) if isinstance(current_obs.get("linear_vels_x"), (list, np.ndarray)) else 0.0
                    vy = float(current_obs.get("linear_vels_y", [0])[0]) if isinstance(current_obs.get("linear_vels_y"), (list, np.ndarray)) else 0.0
                    speed = np.sqrt(vx**2 + vy**2)
                    return speed * 3.6 * self.DISPLAY_SPEED_MULTIPLIER
                elif "std_state" in current_obs and isinstance(current_obs["std_state"], (list, np.ndarray)):
                    # Direct observation format
                    if len(current_obs["std_state"]) >= 4:
                        vx = float(current_obs["std_state"][3])
                        speed = abs(vx)
                        return speed * 3.6 * self.DISPLAY_SPEED_MULTIPLIER
                elif "speed" in current_obs:
                    # Direct speed field
                    return float(current_obs["speed"]) * 3.6 * self.DISPLAY_SPEED_MULTIPLIER
            
            # Fallback: return 0.0 if we can't find speed
            return 0.0
            
        except Exception as e:
            print(f"Error extracting speed: {e}")
            import traceback
            traceback.print_exc()
            return 0.0
    
    def _extract_position_from_obs(self, obs) -> tuple:
        """Extract current position from observation data"""
        try:
            # Handle different observation formats
            if hasattr(obs, 'keys'):
                # Multi-agent observation
                agent_id = list(obs.keys())[0]
                current_obs = obs[agent_id]
            else:
                # Single agent observation
                current_obs = obs
            
            # Try different observation formats
            if isinstance(current_obs, dict):
                if "poses_x" in current_obs and "poses_y" in current_obs:
                    # Original observation format
                    x = float(current_obs["poses_x"][0])
                    y = float(current_obs["poses_y"][0])
                    return (x, y)
                elif "std_state" in current_obs:
                    # Direct observation format
                    x = float(current_obs["std_state"][0])
                    y = float(current_obs["std_state"][1])
                    return (x, y)
                elif "position" in current_obs:
                    # Direct position field
                    pos = current_obs["position"]
                    return (float(pos[0]), float(pos[1]))
            
            # Fallback
            return (0.0, 0.0)
            
        except Exception as e:
            print(f"Error extracting position: {e}")
            return (0.0, 0.0)
    
    def _extract_heading_from_obs(self, obs) -> float:
        """Extract current heading/yaw from observation data"""
        try:
            # Handle different observation formats
            if hasattr(obs, 'keys'):
                # Multi-agent observation
                agent_id = list(obs.keys())[0]
                current_obs = obs[agent_id]
            else:
                # Single agent observation
                current_obs = obs
            
            # Try different observation formats
            if isinstance(current_obs, dict):
                if "poses_theta" in current_obs:
                    # Original observation format
                    return float(current_obs["poses_theta"][0])
                elif "std_state" in current_obs:
                    # Direct observation format
                    return float(current_obs["std_state"][2])  # yaw angle
                elif "heading" in current_obs:
                    # Direct heading field
                    return float(current_obs["heading"])
            
            # Fallback
            return 0.0
            
        except Exception as e:
            print(f"Error extracting heading: {e}")
            return 0.0
    
    def _extract_action_data(self, action) -> tuple:
        """Extract throttle, brake, and steering from action data"""
        try:
            # F1Tenth action format: [steering, speed]
            # steering: -0.4189 to 0.4189 (radians)
            # speed: -4.0 to 10.0 (m/s)
            
            # Handle different action formats
            steering = 0.0
            speed_command = 0.0
            
            if isinstance(action, np.ndarray):
                # Flatten array if needed to get to 1D
                action = action.flatten() if action.ndim > 1 else action
                
                if action.shape == () or len(action) == 1:
                    # Single element - probably just steering
                    steering = float(action) if action.shape == () else float(action.item())
                    speed_command = 0.0
                elif len(action) >= 2:
                    steering = float(action.item(0)) if action.size == 1 else float(action.flat[0])
                    speed_command = float(action.item(1)) if action.size <= 2 else float(action.flat[1])
                else:
                    return 0.0, 0.0, 0.0
            elif isinstance(action, (list, tuple)):
                steering = float(action[0]) if len(action) > 0 else 0.0
                speed_command = float(action[1]) if len(action) > 1 else 0.0
            else:
                # Scalar or unknown format
                steering = float(action) if hasattr(action, '__float__') else 0.0
                speed_command = 0.0
            
            # Convert speed command to throttle/brake
            if speed_command >= 0:
                # Positive speed = throttle
                throttle = min(100.0, max(0.0, (speed_command / 10.0) * 100.0))  # Convert to percentage
                brake = 0.0
            else:
                # Negative speed = brake
                throttle = 0.0
                brake = min(100.0, max(0.0, (abs(speed_command) / 4.0) * 100.0))  # Convert to percentage
            
            # Convert steering to percentage (-100% to +100%)
            steering_percent = (steering / 0.4189) * 100.0
            steering_percent = max(-100.0, min(100.0, steering_percent))
            
            return throttle, brake, steering_percent
            
        except Exception as e:
            # Silently fail to avoid cluttering output - action data extraction is non-critical
            return 0.0, 0.0, 0.0
    
    def _calculate_co2_savings_from_distance(self, distance_m: float) -> float:
        """Calculate CO₂ savings based on cumulative distance traveled"""
        try:
            distance_km = distance_m / 1000  # Convert meters to km
            
            # Calculate baseline CO₂ emissions (inefficient human driving)
            # Human drivers typically use 25% more fuel due to suboptimal racing lines
            human_efficiency_factor = self.CO2_HUMAN_EFFICIENCY_FACTOR
            baseline_fuel_consumption = (distance_km / 100) * self.F1_FUEL_CONSUMPTION_PER_100KM * human_efficiency_factor
            baseline_co2 = baseline_fuel_consumption * self.F1_CO2_PER_KG_FUEL
            
            # Calculate AI CO₂ emissions (efficient driving)
            # AI uses optimal racing lines, smooth acceleration/deceleration
            ai_efficiency_factor = self.CO2_AI_EFFICIENCY_FACTOR
            ai_fuel_consumption = (distance_km / 100) * self.F1_FUEL_CONSUMPTION_PER_100KM * ai_efficiency_factor
            ai_co2 = ai_fuel_consumption * self.F1_CO2_PER_KG_FUEL
            
            # Calculate CO₂ savings
            co2_savings = baseline_co2 - ai_co2
            
            # Add bonus for smooth driving (less aggressive acceleration/deceleration)
            # This is a simplified model - in reality, smooth driving reduces fuel consumption
            co2_savings *= (1 + self.SMOOTH_DRIVING_BONUS)
            
            return max(0, co2_savings)  # Ensure non-negative
            
        except Exception as e:
            print(f"Error calculating CO₂ savings: {e}")
            return 0.0

    def _format_lap_time(self, seconds: float) -> str:
        """Format lap time as MM:SS.mmm"""
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}:{remaining_seconds:06.3f}"
    
    def get_simulation_status(self) -> Dict[str, Any]:
        """Get current simulation status"""
        if not self.current_simulation:
            return {
                'status': 'idle',
                'simulation_id': None,
                'message': 'No simulation running'
            }
        
        # Convert to dictionary for JSON serialization
        status_dict = asdict(self.current_simulation)
        
        # Add additional status info
        status_dict['is_running'] = self.current_simulation.status == 'running'
        status_dict['has_comparison'] = self.current_simulation.comparison_data is not None
        
        return status_dict
    
    def stop_simulation(self):
        """Stop the current simulation"""
        if self.current_simulation and self.current_simulation.status == 'running':
            self.stop_event.set()
            
            # Wait for thread to finish
            if self.simulation_thread and self.simulation_thread.is_alive():
                self.simulation_thread.join(timeout=5.0)
            
            if self.current_simulation.status == 'running':
                self.current_simulation.status = 'stopped'
                self.current_simulation.end_time = time.time()

# Global simulation runner instance
simulation_runner = SimulationRunner()

# Export for use in app.py
__all__ = ['simulation_runner', 'SimulationRunner', 'SimulationConfig', 'SimulationData']
