from flask import Flask, jsonify, request, Response, send_file
from flask_cors import CORS
import json
import os
import time
import re
from datetime import datetime
import random
import shutil
from pathlib import Path

# Try to import psutil for system monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("psutil not available - system monitoring disabled. Install with: pip install psutil")

# Import FastF1 service if available
try:
    from fastf1_service import fastf1_service
    FASTF1_AVAILABLE = True
    print("FastF1 service loaded")
except ImportError as e:
    print(f"FastF1 service not available: {e}")
    FASTF1_AVAILABLE = False
    fastf1_service = None

# Import web simulation if available
try:
    from web_simulation import simulation_runner
    WEB_SIMULATION_AVAILABLE = True
    print("Web simulation module loaded")
except ImportError as e:
    print(f"Web simulation not available: {e}")
    WEB_SIMULATION_AVAILABLE = False

app = Flask(__name__)

# Configure CORS - allow specific origins in production, all in development
cors_origins = os.environ.get('CORS_ORIGINS', '').split(',') if os.environ.get('CORS_ORIGINS') else None
if cors_origins:
    # Filter out empty strings
    cors_origins = [origin.strip() for origin in cors_origins if origin.strip()]
    if cors_origins:
        CORS(app, resources={r"/api/*": {"origins": cors_origins}})
    else:
        CORS(app)  # Allow all if CORS_ORIGINS is set but empty
else:
    CORS(app)  # Allow all origins in development

# Mock data storage
simulation_data = {
    'logs': [],
    'models': {
        'ppo': {'accuracy': 94.2, 'efficiency': 92.5, 'consistency': 96.1},
        'sac': {'accuracy': 96.8, 'efficiency': 94.7, 'consistency': 93.8},
        'td3': {'accuracy': 95.5, 'efficiency': 93.2, 'consistency': 95.4}
    },
    'system_stats': {
        'total_simulations': 1247,
        'active_users': 89,
        'average_session_time': 42.3,
        'system_uptime': 99.8,
        'data_processed': 2.4
    }
}

@app.route('/')
def index():
    return jsonify({
        'message': 'EcoDrive Simulator API',
        'version': '1.0.0',
        'status': 'running'
    })

@app.route('/api/simulation/start', methods=['POST'])
def start_simulation():
    """Start a new simulation with given parameters"""
    data = request.get_json()
    
    # Validate required parameters
    required_params = ['track', 'model_version']
    if not all(param in data for param in required_params):
        return jsonify({'error': 'Missing required parameters'}), 400
    
    # Generate simulation ID
    simulation_id = f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}"
    
    # Try to start web simulation if available
    if WEB_SIMULATION_AVAILABLE:
        try:
            # Add simulation ID to config
            data['simulation_id'] = simulation_id
            
            # Create simulation log entry BEFORE starting (so it exists for updates)
            log_entry = {
                'id': simulation_id,
                'timestamp': datetime.now().isoformat(),
                'user': data.get('user', 'anonymous'),
                'track': data['track'],
                'model_version': data['model_version'],
                'status': 'running',
                'duration': 0,
                'ai_lap_time': None,
                'hamilton_lap_time': None,
                'time_difference': None,
                'ai_faster': None,
                'web_simulation': True
            }
            
            simulation_data['logs'].append(log_entry)
            
            # Pass logs_list to start_simulation so it can update the log entry
            result = simulation_runner.start_simulation(data, simulation_data['logs'])
            
            if result['success']:
                
                return jsonify({
                    'simulation_id': simulation_id,
                    'status': 'started',
                    'message': f"Simulation started with {data['model_version'].upper()} model on {data['track']}",
                    'web_simulation': True
                })
            else:
                return jsonify({
                    'simulation_id': simulation_id,
                    'status': 'error',
                    'error': result.get('error', 'Unknown error'),
                    'web_simulation': True
                }), 400
                
        except Exception as e:
            print(f"Web simulation failed: {e}")
            return jsonify({
                'simulation_id': simulation_id,
                'status': 'error',
                'error': str(e),
                'web_simulation': True
            }), 500
    
    # Fallback to mock simulation
    log_entry = {
        'id': simulation_id,
        'timestamp': datetime.now().isoformat(),
        'user': data.get('user', 'anonymous'),
        'track': data['track'],
        'model_version': data['model_version'],
        'status': 'running',
        'duration': 0,
        'ai_lap_time': None,
        'hamilton_lap_time': None,
        'time_difference': None,
        'ai_faster': None,
        'web_simulation': False
    }
    
    simulation_data['logs'].append(log_entry)
    
    return jsonify({
        'simulation_id': simulation_id,
        'status': 'started',
        'message': 'Mock simulation started successfully',
        'web_simulation': False
    })

@app.route('/api/simulation/<simulation_id>/update', methods=['POST'])
def update_simulation(simulation_id):
    """Update simulation progress"""
    data = request.get_json()
    
    # Find simulation log
    log_entry = next((log for log in simulation_data['logs'] if log['id'] == simulation_id), None)
    if not log_entry:
        return jsonify({'error': 'Simulation not found'}), 404
    
    # Update simulation data
    if 'ai_metrics' in data:
        log_entry['ai_lap_time'] = data['ai_metrics'].get('lap_time')
        log_entry['ai_energy'] = data['ai_metrics'].get('energy_consumption')
        log_entry['ai_co2_saved'] = data['ai_metrics'].get('co2_saved')
    
    if 'human_metrics' in data:
        log_entry['human_lap_time'] = data['human_metrics'].get('lap_time')
        log_entry['human_energy'] = data['human_metrics'].get('energy_consumption')
        log_entry['human_co2_saved'] = data['human_metrics'].get('co2_saved')
    
    if 'duration' in data:
        log_entry['duration'] = data['duration']
    
    return jsonify({
        'status': 'updated',
        'simulation_id': simulation_id
    })

@app.route('/api/simulation/<simulation_id>/complete', methods=['POST'])
def complete_simulation(simulation_id):
    """Mark simulation as completed"""
    data = request.get_json()
    
    # Find simulation log
    log_entry = next((log for log in simulation_data['logs'] if log['id'] == simulation_id), None)
    if not log_entry:
        return jsonify({'error': 'Simulation not found'}), 404
    
    # Update final results
    log_entry['status'] = 'completed'
    log_entry['duration'] = data.get('duration', log_entry['duration'])
    log_entry['ai_lap_time'] = data.get('ai_lap_time')
    log_entry['hamilton_lap_time'] = data.get('hamilton_lap_time')
    log_entry['time_difference'] = data.get('time_difference')
    log_entry['ai_faster'] = data.get('ai_faster')
    
    # Update system stats
    simulation_data['system_stats']['total_simulations'] += 1
    
    return jsonify({
        'status': 'completed',
        'simulation_id': simulation_id,
        'results': {
            'ai_lap_time': log_entry['ai_lap_time'],
            'hamilton_lap_time': log_entry['hamilton_lap_time'],
            'time_difference': log_entry['time_difference'],
            'ai_faster': log_entry['ai_faster']
        }
    })

@app.route('/api/simulation/logs', methods=['GET'])
def get_simulation_logs():
    """Get all simulation logs"""
    limit = request.args.get('limit', 50, type=int)
    offset = request.args.get('offset', 0, type=int)
    
    logs = simulation_data['logs'][offset:offset + limit]
    return jsonify({
        'logs': logs,
        'total': len(simulation_data['logs']),
        'limit': limit,
        'offset': offset
    })

@app.route('/api/models/performance', methods=['GET'])
def get_model_performance():
    """Get AI model performance metrics"""
    return jsonify(simulation_data['models'])

@app.route('/api/models/<model_name>/retrain', methods=['POST'])
def retrain_model(model_name):
    """Trigger model retraining"""
    if model_name not in simulation_data['models']:
        return jsonify({'error': 'Model not found'}), 404
    
    # Simulate retraining process
    return jsonify({
        'status': 'retraining_started',
        'model': model_name,
        'estimated_time': '2-4 hours',
        'message': f'{model_name.upper()} model retraining initiated'
    })

@app.route('/api/telemetry/generate', methods=['POST'])
def generate_telemetry():
    """Get real telemetry data from simulation or generate mock data"""
    try:
        # First try to get real simulation data
        if WEB_SIMULATION_AVAILABLE and simulation_runner.current_simulation:
            sim_data = simulation_runner.get_simulation_status()
            if sim_data.get('telemetry_data'):
                # Use real telemetry data
                telemetry = sim_data['telemetry_data']
                
                # Add Hamilton comparison data (mock for now)
                hamilton_telemetry = []
                for point in telemetry:
                    hamilton_telemetry.append({
                        'time': point['time'],
                        'speed': point['speed'] * 0.9 + random.uniform(-10, 10),  # Slightly slower with variation
                        'throttle': max(0, min(100, point['throttle'] + random.uniform(-10, 10))),
                        'brake': max(0, min(100, point['brake'] + random.uniform(-5, 5))),
                        'steering': point['steering'] + random.uniform(-5, 5),
                        'energy': point['time'] * 0.7 + random.uniform(-0.5, 0.5)
                    })
                
                return jsonify({
                    'success': True,
                    'ai_telemetry': telemetry,
                    'hamilton_telemetry': hamilton_telemetry,
                    'duration': telemetry[-1]['time'] if telemetry else 0,
                    'points': len(telemetry)
                })
        
        # Fallback to mock data if no simulation data available
        data = request.get_json()
        duration = data.get('duration', 30)  # seconds
        points = data.get('points', 100)
        
        telemetry = []
        for i in range(points):
            time = (i / points) * duration
            telemetry.append({
                'time': time,
                'speed': 200 + random.uniform(-50, 100) + 50 * (i % 10) / 10,  # F1-scaled speeds
                'throttle': max(0, min(100, 50 + random.uniform(-30, 30))),
                'brake': max(0, min(100, 20 + random.uniform(-15, 15))),
                'steering': random.uniform(-30, 30),
                'energy': time * 0.5 + random.uniform(-1, 1),
                'co2_saved': time * 0.01 + random.uniform(-0.005, 0.005)
            })
        
        return jsonify({
            'success': True,
            'ai_telemetry': telemetry,
            'hamilton_telemetry': [],  # No Hamilton data for mock
            'duration': duration,
            'points': points
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/models/available', methods=['GET'])
def get_available_models():
    """Get available AI models"""
    import os
    
    # Use absolute paths to match web_simulation.py
    models = {
        'ppo': {
            'name': 'PPO Multi-Track',
            'description': 'Proximal Policy Optimization trained on multiple tracks',
            'path': os.path.join(os.path.dirname(__file__), 'models', 'multi_track_ppo_2025-10-12_02-58-46', 'final_model.zip'),
            'trained_tracks': ['Catalunya', 'Spielberg', 'Silverstone', 'Monza', 'Spa', 'Hockenheim', 'Budapest', 'Melbourne', 'Mexico City', 'Montreal', 'Nuerburgring', 'Oschersleben', 'Sakhir', 'SaoPaulo', 'Sepang', 'Shanghai', 'Sochi', 'YasMarina', 'Zandvoort', 'Austin']
        },
        'sac': {
            'name': 'SAC Single-Track',
            'description': 'Soft Actor-Critic trained on Catalunya track',
            'path': os.path.join(os.path.dirname(__file__), 'models', '2025-10-08_21-27-56', 'model.zip'),
            'trained_tracks': ['Catalunya']
        }
    }
    return jsonify(models)

@app.route('/api/tracks', methods=['GET'])
def get_tracks():
    """Get available tracks with detailed information"""
    import os
    import yaml
    
    tracks_dir = os.path.join(os.path.dirname(__file__), 'maps')
    tracks = []
    
    try:
        # Read track directories
        for track_name in os.listdir(tracks_dir):
            track_path = os.path.join(tracks_dir, track_name)
            if os.path.isdir(track_path) and not track_name.startswith('.'):
                track_info = {
                    'name': track_name,
                    'display_name': track_name.replace('_', ' '),
                    'has_image': False,
                    'has_centerline': False,
                    'has_raceline': False,
                    'image_path': None,
                    'yaml_path': None
                }
                
                # Check for track files
                for file in os.listdir(track_path):
                    if file.endswith('_map.png'):
                        track_info['has_image'] = True
                        track_info['image_path'] = f'/api/tracks/{track_name}/image'
                    elif file.endswith('_centerline.csv'):
                        track_info['has_centerline'] = True
                    elif file.endswith('_raceline.csv'):
                        track_info['has_raceline'] = True
                    elif file.endswith('_map.yaml'):
                        track_info['yaml_path'] = f'/api/tracks/{track_name}/yaml'
                
                tracks.append(track_info)
        
        # Sort tracks alphabetically
        tracks.sort(key=lambda x: x['name'])
        
    except Exception as e:
        print(f"Error reading tracks: {e}")
        # Fallback to basic list
        tracks = [{'name': name, 'display_name': name} for name in [
            'Catalunya', 'Spielberg', 'Silverstone', 'Monza', 'Spa', 'YasMarina',
            'Austin', 'BrandsHatch', 'Budapest', 'Hockenheim', 'Melbourne',
            'Mexico City', 'Montreal', 'MoscowRaceway', 'Nuerburgring',
            'Oschersleben', 'Sakhir', 'SaoPaulo', 'Sepang', 'Shanghai',
            'Sochi', 'Zandvoort'
        ]]
    
    return jsonify(tracks)

@app.route('/api/tracks/<track_name>/image', methods=['GET'])
def get_track_image(track_name):
    """Get track image"""
    import os
    from flask import send_file
    
    try:
        tracks_dir = os.path.join(os.path.dirname(__file__), 'maps')
        track_path = os.path.join(tracks_dir, track_name)
        
        # Look for the map image
        for file in os.listdir(track_path):
            if file.endswith('_map.png'):
                image_path = os.path.join(track_path, file)
                return send_file(image_path, mimetype='image/png')
        
        return jsonify({'error': 'Track image not found'}), 404
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/tracks/<track_name>/info', methods=['GET'])
def get_track_info(track_name):
    """Get detailed track information"""
    import os
    import yaml
    import csv
    
    try:
        tracks_dir = os.path.join(os.path.dirname(__file__), 'maps')
        track_path = os.path.join(tracks_dir, track_name)
        
        if not os.path.exists(track_path):
            return jsonify({'error': 'Track not found'}), 404
        
        track_info = {
            'name': track_name,
            'display_name': track_name.replace('_', ' '),
            'files': [],
            'yaml_data': None,
            'centerline_points': 0,
            'raceline_points': 0
        }
        
        # Read YAML file
        yaml_file = None
        for file in os.listdir(track_path):
            if file.endswith('_map.yaml'):
                yaml_file = os.path.join(track_path, file)
                break
        
        if yaml_file:
            with open(yaml_file, 'r') as f:
                track_info['yaml_data'] = yaml.safe_load(f)
        
        # Count centerline points
        centerline_file = None
        for file in os.listdir(track_path):
            if file.endswith('_centerline.csv'):
                centerline_file = os.path.join(track_path, file)
                break
        
        if centerline_file:
            with open(centerline_file, 'r') as f:
                reader = csv.reader(f)
                track_info['centerline_points'] = sum(1 for row in reader)
        
        # Count raceline points
        raceline_file = None
        for file in os.listdir(track_path):
            if file.endswith('_raceline.csv'):
                raceline_file = os.path.join(track_path, file)
                break
        
        if raceline_file:
            with open(raceline_file, 'r') as f:
                reader = csv.reader(f)
                track_info['raceline_points'] = sum(1 for row in reader)
        
        # List all files
        track_info['files'] = os.listdir(track_path)
        
        return jsonify(track_info)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/insights/rl', methods=['GET'])
def get_rl_insights():
    """Get reinforcement learning educational content"""
    insights = {
        'title': 'Reinforcement Learning Fundamentals',
        'sections': [
            {
                'title': 'What is Reinforcement Learning?',
                'content': 'Reinforcement Learning (RL) is a type of machine learning where an AI agent learns to make decisions by interacting with an environment.',
                'points': [
                    'Agent: The AI system making decisions',
                    'Environment: The world the agent interacts with',
                    'Actions: What the agent can do',
                    'Rewards: Feedback that guides learning',
                    'Policy: The strategy the agent uses to choose actions'
                ]
            }
        ]
    }
    
    return jsonify(insights)

@app.route('/api/hamilton/<track>', methods=['GET'])
def get_hamilton_data(track):
    """Get Lewis Hamilton's performance data for a specific track"""
    if not FASTF1_AVAILABLE:
        # Return mock data if FastF1 is not available
        mock_data = {
            'driver_name': 'Lewis Hamilton',
            'lap_time': 87.5 + random.uniform(-2, 2),
            'lap_time_formatted': f"1:{int(87.5 + random.uniform(-2, 2)):02d}.{int(random.uniform(0, 999)):03d}",
            'energy_consumption': 2.8 + random.uniform(-0.2, 0.2),
            'average_speed': 185.5 + random.uniform(-5, 5),
            'data_source': 'Mock Data (FastF1 unavailable)',
            'is_real_data': False
        }
        return jsonify({
            'success': True,
            'data': mock_data
        })
    
    try:
        hamilton_data = fastf1_service.get_hamilton_comparison_data(track)
        return jsonify({
            'success': True,
            'data': hamilton_data
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'data': fastf1_service.get_hamilton_comparison_data(track)
        }), 500

@app.route('/api/hamilton/lap-data', methods=['POST'])
def get_hamilton_lap_data():
    """Get detailed Lewis Hamilton lap data"""
    data = request.get_json()
    track = data.get('track', 'Silverstone')
    year = data.get('year', 2023)
    session = data.get('session', 'Q')
    
    try:
        lap_data = fastf1_service.get_hamilton_lap_data(year, track, session)
        return jsonify({
            'success': True,
            'data': lap_data
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/comparison/<track>', methods=['GET'])
def get_ai_vs_hamilton_comparison(track):
    """Get AI vs Lewis Hamilton comparison data"""
    try:
        # First try to get real simulation data if available
        if WEB_SIMULATION_AVAILABLE and simulation_runner.current_simulation:
            sim_data = simulation_runner.get_simulation_status()
            if sim_data.get('comparison_data'):
                # Use real simulation data
                comparison = sim_data['comparison_data']
                hamilton_data = sim_data.get('hamilton_data', {})
                
                comparison_data = {
                    'track': track,
                    'hamilton': {
                        'driver_name': hamilton_data.get('driver_name', 'Lewis Hamilton'),
                        'lap_time': comparison['hamilton_lap_time'],
                        'lap_time_formatted': comparison['hamilton_lap_time_formatted'],
                        'energy_consumption': hamilton_data.get('energy_consumption', 2.8),
                        'average_speed': hamilton_data.get('average_speed', 185.5),
                        'data_source': hamilton_data.get('data_source', 'FastF1 Data'),
                        'is_real_data': hamilton_data.get('is_real_data', True)
                    },
                    'ai': {
                        'model': 'AI Eco-Drive',
                        'lap_time': comparison['ai_lap_time'],
                        'lap_time_formatted': comparison['ai_lap_time_formatted'],
                        'energy_consumption': sim_data.get('energy_consumption', 2.4),
                        'average_speed': sim_data.get('average_speed', sim_data.get('current_speed', 200.0)),
                        'efficiency_gain': round(((hamilton_data.get('energy_consumption', 2.8) - sim_data.get('energy_consumption', 2.4)) / hamilton_data.get('energy_consumption', 2.8)) * 100, 1)
                    },
                    'comparison': {
                        'time_difference': comparison['time_difference'],
                        'energy_saved': round(hamilton_data.get('energy_consumption', 2.8) - sim_data.get('energy_consumption', 2.4), 2),
                        'co2_saved': comparison['co2_saved'],
                        'co2_saved_formatted': comparison['co2_saved_formatted'],
                        'ai_faster': comparison['ai_faster'],
                        'ai_more_efficient': sim_data.get('energy_consumption', 2.4) < hamilton_data.get('energy_consumption', 2.8),
                        'simulation_notes': comparison.get('simulation_notes', 'Real simulation data')
                    }
                }
                
                return jsonify({
                    'success': True,
                    'data': comparison_data
                })
        
        # Fallback to mock data if no simulation data available
        if FASTF1_AVAILABLE:
            hamilton_data = fastf1_service.get_hamilton_comparison_data(track)
        else:
            # Use mock data if FastF1 is not available
            hamilton_data = {
                'driver_name': 'Lewis Hamilton',
                'lap_time': 87.5 + random.uniform(-2, 2),
                'lap_time_formatted': f"1:{int(87.5 + random.uniform(-2, 2)):02d}.{int(random.uniform(0, 999)):03d}",
                'energy_consumption': 2.8 + random.uniform(-0.2, 0.2),
                'average_speed': 185.5 + random.uniform(-5, 5),
                'data_source': 'Mock Data (FastF1 unavailable)',
                'is_real_data': False
            }
        
        # Generate AI performance data (simulated)
        ai_lap_time = hamilton_data['lap_time'] * random.uniform(0.95, 1.05)  # AI within 5% of Hamilton
        ai_energy = hamilton_data['energy_consumption'] * random.uniform(0.85, 0.95)  # AI more efficient
        
        comparison_data = {
            'track': track,
            'hamilton': {
                'driver_name': hamilton_data['driver_name'],
                'lap_time': hamilton_data['lap_time'],
                'lap_time_formatted': hamilton_data['lap_time_formatted'],
                'energy_consumption': hamilton_data['energy_consumption'],
                'average_speed': hamilton_data['average_speed'],
                'data_source': hamilton_data['data_source'],
                'is_real_data': hamilton_data['is_real_data']
            },
            'ai': {
                'model': 'PPO Eco-Drive',
                'lap_time': round(ai_lap_time, 3),
                'lap_time_formatted': f"{int(ai_lap_time//60)}:{ai_lap_time%60:06.3f}",
                'energy_consumption': round(ai_energy, 2),
                'average_speed': hamilton_data['average_speed'] * random.uniform(0.98, 1.02),
                'efficiency_gain': round(((hamilton_data['energy_consumption'] - ai_energy) / hamilton_data['energy_consumption']) * 100, 1)
            },
            'comparison': {
                'time_difference': round(ai_lap_time - hamilton_data['lap_time'], 3),
                'energy_saved': round(hamilton_data['energy_consumption'] - ai_energy, 2),
                'co2_saved': 0.0,  # Mock data
                'co2_saved_formatted': '0.000 kg',
                'ai_faster': ai_lap_time < hamilton_data['lap_time'],
                'ai_more_efficient': ai_energy < hamilton_data['energy_consumption'],
                'simulation_notes': 'Mock data - run simulation for real results'
            }
        }
        
        return jsonify({
            'success': True,
            'data': comparison_data
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/fastf1/seasons', methods=['GET'])
def get_available_seasons():
    """Get available F1 seasons"""
    if not FASTF1_AVAILABLE:
        # Return mock seasons if FastF1 is not available
        return jsonify({
            'success': True,
            'seasons': list(range(2018, 2024))
        })
    
    try:
        seasons = fastf1_service.get_available_seasons()
        return jsonify({
            'success': True,
            'seasons': seasons
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/fastf1/tracks/<int:year>', methods=['GET'])
def get_available_tracks(year):
    """Get available tracks for a specific year"""
    if not FASTF1_AVAILABLE:
        # Return mock tracks if FastF1 is not available
        mock_tracks = [
            'Silverstone', 'Monza', 'Spa', 'Catalunya', 'YasMarina', 
            'Austin', 'Budapest', 'Melbourne', 'Montreal', 'Sakhir'
        ]
        return jsonify({
            'success': True,
            'tracks': mock_tracks,
            'year': year
        })
    
    try:
        tracks = fastf1_service.get_available_tracks(year)
        return jsonify({
            'success': True,
            'tracks': tracks,
            'year': year
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Web Simulation Endpoints
@app.route('/api/web-simulation/status', methods=['GET'])
def get_web_simulation_status():
    """Get current web simulation status"""
    if not WEB_SIMULATION_AVAILABLE:
        return jsonify({'error': 'Web simulation not available'}), 503
    
    try:
        status = simulation_runner.get_simulation_status()
        return jsonify(status)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/web-simulation/stop', methods=['POST'])
def stop_web_simulation():
    """Stop the current web simulation"""
    if not WEB_SIMULATION_AVAILABLE:
        return jsonify({'error': 'Web simulation not available'}), 503
    
    try:
        simulation_runner.stop_simulation()
        return jsonify({'status': 'stopped'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/web-simulation/stream', methods=['GET'])
def get_web_simulation_stream():
    """Get simulation data as Server-Sent Events stream"""
    if not WEB_SIMULATION_AVAILABLE:
        return jsonify({'error': 'Web simulation not available'}), 503
    
    try:
        def generate_stream():
            while True:
                status = simulation_runner.get_simulation_status()
                
                # Send current status
                yield f"data: {json.dumps(status)}\n\n"
                
                # Break if simulation is finished
                if status['status'] in ['completed', 'error', 'stopped']:
                    break
                
                # Small delay to prevent overwhelming
                time.sleep(0.1)  # 10 FPS
        
        return Response(generate_stream(), mimetype='text/event-stream', headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Cache-Control'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/web-simulation/frames', methods=['GET'])
def get_web_simulation_frames():
    """Get simulation frames as streaming response"""
    if not WEB_SIMULATION_AVAILABLE:
        return jsonify({'error': 'Web simulation not available'}), 503
    
    try:
        def generate_frames():
            while True:
                status = simulation_runner.get_simulation_status()
                if status['status'] in ['completed', 'error', 'stopped']:
                    break
                    
                if status.get('current_frame'):
                    # Send latest frame
                    yield f"data: {json.dumps({'frame': status['current_frame'], 'status': status['status'], 'lap_time': status['lap_time'], 'progress': status['progress']})}\n\n"
                
                time.sleep(0.1)  # 10 FPS
        
        return Response(generate_frames(), mimetype='text/event-stream')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Developer download endpoints
@app.route('/api/developer/download/<resource_type>', methods=['GET'])
def download_resource(resource_type):
    """Download resources for developers"""
    try:
        # Get base paths
        base_dir = Path(__file__).parent
        models_dir = base_dir / 'models'
        maps_dir = base_dir / 'maps'
        
        # Resource mapping
        resource_map = {
            'models-ppo': {
                'path': models_dir / 'multi_track_ppo_2025-10-12_02-58-46' / 'final_model.zip',
                'filename': 'multi_track_ppo_model.zip'
            },
            'models-sac': {
                'path': models_dir / 'multi_track_sac_2025-10-22_19-11-32' / 'final_model.zip',
                'filename': 'multi_track_sac_model.zip'
            },
            'maps': {
                'path': maps_dir,
                'filename': 'track_maps.zip',
                'is_dir': True
            },
            'environment': {
                'path': base_dir.parent / 'f1tenth_rl-main',
                'filename': 'environment.zip',
                'is_dir': True
            },
            'documentation': {
                'path': base_dir.parent / 'README.md',
                'filename': 'api_documentation.pdf'
            }
        }
        
        if resource_type not in resource_map:
            return jsonify({'error': 'Resource not found'}), 404
        
        resource = resource_map[resource_type]
        file_path = resource['path']
        
        # Check if file/directory exists
        if not file_path.exists():
            return jsonify({'error': 'Resource not available'}), 404
        
        # Handle directory resources (zip them first)
        if resource.get('is_dir'):
            import tempfile
            import zipfile
            
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
            temp_file.close()
            
            with zipfile.ZipFile(temp_file.name, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                if file_path.is_dir():
                    for root, dirs, files in file_path.rglob('*'):
                        for file in files:
                            file_path_full = Path(root) / file
                            arcname = file_path_full.relative_to(file_path)
                            zip_file.write(file_path_full, arcname)
            
            return send_file(
                temp_file.name,
                mimetype='application/zip',
                as_attachment=True,
                download_name=resource['filename']
            )
        
        # Handle regular files
        return send_file(
            str(file_path),
            mimetype='application/octet-stream',
            as_attachment=True,
            download_name=resource['filename']
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Admin Panel Endpoints
@app.route('/api/admin/system-stats', methods=['GET'])
def get_admin_system_stats():
    """Get real-time system statistics for admin panel"""
    try:
        # Get system stats using psutil if available
        if PSUTIL_AVAILABLE:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            cpu_usage = f"{cpu_percent:.1f}%"
            memory_usage = f"{memory.percent:.1f}%"
            disk_usage = f"{(disk.used / disk.total * 100):.1f}%"
        else:
            cpu_usage = "N/A"
            memory_usage = "N/A"
            disk_usage = "N/A"
        
        # Count active simulations
        active_simulations = 1 if (WEB_SIMULATION_AVAILABLE and simulation_runner.current_simulation and 
                                   simulation_runner.current_simulation.status == 'running') else 0
        
        # Get total simulation count from logs
        total_simulations = len(simulation_data['logs'])
        
        # Calculate system uptime (simplified - time since app started)
        # In production, you'd track actual server start time
        uptime_percent = "99.8%"  # Could be tracked with start_time
        
        return jsonify({
            'active_simulations': active_simulations,
            'total_simulations': total_simulations,
            'total_users': total_simulations,  # Use simulation count as proxy
            'system_uptime': uptime_percent,
            'cpu_usage': cpu_usage,
            'memory_usage': memory_usage,
            'disk_usage': disk_usage,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/models', methods=['GET'])
def get_models():
    """Get available models information by scanning models directory"""
    try:
        models_dir = Path(__file__).parent / 'models'
        models = []
        
        if not models_dir.exists():
            return jsonify({
                'models': [],
                'total_models': 0
            })
        
        # Scan all model directories
        for model_folder in models_dir.iterdir():
            if model_folder.is_dir():
                final_model_path = model_folder / 'final_model.zip'
                best_model_path = model_folder / 'best_model' / 'best_model.zip'
                
                # Check if either final_model.zip or best_model.zip exists
                if final_model_path.exists():
                    model_file = final_model_path
                elif best_model_path.exists():
                    model_file = best_model_path
                else:
                    continue
                
                # Extract model info from folder name (format: type_track_date_time)
                folder_name = model_folder.name
                
                # Determine model type and track
                if 'ppo' in folder_name.lower():
                    model_type = 'ppo'
                    if 'multi_track' in folder_name:
                        name = 'PPO Multi-Track'
                        description = 'Proximal Policy Optimization trained on multiple tracks'
                    else:
                        # Extract track name from folder name
                        name = 'PPO Single-Track'
                        description = 'Proximal Policy Optimization model'
                elif 'sac' in folder_name.lower():
                    model_type = 'sac'
                    if 'multi_track' in folder_name:
                        name = 'SAC Multi-Track'
                        description = 'Soft Actor-Critic trained on multiple tracks'
                    else:
                        # Extract track name from folder name (e.g., sac_silverstone_2025-10-22_13-44-55)
                        track_name = 'Unknown'
                        name_parts = folder_name.split('_')
                        if len(name_parts) > 1:
                            track_name = name_parts[1].replace('sac', '').title()
                        name = f'SAC - {track_name}'
                        description = f'Soft Actor-Critic model for {track_name}'
                else:
                    model_type = 'unknown'
                    name = model_folder.name
                    description = 'Model training directory'
                
                # Extract date from folder name (format: YYYY-MM-DD)
                date_match = re.search(r'(\d{4}-\d{2}-\d{2})', folder_name)
                last_trained = date_match.group(1) if date_match else 'Unknown'
                
                # Get file size
                try:
                    size_mb = model_file.stat().st_size / 1024 / 1024
                    size_str = f"{size_mb:.1f} MB"
                except:
                    size_str = 'N/A'
                
                # Determine accuracy based on model type and track
                accuracy = '96.8%' if model_type == 'sac' else '94.2%'
                
                # Extract version number from folder name or use default
                version_match = re.search(r'v?(\d+\.\d+\.\d+)', folder_name)
                version = version_match.group(0) if version_match else 'v1.0.0'
                
                models.append({
                    'name': name,
                    'status': 'Available',
                    'last_trained': last_trained,
                    'accuracy': accuracy,
                    'version': version,
                    'type': model_type,
                    'size': size_str,
                    'description': description,
                    'path': str(model_file.relative_to(Path(__file__).parent))
                })
        
        # Sort models by last_trained date (newest first)
        models.sort(key=lambda x: x['last_trained'], reverse=True)
        
        return jsonify({
            'models': models,
            'total_models': len(models)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/simulations', methods=['GET'])
def get_simulations():
    """Get simulation logs"""
    try:
        limit = request.args.get('limit', 50, type=int)
        logs = simulation_data['logs'][-limit:]  # Get last N logs
        logs.reverse()  # Most recent first
        
        return jsonify({
            'simulations': logs,
            'total': len(simulation_data['logs']),
            'returned': len(logs)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)