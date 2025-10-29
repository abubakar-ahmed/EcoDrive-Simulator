import gymnasium as gym
import datetime
import torch
import sys
import os
import numpy as np
import warnings
import time
from typing import Optional
import fire

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

if 'f1tenth_gym' in sys.modules:
    del sys.modules['f1tenth_gym']

local_f1tenth_path = os.path.join(os.path.dirname(__file__), '..', 'f1tenth_gym')
local_src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
if local_f1tenth_path not in sys.path:
    sys.path.insert(0, local_f1tenth_path)
if local_src_path not in sys.path:
    sys.path.insert(0, local_src_path)

from stable_baselines3 import SAC
from stable_baselines3.common.utils import set_random_seed

import f1tenth_gym
from f1tenth_wrapper.env import F1TenthWrapper

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

def format_lap_time(seconds: float) -> str:
    """Format lap time in F1 standard format (MM:SS.mmm)"""
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    return f"{minutes}:{remaining_seconds:06.3f}"

class SafeRacingRewardWrapper(gym.Wrapper):
    """Wrapper that modifies rewards for safe racing with good braking"""
    
    def __init__(self, env):
        super().__init__(env)
        self.prev_speed = 0.0
        self.speed_history = []
        self.prev_steering = 0.0
        
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        
        original_obs = info.get("original_obs", {})
        safe_reward_modifier = self._calculate_safe_rewards(original_obs, action)
        reward += safe_reward_modifier
        
        return obs, reward, done, truncated, info
    
    def _calculate_safe_rewards(self, original_obs, action):
        """Calculate safe racing reward modifications focused on good braking"""
        reward_modifier = 0.0
        
        current_speed = np.sqrt(original_obs["linear_vels_x"][0]**2 + original_obs["linear_vels_y"][0]**2)
        
        survival_reward = 1.0
        reward_modifier += survival_reward
        
        if current_speed > 0.5:
            speed_reward = min(current_speed * 0.1, 2.0)
            reward_modifier += speed_reward
        
        if len(action) > 0:
            try:
                speed_action = action[1] if hasattr(action, '__len__') and len(action) > 1 else 0
                if hasattr(speed_action, '__len__') and len(speed_action) > 0:
                    speed_action_scalar = float(speed_action[0])
                else:
                    speed_action_scalar = float(speed_action)
                
                if speed_action_scalar < 0 and current_speed > 2.0:
                    braking_reward = abs(speed_action_scalar) * 0.2
                    reward_modifier += braking_reward
            except (ValueError, TypeError):
                pass
        
        if len(action) > 0:
            try:
                steering_action = action[0] if hasattr(action, '__len__') else action
                if hasattr(steering_action, '__len__') and len(steering_action) > 0:
                    steering_action_scalar = float(steering_action[0])
                else:
                    steering_action_scalar = float(steering_action)
                
                steering_penalty = abs(steering_action_scalar) * 0.05
                reward_modifier -= steering_penalty
            except (ValueError, TypeError):
                pass
        
        return reward_modifier
    
    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.prev_speed = 0.0
        self.speed_history = []
        self.prev_steering = 0.0
        return obs, info

def make_track_env(track_name: str, rank: int = 0, seed: int = 0):
    """Create environment fixed to specified track"""

    base_config = {
        "num_agents": 1,
        "timestep": 0.01,
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
        },
        "reset_config": {"type": "rl_grid_static"},
        "seed": seed + rank,
    }

    def _init():
        config = base_config.copy()
        config["map"] = track_name

        env = F1TenthWrapper(config=config, render_mode="rgb_array")
        env = SafeRacingRewardWrapper(env)
        return env

    return _init

class SACWrapper:
    def __init__(self, model_path: str, track_name: str = "Monza", horizon: int = 10):
        temp_env = gym.make(
            "f1tenth-RL-v0",
            config={
                "map": track_name,
                "num_agents": 1,
                "timestep": 0.01,
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
                },
                "reset_config": {"type": "rl_grid_static"},
                "max_laps": 1,
            },
        )
        
        print(f"Action space: {temp_env.action_space}")
        
        try:
            print("Loading SAC model...")
            self.model = SAC.load(model_path, device=device)
            print("✅ Successfully loaded trained SAC model!")
            
        except Exception as e:
            print(f"Failed to load SAC model: {e}")
            print("Creating new SAC model...")
            
            self.model = SAC("MlpPolicy", temp_env, verbose=1)
            
            try:
                import zipfile
                import io
                
                with zipfile.ZipFile(model_path, 'r') as zip_file:
                    policy_data = zip_file.read('policy.pth')
                    policy_state = torch.load(io.BytesIO(policy_data), map_location='cpu')
                    self.model.policy.load_state_dict(policy_state)
                    print("✅ Successfully loaded SAC policy weights")
                    
            except Exception as e2:
                print(f"Failed to load policy weights: {e2}")
                print("⚠️ Using untrained SAC model")
                self.model = None
        
        temp_env.close()

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        if self.model is not None:
            action = self.model.predict(obs, deterministic=True)[0]
            
            try:
                # Ensure action is 1D array with 2 elements
                if len(action.shape) > 1:
                    action = action.flatten()
                
                # Convert to numpy array for easier manipulation
                action = np.array(action, dtype=np.float32)
                    
                if len(action) >= 2:
                    # Clip steering angle to valid bounds
                    action[0] = float(np.clip(action[0], -0.4189, 0.4189))
                    
                    # Process speed action with proper bounds
                    if action[1] < 0:
                        # Braking: amplify but respect v_min = -4.0
                        action[1] = float(np.clip(action[1] * 1.2, -4.0, 0.0))
                    else:
                        # Acceleration: scale down but respect v_max = 10.0
                        action[1] = float(np.clip(action[1] * 0.8, 0.0, 10.0))
                else:
                    print(f"Warning: Action has unexpected length: {len(action)}")
                    
            except Exception as e:
                print(f"Action processing error: {e}")
                # Fallback to safe default action
                action = np.array([0.0, 0.0], dtype=np.float32)
                
            return action
        else:
            print("Warning: No SAC model loaded, using random actions")
            return np.random.uniform(
                low=[-0.4189, -4.0], 
                high=[0.4189, 10.0], 
                size=(2,)
            ).astype(np.float32)

def main(
    model_path: str = "models/sac_monza_2025-10-28_16-39-22/best_model/best_model.zip",
    track: str = "Monza",
    video_recording: bool = False,
    n_episodes: int = 5,
):
    TARGET_TRACK = track
    seed = 1
    set_random_seed(seed)

    # Create environment using the same setup as training (with SafeRacingRewardWrapper)
    base_config = {
        "num_agents": 1,
        "timestep": 0.01,
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
        },
        "reset_config": {"type": "rl_grid_static"},
        "seed": seed,
        "map": TARGET_TRACK,
        "max_laps": 1,
    }
    
    env = F1TenthWrapper(config=base_config, render_mode="rgb_array" if video_recording else "human")
    env = SafeRacingRewardWrapper(env)

    if video_recording:
        if not os.path.exists("videos"):
            os.makedirs("videos")
        video_path = os.path.join("videos", f"SAC_{TARGET_TRACK}_{time.time()}")
        env = gym.wrappers.RecordVideo(env, video_path)

    sac_wrapper = SACWrapper(model_path, track_name=TARGET_TRACK, horizon=30)
    
    # Helper function to safely get the F1TenthWrapper core
    def get_core_env(env):
        """Unwrap all wrappers to get the F1TenthWrapper"""
        unwrapped = env
        while hasattr(unwrapped, 'unwrapped') and unwrapped.unwrapped != unwrapped:
            unwrapped = unwrapped.unwrapped
        return unwrapped
    
    # Access the unwrapped environment (F1TenthWrapper) to get track info
    try:
        unwrapped_env = get_core_env(env)
        if hasattr(unwrapped_env, 'track'):
            track = unwrapped_env.track
            if hasattr(track, "raceline") and track.raceline is not None:
                unwrapped_env.add_render_callback(track.raceline.render_waypoints)
    except Exception as e:
        print(f"Skipping raceline render callback: {e}")

    episode_rewards = []
    episode_lengths = []
    lap_times = []

    print(f"\nTesting SAC model on {TARGET_TRACK} for {n_episodes} episodes...")
    print("=" * 60)

    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        env.render()

        episode_reward = 0
        episode_length = 0
        unwrapped_env = get_core_env(env)
        time_step = unwrapped_env.core.config["timestep"] if hasattr(unwrapped_env, 'core') else 0.01
        lap_completed = False
        lap_start_time = 0
        elapsed_time = 0

        # Get initial toggle_list (starts at 0, reaches 2 after one lap, 4 after two laps)
        try:
            # Access toggle_list directly from unwrapped environment
            if hasattr(unwrapped_env, 'core') and hasattr(unwrapped_env.core, 'toggle_list'):
                initial_toggle = unwrapped_env.core.toggle_list[0]
            else:
                initial_toggle = 0
            lap_start_time = unwrapped_env.core.current_time if hasattr(unwrapped_env, 'core') else 0
            print(f"Starting lap. Initial toggle: {initial_toggle}")
        except Exception as e:
            print(f"Exception getting initial toggle: {e}")
            initial_toggle = 0
            lap_start_time = 0

        while not done and not lap_completed:
            action = sac_wrapper.get_action(obs)
            
            # Calculate elapsed time (stopwatch style)
            try:
                unwrapped_env = get_core_env(env)
                current_time = unwrapped_env.core.current_time if hasattr(unwrapped_env, 'core') else 0
                elapsed_time = current_time - lap_start_time
            except:
                elapsed_time = episode_length * time_step
            
            if episode_length % 50 == 0:
                print(f"Step {episode_length}: Action = {action}, Lap Time: {format_lap_time(elapsed_time)}")
                
            obs, step_reward, done, truncated, info = env.step(action)
            episode_reward += step_reward
            episode_length += 1
            env.render()
            
            # Check if one lap has been completed (toggle_list reaches 2)
            try:
                # Access toggle_list after step
                unwrapped_env = get_core_env(env)
                if hasattr(unwrapped_env, 'core') and hasattr(unwrapped_env.core, 'toggle_list'):
                    current_toggle = unwrapped_env.core.toggle_list[0]
                else:
                    current_toggle = 0
                
                if episode_length % 50 == 0:
                    print(f"  Current toggle: {current_toggle}")
                if current_toggle >= 2:
                    lap_completed = True
                    print(f"\n✅ Lap completed! Toggle: {current_toggle}, Lap Time: {format_lap_time(elapsed_time)}")
            except Exception as e:
                if episode_length % 500 == 0:
                    print(f"Error checking toggle: {e}")
                pass
            
            done = done or truncated or lap_completed
            
            if episode_length > 10000:
                print("Max steps reached")
                break

        # Capture final lap time
        try:
            unwrapped_env = get_core_env(env)
            lap_time = elapsed_time if lap_completed else (unwrapped_env.core.current_time if hasattr(unwrapped_env, 'core') else elapsed_time)
        except:
            lap_time = elapsed_time

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        lap_times.append(lap_time)

        print(f"Episode {episode + 1}:")
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Length: {episode_length} steps")
        print(f"  Lap Time: {format_lap_time(lap_time)}")
        print(f"  Avg Speed: {episode_reward/episode_length:.3f}")
        print("-" * 40)
        
        # Stop after first episode
        if episode == 0:
            print("\n🎬 Stopping after first episode as requested.")
            break

    env.close()

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Track: {TARGET_TRACK}")
    print(f"Model: {model_path}")
    print(f"Episodes: {n_episodes}")
    print(f"Mean Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Mean Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Mean Lap Time: {format_lap_time(np.mean(lap_times))} ± {np.std(lap_times):.3f}s")
    print(f"Best Reward: {np.max(episode_rewards):.2f}")
    print(f"Best Lap Time: {format_lap_time(np.min(lap_times))}")

if __name__ == "__main__":
    fire.Fire(main)