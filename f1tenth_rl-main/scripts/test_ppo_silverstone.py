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

from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import f1tenth_gym
from f1tenth_wrapper.env import F1TenthWrapper

# Import environment creation from training script
sys.path.insert(0, os.path.dirname(__file__))
from train_multi_track_ppo import make_multi_track_env, SafeF1TenthWrapper, SafeRacingRewardWrapper

# Use wrappers from training script for consistency

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

TARGET_TRACK = "Monza"

def format_lap_time(seconds: float) -> str:
    """Format lap time in F1 standard format (MM:SS.mmm)"""
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    return f"{minutes}:{remaining_seconds:06.3f}"

def main(
    model_dir: str = "models/multi_track_ppo_2025-10-28_19-37-56/best_model",
    model_name: str = "best_model.zip",  # Use best_model for eval
    track: str = "Monza",
    video_recording: bool = False,
    n_episodes: int = 5,
):
    seed = 42  # Use consistent seed for testing
    set_random_seed(seed)

    # Construct paths - VecNormalize stats are typically in the parent directory
    # If model_dir points to "best_model" subfolder, stats are in parent
    if model_dir.endswith("best_model") or os.path.basename(model_dir) == "best_model":
        # Stats are in the parent directory (where training saved them)
        parent_dir = os.path.dirname(model_dir)
        vec_normalize_path = os.path.join(parent_dir, "vec_normalize.pkl")
        # Also check root of model_dir in case user provides full path
        possible_stats_paths = [
            vec_normalize_path,  # Parent directory (most common location)
            os.path.join(model_dir, "vec_normalize.pkl"),  # Current directory
            os.path.join(model_dir, "best_model", "vec_normalize.pkl"),  # Nested best_model
        ]
    else:
        # Check multiple possible locations
        possible_stats_paths = [
            os.path.join(model_dir, "vec_normalize.pkl"),  # Root of model_dir
            os.path.join(model_dir, "best_model", "vec_normalize.pkl"),  # In best_model subdir
            os.path.join(os.path.dirname(model_dir), "vec_normalize.pkl"),  # Parent directory
        ]
        vec_normalize_path = possible_stats_paths[0]
    
    model_path = os.path.join(model_dir, model_name)
    
    # Search for VecNormalize stats
    vec_normalize_path = None
    use_vec_normalize = False
    for path in possible_stats_paths:
        if os.path.exists(path):
            vec_normalize_path = path
            use_vec_normalize = True
            break
    
    if vec_normalize_path is None:
        print(f"⚠️  WARNING: VecNormalize stats not found!")
        print(f"   Checked these locations:")
        for path in possible_stats_paths:
            print(f"   - {path}")
        print(f"   Your model was trained with normalized observations.")
        print(f"   Without stats, the model will receive raw data and may crash.")
        print(f"   Suggestion: Run with model_dir pointing to parent folder:")
        if "best_model" in model_dir:
            parent = os.path.dirname(model_dir)
            print(f"   python scripts/test_ppo_silverstone.py --model_dir={parent}")
        use_vec_normalize = False
    else:
        print(f"✅ Found VecNormalize stats at: {vec_normalize_path}")

    # 1. Create environment using the EXACT function from training
    # We need to modify make_multi_track_env to use the specified track
    # For now, we'll create a custom version that uses the target track
    def make_test_env(rank=0, seed=0, target_track=None):
        """Create test environment with specific track - matches training exactly"""
        def _init():
            from train_multi_track_ppo import AVAILABLE_TRACKS
            from gymnasium.wrappers import TimeLimit
            from stable_baselines3.common.monitor import Monitor
            
            # Get base config from training script
            base_config = {
                "num_agents": 1,
                "timestep": 0.01,
                "integrator": "rk4",
                "control_input": ["speed", "steering_angle"],
                "model": "st",
                "observation_config": {"type": "original"},
                "params": {
                    "mu": 1.5, "C_Sf": 5.5, "C_Sr": 6.0, "lf": 0.15875,
                    "lr": 0.17145, "h": 0.074, "m": 3.74, "I": 0.04712,
                    "s_min": -0.4189, "s_max": 0.4189, "sv_min": -3.2,
                    "sv_max": 3.2, "v_switch": 7.319,
                    "a_max": 5.0, "v_min": -1.0, "v_max": 15.0,
                    "width": 0.31, "length": 0.58,
                },
                "reset_config": {"type": "rl_grid_static"},
                "seed": seed + rank,
            }
            
            # Use target track if provided, otherwise use rank-based selection
            if target_track:
                config = base_config.copy()
                config["map"] = target_track
            else:
                track_index = rank % len(AVAILABLE_TRACKS)
                assigned_track = AVAILABLE_TRACKS[track_index]
                config = base_config.copy()
                config["map"] = assigned_track
            
            # Create environment with EXACT same wrapper order as training
            env = SafeF1TenthWrapper(config=config, render_mode="rgb_array" if video_recording else "human")
            
            # 1. Add reward wrapper (matches training)
            env = SafeRacingRewardWrapper(env)
            
            # 2. Add TimeLimit (matches training)
            env = TimeLimit(env, max_episode_steps=5000)
            
            # 3. Add Monitor (matches training)
            env = Monitor(env, filename=None)
            
            return env
        return _init
    
    # 2. Create DummyVecEnv (required for VecNormalize)
    env = DummyVecEnv([make_test_env(rank=0, seed=seed, target_track=track)])
    
    # 3. Load VecNormalize stats if available
    if use_vec_normalize:
        print(f"📊 Loading VecNormalize stats from {vec_normalize_path}...")
        try:
            env = VecNormalize.load(vec_normalize_path, env)
            # CRITICAL: Set evaluation mode
            env.training = False  # Do not update running stats
            env.norm_reward = False  # Do not normalize rewards (we want real scores)
            print("✅ VecNormalize stats loaded successfully!")
        except Exception as e:
            print(f"⚠️  Failed to load VecNormalize stats: {e}")
            print("   Continuing without VecNormalize wrapper (may cause poor performance)")
            # Don't wrap with VecNormalize if stats can't be loaded
            use_vec_normalize = False
    else:
        # If stats weren't found, don't wrap with VecNormalize at all
        # The model will receive unnormalized observations (may not work well)
        print("⚠️  No VecNormalize stats available - using raw environment")
        use_vec_normalize = False
    
    # 4. Add video recording wrapper if enabled
    if video_recording:
        if not os.path.exists("videos"):
            os.makedirs("videos")
        video_path = os.path.join("videos", f"PPO_{track}_{int(time.time())}")
        # RecordVideo needs to wrap the VecEnv, so we use it directly
        env = gym.wrappers.RecordVideo(env, video_path, name_prefix=f"ppo-{track}")
    
    # Debug: Check observation space
    print(f"Environment observation space: {env.observation_space}")
    print(f"Environment action space: {env.action_space}")

    # Load the PPO model
    try:
        print(f"🤖 Loading PPO model from {model_path}...")
        model = PPO.load(model_path, device=device)
        print(f"✅ Successfully loaded trained PPO model!")
        print(f"   Model observation space: {model.observation_space}")
        print(f"   Model action space: {model.action_space}")
        print(f"   Environment observation space: {env.observation_space}")
        print(f"   Environment action space: {env.action_space}")
    except Exception as e:
        print(f"❌ Failed to load PPO model: {e}")
        raise

    episode_rewards = []
    episode_lengths = []
    lap_times = []

    print(f"\n🏁 Testing PPO model on {track} for {n_episodes} episodes...")
    print("=" * 60)
    
    for episode in range(n_episodes):
        # VecEnv.reset() only returns obs (not info)
        obs = env.reset()
        
        # VecEnv returns arrays, get first element if needed
        if isinstance(obs, np.ndarray) and len(obs.shape) > 1:
            obs = obs[0]
        
        done = False
        env.render()
        
        episode_reward = 0
        episode_length = 0
        lap_completed = False
        lap_start_time = 0
        elapsed_time = 0
        time_step = 0.01  # Default timestep
        
        # Get initial toggle_list from core environment using manual unwrapping (compatible with SB3 2.7.0)
        try:
            # Manual unwrapping: VecNormalize -> DummyVecEnv -> Monitor -> TimeLimit -> SafeRacingRewardWrapper -> SafeF1TenthWrapper -> core
            inner_env = env.envs[0]  # Get the base env from the VecEnv
            while hasattr(inner_env, "env"):
                inner_env = inner_env.env
            base_core_env = inner_env.core
            time_step = base_core_env.config["timestep"]
            initial_toggle = base_core_env.toggle_list[0] if hasattr(base_core_env, 'toggle_list') else 0
            lap_start_time = base_core_env.current_time
            print(f"Starting lap. Initial toggle: {initial_toggle}")
        except Exception as e:
            print(f"Exception getting initial toggle: {e}")
            initial_toggle = 0
            lap_start_time = 0
            time_step = 0.01
        
        while not done and not lap_completed:
            # Get action from model
            # VecNormalize handles normalization automatically
            # F1TenthWrapper handles action scaling, so no manual clipping needed
            action, _ = model.predict(obs, deterministic=True)
            
            # Calculate elapsed time and check for lap completion
            try:
                # Manual unwrapping: VecNormalize -> DummyVecEnv -> Monitor -> TimeLimit -> SafeRacingRewardWrapper -> SafeF1TenthWrapper -> core
                inner_env = env.envs[0]  # Get the base env from the VecEnv
                while hasattr(inner_env, "env"):
                    inner_env = inner_env.env
                base_core_env = inner_env.core
                
                current_time = base_core_env.current_time
                elapsed_time = current_time - lap_start_time
                current_toggle = base_core_env.toggle_list[0] if hasattr(base_core_env, 'toggle_list') else 0

                if episode_length % 50 == 0:
                    print(f"   Step {episode_length}: Lap Time: {format_lap_time(elapsed_time)}, Toggle: {current_toggle}")

                if current_toggle >= 2:  # 0 -> 1 (halfway) -> 2 (full lap)
                    lap_completed = True
                    print(f"\n✅ Lap completed! Lap Time: {format_lap_time(elapsed_time)}")

            except Exception as e:
                # Fallback timer (less accurate)
                elapsed_time = episode_length * time_step
                if episode_length % 500 == 0:
                    print(f"Error checking toggle/time (using fallback timer): {e}")
                pass

            # Step the environment (VecEnv returns 4 values: obs, reward, done, info)
            obs, step_reward, done, info = env.step([action])  # VecEnv expects list
            
            # Get truncated from the info dict (it's a list of dicts)
            info_dict = info[0] if isinstance(info, (list, tuple)) else info
            truncated = info_dict.get("TimeLimit.truncated", False) if isinstance(info_dict, dict) else False
            
            # Now process the rest (which are arrays)
            done = done[0] if isinstance(done, (list, np.ndarray)) else done
            step_reward = step_reward[0] if isinstance(step_reward, (list, np.ndarray)) else step_reward
            obs = obs[0] if len(obs.shape) > 1 else obs

            episode_reward += step_reward
            episode_length += 1
            env.render()
            
            done = done or truncated or lap_completed
            
            if episode_length > 10000:
                print("Max steps reached")
                break

        # Capture final lap time
        try:
            if lap_completed:
                lap_time = elapsed_time
            else:
                # Get final time from core using manual unwrapping
                inner_env = env.envs[0]  # Get the base env from the VecEnv
                while hasattr(inner_env, "env"):
                    inner_env = inner_env.env
                base_core_env = inner_env.core
                lap_time = base_core_env.current_time
        except:
            lap_time = elapsed_time

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        lap_times.append(lap_time)

        print(f"Episode {episode + 1}:")
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Length: {episode_length} steps")
        print(f"  Lap Time: {format_lap_time(lap_time)}")
        if episode_length > 0:
            print(f"  Avg Reward: {episode_reward/episode_length:.3f}")
        print("-" * 40)
        
        # Stop after first episode
        if episode == 0:
            print("\n🎬 Stopping after first episode as requested.")
            break

    env.close()

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Track: {track}")
    print(f"Model: {model_path}")
    print(f"Episodes: {n_episodes}")
    print(f"Mean Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Mean Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Mean Lap Time: {format_lap_time(np.mean(lap_times))} ± {np.std(lap_times):.3f}s")
    print(f"Best Reward: {np.max(episode_rewards):.2f}")
    print(f"Best Lap Time: {format_lap_time(np.min(lap_times))}")

if __name__ == "__main__":
    fire.Fire(main)

