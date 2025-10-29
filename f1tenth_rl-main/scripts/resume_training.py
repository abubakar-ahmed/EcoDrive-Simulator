import gymnasium as gym
import datetime
import torch

# Set PyTorch for optimal GPU performance
torch.set_default_dtype(torch.float32)
torch.backends.cudnn.deterministic = False  # Allow non-deterministic for better GPU performance
torch.backends.cudnn.benchmark = True  # Enable cuDNN benchmark for better performance
torch.set_float32_matmul_precision('medium')

import sys
import os
import numpy as np
import warnings
import random
from typing import Optional, List

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Force use of local f1tenth_gym by removing any installed version from sys.modules
if 'f1tenth_gym' in sys.modules:
    del sys.modules['f1tenth_gym']

# Add the f1tenth_gym folder to the Python path (insert at beginning to prioritize local)
# Handle both script execution and interactive environments (Jupyter, etc.)
try:
    script_dir = os.path.dirname(__file__)
except NameError:
    # Running in interactive environment (Jupyter, etc.)
    script_dir = os.getcwd()

local_f1tenth_path = os.path.join(script_dir, '..', 'f1tenth_gym')
local_src_path = os.path.join(script_dir, '..', 'src')
if local_f1tenth_path not in sys.path:
    sys.path.insert(0, local_f1tenth_path)
if local_src_path not in sys.path:
    sys.path.insert(0, local_src_path)

from stable_baselines3 import SAC
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, VecEnvWrapper
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure as configure_logger
from stable_baselines3.common.type_aliases import TrainFreq  # ✅ ADD THIS IMPORT

# Check SB3 version
import stable_baselines3
print(f"Stable-Baselines3 version: {stable_baselines3.__version__}")
if stable_baselines3.__version__.startswith('1.'):
    print("⚠️  Using SB3 1.x - max_grad_norm not supported")
else:
    print("✅ Using SB3 2.x - max_grad_norm supported")

import f1tenth_gym
from f1tenth_wrapper.env import F1TenthWrapper

class SafeF1TenthWrapper(F1TenthWrapper):
    """Wrapper that handles inf observations at the source"""
    
    def _observation(self, original_obs):
        # Sanitize observations before processing
        sanitized_obs = self._sanitize_observation(original_obs)
        
        try:
            obs = super()._observation(sanitized_obs)
            if obs is None or not np.isfinite(obs).all():
                print(f"Warning: Invalid observation after processing, using safe default")
                if hasattr(self, 'observation_space') and hasattr(self.observation_space, 'shape'):
                    obs = np.zeros(self.observation_space.shape, dtype=np.float32)
                else:
                    obs = np.zeros(10, dtype=np.float32)
            return obs
        except ValueError as e:
            if "Invalid observation" in str(e):
                print(f"Warning: Invalid observation detected even after sanitization: {e}")
                obs = np.zeros(self.observation_space.shape, dtype=np.float32)
                return obs
            else:
                raise e
    
    def _sanitize_observation(self, original_obs):
        """Sanitize observations to handle inf/nan values"""
        if not isinstance(original_obs, dict):
            return original_obs
        
        safe_obs = {}
        for key, value in original_obs.items():
            if isinstance(value, np.ndarray):
                # Replace inf/nan values with safe defaults
                safe_value = np.nan_to_num(value, posinf=1000.0, neginf=-1000.0, nan=0.0)
                safe_value = np.clip(safe_value, -1000.0, 1000.0)
                safe_obs[key] = safe_value.astype(np.float32)
            else:
                safe_obs[key] = value
        
        return safe_obs

class SafeRacingRewardWrapper(gym.Wrapper):
    """Stable reward wrapper with balanced scaling"""
    
    def __init__(self, env):
        super().__init__(env)
        self.prev_speed = 0.0
        self.speed_history = []
        self.prev_steering = 0.0
        self.total_steps = 0
        
    def step(self, action):
        action = self._clip_actions(action)
        obs, reward, done, truncated, info = self.env.step(action)
        self.total_steps += 1
        
        # Get original observation for reward calculation
        original_obs = info.get("original_obs", {})
        new_reward = self._calculate_stable_rewards(original_obs, action, done, truncated)
        
        return obs, new_reward, done, truncated, info
    
    def _clip_actions(self, action):
        """Clip actions to valid range"""
        if isinstance(action, np.ndarray):
            if hasattr(self.env, 'action_space'):
                low = self.env.action_space.low
                high = self.env.action_space.high
                tolerance = 1e-6
                if np.any(action < low - tolerance) or np.any(action > high + tolerance):
                    print(f"Warning: Action out of bounds detected. Action: {action}, Bounds: [{low}, {high}]")
                action = np.clip(action, low, high)
            else:
                action = np.clip(action, -1.0, 1.0)
        return action
    
    def _calculate_stable_rewards(self, original_obs, action, done, truncated):
        """Calculate stable rewards with balanced scaling"""
        # Crash penalty
        if done and not truncated: 
            return -10.0
        
        # Base survival reward
        survival_reward = 0.1
        reward_modifier = survival_reward
        
        # Speed reward (log-based for stability)
        current_speed = 0.0 
        try:
            if "linear_vels_x" in original_obs and "linear_vels_y" in original_obs:
                current_speed = np.sqrt(original_obs["linear_vels_x"][0]**2 + original_obs["linear_vels_y"][0]**2)
            
            if current_speed > 1.0: 
                speed_reward = np.log(current_speed + 1) * 0.05
                reward_modifier += min(speed_reward, 0.3)
            
            # Consistency reward (smooth speed changes)
            if len(self.speed_history) > 0:
                speed_change = abs(current_speed - self.speed_history[-1])
                if speed_change < 0.5: 
                    reward_modifier += 0.05
        except (KeyError, IndexError, TypeError):
            pass 
        
        # Steering penalty (encourage smooth steering)
        steering_penalty = 0.0
        if len(action) > 0:
            try:
                steering_action = action[0] if hasattr(action, '__len__') else action
                if hasattr(steering_action, '__len__') and len(steering_action) > 0:
                    steering_action_scalar = float(steering_action[0])
                else:
                    steering_action_scalar = float(steering_action)
                steering_penalty = abs(steering_action_scalar) * 0.01
            except (ValueError, TypeError):
                pass
        
        reward_modifier -= steering_penalty
        
        # Episode length bonus (encourage longer episodes)
        episode_length_bonus = min(self.total_steps * 0.001, 0.5)
        reward_modifier += episode_length_bonus
        
        # Update speed history
        self.speed_history.append(current_speed)
        if len(self.speed_history) > 10:
            self.speed_history = self.speed_history[-10:]
        
        # Final reward clipping
        total_reward = np.clip(reward_modifier, -10.0, 1.0) 
        
        # Safety check for NaN/inf
        if not np.isfinite(total_reward):
            total_reward = 0.0
        
        return total_reward
    
    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.prev_speed = 0.0
        self.speed_history = []
        self.prev_steering = 0.0
        self.total_steps = 0
        
        return obs, info

class NaNMonitorCallback(BaseCallback):
    """Monitor for NaN values and stop training if detected"""
    
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.check_freq = 1000
        
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Check for NaN values in the model
            try:
                for name, param in self.model.policy.actor.named_parameters():
                    if torch.isnan(param).any():
                        print(f"WARNING: NaN detected in actor.{name}")
                        return False
                
                for name, param in self.model.policy.critic.named_parameters():
                    if torch.isnan(param).any():
                        print(f"WARNING: NaN detected in critic.{name}")
                        return False
            except Exception as e:
                if self.verbose > 0:
                    print(f"Error checking for NaN: {e}")
        
        return True

# Available tracks for multi-track training (F1-style tracks only)
AVAILABLE_TRACKS = [
    "Silverstone",    # British GP - High speed, flowing corners
    "Monza",          # Italian GP - Temple of Speed, high speeds
    "Spa",            # Belgian GP - Fast, challenging layout
    "Catalunya",      # Spanish GP - Technical, good for testing
    "Spielberg",      # Austrian GP - Short, fast track
]

def make_multi_track_env(rank: int = 0, seed: int = 0):
    """Create environment with a specific track based on rank"""
    
    # Base configuration with physics stability fixes
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
            "sv_max": 3.2, "v_switch": 7.319,             "a_max": 4.0,  # BALANCED: Good acceleration for learning
            "v_min": -1.0, "v_max": 10.0, "width": 0.31, "length": 0.58,  # BALANCED: Good speed limits for learning
        },
        "reset_config": {"type": "rl_grid_static"},
        "seed": seed + rank,
    }
    
    def _init():
        # Assign different tracks to different parallel environments
        # This ensures each environment trains on a different track
        track_index = rank % len(AVAILABLE_TRACKS)
        assigned_track = AVAILABLE_TRACKS[track_index]
        
        config = base_config.copy()
        config["map"] = assigned_track
        
        env = SafeF1TenthWrapper(config=config, render_mode=None)  # OPTIMIZED: No rendering for training to save memory
        
        # Log action space for verification
        print(f"Track {assigned_track}: Action space = {env.action_space}")
        
        # --- CORRECT WRAPPER ORDER ---
        # 1. Add reward wrapper
        env = SafeRacingRewardWrapper(env)
        # 2. Add Monitor LAST (so it logs the modified reward)
        env = Monitor(env, filename=None)
        # -------------------------
        
        return env
    
    return _init

def create_multi_track_env(n_envs: int = 4, seed: int = 0): # Add seed parameter
    """Create vectorized environment with multi-track support and VecNormalize"""
    
    try:
        if n_envs == 1:
            # Pass seed to make_multi_track_env
            env = DummyVecEnv([make_multi_track_env(0, seed)])
        else:
            import platform
            start_method = "spawn" if platform.system() == "Windows" else "fork"
            # Pass seed to make_multi_track_env in the list comprehension
            env = SubprocVecEnv([make_multi_track_env(i, seed) for i in range(n_envs)], start_method=start_method)
        
        # CRITICAL: Add VecNormalize for stable training
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)
        return env
    except Exception as e:
        print(f"Failed to create parallel environments: {e}")
        print("Falling back to single environment...")
        # Pass seed here too for the fallback
        env = DummyVecEnv([make_multi_track_env(0, seed)])
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)
        return env

def main():
    print("F1Tenth Multi-Track SAC Resume Training System")
    print("=" * 50)

    # --- PATHS FOR RESUMING ---
    # Using the exact path you provided
    run_id_to_resume = "2025-10-25_14-34-08"  # ✅ Use the ID of the run that just froze
    model_dir_to_resume = f"models/multi_track_sac_{run_id_to_resume}"
    
    # !!! CHANGE THIS LINE if final_model.zip wasn't saved correctly !!!
    # checkpoint_path = f"{model_dir_to_resume}/final_model.zip" # Old
    checkpoint_path = f"{model_dir_to_resume}/checkpoints/multi_track_sac_250000_steps.zip" # ✅ Point to the 250k checkpoint
    
    vec_normalize_path = f"{model_dir_to_resume}/vec_normalize.pkl"

    # Check if resuming files exist
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    if not os.path.exists(vec_normalize_path):
         raise FileNotFoundError(f"VecNormalize stats not found: {vec_normalize_path}")

    print(f"Resuming training from checkpoint: {checkpoint_path}")
    print(f"Loading VecNormalize stats from: {vec_normalize_path}")
    # ----------------------------

    # Set up device and GPU memory
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA Version: {torch.version.cuda}")
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Total GPU Memory: {total_memory:.1f}GB")
        
        # Set GPU memory fraction
        memory_fraction = 0.6 if total_memory >= 8.0 else 0.8
        print(f"Setting GPU memory fraction to: {memory_fraction}")
        torch.cuda.set_per_process_memory_fraction(memory_fraction)

    # Set random seed
    seed = 42
    set_random_seed(seed)

    # Create environments (VecNormalize is applied inside)
    n_envs = len(AVAILABLE_TRACKS)
    
    # --- LOAD VECNORMALIZE STATS ---
    # Load the stats *BEFORE* creating the model, but *AFTER* creating the env
    # Pass seed to create_multi_track_env
    env = create_multi_track_env(n_envs, seed=seed) # <-- Pass seed here
    env = VecNormalize.load(vec_normalize_path, env)
    # -------------------------------

    # --- IMPROVED EVAL ENV CREATION WITH VECNORMALIZE ---
    # Create individual eval environments for per-track evaluation
    eval_envs = []
    for i, track in enumerate(AVAILABLE_TRACKS):
        eval_env_func = make_multi_track_env(rank=i, seed=seed + 1000 + i)
        eval_envs.append(eval_env_func)
    
    # Create a single VecEnv containing all evaluation environments
    eval_env = DummyVecEnv(eval_envs)
    
    # --- FIX: WRAP EVAL ENV IN VECNORMALIZE ---
    # It MUST be wrapped so the callback can sync the training stats to it.
    # We set norm_reward=False to log the true, unnormalized rewards.
    # We set training=False so the eval env doesn't update its own stats.
    eval_env = VecNormalize(eval_env,
                           norm_obs=True,
                           norm_reward=False,
                           clip_obs=10.0,
                           training=False)
    # Make eval env use the same stats as the training env
    eval_env.obs_rms = env.obs_rms
    eval_env.ret_rms = env.ret_rms
    eval_env.norm_obs = env.norm_obs
    eval_env.norm_reward = False # Keep this False for eval
    eval_env.clip_obs = env.clip_obs
    eval_env.epsilon = env.epsilon
    # ------------------------------------------------

    # --- Add these lines for debugging ---
    print("-" * 30)
    print(f"DEBUG: Type of training env (env): {type(env)}")
    print(f"DEBUG: Type of evaluation env (eval_env): {type(eval_env)}")
    print(f"DEBUG: Is eval_env VecEnvWrapper? {isinstance(eval_env, VecEnvWrapper)}")
    print(f"DEBUG: Is eval_env VecNormalize? {isinstance(eval_env, VecNormalize)}")
    print("-" * 30)
    # -------------------------------------

    # --- DIRECTORIES FOR THE *NEW* RUN ---
    # It's good practice to save continued training to a new folder
    new_run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_resumed"
    model_dir = f"models/multi_track_sac_{new_run_id}"
    log_dir = f"runs/multi_track_sac_{new_run_id}"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(f"{model_dir}/checkpoints", exist_ok=True) # Create checkpoints subdir
    os.makedirs(f"{model_dir}/eval_logs", exist_ok=True)   # Create eval_logs subdir
    # --------------------------------------

    # --- LOAD THE MODEL ---
    print("Loading model...")
    model = SAC.load(
        checkpoint_path,
        env=env, # Pass the VecNormalize env
        device=device,
        # If you changed hyperparameters, specify them here, otherwise defaults are loaded
        # For example: learning_rate=..., buffer_size=...
    )
    
    # ✅ ADD THESE LINES TO FIX THE TRAINING SPEED
    print("Overriding train_freq and gradient_steps for faster training...")
    model.train_freq = TrainFreq(64, "step")  # ✅ USE THE TrainFreq OBJECT
    model.gradient_steps = 64
    
    # Correct configuration for SB3 v2.x
    new_logger = configure_logger(folder=log_dir, format_strings=["stdout", "tensorboard"])
    model.set_logger(new_logger)
    # ----------------------

    # Set up callbacks (saving to the NEW directory)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{model_dir}/best_model",
        log_path=f"{model_dir}/eval_logs", # Save logs to new folder
        eval_freq=50000,  # ✅ Evaluate less often (was 15000)
        deterministic=True,
        render=False,
        n_eval_episodes=2, # ✅ 2 episodes per track is enough (was 5)
        warn=False
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=f"{model_dir}/checkpoints", # Save checkpoints to new folder
        name_prefix="multi_track_sac_resumed",
    )

    nan_monitor_callback = NaNMonitorCallback(verbose=1)
    callbacks = [eval_callback, checkpoint_callback, nan_monitor_callback]

    print(f"Starting resumed multi-track SAC training with {n_envs} parallel environments...")

    # --- ADJUST TOTAL TIMESTEPS ---
    # Calculate remaining timesteps
    # !!! CHANGE THIS VALUE to match the loaded checkpoint step count !!!
    # already_completed_steps = 75000  # Old/Incorrect
    already_completed_steps = 250000 # ✅ MUST MATCH THE CHECKPOINT FILE
    
    total_target_steps = 2000000
    remaining_timesteps = total_target_steps - already_completed_steps

    print(f"Target total timesteps: {total_target_steps}")
    print(f"Already completed: {already_completed_steps}")
    print(f"Remaining timesteps to train: {remaining_timesteps}")
    if remaining_timesteps <= 0:
         print("Warning: Already trained for the target number of steps or more.")
         remaining_timesteps = 100000 # Train for additional 100k steps
    # ----------------------------

    # Train the model
    try:
        model.learn(
            total_timesteps=remaining_timesteps, # Train for remaining steps
            callback=callbacks,
            progress_bar=True,
            tb_log_name="multi_track_sac",
            log_interval=100,  # ✅ Print logs to console every 100 episodes
            reset_num_timesteps=False # IMPORTANT: Do NOT reset the step counter
        )
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()
    finally:
        # Save the final model and VecNormalize stats
        final_model_path = f"{model_dir}/final_model.zip"
        final_vec_normalize_path = f"{model_dir}/vec_normalize.pkl" # Save to new dir
        try:
            model.save(final_model_path)
            env.save(final_vec_normalize_path) # Save the UPDATED stats
            print(f"Saved final model and VecNormalize stats to {model_dir}")
        except Exception as e:
            print(f"Failed to save final model: {e}")
            import traceback
            print("Full traceback:")
            traceback.print_exc()

    # Close environments
    try:
        env.close()
        eval_env.close()
    except Exception as e:
        print(f"Error closing environments: {e}")

if __name__ == "__main__":
    main()
