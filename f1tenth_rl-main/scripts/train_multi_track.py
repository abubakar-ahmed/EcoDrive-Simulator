import gymnasium as gym
import datetime
import torch

torch.set_default_dtype(torch.float32)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('medium')

import sys
import os
import numpy as np
import warnings
import random
from typing import Optional, List

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

if 'f1tenth_gym' in sys.modules:
    del sys.modules['f1tenth_gym']
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
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

import stable_baselines3
print(f"Stable-Baselines3 version: {stable_baselines3.__version__}")
if stable_baselines3.__version__.startswith('1.'):
    print("⚠️  Using SB3 1.x - max_grad_norm not supported")
else:
    print("✅ Using SB3 2.x - max_grad_norm supported")

import f1tenth_gym
from f1tenth_wrapper.env import F1TenthWrapper

class SafeF1TenthWrapper(F1TenthWrapper):
    """F1Tenth wrapper that handles invalid observations gracefully"""
    
    def _observation(self, original_obs):
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
        """Sanitize observation data to prevent inf/NaN values"""
        if not isinstance(original_obs, dict):
            return original_obs
            
        safe_obs = {}
        for key, value in original_obs.items():
            if isinstance(value, np.ndarray):
                safe_value = np.nan_to_num(value, 
                                        posinf=1000.0, 
                                        neginf=-1000.0,
                                        nan=0.0)
                safe_value = np.clip(safe_value, -1000.0, 1000.0)
                safe_obs[key] = safe_value.astype(np.float32)
            else:
                safe_obs[key] = value
        return safe_obs

class SafeRacingRewardWrapper(gym.Wrapper):
    """Clean reward wrapper for stable multi-track training"""
    
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
        
        original_obs = info.get("original_obs", {})
        new_reward = self._calculate_stable_rewards(original_obs, action, done, truncated)
        
        return obs, new_reward, done, truncated, info
    
    def _clip_actions(self, action):
        """Enhanced action clipping with SAC-specific bounds checking"""
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
        """Calculate balanced rewards for stable training"""

        if done and not truncated: 
            return -10.0

        survival_reward = 0.1
        reward_modifier = survival_reward

        current_speed = 0.0 
        try:
            if "linear_vels_x" in original_obs and "linear_vels_y" in original_obs:
                current_speed = np.sqrt(original_obs["linear_vels_x"][0]**2 + original_obs["linear_vels_y"][0]**2)
            if current_speed > 1.0: 
                speed_reward = np.log(current_speed + 1) * 0.05
                reward_modifier += min(speed_reward, 0.3)
            if len(self.speed_history) > 0:
                speed_change = abs(current_speed - self.speed_history[-1])
                if speed_change < 0.5: 
                    reward_modifier += 0.05
        except (KeyError, IndexError, TypeError):
            pass 

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

        episode_length_bonus = min(self.total_steps * 0.001, 0.5)
        reward_modifier += episode_length_bonus

        self.speed_history.append(current_speed)
        if len(self.speed_history) > 10:
            self.speed_history = self.speed_history[-10:]

        total_reward = np.clip(reward_modifier, -10.0, 1.0) 

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

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"Total GPU Memory: {total_memory:.1f}GB")
    
    if total_memory < 8.0:
        memory_fraction = 0.6
    elif total_memory < 12.0:
        memory_fraction = 0.7
    else:
        memory_fraction = 0.8
    
    print(f"Setting GPU memory fraction to: {memory_fraction}")
    torch.cuda.set_per_process_memory_fraction(memory_fraction)

run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
seed = 42
set_random_seed(seed)

AVAILABLE_TRACKS = [
    "Silverstone",
    "Monza",
    "Spa",
    "Catalunya",
    "Spielberg",
]

base_batch_size = 256
if torch.cuda.is_available():
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if total_memory < 8.0:
        batch_size = 128
    elif total_memory < 12.0:
        batch_size = 192
    else:
        batch_size = base_batch_size
    print(f"Adaptive batch size: {batch_size} (based on {total_memory:.1f}GB GPU memory)")
else:
    batch_size = base_batch_size

multi_track_config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 2000000,
    "env_id": "f1tenth-RL-v0",
    "seed": seed,
    "buffer_size": 200000,
    "learning_starts": 10000,
    "batch_size": batch_size,
    "tau": 0.005,
    "gamma": 0.995,
    "train_freq": (64, "step"),
    "gradient_steps": 64,
    "ent_coef": "auto",
    "target_update_interval": 1,
    "target_entropy": "auto",
    "use_sde": False,
    "sde_sample_freq": -1,
    "use_sde_at_warmup": False,
    "learning_rate": 5e-5,
    "verbose": 1,
}

policy_kwargs = {
    "net_arch": {
        "pi": [256, 256, 128],
        "qf": [256, 256, 128]
    },
    "activation_fn": torch.nn.ReLU,
    "log_std_init": -2.0,
}

def make_multi_track_env(rank: int = 0, seed: int = 0):
    """Create environment with a specific track based on rank"""
    
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
            "sv_max": 3.2, "v_switch": 7.319,             "a_max": 5.0,
            "v_min": -1.0, "v_max": 15.0, "width": 0.31, "length": 0.58,
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
            # ✅ USE DummyVecEnv FOR STABILITY - Avoids SubprocVecEnv freezing issues
            # OLD: env = SubprocVecEnv([make_multi_track_env(i, seed) for i in range(n_envs)], start_method=start_method)
            # NEW: Use DummyVecEnv to avoid multiprocessing bugs that cause freezing
            env = DummyVecEnv([make_multi_track_env(i, seed) for i in range(n_envs)])
        
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
    print("F1Tenth Multi-Track SAC Training System")
    print("=" * 50)
    
    # Create environments - RESTORED MULTI-TRACK TRAINING
    n_envs = len(AVAILABLE_TRACKS)  # Use all 5 tracks for proper multi-track learning
    env = create_multi_track_env(n_envs, seed=seed) # <-- Pass seed here
    
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
    
    # Create model directory
    model_dir = f"models/multi_track_sac_{run_id}"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(f"runs/multi_track_sac_{run_id}", exist_ok=True)
    
    # Initialize SAC model
    try:
        model = SAC(
            "MlpPolicy",  # SB3 1.x compatible
            env,
            buffer_size=multi_track_config["buffer_size"],
            learning_starts=multi_track_config["learning_starts"],
            batch_size=multi_track_config["batch_size"],
            tau=multi_track_config["tau"],
            gamma=multi_track_config["gamma"],
            train_freq=multi_track_config["train_freq"],
            gradient_steps=multi_track_config["gradient_steps"],
            ent_coef=multi_track_config["ent_coef"],
            target_update_interval=multi_track_config["target_update_interval"],
            target_entropy=multi_track_config["target_entropy"],
            use_sde=multi_track_config["use_sde"],
            sde_sample_freq=multi_track_config["sde_sample_freq"],
            use_sde_at_warmup=multi_track_config["use_sde_at_warmup"],
            learning_rate=multi_track_config["learning_rate"],
            verbose=multi_track_config["verbose"],
            tensorboard_log=f"runs/multi_track_sac_{run_id}",
            device=device,
            seed=seed,
            policy_kwargs=policy_kwargs,
        )
    except Exception as e:
        print(f"Failed to initialize SAC model: {e}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()
        raise
    
    # Set up callbacks
    eval_callback = EvalCallback(
        eval_env,  # Single VecEnv containing all tracks
        best_model_save_path=f"{model_dir}/best_model",
        log_path=f"{model_dir}/eval_logs",
        eval_freq=50000,  # ✅ Evaluate less often (was 15000)
        deterministic=True,
        render=False,
        n_eval_episodes=2, # ✅ 2 episodes per track is enough (was 5)
        warn=False # Suppress warnings about multiple eval envs
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=25000,  # ✅ More frequent, smaller checkpoints
        save_path=f"{model_dir}/checkpoints",
        name_prefix="multi_track_sac",
    )
    
    # Add monitoring callbacks
    nan_monitor_callback = NaNMonitorCallback(verbose=1)
    
    callbacks = [eval_callback, checkpoint_callback, nan_monitor_callback]
    
    print(f"Starting multi-track SAC training with {n_envs} parallel environments...")
    print(f"Total timesteps: {multi_track_config['total_timesteps']}")
    print(f"Model will be saved to: {model_dir}")
    
    # Train the model
    try:
        model.learn(
            total_timesteps=multi_track_config["total_timesteps"],
            callback=callbacks,
            progress_bar=True,
            tb_log_name="multi_track_sac",
            log_interval=100  # ✅ Print logs to console every 100 episodes
        )
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()
        raise
    finally:
        # Save the final model and VecNormalize stats
        final_model_path = f"{model_dir}/final_model.zip"
        vec_normalize_path = f"{model_dir}/vec_normalize.pkl"
        try:
            model.save(final_model_path)
            # Save VecNormalize statistics for consistent evaluation
            env.save(vec_normalize_path)
            print(f"Saved model and VecNormalize stats to {model_dir}")
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