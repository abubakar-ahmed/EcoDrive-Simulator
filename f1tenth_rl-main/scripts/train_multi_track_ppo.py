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

from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TimeLimit

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
        """Override to handle invalid observations gracefully with optimized processing"""
        # OPTIMIZED: Sanitize observations BEFORE calling super() to avoid double processing
        sanitized_obs = self._sanitize_observation(original_obs)
        
        try:
            # Try the original observation method with sanitized data
            obs = super()._observation(sanitized_obs)
            
            # CRITICAL: Additional safety check for None or invalid observations
            if obs is None or not np.isfinite(obs).all():
                print(f"Warning: Invalid observation after processing, using safe default")
                obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            
            return obs
        except ValueError as e:
            if "Invalid observation" in str(e):
                print(f"Warning: Invalid observation detected even after sanitization: {e}")
                # Return minimal safe observation
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
                # Replace inf, -inf, and NaN values
                safe_value = np.nan_to_num(value, 
                                        posinf=1000.0, 
                                        neginf=-1000.0,
                                        nan=0.0)
                # Clip to safe bounds
                safe_value = np.clip(safe_value, -1000.0, 1000.0)
                safe_obs[key] = safe_value.astype(np.float32)
            else:
                safe_obs[key] = value
        return safe_obs

class SafeRacingRewardWrapper(gym.Wrapper):
    """Wrapper that modifies rewards for safe racing with good braking"""
    
    def __init__(self, env):
        super().__init__(env)
        self.prev_speed = 0.0
        self.prev_steering = 0.0
        
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        
        # Get original observation for safe racing rewards
        original_obs = info.get("original_obs", {})
        
        # Get current state from observation
        # Use linear_vels_x for *forward* speed, not magnitude
        current_speed = original_obs.get("linear_vels_x", [0.0])[0] 
        
        # Extract scalar action for logic
        # action[0] is speed, action[1] is steering
        try:
            steering_action_scalar = float(action[1])
        except (ValueError, TypeError, IndexError):
            steering_action_scalar = 0.0 # Default to no steering if action is invalid
        
        # Calculate rewards based on state and action
        safe_reward_modifier = self._calculate_safe_rewards(current_speed, steering_action_scalar)
        reward += safe_reward_modifier
        
        # Update history for the *next* step's calculation
        self.prev_speed = current_speed
        self.prev_steering = steering_action_scalar
        
        return obs, reward, done, truncated, info
    
    def _calculate_safe_rewards(self, current_speed, steering_action):
        """Calculate reward modifications for robust multi-track training"""
        reward_modifier = 0.0
        
        # 1. Speed Reward: Encourage forward progress across all tracks
        if current_speed > 0.5:
            speed_reward = current_speed * 0.15  # Stronger reward for multi-track generalization
            reward_modifier += speed_reward
        
        # 2. Penalty for Driving Backward
        if current_speed < -0.1:
            reward_modifier -= 2.0  # Moderate penalty (not too harsh for stability)
        
        # 3. Steering Jerk Penalty (for stability)
        steering_change = abs(steering_action - self.prev_steering)
        jerk_penalty = steering_change * 0.05  # Gentle penalty
        reward_modifier -= jerk_penalty
        
        # 4. Braking Reward (good for tight corners across different tracks)
        is_decelerating = current_speed < self.prev_speed
        if is_decelerating and self.prev_speed > 3.0:
            braking_reward = (self.prev_speed - current_speed) * 0.15
            reward_modifier += braking_reward
            
        return reward_modifier
    
    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.prev_speed = 0.0
        self.prev_steering = 0.0
        return obs, info

class NaNMonitorCallback(BaseCallback):
    """Monitor for NaN values and stop training if detected"""
    
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.check_freq = 1000
        
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            try:
                for name, param in self.model.policy.named_parameters():
                    if torch.isnan(param).any():
                        print(f"WARNING: NaN detected in policy.{name}")
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

base_batch_size = 256  # Increased for multi-track stability
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
    batch_size = 128  # Reduced for CPU

# PPO-specific configuration (cleaned up, no SAC parameters)
multi_track_config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 2000000,
    "env_id": "f1tenth-RL-v0",
    "seed": seed,
    "n_steps": 4096,           # More steps per update for better multi-track diversity
    "batch_size": batch_size,  # PPO: Minibatch size per update
    "n_epochs": 10,             # PPO: Epochs per rollout
    "gamma": 0.995,             # Higher discount for long-term planning across tracks
    "gae_lambda": 0.95,         # PPO: GAE lambda parameter
    "clip_range": 0.2,          # PPO: Policy clipping parameter (keeps updates stable)
    "clip_range_vf": None,      # PPO: Value function clipping (disabled)
    "ent_coef": 0.01,           # Higher entropy for better exploration across diverse tracks
    "vf_coef": 0.5,             # PPO: Value function coefficient
    "max_grad_norm": 0.5,       # PPO: Gradient clipping
    "learning_rate": 3e-4,      # Standard learning rate
    "verbose": 1,
}

policy_kwargs = {
    # Larger network for better generalization across multiple tracks
    "net_arch": {
        "pi": [512, 512, 256],  # Policy network - larger for multi-track generalization
        "vf": [512, 512, 256]   # Value function network
    },
    "activation_fn": torch.nn.ReLU,
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
        # 2. Add TimeLimit to prevent getting stuck
        env = TimeLimit(env, max_episode_steps=5000)
        # 3. Add Monitor LAST (so it logs the modified reward)
        env = Monitor(env, filename=None)
        # -------------------------
        
        return env

    return _init

def create_multi_track_env(n_envs: int = 4, seed: int = 0): # Add seed parameter
    """Create vectorized environment with multi-track support and VecNormalize"""
    # Set the start method to 'spawn' for CUDA safety
    start_method = "spawn"
    
    try:
        if n_envs == 1:
            # Pass seed to make_multi_track_env
            env = DummyVecEnv([make_multi_track_env(0, seed)])
        else:
            # Use SubprocVecEnv for true parallelization (3-5x faster!)
            env_fns = [make_multi_track_env(i, seed) for i in range(n_envs)]
            env = SubprocVecEnv(env_fns, start_method=start_method)
        
        # CRITICAL: Add VecNormalize for stable training
        env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0, clip_reward=10.0)
        return env
    except Exception as e:
        print(f"Failed to create parallel environments: {e}")
        print("Falling back to single environment...")
        # Pass seed here too for the fallback
        env = DummyVecEnv([make_multi_track_env(0, seed)])
        env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0, clip_reward=10.0)
        return env

def main():
    print("F1Tenth Multi-Track PPO Training System")
    print("=" * 50)
    
    # Create environments - RESTORED MULTI-TRACK TRAINING
    # Increased parallelism for better data diversity and faster training
    n_envs = 10  # Run 10 parallel environments for faster training (tracks will cycle)
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
    model_dir = f"models/multi_track_ppo_{run_id}"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(f"runs/multi_track_ppo_{run_id}", exist_ok=True)
    
    # Initialize PPO model
    try:
        model = PPO(
            "MlpPolicy",  # SB3 1.x compatible: string only, no policy_class
            env,
            n_steps=multi_track_config["n_steps"],
            batch_size=multi_track_config["batch_size"],
            n_epochs=multi_track_config["n_epochs"],
            gamma=multi_track_config["gamma"],
            gae_lambda=multi_track_config["gae_lambda"],
            clip_range=multi_track_config["clip_range"],
            clip_range_vf=multi_track_config["clip_range_vf"],
            ent_coef=multi_track_config["ent_coef"],
            vf_coef=multi_track_config["vf_coef"],
            max_grad_norm=multi_track_config["max_grad_norm"],
            learning_rate=multi_track_config["learning_rate"],
            verbose=multi_track_config["verbose"],
            tensorboard_log=f"runs/multi_track_ppo_{run_id}",
            device=device,
            seed=seed,
            policy_kwargs=policy_kwargs,
        )
    except Exception as e:
        print(f"Failed to initialize PPO model: {e}")
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
        name_prefix="multi_track_ppo",
    )
    
    # Add monitoring callbacks
    nan_monitor_callback = NaNMonitorCallback(verbose=1)
    
    callbacks = [eval_callback, checkpoint_callback, nan_monitor_callback]
    
    print(f"Starting multi-track PPO training with {n_envs} parallel environments...")
    print(f"Total timesteps: {multi_track_config['total_timesteps']}")
    print(f"Model will be saved to: {model_dir}")
    
    # Train the model
    try:
        model.learn(
            total_timesteps=multi_track_config["total_timesteps"],
            callback=callbacks,
            progress_bar=True,
            tb_log_name="multi_track_ppo",
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
