import gymnasium as gym
import datetime
import torch
import sys
import os
import numpy as np
import warnings
from typing import Optional

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Force use of local f1tenth_gym by removing any installed version from sys.modules
if 'f1tenth_gym' in sys.modules:
    del sys.modules['f1tenth_gym']

# Add the f1tenth_gym folder to the Python path (insert at beginning to prioritize local)
local_f1tenth_path = os.path.join(os.path.dirname(__file__), '..', 'f1tenth_gym')
local_src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
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

import f1tenth_gym
from f1tenth_wrapper.env import F1TenthWrapper

# Check for GPU availability and optimize settings
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of GPU memory
    torch.cuda.empty_cache()  # Clear GPU cache
else:
    print("GPU not available, using CPU")

run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
seed = 42
set_random_seed(seed)

# Set PyTorch for optimal GPU performance
torch.backends.cudnn.deterministic = False  # Allow non-deterministic for better GPU performance
torch.backends.cudnn.benchmark = True  # Enable cuDNN benchmark for better performance

# Single target track
TARGET_TRACK = "Silverstone"


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
        """Calculate robust reward modifications"""
        reward_modifier = 0.0
        
        # 1. Speed Reward: Only reward *forward* speed
        if current_speed > 0.5:
            # Reward is proportional to forward speed
            speed_reward = current_speed * 0.1
            reward_modifier += speed_reward
        
        # 2. Moderate Penalty for Driving Backward (less harsh)
        if current_speed < -0.1: 
            reward_modifier -= 2.0 # Moderate penalty instead of -10.0
            
        # 3. Remove "stopped" penalty - let episodes naturally end via TimeLimit
        # The TimeLimit wrapper will handle stuck episodes

        # 4. Steering Jerk Penalty (for stability)
        steering_change = abs(steering_action - self.prev_steering)
        jerk_penalty = steering_change * 0.05  # Reduced from 0.2 to be less harsh
        reward_modifier -= jerk_penalty
        
        # 5. Braking Reward
        is_decelerating = current_speed < self.prev_speed
        if is_decelerating and self.prev_speed > 3.0:
            braking_reward = (self.prev_speed - current_speed) * 0.1
            reward_modifier += braking_reward

        return reward_modifier
    
    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.prev_speed = 0.0
        self.prev_steering = 0.0
        return obs, info
    

# PPO configuration
ppo_config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 400000,  # Same as SAC for comparison
    "env_id": "f1tenth-RL-v0",
    "seed": seed,
    "n_steps": 2048,  # Reduced for more frequent updates
    "batch_size": 256,  # More reasonable batch size
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "clip_range_vf": None,
    "ent_coef": 0.01,  # Increased for better exploration
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "learning_rate": 3e-4,
    "verbose": 1,
}

policy_kwargs = {
    "net_arch": {
        "pi": [512, 512, 256],  # Policy network - larger for better speed optimization
        "vf": [512, 512, 256]   # Value function network - larger for better speed optimization
    },
    "activation_fn": torch.nn.ReLU,
}

def make_silverstone_env(rank: int = 0, seed: int = 0):
    """Create environment fixed to Silverstone"""

    base_config = {
        "num_agents": 1,
        "timestep": 0.01,
        "integrator": "rk4",
        "control_input": ["speed", "steering_angle"],
        "model": "st",
        "observation_config": {"type": "original"},
        "params": {
            "mu": 1.5,  # Increased friction for better traction and braking
            "C_Sf": 5.5,  # Increased front cornering stiffness for better steering
            "C_Sr": 6.0,  # Increased rear cornering stiffness for better stability
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
            "a_max": 4.0,  # Conservative acceleration (copied from SAC)
            "v_min": -4.0,
            "v_max": 10.0,  # Conservative max speed (copied from SAC)
            "width": 0.31,
            "length": 0.58,
        },
        "reset_config": {"type": "rl_grid_static"},
        "seed": seed + rank,
    }

    def _init():
        config = base_config.copy()
        config["map"] = TARGET_TRACK
        
        # Note: The car orientation is determined by the track's racing line
        # forward_velocity > 0 means the car is moving forward around the track

        # Use F1TenthWrapper for better observation processing
        env = F1TenthWrapper(config=config, render_mode="rgb_array")
        env = Monitor(env, filename=None)
        
        # Force the episode to end after 5000 steps (50 seconds)
        # This prevents the agent from getting stuck sitting still forever
        env = TimeLimit(env, max_episode_steps=5000)
        
        # Add safe racing reward wrapper
        env = SafeRacingRewardWrapper(env)
        return env

    return _init

def create_env(n_envs: int = 4):
    try:
        if n_envs == 1:
            env = DummyVecEnv([make_silverstone_env(0, seed)])
        else:
            env = SubprocVecEnv([make_silverstone_env(i, seed) for i in range(n_envs)])
        
        # Add VecNormalize for better training stability
        env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
        
        return env
    except Exception as e:
        print(f"Failed to create parallel environments: {e}")
        env = DummyVecEnv([make_silverstone_env(0, seed)])
        env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
        return env

def main():
    n_envs = 4
    env = create_env(n_envs)
    eval_env = DummyVecEnv([make_silverstone_env(0, seed + 1000)])
    # Wrap eval env with VecNormalize and sync stats from training env
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)
    eval_env.obs_rms = env.obs_rms
    eval_env.ret_rms = env.ret_rms
    eval_env.norm_obs = env.norm_obs
    eval_env.clip_obs = env.clip_obs
    eval_env.epsilon = env.epsilon

    model_dir = f"models/ppo_silverstone_{run_id}"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(f"runs/ppo_silverstone_{run_id}", exist_ok=True)

    try:
        model = PPO(
            ppo_config["policy_type"],
            env,
            n_steps=ppo_config["n_steps"],
            batch_size=ppo_config["batch_size"],
            n_epochs=ppo_config["n_epochs"],
            gamma=ppo_config["gamma"],
            gae_lambda=ppo_config["gae_lambda"],
            clip_range=ppo_config["clip_range"],
            clip_range_vf=ppo_config["clip_range_vf"],
            ent_coef=ppo_config["ent_coef"],
            vf_coef=ppo_config["vf_coef"],
            max_grad_norm=ppo_config["max_grad_norm"],
        learning_rate=ppo_config["learning_rate"],
        verbose=ppo_config["verbose"],
        tensorboard_log=f"runs/ppo_silverstone_{run_id}",
        device=device,
        seed=seed,
        policy_kwargs=policy_kwargs,
    )
    except Exception as e:
        print(f"Failed to initialize PPO model: {e}")
        raise

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{model_dir}/best_model",
        log_path=f"{model_dir}/eval_logs",
        eval_freq=10000,
        deterministic=True,
        render=False,
        n_eval_episodes=5,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=40000,
        save_path=f"{model_dir}/checkpoints",
        name_prefix="ppo_silverstone",
    )

    callbacks = [eval_callback, checkpoint_callback]

    try:
        model.learn(
            total_timesteps=ppo_config["total_timesteps"],
            callback=callbacks,
            progress_bar=True,
            tb_log_name="ppo_silverstone",
        )
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise
    finally:
        final_model_path = f"{model_dir}/final_model.zip"
        vec_normalize_path = f"{model_dir}/vec_normalize.pkl"
        try:
            model.save(final_model_path)
            # Save VecNormalize statistics for consistent evaluation
            env.save(vec_normalize_path)
            print(f"Saved model and VecNormalize stats to {model_dir}")
        except Exception as e:
            print(f"Failed to save final model or stats: {e}")

        print(f"Final model: {final_model_path}")
        print(f"Best model: {model_dir}/best_model.zip")
        print(f"VecNormalize stats: {vec_normalize_path}")
        print(f"TensorBoard logs: runs/ppo_silverstone_{run_id}")
        print(f"Trained on track: {TARGET_TRACK}")

    try:
        env.close()
        eval_env.close()
    except Exception as e:
        print(f"Error closing environments: {e}")

if __name__ == "__main__":
    main()

