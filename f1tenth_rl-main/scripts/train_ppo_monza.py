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
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

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
TARGET_TRACK = "Monza"


class SafeRacingRewardWrapper(gym.Wrapper):
    """Wrapper that modifies rewards for safe racing with good braking"""
    
    def __init__(self, env):
        super().__init__(env)
        self.prev_speed = 0.0
        self.speed_history = []
        self.prev_steering = 0.0
        
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        
        # Get original observation for safe racing rewards
        original_obs = info.get("original_obs", {})
        
        # Calculate safe racing reward modifications
        safe_reward_modifier = self._calculate_safe_rewards(original_obs, action)
        reward += safe_reward_modifier
        
        return obs, reward, done, truncated, info
    
    def _calculate_safe_rewards(self, original_obs, action):
        """Calculate safe racing reward modifications focused on good braking"""
        reward_modifier = 0.0
        
        # Get current state
        current_speed = np.sqrt(original_obs["linear_vels_x"][0]**2 + original_obs["linear_vels_y"][0]**2)
        
        # Simplified reward system for more stable training
        reward_modifier = 0.0
        
        # 1. Basic survival reward (most important)
        survival_reward = 1.0
        reward_modifier += survival_reward
        
        # 2. Simple speed reward (encourage forward progress)
        if current_speed > 0.5:  # Only if moving
            speed_reward = min(current_speed * 0.1, 2.0)  # Cap at 2.0
            reward_modifier += speed_reward
        
        # 3. Simple braking reward (encourage braking when needed)
        if len(action) > 0:
            try:
                speed_action = action[1] if hasattr(action, '__len__') and len(action) > 1 else 0
                if hasattr(speed_action, '__len__') and len(speed_action) > 0:
                    speed_action_scalar = float(speed_action[0])
                else:
                    speed_action_scalar = float(speed_action)
                
                if speed_action_scalar < 0 and current_speed > 2.0:  # Braking when fast
                    braking_reward = abs(speed_action_scalar) * 0.2
                    reward_modifier += braking_reward
            except (ValueError, TypeError):
                pass
        
        # 4. Simple steering penalty (discourage excessive steering)
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
    

# PPO configuration
ppo_config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 400000,  # Reduced for faster iteration and testing
    "env_id": "f1tenth-RL-v0",
    "seed": seed,
    
    # --- PPO-Specific Hyperparameters ---
    "n_steps": 4096,           # (Replaces buffer_size) Steps per env per update
    "batch_size": 128,          # Mini-batch size for gradient descent
    "n_epochs": 10,             # (Replaces gradient_steps) Epochs to train on collected data
    "gae_lambda": 0.95,         # GAE parameter
    "clip_range": 0.2,          # PPO clipping parameter
    "vf_coef": 0.5,             # Value function loss coefficient
    "ent_coef": 0.001,          # Entropy coefficient (often lower for PPO)
    "gamma": 0.99,
    "use_sde": False,
    "sde_sample_freq": -1,
    "learning_rate": 1e-4,      # 1e-4 or 3e-4 is a good starting point
    "verbose": 1,
}

policy_kwargs = {
    "net_arch": {
        "pi": [512, 512, 256, 128],
        "vf": [512, 512, 256, 128]  # PPO uses Value Function (vf) instead of Q-Function (qf)
    },
    "activation_fn": torch.nn.ReLU,
}

def make_monza_env(rank: int = 0, seed: int = 0):
    """Create environment fixed to Monza"""

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
            "a_max": 4.0,  # Very conservative acceleration for stable training
            "v_min": -4.0,
            "v_max": 10.0,  # Very conservative max speed for stable training
            "width": 0.31,
            "length": 0.58,
        },
        # Start at the start line on the racing line, facing forward
        "reset_config": {"type": "rl_grid_static"},
        "seed": seed + rank,
    }

    def _init():
        config = base_config.copy()
        config["map"] = TARGET_TRACK

        # Use F1TenthWrapper for better observation processing
        env = F1TenthWrapper(config=config, render_mode="rgb_array")
        env = Monitor(env, filename=None)
        # Add safe racing reward wrapper focused on good braking
        env = SafeRacingRewardWrapper(env)
        return env

    return _init

def create_env(n_envs: int = 4):
    try:
        if n_envs == 1:
            env = DummyVecEnv([make_monza_env(0, seed)])
        else:
            env = SubprocVecEnv([make_monza_env(i, seed) for i in range(n_envs)])
        return env
    except Exception as e:
        print(f"Failed to create parallel environments: {e}")
        return DummyVecEnv([make_monza_env(0, seed)])

def main():
    n_envs = 4
    env = create_env(n_envs)
    eval_env = DummyVecEnv([make_monza_env(0, seed + 1000)])

    model_dir = f"models/ppo_monza_{run_id}"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(f"runs/ppo_monza_{run_id}", exist_ok=True)

    try:
        model = PPO(
            ppo_config["policy_type"],
            env,
            n_steps=ppo_config["n_steps"],
            batch_size=ppo_config["batch_size"],
            n_epochs=ppo_config["n_epochs"],
            gae_lambda=ppo_config["gae_lambda"],
            clip_range=ppo_config["clip_range"],
            vf_coef=ppo_config["vf_coef"],
            ent_coef=ppo_config["ent_coef"],
            gamma=ppo_config["gamma"],
            use_sde=ppo_config["use_sde"],
            sde_sample_freq=ppo_config["sde_sample_freq"],
            learning_rate=ppo_config["learning_rate"],
            verbose=ppo_config["verbose"],
            tensorboard_log=f"runs/ppo_monza_{run_id}",
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
        name_prefix="ppo_monza",
    )

    callbacks = [eval_callback, checkpoint_callback]

    try:
        model.learn(
            total_timesteps=ppo_config["total_timesteps"],
            callback=callbacks,
            progress_bar=True,
            tb_log_name="ppo_monza",
        )
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise
    finally:
        final_model_path = f"{model_dir}/final_model.zip"
        try:
            model.save(final_model_path)
        except Exception as e:
            print(f"Failed to save final model: {e}")

        print(f"Final model: {final_model_path}")
        print(f"Best model: {model_dir}/best_model.zip")
        print(f"TensorBoard logs: runs/ppo_monza_{run_id}")
        print(f"Trained on track: {TARGET_TRACK}")

    try:
        env.close()
        eval_env.close()
    except Exception as e:
        print(f"Error closing environments: {e}")

if __name__ == "__main__":
    main()

