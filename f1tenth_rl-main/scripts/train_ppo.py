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
from stable_baselines3.common.env_util import make_vec_env

import f1tenth_gym

maps_dir = os.path.join(os.path.dirname(f1tenth_gym.__file__), "maps", "Catalunya")
print(f"Maps directory exists: {os.path.exists(maps_dir)}")

if not os.path.exists(maps_dir):
    raise FileNotFoundError(f"Catalunya map directory not found: {maps_dir}")

map_yaml = os.path.join(maps_dir, "Catalunya_map.yaml")
if not os.path.exists(map_yaml):
    raise FileNotFoundError(f"Catalunya_map.yaml not found in {maps_dir}")

from f1tenth_wrapper.env import F1TenthWrapper

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    # Set memory fraction to avoid OOM
    torch.cuda.set_per_process_memory_fraction(0.8)

run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

seed = 42
set_random_seed(seed)

# Set PyTorch to use deterministic algorithms for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class TrainingProgressCallback(BaseCallback):
    """Custom callback to track training progress and handle early stopping"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.best_mean_reward = -np.inf
        self.patience = 100000  # Steps to wait without improvement
        self.patience_counter = 0
        
    def _on_step(self) -> bool:
        # Log episode info if available
        if 'episode' in self.locals:
            episode_info = self.locals['episode']
            if 'r' in episode_info:
                self.episode_rewards.append(episode_info['r'])
            if 'l' in episode_info:
                self.episode_lengths.append(episode_info['l'])
                
        # Check for improvement every 1000 steps
        if self.num_timesteps % 1000 == 0:
            if len(self.episode_rewards) > 10:
                mean_reward = np.mean(self.episode_rewards[-10:])
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    self.patience_counter = 0
                    if self.verbose > 0:
                        print(f"New best mean reward: {mean_reward:.2f}")
                else:
                    self.patience_counter += 1000
                    
        return True

# PPO-specific configuration optimized for F1Tenth
ppo_config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 2000000,  # PPO typically needs fewer timesteps than SAC
    "env_id": "f1tenth-RL-v0",
    "seed": seed,
    "n_steps": 2048,  # Number of steps to collect per rollout
    "batch_size": 64,  # Batch size for training
    "n_epochs": 10,  # Number of epochs to train on each batch
    "gamma": 0.99,  # Discount factor
    "gae_lambda": 0.95,  # GAE lambda parameter
    "clip_range": 0.2,  # PPO clipping parameter
    "ent_coef": 0.01,  # Entropy coefficient
    "vf_coef": 0.5,  # Value function coefficient
    "max_grad_norm": 0.5,  # Maximum gradient norm for clipping
    "learning_rate": 3e-4,  # Learning rate
    "verbose": 1,
    # Additional PPO parameters for stability
    "clip_range_vf": None,  # Clipping for value function
    "normalize_advantage": True,  # Normalize advantages
    "target_kl": None,  # Target KL divergence
    "tensorboard_log": None,  # Will be set later
}

# Policy network architecture
policy_kwargs = {
    "net_arch": {
        "pi": [256, 256, 128],  # Policy network
        "vf": [256, 256, 128]   # Value network
    },
    "activation_fn": torch.nn.ReLU,
    "ortho_init": True,  # Orthogonal initialization
    "log_std_init": 0.0,  # Initial log standard deviation
}

def make_env(rank: int = 0, seed: int = 0):
    """Create and return the F1Tenth environment with proper seeding"""
    def _init():
        env = gym.make(
            "f1tenth-RL-v0",
            config={
                "map": maps_dir,
                "num_agents": 1,
                "timestep": 0.01,
                "integrator": "rk4",
                "control_input": ["speed", "steering_angle"],
                "model": "st",
                "observation_config": {"type": "original"},
                "params": {"mu": 1.0},
                "reset_config": {"type": "rl_random_static"},
                "seed": seed + rank,  # Different seed for each environment
            },
            render_mode="rgb_array",
        )
        # Wrap with Monitor for logging
        env = Monitor(env, filename=None)
        return env
    return _init

# Create vectorized environment for faster training
# Use DummyVecEnv for single environment or SubprocVecEnv for multiple parallel environments
n_envs = 4  # Number of parallel environments
print(f"Creating {n_envs} parallel environments...")

try:
    if n_envs == 1:
        env = DummyVecEnv([make_env(0, seed)])
    else:
        env = SubprocVecEnv([make_env(i, seed) for i in range(n_envs)])
    print("✅ Vectorized environment created successfully")
except Exception as e:
    print(f"⚠️ Failed to create parallel environments: {e}")
    print("Falling back to single environment...")
    env = DummyVecEnv([make_env(0, seed)])
    n_envs = 1

# Create evaluation environment
eval_env = DummyVecEnv([make_env(0, seed + 1000)])  # Different seed for eval

# Create model directory
model_dir = f"models/ppo_{run_id}"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(f"runs/ppo_{run_id}", exist_ok=True)

print(f"Model directory: {model_dir}")
print(f"TensorBoard logs: runs/ppo_{run_id}")

# Initialize PPO model with custom hyperparameters
print("Initializing PPO model...")
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
        ent_coef=ppo_config["ent_coef"],
        vf_coef=ppo_config["vf_coef"],
        max_grad_norm=ppo_config["max_grad_norm"],
        learning_rate=ppo_config["learning_rate"],
        verbose=ppo_config["verbose"],
        tensorboard_log=f"runs/ppo_{run_id}",
        device=device,
        seed=seed,
        policy_kwargs=policy_kwargs,
        clip_range_vf=ppo_config["clip_range_vf"],
        normalize_advantage=ppo_config["normalize_advantage"],
        target_kl=ppo_config["target_kl"],
    )
    print("✅ PPO model initialized successfully")
except Exception as e:
    print(f"❌ Failed to initialize PPO model: {e}")
    raise

# Set up callbacks
print("Setting up callbacks...")
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=f"{model_dir}/best_model",
    log_path=f"{model_dir}/eval_logs",
    eval_freq=10000,  # Evaluate every 10k steps
    deterministic=True,
    render=False,
    n_eval_episodes=5,  # Number of episodes for evaluation
)

checkpoint_callback = CheckpointCallback(
    save_freq=50000,  # Save checkpoint every 50k steps
    save_path=f"{model_dir}/checkpoints",
    name_prefix="ppo_model",
)

progress_callback = TrainingProgressCallback(verbose=1)

callbacks = [eval_callback, checkpoint_callback, progress_callback]

print(f"Starting PPO training with {n_envs} parallel environments...")
print(f"Total timesteps: {ppo_config['total_timesteps']}")
print(f"Model will be saved to: {model_dir}")
print(f"Evaluation frequency: every 10k steps")
print(f"Checkpoint frequency: every 50k steps")
print("-" * 50)

# Train the model with error handling
try:
    model.learn(
        total_timesteps=ppo_config["total_timesteps"],
        callback=callbacks,
        progress_bar=True,
        tb_log_name="ppo_f1tenth",
    )
    print("✅ Training completed successfully!")
    
except KeyboardInterrupt:
    print("⚠️ Training interrupted by user")
except Exception as e:
    print(f"❌ Training failed with error: {e}")
    raise
finally:
    # Save the final model
    final_model_path = f"{model_dir}/final_model.zip"
    try:
        model.save(final_model_path)
        print(f"✅ Final model saved to: {final_model_path}")
    except Exception as e:
        print(f"⚠️ Failed to save final model: {e}")
    
    # Print training summary
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    print(f"Final model: {final_model_path}")
    print(f"Best model: {model_dir}/best_model.zip")
    print(f"TensorBoard logs: runs/ppo_{run_id}")
    print(f"Evaluation logs: {model_dir}/eval_logs")
    print(f"Checkpoints: {model_dir}/checkpoints/")
    
    if hasattr(progress_callback, 'best_mean_reward'):
        print(f"Best mean reward: {progress_callback.best_mean_reward:.2f}")
    
    print("\nTo view training progress:")
    print(f"tensorboard --logdir runs/ppo_{run_id}")
    print("="*50)

# Close environments
try:
    env.close()
    eval_env.close()
    print("✅ Environments closed successfully")
except Exception as e:
    print(f"⚠️ Error closing environments: {e}")
