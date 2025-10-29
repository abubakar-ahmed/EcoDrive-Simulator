import gymnasium as gym
import datetime
import torch
import sys
import os

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

from stable_baselines3 import SAC
from stable_baselines3.common.utils import set_random_seed

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

run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

seed = 42
set_random_seed(seed)

wandb_config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 3000000,
    "env_id": "f1tenth-RL-v0",
    "seed": seed,
}

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
        "seed": seed,
    },
    render_mode="rgb_array",
)


model = SAC(
    wandb_config["policy_type"], env, verbose=1, tensorboard_log=f"runs/{run_id}", device=device
)

model.learn(
    total_timesteps=wandb_config["total_timesteps"],
    progress_bar=True,
)

model_path = f"models/{run_id}/model.zip"
model.save(model_path)
