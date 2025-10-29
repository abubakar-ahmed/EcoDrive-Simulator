from typing import Tuple, List
import fire
import time
import os
import sys
import io
import torch
import numpy as np
import gymnasium as gym
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3 import PPO

# --- Force use of LOCAL f1tenth_gym instead of installed one ---
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
local_f1tenth_path = os.path.join(root_dir, "f1tenth_gym")
local_src_path = os.path.join(root_dir, "src")

# Remove any already-imported f1tenth_gym
if "f1tenth_gym" in sys.modules:
    del sys.modules["f1tenth_gym"]

# Prepend local paths to sys.path
sys.path.insert(0, local_f1tenth_path)
sys.path.insert(0, local_src_path)
sys.path.insert(0, root_dir)

import f1tenth_gym
print(f"✅ Using f1tenth_gym from: {f1tenth_gym.__file__}")

from f1tenth_wrapper.env import F1TenthWrapper


class RLWrapper:
    def __init__(
        self, model_path: str, original_env: F1TenthWrapper = None, horizon: int = 10
    ):
        # Create a temporary environment with correct spaces
        temp_env = gym.make(
            "f1tenth-RL-v0",
            config={
                "map": "Spa",
                "num_agents": 1,
                "timestep": 0.01,
                "integrator": "rk4",
                "control_input": ["speed", "steering_angle"],
                "model": "st",
                "observation_config": {"type": "original"},
                "params": {"mu": 1.0},
                "reset_config": {"type": "rl_random_static"},
            },
        )
        
        # Debug: Print action space info
        print(f"Action space: {temp_env.action_space}")
        print(f"Action space type: {type(temp_env.action_space)}")
        print(f"Action space shape: {temp_env.action_space.shape}")
        
        # Try to load the model with current stable-baselines3 version
        try:
            print("Attempting to load model with current stable-baselines3...")
            self.model = PPO.load(model_path)
            print("✅ Successfully loaded trained model!")
            
        except Exception as e:
            print(f"Failed to load model: {e}")
            print("Attempting to create new model and load policy weights...")
            
            # Create a new model with correct architecture first
            self.model = PPO("MlpPolicy", temp_env, verbose=1)
            
            # Try to load just the policy weights
            try:
                import zipfile
                import torch
                
                with zipfile.ZipFile(model_path, 'r') as zip_file:
                    # Extract policy weights
                    policy_data = zip_file.read('policy.pth')
                    policy_state = torch.load(io.BytesIO(policy_data), map_location='cpu')
                    
                    # Load policy weights into new model
                    self.model.policy.load_state_dict(policy_state)
                    print("✅ Successfully loaded policy weights from saved model")
                    
            except Exception as e2:
                print(f"Failed to load policy weights: {e2}")
                print("⚠️ Using untrained model - the car will drive randomly")
                self.model = None
        
        temp_env.close()

        if original_env is not None:
            self.env = original_env.unwrapped.clone(render_mode="rgb_array")
            _, _ = self.env.reset()
            self._predictive_state = np.zeros((horizon, 2))
            self._predictive_action = np.zeros((horizon, 2))
            self._horizon = horizon


    def get_action(self, obs: np.ndarray) -> np.ndarray:
        if self.model is not None:
            # Use the full model if available
            action = self.model.predict(obs, deterministic=True)[0]
            return action
        else:
            # Fallback: return random actions if no model is loaded
            print("Warning: No model loaded, using random actions")
            return np.random.uniform(
                low=[-0.4189, -5.0], 
                high=[0.4189, 20.0], 
                size=(2,)
            ).astype(np.float32)



def main(
    model_path: str = os.path.join(
        "models", "multi_track_ppo_2025-10-12_02-58-46", "final_model.zip"
    ),
    video_recording: bool = False,
):
    seed = 1
    set_random_seed(seed)
#monza
#silverstone
#spa
#catalunya
#spielberg
    # create the environment
    env = gym.make(
        "f1tenth-RL-v0",
        config={
            "map": "Catalunya",
            "num_agents": 1,
            "timestep": 0.02,
            "integrator": "rk4",
            "control_input": ["speed", "steering_angle"],
            "model": "st",
            "observation_config": {"type": "original"},
            "params": {"mu": 1.0},
            "reset_config": {"type": "rl_grid_static"},
            "seed": seed,
        },
        render_mode="rgb_array" if video_recording else "human",
    )

    if video_recording:
        # check folder exists
        if not os.path.exists("videos"):
            os.makedirs("videos")
        video_path = os.path.join("videos", f"RL_{time.time()}")
        env = gym.wrappers.RecordVideo(env, video_path)

    # RL model
    rl_wrapper = RLWrapper(model_path, original_env=env, horizon=30)
    track = env.unwrapped.track
    env.unwrapped.add_render_callback(track.raceline.render_waypoints)

    obs, info = env.reset()
    done = False
    env.render()

    laptime = 0.0
    time_step = env.unwrapped.core.config["timestep"]
    end_time = 15.0 if video_recording else float("inf")

    while laptime < end_time and not done:
        action = rl_wrapper.get_action(obs)
        obs, step_reward, done, truncated, info = env.step(action)
        laptime += time_step
        env.render()

    env.close()


if __name__ == "__main__":
    fire.Fire(main)
