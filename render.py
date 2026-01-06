import gymnasium as gym
import PyFlyt.gym_envs
from PyFlyt.gym_envs import FlattenWaypointEnv

import sys

# import numpy as np
# import matplotlib.pyplot as plt

from stable_baselines3 import PPO

env = gym.make(
        "PyFlyt/QuadX-Waypoints-v4",
        num_targets=4,
        sparse_reward=False,
        use_yaw_targets=False,
        goal_reach_distance=0.2,
        goal_reach_angle=0.1,
        flight_mode=0,
        flight_dome_size=5.0,
        max_duration_seconds=10.0,
        angle_representation="quaternion",
        agent_hz=30,
        render_mode="human", # "rgb_array" -> video
    render_resolution=(480, 480),
)
    
env = FlattenWaypointEnv(env, context_length=2)
    
model = PPO.load(sys.argv[1], env=env)
    
obs, _ = env.reset()
term, trunc = False, False
    
while not (term or trunc):
    action, _ = model.predict(obs)
    obs, reward, term, trunc, info = env.step(action)
    
env.close()