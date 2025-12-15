import gymnasium
import PyFlyt.gym_envs
from PyFlyt.gym_envs import FlattenWaypointEnv

env = gymnasium.make("PyFlyt/QuadX-Waypoints-v4", render_mode="human")
env = FlattenWaypointEnv(env, context_length=2)

term, trunc = False, False
obs, _ = env.reset()

while not (term or trunc):
    obs, rew, term, trunc, _ = env.step(env.action_space.sample())
    print(f"Reward: {rew}")
    