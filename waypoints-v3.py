from xml.parsers.expat import model
import gymnasium as gym
import PyFlyt.gym_envs
import imageio
from PyFlyt.gym_envs import FlattenWaypointEnv

import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback


# Callback para recolectar metricas por episodio
class MetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.targets_reached = []

        self._current_reward = 0.0
        self._current_length = 0

    def _on_step(self) -> bool:
        self._current_reward += self.locals["rewards"][0]
        self._current_length += 1

        done = self.locals["dones"][0]
        info = self.locals["infos"][0]

        if done:
            self.episode_rewards.append(self._current_reward)
            self.episode_lengths.append(self._current_length)
            self.targets_reached.append(info.get("num_targets_reached", 0))

            self._current_reward = 0.0
            self._current_length = 0

        return True

# entrenamiento
def entrenamiento():
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
        render_mode=None, # "rgb_array" -> video
        render_resolution=(480, 480),
    )

    # Wapper 
    env = FlattenWaypointEnv(env, context_length=2)

    callback = MetricsCallback()

    # PPO
    model = PPO(
        "MlpPolicy",
        env,
        device="cpu",
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=128,
        gamma=0.85,
        verbose=0,
    )

    model.learn(total_timesteps=500_000, callback=callback, progress_bar=True)

    model.save("ppo_quadx_waypoints")

    # frames = []

    # term, trunc = False, False
    # obs, info = env.reset()

    # while not (term or trunc):
    #     obs, rew, term, trunc, info = env.step(env.action_space.sample())
    #     frame = env.render()
    #     if frame is not None and frame.shape[-1] == 4:
    #         frame = frame[:, :, :3]  
    #     frames.append(frame)

    env.close()

    

    # imageio.mimsave(
    #     "quadx_waypoints.mp4",
    #     frames,
    #     fps=30
    # )

    return callback


#graficos
def metrics(callback: MetricsCallback):
    rewards = np.array(callback.episode_rewards)
    lengths = np.array(callback.episode_lengths)
    targets = np.array(callback.targets_reached)  

    episodes = np.arange(len(rewards))

    window = 10 # ventana para suavizado de los ultimos 10 episodios
    if len(rewards) >= window:
        rewards_smooth = np.convolve(rewards, np.ones(window)/window, mode='valid')
    else:
        rewards_smooth = rewards

    plt.figure(figsize=(14, 10))

    # Recompensa por episodio
    plt.subplot(3, 1, 1)
    plt.plot(episodes, rewards, alpha=0.4, label="Reward por episodio")
    if len(rewards_smooth) > 0:
        plt.plot(
            np.arange(len(rewards_smooth)),
            rewards_smooth,
            linewidth=2,
            label="Reward suavizada"
        )
    plt.ylabel("Recompensa")
    plt.legend()
    plt.grid(True)

    # Longitud por episodio -> Duracion
    plt.subplot(3, 1, 2)
    plt.plot(episodes, lengths)
    plt.title("Longitud por Episodio")
    plt.grid(True)

    # waypoints alcanzados
    plt.subplot(3, 1, 3)
    plt.plot(episodes, targets)
    plt.xlabel("Episodio")
    plt.ylabel("Waypoints alcanzados")
    plt.grid(True)


    plt.tight_layout()
    print("Mostrando gráficos...")
    plt.savefig("training_metrics.png", dpi=150)
    plt.close()
    


# main
if __name__ == "__main__":
    callback = entrenamiento()
    metrics(callback)
    
    