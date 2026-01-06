import gymnasium as gym
import PyFlyt.gym_envs
from PyFlyt.gym_envs import FlattenWaypointEnv
import numpy as np
import matplotlib.pyplot as plt
import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback

# --- CONFIGURACIÓN ---
# Número de pasos para probar cada configuración en Optuna (menos que el entrenamiento final para ir rápido)
OPTUNA_TIMESTEPS = 50_000 
# Número de pruebas que hará Optuna
N_TRIALS = 20  
# Entrenamiento final con los mejores params
FINAL_TIMESTEPS = 500_000

# Callback original (lo mantenemos para el entrenamiento final)
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

# Función auxiliar para crear el entorno (usada tanto por Optuna como por el entrenamiento final)
def make_env():
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
        render_mode=None,
        render_resolution=(480, 480),
    )
    env = FlattenWaypointEnv(env, context_length=2)
    return env

# --- FUNCIÓN OBJECTIVE DE OPTUNA ---
def objective(trial):
    # 1. Sugerir Hiperparámetros
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    gamma = trial.suggest_float("gamma", 0.8, 0.999)
    gae_lambda = trial.suggest_float("gae_lambda", 0.8, 1.0)
    ent_coef = trial.suggest_float("ent_coef", 1e-8, 1e-1, log=True)
    
    # Batch size y n_steps suelen estar relacionados
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    n_steps = trial.suggest_categorical("n_steps", [1024, 2048])
    
    # Validar que n_steps sea mayor que batch_size (requisito de PPO)
    if batch_size > n_steps:
        batch_size = n_steps // 2

    # 2. Crear entorno y modelo
    env = make_env()
    
    model = PPO(
        "MlpPolicy",
        env,
        device="cpu",
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        verbose=0,
    )

    # 3. Entrenar (usamos menos pasos para probar rápido)
    try:
        model.learn(total_timesteps=OPTUNA_TIMESTEPS)
        
        # 4. Evaluar el modelo
        # Evaluamos en 10 episodios para obtener una media robusta
        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
    except Exception as e:
        # A veces los parámetros explotan (NaNs), manejamos el error devolviendo una recompensa baja
        print(f"Trial fallido: {e}")
        mean_reward = -np.inf
    finally:
        env.close()

    return mean_reward

# --- FUNCIÓN DE GRÁFICOS ---
def metrics(callback: MetricsCallback):
    rewards = np.array(callback.episode_rewards)
    lengths = np.array(callback.episode_lengths)
    targets = np.array(callback.targets_reached)  
    episodes = np.arange(len(rewards))

    window = 10 
    if len(rewards) >= window:
        rewards_smooth = np.convolve(rewards, np.ones(window)/window, mode='valid')
    else:
        rewards_smooth = rewards

    plt.figure(figsize=(14, 10))

    plt.subplot(3, 1, 1)
    plt.plot(episodes, rewards, alpha=0.4, label="Reward por episodio")
    if len(rewards_smooth) > 0:
        plt.plot(np.arange(len(rewards_smooth)), rewards_smooth, linewidth=2, label="Reward suavizada")
    plt.ylabel("Recompensa")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(episodes, lengths)
    plt.title("Longitud por Episodio")
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(episodes, targets)
    plt.xlabel("Episodio")
    plt.ylabel("Waypoints alcanzados")
    plt.grid(True)

    plt.tight_layout()
    print("Guardando gráficos del mejor modelo...")
    plt.savefig("optimized_training_metrics.png", dpi=150)
    plt.close()

# --- MAIN ---
if __name__ == "__main__":
    print("Iniciando optimización con Optuna...")
    
    # 1. Crear estudio y optimizar
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS)

    print("\n------------------------------------------------")
    print("Mejores parámetros encontrados:")
    print(study.best_params)
    print("------------------------------------------------\n")

    # 2. Entrenar modelo final con los mejores parámetros
    print("Iniciando entrenamiento final con los mejores parámetros...")
    
    best_params = study.best_params
    
    # Recalculamos validación n_steps vs batch_size por seguridad
    if best_params["batch_size"] > best_params["n_steps"]:
        best_params["batch_size"] = best_params["n_steps"] // 2

    env = make_env()
    callback = MetricsCallback()

    model = PPO(
        "MlpPolicy",
        env,
        device="cpu",
        learning_rate=best_params["learning_rate"],
        n_steps=best_params["n_steps"],
        batch_size=best_params["batch_size"],
        gamma=best_params["gamma"],
        gae_lambda=best_params["gae_lambda"],
        ent_coef=best_params["ent_coef"],
        verbose=1,
    )

    model.learn(total_timesteps=FINAL_TIMESTEPS, callback=callback)
    
    model.save("ppo_quadx_waypoints_optimized")
    env.close()

    # 3. Generar gráficas
    metrics(callback)
    print("¡Proceso completado!")