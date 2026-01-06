import gymnasium
import PyFlyt.gym_envs
from PyFlyt.gym_envs import FlattenWaypointEnv
from stable_baselines3 import PPO
import os

# Crear directorios para guardar modelos y logs (opcional pero recomendado)
models_dir = "models/PPO"
log_dir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 1. Configuración del Entorno
# IMPORTANTE: Usamos render_mode=None para entrenar rápido.
env = gymnasium.make("PyFlyt/QuadX-Waypoints-v4", render_mode=None)

# 2. Aplicar el Wrapper
# context_length=2 le da al agente información sobre los próximos 2 waypoints
env = FlattenWaypointEnv(env, context_length=2)

# 3. Inicializar el Modelo PPO
# Usamos "MlpPolicy" porque el wrapper aplana la entrada a un vector numérico.
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=log_dir,
    learning_rate=0.0003, # Ajustable
    n_steps=2048,         # Pasos por actualización
    batch_size=64
)

# 4. Entrenar el Modelo
TIMESTEPS = 100000 # Define cuánto tiempo quieres entrenar
print("Iniciando entrenamiento...")
model.learn(total_timesteps=TIMESTEPS)

# 5. Guardar el Modelo
model_path = f"{models_dir}/ppo_quadx_waypoints"
model.save(model_path)
print(f"Modelo guardado en {model_path}")

env.close()

## 6. Cargar y Probar el Modelo Guardado

final_env = gymnasium.make("PyFlyt/QuadX-Waypoints-v4", render_mode="human")
final_env = FlattenWaypointEnv(final_env, context_length=2)

final_model = PPO.load(model_path, env=env)

obs, _ = final_env.reset()
term, trunc = False, False

while not (term or trunc):
    # El modelo predice la acción basada en la observación.
    # deterministic=True es mejor para evaluación (quita el ruido de exploración).
    action, _ = final_model.predict(obs, deterministic=True)
    
    obs, reward, term, trunc, info = final_env.step(action)

final_env.close()