
import os
import torch
import optuna
import gymnasium as gym
import numpy as np
import PyFlyt.gym_envs
from PyFlyt.gym_envs import FlattenWaypointEnv
import time
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from optuna.visualization import plot_optimization_history, plot_param_importances

# Optimize CPU usage
device = torch.device("cpu")
torch.set_num_threads(2)
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'

def evaluate_agent(model, num_episodes=20):
    test_env = gym.make("PyFlyt/QuadX-Waypoints-v4", max_duration_seconds=15.0)
    test_env = FlattenWaypointEnv(test_env, context_length=2)

    successes = 0
    total_lengths = []
    total_steps = 0

    for i in range(num_episodes):
        obs, _ = test_env.reset()
        episode_length = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = test_env.step(action)
            episode_length += 1
            total_steps += 1
            done = terminated or truncated

            if reward > 0:
                successes += 1
                break

        total_lengths.append(episode_length)

    test_env.close()
    success_rate = successes / num_episodes
    avg_length = np.mean(total_lengths)

    return success_rate, avg_length  # Removed efficiency from return values

def objective(trial):
    """Optuna objective function with more discriminative evaluation"""
    env = gym.make("PyFlyt/QuadX-Waypoints-v4",  max_duration_seconds=15.0)
    env = FlattenWaypointEnv(env, context_length=2)
    env = Monitor(env, filename=None)

    # Use powers of 2 for n_steps and batch_size
    n_steps = trial.suggest_categorical("n_steps", [64, 128, 256, 512])
    batch_size = trial.suggest_categorical("batch_size", [32, 64])

    # Other hyperparameters with adjusted ranges
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    n_epochs = trial.suggest_int("n_epochs", 3, 10)
    gamma = trial.suggest_float("gamma", 0.15, 0.20)
    gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.99)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.3)
    ent_coef = trial.suggest_float("ent_coef", 0.001, 0.01)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        policy_kwargs=dict(
            net_arch=[64, 64],
            activation_fn=torch.nn.ReLU
        ),
        verbose=0
    )

    try:
        # Train with reduced timesteps during optimization
        model.learn(total_timesteps=25000, log_interval=100)

        # Evaluate with more comprehensive metrics
        success_rate, avg_length = evaluate_agent(model)

        # Create a more discriminative score that favors both high success rates and efficient paths
        score = success_rate / avg_length

        # Add a small random noise to break ties
        score += np.random.uniform(0, 0.001)

        # Clean up
        env.close()

        return score

    except Exception as e:
        print(f"Trial failed: {e}")
        return float('-inf')

def main():
    # Create study with TPE sampler
    study = optuna.create_study(
        direction="maximize",
        study_name="frozen_lake_ppo_optimization",
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    # Optimize with fewer trials initially to test
    n_trials = 5
    
    # Calculate and show estimated time
    # Do a quick test run to get average trial time
    print("Running test trial to estimate total time...")
    start_time = time.time()
    _ = objective(study.ask())
    trial_time = time.time() - start_time
    estimated_total_time = trial_time * n_trials
    print(f"Estimated total time: {estimated_total_time/3600:.1f} hours ({estimated_total_time/60:.1f} minutes)")

    try:
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    except KeyboardInterrupt:
        print("Optimization stopped early by user.")

    # Rest of the code remains the same...
    print("\nBest trial:")
    trial = study.best_trial

    print(f"Value: {trial.value:.3f}")
    print("\nBest hyperparameters:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Save visualization plots
    try:
        os.makedirs("Resultados/OptunaFrozenPPO", exist_ok=True)

        # Plot optimization history
        fig1 = plot_optimization_history(study)
        fig1.write_image("Resultados/OptunaFrozenPPO/optimization_history.png")

        # Plot parameter importances
        fig2 = plot_param_importances(study)
        fig2.write_image("Resultados/OptunaFrozenPPO/param_importances.png")

    except Exception as e:
        print(f"Error saving plots: {e}")

    # Train final model with best parameters
    print("\nTraining final model with best parameters...")
    env = gym.make("FrozenLake-v1", render_mode=None, is_slippery=False, max_episode_steps=50)
    env = Monitor(env, filename=None)

    final_model = PPO(
        "MlpPolicy",
        env,
        **trial.params,
        policy_kwargs=dict(
            net_arch=[64, 64],
            activation_fn=torch.nn.ReLU
        ),
        verbose=1
    )

    final_model.learn(total_timesteps=50000)

    # Evaluate final model
    success_rate, avg_length = evaluate_agent(final_model, num_episodes=16)
    print(f"\nFinal Model Performance:")
    print(f"Success Rate: {success_rate:.1%}")
    print(f"Average Episode Length: {avg_length:.2f}")

    # Save the best model
    final_model.save("Resultados/OptunaFrozenPPO/best_model")
    env.close()

if __name__ == "__main__":
    main()