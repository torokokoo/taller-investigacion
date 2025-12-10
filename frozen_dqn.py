
import os
import torch
import optuna
import gymnasium as gym
import numpy as np
import time
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from optuna.visualization import plot_optimization_history, plot_param_importances

# Optimize CPU usage
device = torch.device("cpu")
torch.set_num_threads(4)
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'

def evaluate_agent(model, num_episodes=20):
    """Evaluate agent performance with more episodes for better discrimination"""
    test_env = gym.make("FrozenLake-v1", render_mode=None, is_slippery=False, max_episode_steps=15)

    successes = 0
    total_lengths = []

    for _ in range(num_episodes):
        obs, _ = test_env.reset()
        episode_length = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = test_env.step(int(action))
            episode_length += 1
            done = terminated or truncated

            if reward > 0:
                successes += 1
                break

        total_lengths.append(episode_length)

    test_env.close()
    success_rate = successes / num_episodes
    avg_length = np.mean(total_lengths)

    return success_rate, avg_length

def objective(trial):
    """Optuna objective function with more discriminative evaluation"""
    env = gym.make("FrozenLake-v1", render_mode=None, is_slippery=False, max_episode_steps=50)
    env = Monitor(env, filename=None)

    # Suggest hyperparameters with adjusted ranges
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    buffer_size = trial.suggest_int("buffer_size", 10000, 100000)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])  # Using power-of-2 values
    learning_starts = trial.suggest_int("learning_starts", 500, 2000)
    gamma = trial.suggest_float("gamma", 0.15, 0.20)  # Narrowed range like PPO
    exploration_fraction = trial.suggest_float("exploration_fraction", 0.1, 0.5)
    exploration_final_eps = trial.suggest_float("exploration_final_eps", 0.01, 0.1)
    target_update_interval = trial.suggest_int("target_update_interval", 100, 1000)
    train_freq = trial.suggest_int("train_freq", 1, 16)
    gradient_steps = trial.suggest_int("gradient_steps", 1, 8)

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=batch_size,
        gamma=gamma,
        exploration_fraction=exploration_fraction,
        exploration_initial_eps=1.0,
        exploration_final_eps=exploration_final_eps,
        target_update_interval=target_update_interval,
        train_freq=train_freq,
        gradient_steps=gradient_steps,
        policy_kwargs=dict(
            net_arch=[64, 64]  # Smaller network like in PPO
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
        study_name="frozen_lake_dqn_optimization",
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    # Optimize with fewer trials
    n_trials = 50  # Reduced from 100 to match PPO

    # Calculate and show estimated time
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

    # Print results
    print("\nBest trial:")
    trial = study.best_trial

    print(f"Value: {trial.value:.3f}")
    print("\nBest hyperparameters:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Save visualization plots
    try:
        os.makedirs("Resultados/OptunaFrozenDQN", exist_ok=True)

        # Plot optimization history
        fig1 = plot_optimization_history(study)
        fig1.write_image("Resultados/OptunaFrozenDQN/optimization_history.png")

        # Plot parameter importances
        fig2 = plot_param_importances(study)
        fig2.write_image("Resultados/OptunaFrozenDQN/param_importances.png")

    except Exception as e:
        print(f"Error saving plots: {e}")

    # Train final model with best parameters
    print("\nTraining final model with best parameters...")
    env = gym.make("FrozenLake-v1", render_mode=None, is_slippery=False, max_episode_steps=50)
    env = Monitor(env, filename=None)

    final_model = DQN(
        "MlpPolicy",
        env,
        **trial.params,
        policy_kwargs=dict(
            net_arch=[64, 64]  # Smaller network like in PPO
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
    final_model.save("Resultados/OptunaFrozenDQN/best_model")
    env.close()

if __name__ == "__main__":
    main()