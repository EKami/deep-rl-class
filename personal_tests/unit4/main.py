import os
import numpy as np
from tqdm import tqdm
import json

from typing import Any, Dict
from collections import deque
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# Gym
import gymnasium as gym
import imageio

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.visualization import plot_optimization_history, plot_param_importances

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cur_dir = Path(os.path.dirname(os.path.abspath(__file__)))
BEST_MODEL_PATH = cur_dir / "best_cartpole_model.pth"
best_mean_reward_global = float("-inf")


class Policy(nn.Module):
    def __init__(self, s_size, a_size, h_size, n_fc_layers=1):
        super(Policy, self).__init__()
        self.n_fc_layers = n_fc_layers

        layers = []
        # Input layer
        layers.append(nn.Linear(s_size, h_size))
        layers.append(nn.ReLU())

        # Hidden layers
        for _ in range(n_fc_layers - 1):
            layers.append(nn.Linear(h_size, h_size))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(h_size, a_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = self.network(x)
        return F.softmax(x, dim=1)

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)


def record_video(env_id, policy, out_directory, fps=30):
    images = []
    # Use a new environment for recording to avoid state issues
    record_env = gym.make(env_id, render_mode="rgb_array")
    state, _ = record_env.reset()
    terminated = False
    truncated = False
    episode_reward = 0
    for _ in range(500):  # Max steps for CartPole
        img = record_env.render()
        images.append(img)
        action, _ = policy.act(state)
        state, reward, terminated, truncated, _ = record_env.step(action)
        episode_reward += reward
        if terminated or truncated:
            break
    record_env.close()
    print(f"Video recording: episode reward: {episode_reward}")
    imageio.mimsave(
        out_directory, [np.array(img) for i, img in enumerate(images)], fps=fps
    )


def reinforce(
    env,
    policy,
    optimizer,
    n_training_episodes,
    max_t,
    gamma,
    print_every,
    trial: optuna.Trial,
    eval_env: gym.Env,
    eval_freq_episodes: int,
    n_eval_episodes_optuna: int,
):
    scores_deque = deque(maxlen=100)
    all_scores = []

    for i_episode in tqdm(range(1, n_training_episodes + 1), desc="Training"):
        saved_log_probs = []
        rewards = []
        state, _ = env.reset()

        for t in range(max_t):
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            state, reward, terminated, truncated, _ = env.step(action)
            rewards.append(reward)
            if terminated or truncated:
                break

        current_episode_reward = sum(rewards)
        scores_deque.append(current_episode_reward)
        all_scores.append(current_episode_reward)

        returns = deque(maxlen=max_t)
        n_steps = len(rewards)
        for t in range(n_steps)[::-1]:
            disc_return_t = returns[0] if len(returns) > 0 else 0
            returns.appendleft(gamma * disc_return_t + rewards[t])

        eps = np.finfo(np.float32).eps.item()
        returns_tensor = torch.tensor(
            list(returns), device=device, dtype=torch.float32
        )  # Ensure on device
        returns_tensor = (returns_tensor - returns_tensor.mean()) / (
            returns_tensor.std() + eps
        )

        policy_loss = []
        for log_prob, disc_return in zip(saved_log_probs, returns_tensor):
            policy_loss.append(-log_prob * disc_return)

        if not policy_loss:  # Should not happen if episode runs at least 1 step
            # print(f"Warning: Episode {i_episode} had no steps or policy_loss was empty.")
            if trial:  # If in optuna context, and something is weird, prune early
                trial.report(0, i_episode)  # Report a bad score
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
            continue

        policy_loss = torch.stack(
            policy_loss
        ).sum()  # Use torch.stack for list of tensors

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if i_episode % print_every == 0:
            print(
                f"Episode {i_episode}\tAverage Score (last {scores_deque.maxlen}): {np.mean(scores_deque):.2f}"
            )

        # Optuna pruning check
        if trial and i_episode % eval_freq_episodes == 0:
            # Evaluate the current policy for Optuna
            # Use a simplified evaluation or reuse evaluate_agent
            # For simplicity here, let's use the scores_deque mean if it's sufficiently full
            # Or, better, run a proper evaluation:
            intermediate_mean_reward, _ = evaluate_agent(
                eval_env,
                max_t,
                n_eval_episodes_optuna,
                policy,
                is_optuna_eval=True,
            )

            trial.report(intermediate_mean_reward, i_episode)
            if trial.should_prune():
                print(
                    f"Trial pruned at episode {i_episode} with reward {intermediate_mean_reward}"
                )
                raise optuna.exceptions.TrialPruned()

    return all_scores


def evaluate_agent(env, max_steps, n_eval_episodes, policy, is_optuna_eval=False):
    episode_rewards = []
    desc = "Optuna Intermediate Eval" if is_optuna_eval else "Final Evaluation"
    for episode in tqdm(
        range(n_eval_episodes), desc=desc, leave=False, disable=is_optuna_eval
    ):  # Disable tqdm for frequent optuna evals
        state, _ = env.reset()
        total_rewards_ep = 0
        for step in range(max_steps):
            action, _ = policy.act(state)  # Assumes policy.act handles device placement
            new_state, reward, terminated, truncated, _ = env.step(action)
            total_rewards_ep += reward
            if terminated or truncated:
                break
            state = new_state
        episode_rewards.append(total_rewards_ep)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    return mean_reward, std_reward


def sample_params(env_id, s_size, a_size, trial: optuna.Trial) -> Dict[str, Any]:
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.95, 0.999, log=True)  # Slightly narrowed
    h_size = trial.suggest_categorical("net_arch", [64, 128, 256])  # Increased capacity
    n_fc_layers = trial.suggest_categorical("n_fc_layers", [1, 2])

    n_training_episodes = trial.suggest_categorical("n_training_episodes", [3000, 5000])
    # Optuna evaluation frequency (in episodes)
    # Should be frequent enough for pruning but not too frequent to slow down too much
    eval_freq_episodes_optuna = trial.suggest_categorical(
        "eval_freq_episodes_optuna", [250, 500]
    )

    return {
        "h_size": h_size,
        "n_fc_layers": n_fc_layers,
        "n_training_episodes": n_training_episodes,
        "n_evaluation_episodes_final": 20,  # For final evaluation after training
        "n_evaluation_episodes_optuna": 5,  # For intermediate Optuna pruning evaluations
        "eval_freq_episodes_optuna": eval_freq_episodes_optuna,
        "max_t": 500,  # CartPole-v1 environment limit
        "gamma": gamma,
        "lr": lr,
        "env_id": env_id,
        "state_space": s_size,
        "action_space": a_size,
    }


def objective(env, eval_env, s_size, a_size):
    def inner(trial: optuna.Trial) -> float:
        global best_mean_reward_global  # Use the renamed global
        hyperparameters = sample_params(env.spec.id, s_size, a_size, trial)

        # Pass actual number of hidden layers to Policy
        # If n_fc_layers = 1, it means 1 hidden layer.
        # If n_fc_layers = 0, it means input -> output (but our Policy has at least one fc1)
        # My Policy class takes n_fc_layers as the number of *additional* hidden layers after fc1.
        # Let's adjust Policy init for clarity: n_fc_layers = total hidden layers.
        # So if trial.suggests 1, policy has 1 hidden layer. If 2, policy has 2.
        cartpole_policy = Policy(
            hyperparameters["state_space"],
            hyperparameters["action_space"],
            hyperparameters["h_size"],
            hyperparameters["n_fc_layers"],
        ).to(device)

        cartpole_optimizer = optim.Adam(
            cartpole_policy.parameters(), lr=hyperparameters["lr"]
        )

        nan_encountered = False
        try:
            # Pass trial and eval_env for intermediate evaluations and pruning
            scores = reinforce(
                env,
                cartpole_policy,
                cartpole_optimizer,
                hyperparameters["n_training_episodes"],
                hyperparameters["max_t"],
                hyperparameters["gamma"],
                print_every=hyperparameters["n_training_episodes"]
                // 10,  # Print 10 times during training
                trial=trial,  # Pass trial for pruning
                eval_env=eval_env,  # Pass eval_env for intermediate evals
                eval_freq_episodes=hyperparameters["eval_freq_episodes_optuna"],
                n_eval_episodes_optuna=hyperparameters["n_evaluation_episodes_optuna"],
            )
        except AssertionError as e:
            print(f"AssertionError in training: {e}")
            nan_encountered = True
        except optuna.exceptions.TrialPruned:
            # If reinforce raises TrialPruned, re-raise it so Optuna handles it
            raise
        except Exception as e:
            print(f"Unexpected error during training for trial {trial.number}: {e}")
            nan_encountered = True  # Treat other errors as failures

        if nan_encountered:
            return float("nan")  # Optuna will handle this as a failed trial

        # Final evaluation after successful training
        mean_reward, std_reward = evaluate_agent(
            eval_env,
            hyperparameters["max_t"],
            hyperparameters["n_evaluation_episodes_final"],
            cartpole_policy,
        )

        print(
            f"Trial {trial.number} finished. Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}"
        )

        # Save the best model and std_reward so far
        # Ensure this comparison is with the global best, not a trial-local one
        if mean_reward > best_mean_reward_global:
            best_mean_reward_global = mean_reward
            print(f"New best model found! Mean reward: {best_mean_reward_global:.2f}")
            torch.save(
                {
                    "state_dict": cartpole_policy.state_dict(),
                    "mean_reward": mean_reward,
                    "std_reward": std_reward,
                    "hyperparameters": hyperparameters,
                },
                BEST_MODEL_PATH,
            )
        return mean_reward

    return inner


def cartpole():
    global best_mean_reward_global  # Ensure we modify the global
    best_mean_reward_global = float("-inf")  # Reset for each run of cartpole()

    env_id = "CartPole-v1"
    env = gym.make(env_id)  # No render_mode needed for training env
    eval_env = gym.make(env_id)  # No render_mode for eval env unless debugging

    s_size = env.observation_space.shape[0]
    a_size = env.action_space.n

    print(f"State Space: {s_size}, Action Space: {a_size}")

    # Optuna settings
    N_STARTUP_TRIALS = 10  # Allow more random exploration initially
    # N_EVALUATIONS_PRUNING is now controlled by eval_freq_episodes_optuna and n_training_episodes
    N_TRIALS = 50  # Adjust as needed, more trials = better search but longer
    N_JOBS = 1
    TIMEOUT = 7200  # 2 hours timeout for the whole study

    torch.set_num_threads(1)
    sampler = TPESampler(
        n_startup_trials=N_STARTUP_TRIALS, seed=42
    )  # Add seed for reproducibility

    # Pruning: n_warmup_steps is the number of intermediate reports to wait for.
    # If eval_freq_episodes_optuna is 500, and n_training_episodes is 3000,
    # there will be 3000/500 = 6 reports.
    # Let's say we want to wait for at least 2 reports before pruning.
    pruner = MedianPruner(
        n_startup_trials=N_STARTUP_TRIALS,
        n_warmup_steps=2,  # Wait for 2 intermediate results
    )
    study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize")

    try:
        study.optimize(
            objective(env, eval_env, s_size, a_size),
            n_trials=N_TRIALS,
            n_jobs=N_JOBS,
            timeout=TIMEOUT,
        )
    except KeyboardInterrupt:
        print("Optimization interrupted by user.")
    finally:
        env.close()
        eval_env.close()

    print("Number of finished trials: ", len(study.trials))

    # Filter out pruned/failed trials before accessing best_trial
    completed_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]
    if not completed_trials:
        print("No trials completed successfully.")
        return None, None, -1, -1  # Indicate failure

    # Optuna's study.best_trial should correctly identify the best among completed trials
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    best_params_from_study = trial.params  # These are the sampled ones

    # Load the model saved via best_mean_reward_global logic
    if not BEST_MODEL_PATH.exists():
        print("Best model path does not exist. Cannot load best model.")
        # Fallback to recreating from study's best_params if no model was saved
        # (e.g., if all trials were worse than initial -inf, or error in saving)
        print("Recreating model from study's best_params as a fallback.")
        best_policy = Policy(
            s_size,
            a_size,
            best_params_from_study["h_size"],
            best_params_from_study["n_fc_layers"],
        ).to(device)
        # Note: This fallback model is untrained or partially trained from the last trial.
        # The primary path is to load from BEST_MODEL_PATH.
        mean_reward_loaded = trial.value  # This is the value Optuna tracked
        std_reward_loaded = "N/A (fallback)"
    else:
        print(f"Loading best model from {BEST_MODEL_PATH}")
        checkpoint = torch.load(
            BEST_MODEL_PATH, map_location=device, weights_only=False
        )
        saved_hyperparams = checkpoint.get("hyperparameters", best_params_from_study)

        best_policy = Policy(
            s_size,
            a_size,
            saved_hyperparams["h_size"],
            saved_hyperparams["n_fc_layers"],
        ).to(device)
        best_policy.load_state_dict(checkpoint["state_dict"])
        std_reward_loaded = checkpoint["std_reward"]
        # The 'trial.value' is the one Optuna knows.
        # 'best_mean_reward_global' should ideally match 'trial.value' if saving logic is robust.
        mean_reward_loaded = trial.value

    return env_id, best_policy, mean_reward_loaded, std_reward_loaded


def main():
    result = cartpole()
    if result is None or result[1] is None:
        print("Cartpole optimization failed or produced no model.")
        return

    env_id, model, mean_reward, std_reward = result

    print(
        f"Optimization finished. Best Mean Reward: {mean_reward:.2f}, Std Reward: {std_reward}"
    )

    # It's possible that even the best trial doesn't reach 200.
    if mean_reward < 100:
        print(
            f"Mean reward {mean_reward} is low, video might not show successful behavior."
        )

    if model is not None:
        video_path = cur_dir / "replay.mp4"
        print(f"Recording video for model with mean reward: {mean_reward}")
        record_video(env_id, model, video_path, fps=30)
        print(f"Video saved at {video_path}")
    else:
        print("No model available to record video.")


if __name__ == "__main__":
    # It's good practice to handle potential issues if __file__ is not defined (e.g. in some interpreters)
    if "__file__" not in globals():
        __file__ = "dummy_filename_for_interactive.py"  # Or handle appropriately
        cur_dir = Path(os.getcwd())
        BEST_MODEL_PATH = cur_dir / "best_cartpole_model.pth"

    main()
