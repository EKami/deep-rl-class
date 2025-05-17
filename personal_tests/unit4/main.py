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
best_mean_reward = float("-inf")


class Policy(nn.Module):
    def __init__(self, s_size, a_size, h_size, n_fc_layers=1):
        super(Policy, self).__init__()
        self.n_fc_layers = n_fc_layers

        # First layer always goes from state size to hidden size
        self.fc1 = nn.Linear(s_size, h_size)

        # Add intermediate layers if n_fc_layers > 1
        self.fc_layers = nn.ModuleList(
            [nn.Linear(h_size, h_size) for _ in range(n_fc_layers - 1)]
        )

        # Final layer always goes to action size
        self.fc_out = nn.Linear(h_size, a_size)

    def forward(self, x):
        # Apply first layer with ReLU
        x = F.relu(self.fc1(x))

        # Apply intermediate layers with ReLU
        for layer in self.fc_layers:
            x = F.relu(layer(x))

        # Apply final layer
        x = self.fc_out(x)
        return F.softmax(x, dim=1)

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)


class TrialEvalCallback:
    """
    Callback used for evaluating and reporting a trial.

    :param eval_env: Evaluation environement
    :param trial: Optuna trial object
    :param n_eval_episodes: Number of evaluation episodes
    :param eval_freq:   Evaluate the agent every ``eval_freq`` call of the callback.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic policy.
    :param verbose:
    """

    def __init__(
        self,
        eval_env: gym.Env,
        trial: optuna.Trial,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        deterministic: bool = True,
        verbose: int = 0,
    ):
        # Initialize attributes directly without super() call
        self.eval_env = eval_env
        self.trial = trial
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.deterministic = deterministic
        self.verbose = verbose
        self.n_calls = 0
        self.last_mean_reward = -float("inf")
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Evaluate policy (done in the parent class)
            super()._on_step()
            self.eval_idx += 1
            # Send report to Optuna
            self.trial.report(self.last_mean_reward, self.eval_idx)
            # Prune trial if need
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True


def record_video(env_id, policy, out_directory, fps=30):
    """
    Generate a replay video of the agent
    :param env
    :param Qtable: Qtable of our agent
    :param out_directory
    :param fps: how many frame per seconds (with taxi-v3 and frozenlake-v1 we use 1)
    """
    images = []
    env = gym.make(env_id, render_mode="rgb_array")
    state, _ = env.reset()
    terminated = False
    truncated = False
    while not (terminated or truncated):
        img = env.render()
        images.append(img)
        action, _ = policy.act(state)
        state, reward, _, terminated, truncated = env.step(action)
    env.close()
    imageio.mimsave(
        out_directory, [np.array(img) for i, img in enumerate(images)], fps=fps
    )


def reinforce(env, policy, optimizer, n_training_episodes, max_t, gamma, print_every):
    # Help us to calculate the score during the training
    scores_deque = deque(maxlen=100)
    scores = []
    # Line 3 of pseudocode
    for i_episode in tqdm(range(1, n_training_episodes + 1), desc="Training"):
        saved_log_probs = []
        rewards = []
        state, _ = env.reset()

        # Line 4 of pseudocode
        for t in range(max_t):
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            state, reward, _, terminated, truncated = env.step(action)
            rewards.append(reward)
            if terminated or truncated:
                break
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        # Line 6 of pseudocode: calculate the return
        returns = deque(maxlen=max_t)
        n_steps = len(rewards)
        # Compute the discounted returns at each timestep,
        # as
        #      the sum of the gamma-discounted return at time t (G_t) + the reward at time t
        #
        # In O(N) time, where N is the number of time steps
        # (this definition of the discounted return G_t follows the definition of this quantity
        # shown at page 44 of Sutton&Barto 2017 2nd draft)
        # G_t = r_(t+1) + r_(t+2) + ...

        # Given this formulation, the returns at each timestep t can be computed
        # by re-using the computed future returns G_(t+1) to compute the current return G_t
        # G_t = r_(t+1) + gamma*G_(t+1)
        # G_(t-1) = r_t + gamma* G_t
        # (this follows a dynamic programming approach, with which we memorize solutions in order
        # to avoid computing them multiple times)

        # This is correct since the above is equivalent to (see also page 46 of Sutton&Barto 2017 2nd draft)
        # G_(t-1) = r_t + gamma*r_(t+1) + gamma*gamma*r_(t+2) + ...

        ## Given the above, we calculate the returns at timestep t as:
        #               gamma[t] * return[t] + reward[t]
        #
        ## We compute this starting from the last timestep to the first, in order
        ## to employ the formula presented above and avoid redundant computations that would be needed
        ## if we were to do it from first to last.

        ## Hence, the queue "returns" will hold the returns in chronological order, from t=0 to t=n_steps
        ## thanks to the appendleft() function which allows to append to the position 0 in constant time O(1)
        ## a normal python list would instead require O(N) to do this.
        for t in range(n_steps)[::-1]:
            disc_return_t = returns[0] if len(returns) > 0 else 0
            returns.appendleft(gamma * disc_return_t + rewards[t])

        ## standardization of the returns is employed to make training more stable
        eps = np.finfo(np.float32).eps.item()
        ## eps is the smallest representable float, which is
        # added to the standard deviation of the returns to avoid numerical instabilities
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)

        # Line 7:
        policy_loss = []
        for log_prob, disc_return in zip(saved_log_probs, returns):
            policy_loss.append(-log_prob * disc_return)
        policy_loss = torch.cat(policy_loss).sum()

        # Line 8: PyTorch prefers gradient descent
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if i_episode % print_every == 0:
            print(
                "Episode {}\tAverage Score: {:.2f}".format(
                    i_episode, np.mean(scores_deque)
                )
            )

    return scores


def evaluate_agent(env, eval_callback, max_steps, n_eval_episodes, policy):
    """
    Evaluate the agent for ``n_eval_episodes`` episodes and returns average reward and std of reward.
    :param env: The evaluation environment
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param policy: The Reinforce agent
    """
    episode_rewards = []
    for episode in range(n_eval_episodes):
        state, _ = env.reset()
        total_rewards_ep = 0

        for step in range(max_steps):
            action, _ = policy.act(state)
            new_state, reward, terminated, truncated, info = env.step(action)
            total_rewards_ep += reward
            if terminated or truncated:
                break
            state = new_state
        episode_rewards.append(total_rewards_ep)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    # --- Use the callback here if provided ---
    if eval_callback is not None:
        # Update n_calls for proper timing
        if hasattr(eval_callback, "n_calls"):
            eval_callback.n_calls += 1

        # Set the last mean reward so the callback can access it
        if hasattr(eval_callback, "last_mean_reward"):
            eval_callback.last_mean_reward = mean_reward

        # Optionally, call _on_step if the callback has it
        if hasattr(eval_callback, "_on_step"):
            eval_callback._on_step()

    return mean_reward, std_reward


def sample_params(env_id, s_size, a_size, trial: optuna.Trial) -> Dict[str, Any]:
    lr = trial.suggest_float("lr", 1e-5, 0.1, log=True)
    gamma = trial.suggest_float("gamma", 0.9, 0.99999, log=True)
    h_size = trial.suggest_categorical("net_arch", [8, 16, 32])
    n_fc_layers = trial.suggest_categorical("n_fc_layers", [2, 4, 8])
    n_training_episodes = trial.suggest_categorical(
        "n_training_episodes", [1000, 2000, 3000]
    )

    return {
        "h_size": h_size,
        "n_fc_layers": n_fc_layers,
        "n_training_episodes": n_training_episodes,
        "n_evaluation_episodes": 10,
        "max_t": 1000,
        "gamma": gamma,
        "lr": lr,
        "env_id": env_id,
        "state_space": s_size,
        "action_space": a_size,
    }


def objective(env, eval_env, s_size, a_size):
    def inner(trial: optuna.Trial) -> float:
        """
        Objective function using by Optuna to evaluate
        one configuration (i.e., one set of hyperparameters).

        Given a trial object, it will sample hyperparameters,
        evaluate it and report the result (mean episodic reward after training)

        :param trial: Optuna trial object
        :return: Mean episodic reward after training
        """
        global best_mean_reward
        hyperparameters = sample_params(env.spec.id, s_size, a_size, trial)
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
            scores = reinforce(
                env,
                cartpole_policy,
                cartpole_optimizer,
                hyperparameters["n_training_episodes"],
                hyperparameters["max_t"],
                hyperparameters["gamma"],
                100,
            )
        except AssertionError as e:
            # Sometimes, random hyperparams can generate NaN
            print(e)
            nan_encountered = True

        # Tell the optimizer that the trial failed
        if nan_encountered:
            return float("nan")

        # Create the evaluation callback
        eval_callback = TrialEvalCallback(
            eval_env, trial, hyperparameters["n_evaluation_episodes"]
        )

        if eval_callback.is_pruned:
            raise optuna.exceptions.TrialPruned()

        # Now evaluate
        mean_reward, std_reward = evaluate_agent(
            eval_env,
            eval_callback,
            hyperparameters["max_t"],
            hyperparameters["n_evaluation_episodes"],
            cartpole_policy,
        )

        # Save the best model and std_reward so far
        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            torch.save(
                {
                    "state_dict": cartpole_policy.state_dict(),
                    "std_reward": std_reward,
                },
                BEST_MODEL_PATH,
            )

        return mean_reward

    return inner


def cartpole():
    # https://gymnasium.farama.org/environments/classic_control/cart_pole/
    env_id = "CartPole-v1"
    # Create the env
    env = gym.make(env_id, render_mode="rgb_array")

    # Create the evaluation env
    eval_env = gym.make(env_id, render_mode="rgb_array")

    # Get the state space and action space
    s_size = env.observation_space.shape[0]
    a_size = env.action_space.n

    print("_____OBSERVATION SPACE_____ \n")
    print("The State Space is: ", s_size)
    print(
        "Sample observation", env.observation_space.sample()
    )  # Get a random observation

    print("\n _____ACTION SPACE_____ \n")
    print("The Action Space is: ", a_size)
    # Take a random action
    print("Action Space Sample", env.action_space.sample())

    N_STARTUP_TRIALS = 5  # Stop random sampling after N_STARTUP_TRIALS
    N_EVALUATIONS = 2  # Number of evaluations during the training
    N_TRIALS = 100  # Maximum number of trials
    N_JOBS = 1  # Number of jobs to run in parallel
    TIMEOUT = None  # No timeout, or set to a much larger value like 7200 (2 hours)

    # Set pytorch num threads to 1 for faster training
    torch.set_num_threads(1)
    # Select the sampler, can be random, TPESampler, CMAES, ...
    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
    # Do not prune before 1/3 of the max budget is used
    pruner = MedianPruner(
        n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_EVALUATIONS // 3
    )
    # Create the study and start the hyperparameter optimization
    study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize")
    try:
        study.optimize(
            objective(env, eval_env, s_size, a_size),
            n_trials=N_TRIALS,
            n_jobs=N_JOBS,
            timeout=TIMEOUT,
        )
    except KeyboardInterrupt:
        pass

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")

    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    best_params = trial.params
    # Recreate the best model
    best_policy = Policy(
        s_size,
        a_size,
        best_params["net_arch"],
        best_params["n_fc_layers"],
    ).to(device)
    checkpoint = torch.load(BEST_MODEL_PATH, weights_only=False)
    best_policy.load_state_dict(checkpoint["state_dict"])
    std_reward = checkpoint["std_reward"]
    mean_reward = trial.value

    return env_id, best_policy, mean_reward, std_reward


def main():
    env_id, model, mean_reward, std_reward = cartpole()
    if mean_reward < 200:
        raise ValueError("Mean reward is less than 200, can't record video")

    print(f"Obtained mean reward: {mean_reward} with std: {std_reward}")
    video_path = cur_dir / "replay.mp4"
    record_video(env_id, model, video_path, fps=30)
    print(f"Video saved at {video_path}")


if __name__ == "__main__":
    main()
