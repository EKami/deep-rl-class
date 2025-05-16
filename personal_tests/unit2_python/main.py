import os
import imageio
import numpy as np
import random
from pathlib import Path
import numpy as np
import gymnasium as gym
import random
import imageio
from tqdm import tqdm
from PIL import Image


def record_video(env, Qtable, out_directory, fps=1):
    """
    Generate a replay video of the agent
    :param env
    :param Qtable: Qtable of our agent
    :param out_directory
    :param fps: how many frame per seconds (with taxi-v3 and frozenlake-v1 we use 1)
    """
    images = []
    terminated = False
    truncated = False
    state, info = env.reset(seed=random.randint(0, 500))
    img = env.render()
    images.append(img)
    while not terminated or truncated:
        # Take the action (index) that have the maximum expected future reward given that state
        action = np.argmax(Qtable[state][:])
        # We directly put next_state = state for recording logic
        state, _, terminated, truncated, _ = env.step(action)
        img = env.render()
        images.append(img)

    # Convert list to numpy array for batch processing
    images = np.array(images)
    h, w = images.shape[1:3]
    # Calculate new dimensions that are divisible by 16
    new_h = ((h + 15) // 16) * 16
    new_w = ((w + 15) // 16) * 16

    # Resize all images at once if needed
    if new_h != h or new_w != w:
        images = np.array(
            [np.array(Image.fromarray(img).resize((new_w, new_h))) for img in images]
        )

    imageio.mimsave(out_directory, images, fps=fps)
    return out_directory


def initialize_q_table(state_space, action_space):
    Qtable = np.zeros((state_space, action_space))
    return Qtable


def greedy_policy(Qtable, state):
    # Exploitation: take the action with the highest state, action value
    action = np.argmax(Qtable[state])

    return action


def epsilon_greedy_policy(env, Qtable, state, epsilon):
    # Randomly generate a number between 0 and 1
    random_num = np.random.rand()
    # if random_num > greater than epsilon --> exploitation
    if random_num > epsilon:
        # Take the action with the highest value given a state
        # np.argmax can be useful here
        action = greedy_policy(Qtable, state)
    # else --> exploration
    else:
        action = env.action_space.sample()

    return action


def train(
    n_training_episodes,
    min_epsilon,
    max_epsilon,
    decay_rate,
    env,
    max_steps,
    Qtable,
    gamma,
    learning_rate,
):
    for episode in tqdm(range(n_training_episodes)):
        # Reduce epsilon (because we need less and less exploration)
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(
            -decay_rate * episode
        )
        # Reset the environment
        state, _ = env.reset()
        terminated = False
        truncated = False

        # repeat
        for _ in range(max_steps):
            # Choose the action At using epsilon greedy policy
            action = epsilon_greedy_policy(env, Qtable, state, epsilon)

            # Take action At and observe Rt+1 and St+1
            # Take the action (a) and observe the outcome state(s') and reward (r)
            new_state, reward, terminated, truncated, info = env.step(action)

            # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
            difference = (
                reward + gamma * np.max(Qtable[new_state]) - Qtable[state][action]
            )
            Qtable[state][action] = Qtable[state][action] + learning_rate * difference

            # If terminated or truncated finish the episode
            if terminated or truncated:
                break

            # Our next state is the new state
            state = new_state
    return Qtable


def evaluate_agent(env, max_steps, n_eval_episodes, Q, seed=[]):
    """
    Evaluate the agent for ``n_eval_episodes`` episodes and returns average reward and std of reward.
    :param env: The evaluation environment
    :param max_steps: Maximum number of steps per episode
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param Q: The Q-table
    :param seed: The evaluation seed array (for taxi-v3)
    """
    episode_rewards = []
    for episode in tqdm(range(n_eval_episodes)):
        if seed:
            state, _ = env.reset(seed=seed[episode])
        else:
            state, _ = env.reset()
        truncated = False
        terminated = False
        total_rewards_ep = 0

        for _ in range(max_steps):
            # Take the action (index) that have the maximum expected future reward given that state
            action = greedy_policy(Q, state)
            new_state, reward, terminated, truncated, _ = env.step(action)
            total_rewards_ep += reward

            if terminated or truncated:
                break
            state = new_state
        episode_rewards.append(total_rewards_ep)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward


def frozenlake_q_learning():
    env_id = "FrozenLake-v1"
    env = gym.make(env_id, map_name="4x4", is_slippery=False, render_mode="rgb_array")
    print("_____OBSERVATION SPACE_____ \n")
    print("Observation Space", env.observation_space)
    print("Sample observation", env.observation_space.sample())
    print("\n _____ACTION SPACE_____ \n")
    print("Action Space Shape", env.action_space.n)
    print("Action Space Sample", env.action_space.sample())

    state_space = env.observation_space.n
    print("There are ", state_space, " possible states")

    action_space = env.action_space.n
    print("There are ", action_space, " possible actions")

    # Training parameters
    n_training_episodes = 10000  # Total training episodes
    learning_rate = 0.7  # Learning rate

    # Evaluation parameters
    n_eval_episodes = 100  # Total number of test episodes

    # Environment parameters
    max_steps = 99  # Max steps per episode
    gamma = 0.95  # Discounting rate

    # Exploration parameters
    max_epsilon = 1.0  # Exploration probability at start
    min_epsilon = 0.05  # Minimum exploration probability
    decay_rate = 0.0005  # Exponential decay rate for exploration prob

    Qtable_frozenlake = initialize_q_table(state_space, action_space)
    Qtable_frozenlake = train(
        n_training_episodes,
        min_epsilon,
        max_epsilon,
        decay_rate,
        env,
        max_steps,
        Qtable_frozenlake,
        gamma,
        learning_rate,
    )
    return env, max_steps, n_eval_episodes, Qtable_frozenlake


def taxi_q_learning():
    env_id = "Taxi-v3"
    env = gym.make(env_id, render_mode="rgb_array")
    print("_____OBSERVATION SPACE_____ \n")
    print("Observation Space", env.observation_space)
    print("Sample observation", env.observation_space.sample())
    print("\n _____ACTION SPACE_____ \n")
    print("Action Space Shape", env.action_space.n)
    print("Action Space Sample", env.action_space.sample())

    state_space = env.observation_space.n
    print("There are ", state_space, " possible states")

    action_space = env.action_space.n
    print("There are ", action_space, " possible actions")

    # Training parameters
    n_training_episodes = 25000  # Total training episodes
    learning_rate = 0.7  # Learning rate

    # Environment parameters
    max_steps = 99  # Max steps per episode
    gamma = 0.95  # Discounting rate

    # Exploration parameters
    max_epsilon = 1.0  # Exploration probability at start
    min_epsilon = 0.05  # Minimum exploration probability
    decay_rate = 0.005  # Exponential decay rate for exploration prob

    # Evaluation parameters
    n_eval_episodes = 100

    Qtable_taxi = initialize_q_table(state_space, action_space)
    print("Q-table taxi: ", Qtable_taxi.shape)
    np.set_printoptions(precision=2, suppress=True)
    print("Initial Q-table:")
    print(Qtable_taxi)

    Qtable_taxi = train(
        n_training_episodes,
        min_epsilon,
        max_epsilon,
        decay_rate,
        env,
        max_steps,
        Qtable_taxi,
        gamma,
        learning_rate,
    )
    print("Q-table taxi (after training): ", Qtable_taxi.shape)
    print(Qtable_taxi)
    return env, max_steps, n_eval_episodes, Qtable_taxi


def main():
    print("Starting the main function")
    # env, max_steps, n_eval_episodes, q_table = frozenlake_q_learning()
    env, max_steps, n_eval_episodes, q_table = taxi_q_learning()

    # Evaluate our Agent
    mean_reward, std_reward = evaluate_agent(env, max_steps, n_eval_episodes, q_table)
    print(f"Mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

    # Record a video now
    cur_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    video_path = cur_dir / "replay.mp4"
    video_fps = 1
    out_directory = record_video(env, q_table, video_path, video_fps)
    print("Video saved at: ", out_directory)


if __name__ == "__main__":
    main()
