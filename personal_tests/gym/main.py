from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import gymnasium as gym
import ale_py
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy

gym.register_envs(ale_py)


def display_video(frames):
    if not frames:
        print("No frames collected, cannot create animation.")
    else:
        print("Creating animation...")
        fig, ax = plt.subplots()
        ax.axis('off')  # Hide axes
        # Initialize the image plot with the first frame
        img = ax.imshow(frames[0])

        def update(frame_index):
            """Update the image data for each frame."""
            img.set_array(frames[frame_index])
            return img,

        # Calculate interval based on a desired FPS (e.g., 30 FPS)
        # interval = 1000 ms / FPS
        interval = 1000 / 30

        # Create the animation object
        output_path = Path("./data/animation.mp4")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=interval, blit=True)
        ani.save(output_path, writer='ffmpeg', fps=30)
        print(f"File stored in {output_path}")


def main():
    #env = gym.make('ALE/Breakout-v5', render_mode="rgb_array")
    env = make_atari_env('ALE/Breakout-v5', n_envs=16)
    env = VecFrameStack(env, n_stack=4)

    # Observation and action space
    obs_space = env.observation_space
    print(f"The observation space: {obs_space}")
    print("Upper Bound for Env Observation", env.observation_space.high)
    print("Lower Bound for Env Observation", env.observation_space.low)
    action_space = env.action_space
    print(f"The action space: {action_space}")

    obs = env.reset()
    print(f"The initial observation shape is {obs.shape}")

    model = PPO(
        policy='CnnPolicy',        # Correct for image input
        env=env,                   # Use the preprocessed & stacked env
        n_steps=128,               # Steps per env per update (common for Atari)
        batch_size=256,
        n_epochs=4,                # Optimization epochs per update
        gamma=0.99,                # Standard discount factor for Atari
        gae_lambda=0.95,           # Standard GAE factor
        clip_range=0.1,            # PPO clip range (often 0.1 for Atari)
        ent_coef=0.01,             # Entropy coefficient
        learning_rate=2.5e-4,      # Crucial: Often tuned, 2.5e-4 is a common start
                                   # Consider using a linear schedule: learning_rate=lambda f: f * 2.5e-4
        vf_coef=0.5,               # Value function coefficient (default)
        max_grad_norm=0.5,         # Gradient clipping (common practice)
        verbose=1
    )
    model.learn(total_timesteps=10_000_000, log_interval=10)
    model_name = "ppo-breakout"
    model.save(model_name)

    # Evaluate the agent
    eval_env = Monitor(gym.make("ALE/Breakout-v5", render_mode='rgb_array'))
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
    print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

    print("Generating video with trained agent...")
    frames = []
    num_steps_video = 1500  # How many steps to record

    # Use the evaluation environment for rendering
    obs, _ = eval_env.reset()
    for step in range(num_steps_video):
        # Predict the action using the trained model
        # Use deterministic=True for the "best" action according to the policy
        action, _states = model.predict(obs, deterministic=True)

        # Take the action in the environment
        observation, reward, terminated, truncated, info = eval_env.step(action)

        # Render the current frame
        # Make sure the eval_env was created with render_mode='rgb_array'
        frame = eval_env.render()
        if frame is not None:
            frames.append(frame)
        else:
            print(f"Warning: eval_env.render() returned None at step {step}")

        # Update the observation for the next iteration
        obs = observation

        # If the episode ends, reset the environment
        if terminated or truncated:
            print(f"Episode finished at step {step+1}. Resetting...")
            obs, _ = eval_env.reset()
            # Optional: You could break here if you only want one episode in the video
            # break

        # Optional small delay
        # time.sleep(0.01)

    eval_env.close() # Close the evaluation environment
    env.close()      # Close the training environment
    print(f"Simulation finished. Collected {len(frames)} frames.")
    display_video(frames)


if __name__ == "__main__":
    main()
