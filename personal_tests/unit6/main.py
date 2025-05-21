import os

import gymnasium as gym

# Do not remove this import
import panda_gym as panda_gym
from pathlib import Path

from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
import torch
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import numpy as np
import pickle  # or torch.load if saved as a PyTorch file


def train(cur_dir, model, env):
    model_file = str(cur_dir / "a2c-PandaReachDense-v3")
    env_file = str(cur_dir / "vec_normalize.pkl")
    best_model_file = str(cur_dir / "best_model" / "best_model")

    if not os.path.exists(model_file) and not os.path.exists(env_file):
        # Create evaluation environment
        eval_env = DummyVecEnv([lambda: gym.make("PandaReachDense-v3")])
        eval_env = VecNormalize(eval_env, training=False, norm_reward=False)

        # Create callbacks
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(cur_dir / "best_model"),
            log_path=str(cur_dir / "logs"),
            eval_freq=10000,
            deterministic=True,
            render=False,
        )

        checkpoint_callback = CheckpointCallback(
            save_freq=50000,
            save_path=str(cur_dir / "checkpoints"),
            name_prefix="a2c_panda",
        )

        print("Training the model...")
        # Train with callbacks
        model.learn(
            total_timesteps=2_000_000,
            callback=[eval_callback, checkpoint_callback],
            progress_bar=True,
        )

        # Save the model and VecNormalize statistics
        model.save(model_file)
        env.save(env_file)

    # Return the best model file path if it exists, otherwise return the regular model file
    if os.path.exists(best_model_file):
        return best_model_file, env_file
    return model_file, env_file


def evaluate(model_file, env_file):
    # Load the saved statistics
    eval_env = DummyVecEnv([lambda: gym.make("PandaReachDense-v3")])
    eval_env = VecNormalize.load(env_file, eval_env)

    # We need to override the render_mode
    eval_env.render_mode = "rgb_array"

    #  do not update them at test time
    eval_env.training = False
    # reward normalization is not needed at test time
    eval_env.norm_reward = False

    # Load the agent
    model = A2C.load(model_file)

    # Run multiple evaluation episodes
    n_eval_episodes = 10
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=n_eval_episodes, deterministic=True
    )

    print(f"Evaluation over {n_eval_episodes} episodes:")
    print(f"Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")

    # Additional metrics
    success_count = 0
    total_steps = 0

    for _ in range(n_eval_episodes):
        obs = eval_env.reset()
        done = False
        episode_steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            episode_steps += 1

            if info[0].get("is_success", False):
                success_count += 1

        total_steps += episode_steps

    success_rate = success_count / n_eval_episodes * 100
    avg_steps = total_steps / n_eval_episodes

    print(f"Success rate: {success_rate:.1f}%")
    print(f"Average steps per episode: {avg_steps:.1f}")


def record_video(model_file, env_file, video_folder="videos"):
    """
    Record a video of the trained model in its environment.

    Args:
        model_file: Path to the saved model file
        env_file: Path to the VecNormalize statistics file
        video_folder: Folder where the video will be saved
    """

    os.makedirs(video_folder, exist_ok=True)

    # Create a single environment for recording
    env = gym.make("PandaReachDense-v3", render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        video_folder,
        name_prefix="panda_reach",
        disable_logger=True,
    )

    # Wrap in DummyVecEnv before applying VecNormalize
    env = DummyVecEnv([lambda: env])
    env = VecNormalize.load(env_file, env)
    env.training = False
    env.norm_reward = False

    # Load the model
    model = A2C.load(model_file)

    obs = env.reset()
    done = False
    total_reward = 0
    steps = 0
    success_count = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward[0]  # Get reward from first (and only) env
        steps += 1

        if info[0].get("is_success", False):  # Get info from first (and only) env
            success_count += 1
            print(f"Task succeeded at step {steps}!")

    print(f"Episode finished with reward {total_reward:.2f} in {steps} steps")
    print(f"Number of successes: {success_count}")

    env.close()
    print(f"Video saved in {video_folder}")


def main():
    cur_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    env_id = "PandaReachDense-v3"

    # Create the env
    env = gym.make(env_id)

    # Get the state space and action space
    s_size = env.observation_space.shape
    a_size = env.action_space

    print("_____OBSERVATION SPACE_____ \n")
    print("The State Space is: ", s_size)
    print(
        "Sample observation", env.observation_space.sample()
    )  # Get a random observation

    print("\n _____ACTION SPACE_____ \n")
    print("The Action Space is: ", a_size)
    print("Action Space Sample", env.action_space.sample())  # Take a random action

    env = make_vec_env(env_id, n_envs=16)

    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = A2C(
        policy="MultiInputPolicy",
        env=env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        rms_prop_eps=1e-5,
        use_rms_prop=True,
        use_sde=False,
        sde_sample_freq=-1,
        normalize_advantage=True,
        device=device,
        policy_kwargs=dict(
            net_arch=[dict(pi=[256, 256], vf=[256, 256])],
            activation_fn=torch.nn.ReLU,
        ),
    )

    best_model_file, env_file = train(cur_dir, model, env)

    evaluate(best_model_file, env_file)

    video_folder = str(cur_dir / "videos")
    record_video(best_model_file, env_file, video_folder)


if __name__ == "__main__":
    main()
