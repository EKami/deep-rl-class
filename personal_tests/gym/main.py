import gymnasium as gym
import matplotlib.pyplot as plt

env = gym.make('MountainCar-v0', render_mode="rgb_array")

# Observation and action space
obs_space = env.observation_space
action_space = env.action_space
print(f"The observation space: {obs_space}")
print(f"The action space: {action_space}")

# reset the environment and see the initial observation
obs, info = env.reset()
print(f"The initial observation is {obs}")

# Sample a random action from the entire action space
random_action = env.action_space.sample()

# # Take the action and get the new observation space
new_obs, reward, terminated, truncated, newInfo = env.step(random_action)
print(f"The new observation is {new_obs}")
rgb_array = env.render()
plt.imshow(rgb_array)
env.close()