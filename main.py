import gym
from gym import Env
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete

import numpy as np
import random
import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

from Environments.environments import PerishEnv, PerishEnvFullInfo

full_information = True

# Instantiate Environment
if full_information:
    env = PerishEnvFullInfo()
    save_path = os.path.join("Training", "Saved Models", "PPO_perish_full_information")
else:
    env = PerishEnv()
    save_path = os.path.join("Training", "Saved Models", "PPO_perish")

#for faster training
env = DummyVecEnv([lambda: env])

# # Train Model
# model = PPO.load(save_path, env=env)
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./Training/Logs/PPO_perish_log")
model.learn(total_timesteps=5e5) 

# # Save Model
model.save(save_path)

# # Load Model
# model = PPO.load(save_path, env=env)

# # Evaluate Model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=30)
print(f"Mean Reward: {mean_reward}, Std Reward: {std_reward}")

# Test Model
# episodes = 5
# for episode in range(1, episodes + 1):
#     obs = env.reset()
#     done = False
#     score = 0

#     while not done:
#         action, _ = model.predict(obs)
#         obs, reward, done, info = env.step(action)
#         score += reward
#         print(f"Action: {action}, Obs: {obs}, Reward: {reward}")

#     print(f"Episode: {episode}, Score: {score}")
