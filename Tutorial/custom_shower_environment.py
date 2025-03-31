import gym
from gym import Env
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete

import numpy as np
import random
import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

#types of spaces
# print(Discrete(3).sample())

class ShowerEnv(Env):
    def __init__(self):
        super().__init__()
        self.action_space = Discrete(3)  # 0 = decrease, 1 = maintain, 2 = increase
        self.observation_space = Box(low=np.array([0]), high=np.array([100]), dtype=np.float32)
        self.state = 38 + random.randint(-3, 3)
        self.shower_length = 60

    def step(self, action):
        self.state += action - 1  
        self.shower_length -= 1

        # Reward function: Higher reward for being closer to 38
        reward = -abs(self.state - 38)

        done = self.shower_length <= 0 
        truncated = False
        info = {}

        return np.array([self.state]), reward, done, truncated, info

    def reset(self, seed=None, options=None):
        self.shower_length = 60
        self.state = np.array([38 + random.randint(-3, 3)]).astype(float)
        info = {}
        return self.state, info

    def render(self):
        pass

# Instantiate Environment
env = ShowerEnv()
env = DummyVecEnv([lambda: env])

save_path = os.path.join("Training", "Saved Models", "PPO_Shower")

# # Train Model
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./Training/Logs/PPO_Shower")
model.learn(total_timesteps=500000) 

# Save Model
model.save(save_path)

# Load Model
model = PPO.load(save_path, env=env)

# Evaluate Model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean Reward: {mean_reward}, Std Reward: {std_reward}")

# Test Model
episodes = 5
for episode in range(1, episodes + 1):
    obs = env.reset()
    done = False
    score = 0

    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        score += reward
        print(f"Action: {action}, Obs: {obs}, Reward: {reward}")

    print(f"Episode: {episode}, Score: {score}")
