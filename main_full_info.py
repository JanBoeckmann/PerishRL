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

order_quantity = 100
demand_mean = 40
demand_std = 10

advertising_increase_factor = 2

gain_per_sales = 10
gain_per_sale_in_advertisement = 4
loss_per_lost_sales = 5
loss_per_perished_product = 30

class PerishEnvFullInfo(Env):
    def __init__(self):
        super().__init__()
        self.action_space = Discrete(4) #0 = nothig, 1=reorder, 2=advertise, 3=reorder and advertise
        self.observation_space = Box(low=np.array([0, 0, 0]), high=np.array([100, 200, 300]), dtype=np.float32)
        self.state = [order_quantity, 0, 0] 
        self.number_of_periods = 100  

    def step(self, action):
        # action = tuple(action)  # Convert to tuple to ensure unpacking works correctly
        
        order, advertise = 0, 0  # Correctly unpack action

        if action in [1, 3]:
            order = 1
        if action in [2, 3]:
            advertise = 1

        # Your demand and inventory update logic
        demand = round(np.random.normal(demand_mean, demand_std))
        demand = max(demand, 0)  # Ensure demand is non-negative

        if advertise:
            demand = round(demand * advertising_increase_factor)

        original_demand = demand

        for s in range(3):
            if s < 2:
                reverse_index = -1 - s
            else:
                reverse_index = 0
            if self.state[reverse_index] > demand:
                self.state[reverse_index] -= demand
                demand = 0
            else:
                demand -= self.state[reverse_index]
                self.state[reverse_index] = 0

        self.number_of_periods -= 1

        actual_gain = gain_per_sales
        if advertise:
            actual_gain = gain_per_sale_in_advertisement

        reward = actual_gain * (original_demand - demand) - loss_per_perished_product * self.state[2]
        self.state = np.roll(self.state, 1)  # Shift all elements to the right
        self.state[0] = order * order_quantity  # Set the new first element 

        # print("demand:", original_demand, "state:", self.state)


        done = self.number_of_periods <= 0  
        truncated = False
        info = {}

        return np.array(self.state, dtype=np.float32), reward, done, truncated, info  # Ensure correct format

    def reset(self, seed=None, options=None):
        self.number_of_periods = 100  
        self.state = np.array([order_quantity, 0, 0], dtype=np.float32)
        observation = np.array(self.state, dtype=np.float32)  
        return observation, {}

# Instantiate Environment
env = PerishEnvFullInfo()
env = DummyVecEnv([lambda: env])

save_path = os.path.join("Training", "Saved Models", "PPO_perish_full_info")

# # Train Model
# model = PPO.load(save_path, env=env)
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./Training/Logs/PPO_perish_log")
model.learn(total_timesteps=1e7) 

# # # Save Model
model.save(save_path)

# # Load Model
# model = PPO.load(save_path, env=env)

# # Evaluate Model
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
