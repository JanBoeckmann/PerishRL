from gym import Env
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete

import numpy as np

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

percentage_of_picking = 0.5

def update_state(action, state, number_of_periods):
    order, advertise = 0, 0

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

    demand_picking = round(percentage_of_picking * demand)
    demand_non_picking = demand - demand_picking

    # update inventory with non-picking
    for s in range(3):
        if s < 2:
            reverse_index = -1 - s
        else:
            reverse_index = 0
        if state[reverse_index] > demand_non_picking:
            state[reverse_index] -= demand_non_picking
            demand_non_picking = 0
        else:
            demand_non_picking -= state[reverse_index]
            state[reverse_index] = 0

    # update inventory with picking
    for s in range(3):
        if state[s] < demand_picking:
            demand_picking -= state[s]
            state[s] = 0
        else:
            state[s] -= demand_picking
            demand_picking = 0

    demand = demand_non_picking + demand_picking

    number_of_periods -= 1

    actual_gain = gain_per_sales
    if advertise:
        actual_gain = gain_per_sale_in_advertisement

    reward = actual_gain * (original_demand - demand) - loss_per_perished_product * state[2]
    state = np.roll(state, 1)  # Shift all elements to the right
    state[0] = order * order_quantity  # Set the new first element 

    # print("demand:", original_demand, "state:", self.state)


    done = number_of_periods <= 0  
    truncated = False
    info = {}

    return reward, done, truncated, info, state, number_of_periods



class PerishEnv(Env):
    def __init__(self):
        super().__init__()
        self.action_space = Discrete(4) #0 = nothig, 1=reorder, 2=advertise, 3=reorder and advertise
        self.observation_space = Box(low=np.array([0]), high=np.array([300]), dtype=np.float32)
        self.state = [order_quantity, 0, 0] 
        self.number_of_periods = 100  

    def step(self, action):
        reward, done, truncated, info, self.state, self.number_of_periods = update_state(action, self.state, self.number_of_periods)
        return np.array([sum(self.state)], dtype=np.float32), reward, done, truncated, info  # Ensure correct format

    def reset(self, seed=None, options=None):
        self.number_of_periods = 100  
        self.state = np.array([order_quantity, 0, 0], dtype=np.float32)
        observation = np.array([self.state.sum()], dtype=np.float32)  
        return observation, {}

class PerishEnvFullInfo(Env):
    def __init__(self):
        super().__init__()
        self.action_space = Discrete(4) #0 = nothig, 1=reorder, 2=advertise, 3=reorder and advertise
        self.observation_space = Box(low=np.array([0, 0, 0]), high=np.array([100, 200, 300]), dtype=np.float32)
        self.state = [order_quantity, 0, 0] 
        self.number_of_periods = 100  

    def step(self, action):
        reward, done, truncated, info, self.state, self.number_of_periods = update_state(action, self.state, self.number_of_periods)
        return np.array(self.state, dtype=np.float32), reward, done, truncated, info  # Ensure correct format

    def reset(self, seed=None, options=None):
        self.number_of_periods = 100  
        self.state = np.array([order_quantity, 0, 0], dtype=np.float32)
        observation = np.array(self.state, dtype=np.float32)  
        return observation, {}