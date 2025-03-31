import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

environment_name = "CartPole-v1"
env = gym.make(environment_name)

# episodes = 5
# for episode in range(1, episodes+1):
#     state = env.reset()
#     done = False
#     score = 0

#     while not done:
#         # env.render()
#         action = env.action_space.sample()
#         n_state, reward, done, truncated, info = env.step(action)
#         score += reward
#     print(f"Episode: {episode}, Score: {score}")

# env.close()

save_path = os.path.join("Training", "Saved Models", "PPO_CartPole")

# Train the model
# log_path = os.path.join("Training", "Logs")
# env = DummyVecEnv([lambda: env])
# model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_path)

# model.learn(total_timesteps=200000)

# model.save(save_path)

model = PPO.load(save_path, env=env)


# Evaluate the model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean Reward: {mean_reward}, Std Reward: {std_reward}")


# test model
episodes = 5
for episode in range(1, episodes+1):
    obs, info = env.reset()
    done = False
    score = 0

    while not done:
        # env.render()
        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        done = done or truncated
        score += reward
    print(f"Episode: {episode}, Score: {score}")

env.close()