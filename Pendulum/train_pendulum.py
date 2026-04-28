import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("Pendulum-v1")

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

model.save("ppo_pendulum")

env.close()
print("Training complete. Model saved as ppo_pendulum")