import gymnasium as gym
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

def make_env():
    return gym.make("Pendulum-v1")

env = DummyVecEnv([make_env])
env = VecNormalize.load("pendulum_td3_vecnormalize.pkl", env)

env.training = False
env.norm_reward = False

model = TD3.load("td3_pendulum_normalized", env=env)

num_episodes = 20
results = []

for ep in range(num_episodes):
    obs = env.reset()
    total_reward = 0

    for step in range(200):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward[0]

        if done[0]:
            break

    results.append(total_reward)
    print(f"Episode {ep+1}: total_reward={total_reward:.3f}")

print("\nSummary:")
print(f"Mean reward: {np.mean(results):.3f}")
print(f"Best reward: {np.max(results):.3f}")
print(f"Worst reward: {np.min(results):.3f}")

env.close()