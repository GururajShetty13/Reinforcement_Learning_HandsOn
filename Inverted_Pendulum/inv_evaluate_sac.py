import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC

env = gym.make("InvertedPendulum-v5")
model = SAC.load("sac_inverted_pendulum")

num_episodes = 20
results = []

for ep in range(num_episodes):
    obs, info = env.reset()
    total_reward = 0

    for step in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            break

    results.append(total_reward)
    print(f"Episode {ep+1}: total_reward={total_reward}")

print("\nSummary:")
print(f"Mean reward: {np.mean(results):.3f}")
print(f"Best reward: {np.max(results):.3f}")
print(f"Worst reward: {np.min(results):.3f}")

env.close()