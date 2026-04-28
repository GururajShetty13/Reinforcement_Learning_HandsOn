import gymnasium as gym
import time
from stable_baselines3 import PPO

env = gym.make("InvertedPendulum-v5", render_mode="human")
model = PPO.load("ppo_inverted_pendulum")

obs, info = env.reset()

for step in range(500):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    print(f"step={step}, action={action}, reward={reward}, obs={obs}")

    if terminated or truncated:
        print("Episode ended. Resetting.")
        obs, info = env.reset()

    time.sleep(0.02)

env.close()