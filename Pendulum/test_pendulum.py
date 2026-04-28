import gymnasium as gym
import time

env = gym.make("Pendulum-v1", render_mode="human")
obs, info = env.reset()

print("Initial observation:", obs)
print("Action space:", env.action_space)
print("Observation space:", env.observation_space)

for step in range(200):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    print(f"step={step}, action={action}, reward={reward:.3f}, obs={obs}")

    if terminated or truncated:
        print("Episode ended. Resetting.")
        obs, info = env.reset()

    time.sleep(0.03)

env.close()