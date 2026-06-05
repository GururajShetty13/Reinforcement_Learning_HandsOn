import random
import numpy as np
import torch
from stable_baselines3 import DQN
from upright_inverted_pendulum_dqn_fpga import UprightInvertedPendulumDQNFPGAEnv

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

env = UprightInvertedPendulumDQNFPGAEnv()
model = DQN.load("dqn_upright_inverted_pendulum_fpga")

num_episodes = 20
results = []
success_count = 0

for ep in range(num_episodes):
    obs, info = env.reset(seed=SEED + ep)
    total_reward = 0.0
    reached_upright = False
    stable_steps = 0
    max_stable_steps = 0

    for step in range(200):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        angle_error = abs(info["angle_error"])
        theta_dot = abs(info["theta_dot"])

        # Upright stability condition
        if angle_error < 0.10 and theta_dot < 0.50:
            stable_steps += 1
        else:
            stable_steps = 0

        if stable_steps > max_stable_steps:
            max_stable_steps = stable_steps

        if stable_steps >= 10:
            reached_upright = True

        if terminated or truncated:
            break

    if reached_upright:
        success_count += 1

    results.append(total_reward)

    print(
        f"Episode {ep+1}: "
        f"total_reward={total_reward:.3f}, "
        f"reached_upright={reached_upright}, "
        f"max_stable_steps={max_stable_steps}"
    )

print("\nSummary:")
print(f"Mean reward: {np.mean(results):.3f}")
print(f"Best reward: {np.max(results):.3f}")
print(f"Worst reward: {np.min(results):.3f}")
print(f"Reached upright in {success_count}/{num_episodes} episodes")

env.close()