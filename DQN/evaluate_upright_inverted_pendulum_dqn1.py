import numpy as np
from stable_baselines3 import DQN
from upright_inverted_pendulum_dqn import UprightInvertedPendulumDQNEnv

env = UprightInvertedPendulumDQNEnv()
model = DQN.load("dqn_upright_inverted_pendulum_optimized")

num_episodes = 20
results = []
success_counts = 0

for ep in range(num_episodes):
    obs, info = env.reset()
    total_reward = 0.0
    reached_upright = False
    stable_steps = 0

    for step in range(300):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        angle_error = abs(info["angle_error"])
        theta_dot = abs(info["theta_dot"])

        if angle_error < 0.2 and theta_dot < 0.5:
            stable_steps += 1
        else:
            stable_steps = 0

        if stable_steps >= 10:
            reached_upright = True

        if terminated or truncated:
            break

    if reached_upright:
        success_counts += 1

    results.append(total_reward)
    print(
        f"Episode {ep+1}: total_reward={total_reward:.3f}, "
        f"reached_upright={reached_upright}"
    )

print("\nSummary:")
print(f"Mean reward: {np.mean(results):.3f}")
print(f"Best reward: {np.max(results):.3f}")
print(f"Worst reward: {np.min(results):.3f}")
print(f"Reached upright in {success_counts}/{num_episodes} episodes")

env.close()