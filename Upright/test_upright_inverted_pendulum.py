import time
from upright_inverted_pendulum import UprightInvertedPendulumEnv

env = UprightInvertedPendulumEnv()
obs, info = env.reset()

print("Environment created successfully")
print("Action space:", env.action_space)
print("Observation space:", env.observation_space)
print("Initial observation:", obs)

for step in range(50):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    print(
        f"step={step}, "
        f"action={action[0]:.4f}, "
        f"reward={reward:.4f}, "
        f"cos(theta)={obs[0]:.4f}, "
        f"sin(theta)={obs[1]:.4f}, "
        f"theta_dot={obs[2]:.4f}, "
        f"angle_error={info['angle_error']:.4f}"
    )

    if terminated or truncated:
        print("Episode ended. Resetting.")
        obs, info = env.reset()

    time.sleep(0.03)

env.close()