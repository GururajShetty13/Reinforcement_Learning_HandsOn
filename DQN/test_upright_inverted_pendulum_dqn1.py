import time
from upright_inverted_pendulum_dqn import UprightInvertedPendulumDQNEnv

env = UprightInvertedPendulumDQNEnv(render_mode="human")
obs, info = env.reset()

print("Environment created successfully")
print("Action space:", env.action_space)
print("Observation space:", env.observation_space)
print("Discrete torque values:", env.discrete_actions)
print("Initial observation:", obs)

for step in range(60):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    print(
        f"step={step}, "
        f"action_index={action}, "
        f"torque={info['torque']:.2f}, "
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