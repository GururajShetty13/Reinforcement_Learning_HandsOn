import time
from upright_inverted_pendulum import UprightInvertedPendulumEnv

env = UprightInvertedPendulumEnv(render_mode="human")
obs, info = env.reset()

for step in range(200):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    print(
        f"step={step}, action={action[0]:.4f}, reward={reward:.4f}, "
        f"angle_error={info['angle_error']:.4f}"
    )

    if terminated or truncated:
        obs, info = env.reset()

    time.sleep(0.03)

env.close()