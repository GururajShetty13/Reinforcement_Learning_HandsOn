import time
from stable_baselines3 import DQN
from upright_inverted_pendulum_dqn import UprightInvertedPendulumDQNEnv

env = UprightInvertedPendulumDQNEnv(render_mode="human")
model = DQN.load("dqn_upright_inverted_pendulum_optimized")

obs, info = env.reset()

for step in range(300):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    print(
        f"step={step}, "
        f"action_index={int(action)}, "
        f"torque={info['torque']:.2f}, "
        f"reward={reward:.4f}, "
        f"angle_error={info['angle_error']:.4f}"
    )

    if terminated or truncated:
        print("Episode ended. Resetting.")
        obs, info = env.reset()

    time.sleep(0.03)

env.close()