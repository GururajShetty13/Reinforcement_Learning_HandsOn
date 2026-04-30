import time
from stable_baselines3 import SAC
from upright_inverted_pendulum import UprightInvertedPendulumEnv

env = UprightInvertedPendulumEnv()
model = SAC.load("sac_upright_inverted_pendulum")

obs, info = env.reset()

for step in range(300):
    action, _ = model.predict(obs, deterministic=True)
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