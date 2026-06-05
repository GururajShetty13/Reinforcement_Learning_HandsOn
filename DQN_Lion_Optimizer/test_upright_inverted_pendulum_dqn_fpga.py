import random
import numpy as np
import time
from upright_inverted_pendulum_dqn_fpga import UprightInvertedPendulumDQNFPGAEnv

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

env = UprightInvertedPendulumDQNFPGAEnv(render_mode="human")
obs, info = env.reset(seed=SEED)

print("=" * 60)
print("FPGA-FRIENDLY UPRIGHT INVERTED PENDULUM TEST")
print("=" * 60)
print("Action space:", env.action_space)
print("Observation space:", env.observation_space)
print("Discrete torque values:", env.discrete_actions)
print("Initial observation:", obs)
print()

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
        print("Episode ended. Resetting.\n")
        obs, info = env.reset(seed=SEED)

    time.sleep(0.03)

env.close()