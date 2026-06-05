import random
import numpy as np
import torch
import time
from stable_baselines3 import DQN
from upright_inverted_pendulum_dqn_fpga import UprightInvertedPendulumDQNFPGAEnv

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

env = UprightInvertedPendulumDQNFPGAEnv(render_mode="human")
model = DQN.load("dqn_upright_inverted_pendulum_fpga")

obs, info = env.reset(seed=SEED)

print("=" * 60)
print("PLAYING TRAINED FPGA-FRIENDLY DQN POLICY")
print("=" * 60)

for step in range(200):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    print(
        f"step={step}, "
        f"action_index={int(action)}, "
        f"torque={info['torque']:.2f}, "
        f"reward={reward:.4f}, "
        f"angle_error={info['angle_error']:.4f}, "
        f"theta_dot={info['theta_dot']:.4f}"
    )

    if terminated or truncated:
        print("Episode ended. Resetting.\n")
        obs, info = env.reset(seed=SEED)

    time.sleep(0.03)

env.close()