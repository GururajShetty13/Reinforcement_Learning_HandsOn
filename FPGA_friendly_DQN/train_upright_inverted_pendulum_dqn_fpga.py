import random
import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from upright_inverted_pendulum_dqn_fpga import UprightInvertedPendulumDQNFPGAEnv

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

env = Monitor(UprightInvertedPendulumDQNFPGAEnv())

model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=5e-4,
    buffer_size=10000,
    learning_starts=1000,
    batch_size=64,
    gamma=0.99,
    train_freq=4,
    gradient_steps=1,
    target_update_interval=1000,
    exploration_fraction=0.30,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,
    policy_kwargs=dict(net_arch=[32]),
    seed=SEED,
)

model.learn(total_timesteps=800000)
model.save("dqn_upright_inverted_pendulum_fpga")

env.close()
print("Training complete.")
print("Saved model as: dqn_upright_inverted_pendulum_fpga")