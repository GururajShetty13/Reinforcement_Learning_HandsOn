from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from upright_inverted_pendulum_dqn import UprightInvertedPendulumDQNEnv

env = Monitor(UprightInvertedPendulumDQNEnv())

model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=5e-4,
    buffer_size=100000,
    learning_starts=5000,
    batch_size=128,
    gamma=0.99,
    train_freq=4,
    gradient_steps=1,
    target_update_interval=2000,
    exploration_fraction=0.4,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.02,
    policy_kwargs=dict(net_arch=[256, 256]),
)

model.learn(total_timesteps=500000)
model.save("dqn_upright_inverted_pendulum_optimized")

env.close()
print("Training complete. Model saved as dqn_upright_inverted_pendulum_optimized")