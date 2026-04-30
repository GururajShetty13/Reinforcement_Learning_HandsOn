from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from upright_inverted_pendulum import UprightInvertedPendulumEnv

env = Monitor(UprightInvertedPendulumEnv())

model = SAC(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    buffer_size=100000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    train_freq=1,
    gradient_steps=1,
    learning_starts=1000,
    policy_kwargs=dict(net_arch=[256, 256]),
)

model.learn(total_timesteps=500000)
model.save("sac_upright_inverted_pendulum")

env.close()
print("Training complete. Model saved as sac_upright_inverted_pendulum")