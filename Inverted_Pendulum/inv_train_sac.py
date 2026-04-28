import gymnasium as gym
from stable_baselines3 import SAC

env = gym.make("InvertedPendulum-v5")

model = SAC(
    "MlpPolicy",
    env,
    verbose=1,
    policy_kwargs=dict(net_arch=[256, 256])
)

model.learn(total_timesteps=100000)

model.save("sac_inverted_pendulum")

env.close()
print("Training complete. Model saved as sac_inverted_pendulum")