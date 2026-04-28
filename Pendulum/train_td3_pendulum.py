import gymnasium as gym
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

def make_env():
    return gym.make("Pendulum-v1")

env = DummyVecEnv([make_env])
env = VecNormalize(env, norm_obs=True, norm_reward=True)

model = TD3(
    "MlpPolicy",
    env,
    verbose=1,
    policy_kwargs=dict(net_arch=[256, 256])
)

model.learn(total_timesteps=300000)

model.save("td3_pendulum_normalized")
env.save("pendulum_td3_vecnormalize.pkl")

env.close()
print("Training complete. TD3 model and normalization stats saved.")