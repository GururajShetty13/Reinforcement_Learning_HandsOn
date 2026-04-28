import gymnasium as gym
from stable_baselines3 import PPO

def run_random_episode(env, max_steps=500):
    obs, info = env.reset()
    total_reward = 0

    for step in range(max_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            return step + 1, total_reward

    return max_steps, total_reward


def run_trained_episode(env, model, max_steps=500):
    obs, info = env.reset()
    total_reward = 0

    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            return step + 1, total_reward

    return max_steps, total_reward


env = gym.make("CartPole-v1")
model = PPO.load("ppo_cartpole")

num_episodes = 10

print("Random policy results:")
for i in range(num_episodes):
    length, reward = run_random_episode(env)
    print(f"Episode {i+1}: length={length}, total_reward={reward}")

print("\nTrained policy results:")
for i in range(num_episodes):
    length, reward = run_trained_episode(env, model)
    print(f"Episode {i+1}: length={length}, total_reward={reward}")

env.close()