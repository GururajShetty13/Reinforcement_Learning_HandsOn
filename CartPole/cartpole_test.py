import gymnasium as gym
import time

env = gym.make("CartPole-v1", render_mode="human")
obs, info = env.reset()

print("Initial observation:", obs)
print("Action space:", env.action_space)
print("Observation space:", env.observation_space)

episode_step = 0
episode_count = 1
total_reward = 0

for step in range(500):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    cart_position = obs[0]
    cart_velocity = obs[1]
    pole_angle = obs[2]
    pole_angular_velocity = obs[3]

    episode_step += 1
    total_reward += reward

    print(
        f"episode={episode_count}, step={episode_step}, action={action}, reward={reward:.1f}, "
        f"cart_pos={cart_position:.4f}, cart_vel={cart_velocity:.4f}, "
        f"pole_angle={pole_angle:.4f}, pole_ang_vel={pole_angular_velocity:.4f}"
    )

    if terminated or truncated:
        print("\nEpisode ended")
        print(f"Episode number: {episode_count}")
        print(f"Episode length: {episode_step}")
        print(f"Total reward: {total_reward:.1f}")
        print(f"Final cart position: {cart_position:.4f}")
        print(f"Final pole angle: {pole_angle:.4f}")
        print(f"terminated={terminated}, truncated={truncated}")
        print("-" * 60)

        obs, info = env.reset()
        episode_count += 1
        episode_step = 0
        total_reward = 0

    time.sleep(0.03)

env.close()