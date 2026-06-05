import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt


class UprightInvertedPendulumDQNFPGAEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()

        self.max_speed = 8.0
        self.dt = 0.05
        self.g = 10.0
        self.m = 1.0
        self.l = 1.0
        self.max_steps = 200

        self.render_mode = render_mode
        self.step_count = 0
        self.state = None

        # 7-action discrete torque set
        self.discrete_actions = np.array(
            [-2.0, -0.5, -0.1, 0.0, 0.1, 0.5, 2.0],
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(len(self.discrete_actions))

        self.observation_space = spaces.Box(
            low=np.array([-1.0, -1.0, -self.max_speed], dtype=np.float32),
            high=np.array([1.0, 1.0, self.max_speed], dtype=np.float32),
            dtype=np.float32,
        )

        self.fig = None
        self.ax = None
        self.line = None
        self.mass = None

    def angle_normalize(self, x):
        return ((x + np.pi) % (2 * np.pi)) - np.pi

    def _get_obs(self):
        theta, theta_dot = self.state
        return np.array([np.cos(theta), np.sin(theta), theta_dot], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        theta = np.pi + self.np_random.uniform(low=-0.1, high=0.1)
        theta_dot = self.np_random.uniform(low=-0.1, high=0.1)

        self.state = np.array([theta, theta_dot], dtype=np.float64)
        self.step_count = 0

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), {}

    def step(self, action):
        theta, theta_dot = self.state
        u = float(self.discrete_actions[int(action)])

        prev_angle_error = self.angle_normalize(theta)

        new_theta_dot = theta_dot + (
            3.0 * self.g / (2.0 * self.l) * np.sin(theta)
            + 3.0 / (self.m * self.l**2) * u
        ) * self.dt

        new_theta_dot = np.clip(new_theta_dot, -self.max_speed, self.max_speed)
        new_theta = theta + new_theta_dot * self.dt

        self.state = np.array([new_theta, new_theta_dot], dtype=np.float64)
        self.step_count += 1

        angle_error = self.angle_normalize(new_theta)

        # Previous reward system that worked better
        upright_reward = 2.0 * np.cos(angle_error)
        velocity_penalty = 0.01 * (new_theta_dot ** 2)
        torque_penalty = 0.001 * (u ** 2)
        progress_bonus = 0.5 * (abs(prev_angle_error) - abs(angle_error))

        bonus = 0.0
        if abs(angle_error) < 0.3:
            bonus += 2.0
        if abs(angle_error) < 0.15 and abs(new_theta_dot) < 1.0:
            bonus += 4.0
        if abs(angle_error) < 0.08 and abs(new_theta_dot) < 0.5:
            bonus += 6.0

        reward = (
            upright_reward
            - velocity_penalty
            - torque_penalty
            + progress_bonus
            + bonus
        )

        reward = float(np.clip(reward, -10.0, 10.0))

        terminated = False
        truncated = self.step_count >= self.max_steps

        info = {
            "theta": float(new_theta),
            "theta_dot": float(new_theta_dot),
            "angle_error": float(angle_error),
            "torque": float(u),
        }

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        theta, _ = self.state

        x = self.l * np.sin(theta)
        y = self.l * np.cos(theta)

        if self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(5, 5))
            self.ax.set_xlim(-1.2, 1.2)
            self.ax.set_ylim(-1.2, 1.2)
            self.ax.set_aspect("equal")
            self.ax.grid(True)
            self.ax.set_title("Upright Inverted Pendulum - DQN FPGA")

            self.line, = self.ax.plot([0, x], [0, y], lw=3)
            self.mass, = self.ax.plot(x, y, "o", markersize=12)
        else:
            self.line.set_data([0, x], [0, y])
            self.mass.set_data([x], [y])

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def close(self):
        if self.fig is not None:
            plt.ioff()
            plt.close(self.fig)
            self.fig = None