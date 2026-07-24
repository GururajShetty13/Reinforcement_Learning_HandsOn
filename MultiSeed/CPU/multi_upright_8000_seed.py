import gymnasium as gym
import numpy as np
from gymnasium import spaces

MAX_STEPS = 250
MAX_SPEED = 8.0
DT = 0.05
G = 10.0
M = 1.0
L = 1.0

ACTION_TORQUES = np.array(
    [-2.0, -0.5, -0.1, 0.0, 0.1, 0.5, 2.0],
    dtype=np.float32,
)


def angle_normalize(x: float) -> float:
    return ((x + np.pi) % (2.0 * np.pi)) - np.pi


def observation_from_state(theta: float, theta_dot: float) -> np.ndarray:
    return np.array(
        [np.cos(theta), np.sin(theta), theta_dot],
        dtype=np.float32,
    )


class UprightInvertedPendulumDQNFPGAEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        self.action_space = spaces.Discrete(len(ACTION_TORQUES))
        self.observation_space = spaces.Box(
            low=np.array([-1.0, -1.0, -MAX_SPEED], dtype=np.float32),
            high=np.array([1.0, 1.0, MAX_SPEED], dtype=np.float32),
            dtype=np.float32,
        )

        self.theta = np.pi
        self.theta_dot = 0.0
        self.step_count = 0
        self.prev_angle_error = abs(angle_normalize(self.theta))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}

        if "theta" in options and "theta_dot" in options:
            self.theta = float(options["theta"])
            self.theta_dot = float(options["theta_dot"])
        else:
            self.theta = float(np.pi + self.np_random.uniform(low=-0.1, high=0.1))
            self.theta_dot = float(self.np_random.uniform(low=-0.1, high=0.1))

        self.theta_dot = float(np.clip(self.theta_dot, -MAX_SPEED, MAX_SPEED))
        self.step_count = 0
        self.prev_angle_error = abs(angle_normalize(self.theta))

        return observation_from_state(self.theta, self.theta_dot), {}

    def step(self, action):
        action = int(action)
        torque = float(ACTION_TORQUES[action])

        theta_wrapped_before = angle_normalize(self.theta)
        abs_angle_error_before = abs(theta_wrapped_before)

        theta_ddot = (3.0 * G / (2.0 * L)) * np.sin(self.theta) + (
            3.0 / (M * L * L)
        ) * torque

        self.theta_dot = float(self.theta_dot + theta_ddot * DT)
        self.theta_dot = float(np.clip(self.theta_dot, -MAX_SPEED, MAX_SPEED))
        self.theta = float(self.theta + self.theta_dot * DT)

        theta_wrapped_after = angle_normalize(self.theta)
        abs_angle_error_after = abs(theta_wrapped_after)

        upright_reward = 2.0 * np.cos(theta_wrapped_after)
        velocity_penalty = 0.01 * (self.theta_dot**2)
        torque_penalty = 0.001 * (torque**2)
        progress_bonus = 0.5 * (abs_angle_error_before - abs_angle_error_after)

        reward = upright_reward - velocity_penalty - torque_penalty + progress_bonus

        if abs_angle_error_after < 0.30:
            reward += 2.0
        if abs_angle_error_after < 0.15 and abs(self.theta_dot) < 1.0:
            reward += 4.0
        if abs_angle_error_after < 0.08 and abs(self.theta_dot) < 0.5:
            reward += 6.0

        reward = float(np.clip(reward, -10.0, 10.0))

        self.step_count += 1
        terminated = False
        truncated = self.step_count >= MAX_STEPS

        info = {
            "theta_wrapped": theta_wrapped_after,
            "theta_unwrapped": self.theta,
            "theta_dot": self.theta_dot,
            "torque": torque,
        }

        return (
            observation_from_state(self.theta, self.theta_dot),
            reward,
            terminated,
            truncated,
            info,
        )
