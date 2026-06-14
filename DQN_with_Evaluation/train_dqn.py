import csv
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from upright_inverted_pendulum_dqn_fpga_rectified import (
    UprightInvertedPendulumDQNFPGAEnv,
)


SEED = 42

TOTAL_TIMESTEPS = 800_000
MAX_STEPS_PER_EPISODE = 200

N_EPISODES = TOTAL_TIMESTEPS // MAX_STEPS_PER_EPISODE

VALIDATION_INTERVAL_EPISODES = 400

OUTPUT_DIRECTORY = Path("dqn_adam_results")

VALIDATION_INITIAL_STATE = {
    "theta": float(np.pi),
    "theta_dot": 0.0,
}

FINAL_EVALUATION_INITIAL_STATES = [
    {"theta": float(np.pi), "theta_dot": 0.0},
    {"theta": float(np.pi + 0.20), "theta_dot": -2.0},
    {"theta": 1.67, "theta_dot": 2.0},
    {"theta": 1.67, "theta_dot": -1.0},
    {"theta": float(np.pi + 1.67), "theta_dot": 0.0},
]


def normalize_angle(theta: float) -> float:
    return float(
        ((theta + np.pi) % (2.0 * np.pi)) - np.pi
    )


class DQNWithTDTracking(DQN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.td_error_sum_since_episode = 0.0
        self.td_error_count_since_episode = 0

    def consume_episode_td_error(self) -> float:
        if self.td_error_count_since_episode == 0:
            return 0.0

        result = (
            self.td_error_sum_since_episode
            / self.td_error_count_since_episode
        )

        self.td_error_sum_since_episode = 0.0
        self.td_error_count_since_episode = 0

        return float(result)

    def train(
        self,
        gradient_steps: int,
        batch_size: int = 100,
    ) -> None:
        self.policy.set_training_mode(True)

        self._update_learning_rate(
            self.policy.optimizer
        )

        losses = []
        mean_abs_td_errors = []

        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(
                batch_size,
                env=self._vec_normalize_env,
            )

            with torch.no_grad():
                next_q_values = self.q_net_target(
                    replay_data.next_observations
                )

                next_q_values = (
                    next_q_values
                    .max(dim=1)
                    .values
                    .reshape(-1, 1)
                )

                target_q_values = (
                    replay_data.rewards
                    + (1.0 - replay_data.dones)
                    * self.gamma
                    * next_q_values
                )

            current_q_values = self.q_net(
                replay_data.observations
            )

            current_q_values = torch.gather(
                current_q_values,
                dim=1,
                index=replay_data.actions.long(),
            )

            td_error = target_q_values - current_q_values

            mean_abs_td_error = float(
                td_error.abs().mean().item()
            )

            loss = F.smooth_l1_loss(
                current_q_values,
                target_q_values,
            )

            self.policy.optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(),
                self.max_grad_norm,
            )

            self.policy.optimizer.step()

            losses.append(float(loss.item()))
            mean_abs_td_errors.append(
                mean_abs_td_error
            )

        self._n_updates += gradient_steps

        mean_loss = (
            float(np.mean(losses))
            if losses
            else 0.0
        )

        mean_abs_td_error = (
            float(np.mean(mean_abs_td_errors))
            if mean_abs_td_errors
            else 0.0
        )

        self.td_error_sum_since_episode += (
            mean_abs_td_error
        )

        self.td_error_count_since_episode += 1

        self.logger.record(
            "train/n_updates",
            self._n_updates,
            exclude="tensorboard",
        )

        self.logger.record(
            "train/loss",
            mean_loss,
        )

        self.logger.record(
            "train/mean_abs_td_error",
            mean_abs_td_error,
        )


def write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[Dict],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def run_fixed_episode(
    model: DQN,
    initial_state: Dict[str, float],
) -> Tuple[float, List[Dict]]:
    env = UprightInvertedPendulumDQNFPGAEnv()

    observation, reset_info = env.reset(
        seed=SEED,
        options={
            "theta": float(initial_state["theta"]),
            "theta_dot": float(initial_state["theta_dot"]),
        },
    )

    expected_theta = float(initial_state["theta"])
    expected_theta_dot = float(initial_state["theta_dot"])

    actual_theta = float(env.state[0])
    actual_theta_dot = float(env.state[1])

    if not np.isclose(
        actual_theta,
        expected_theta,
        atol=1e-12,
    ):
        raise RuntimeError(
            "Fixed reset failed for theta: "
            f"expected {expected_theta}, "
            f"got {actual_theta}"
        )

    if not np.isclose(
        actual_theta_dot,
        expected_theta_dot,
        atol=1e-12,
    ):
        raise RuntimeError(
            "Fixed reset failed for theta_dot: "
            f"expected {expected_theta_dot}, "
            f"got {actual_theta_dot}"
        )

    if not reset_info.get("fixed_reset_used", False):
        raise RuntimeError(
            "Environment did not confirm fixed reset."
        )

    trajectory = []
    total_reward = 0.0

    for step_index in range(env.max_steps):
        theta_unwrapped = float(env.state[0])
        theta_wrapped = normalize_angle(
            theta_unwrapped
        )
        theta_dot = float(env.state[1])

        action, _ = model.predict(
            observation,
            deterministic=True,
        )

        action = int(action)

        (
            next_observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        next_theta_unwrapped = float(
            info["theta_unwrapped"]
        )

        next_theta_wrapped = float(
            info["theta_wrapped"]
        )

        next_theta_dot = float(
            info["theta_dot"]
        )

        trajectory.append(
            {
                "step": step_index,

                "state_cos_theta":
                    float(observation[0]),

                "state_sin_theta":
                    float(observation[1]),

                "state_theta_dot":
                    float(observation[2]),

                "theta_wrapped":
                    theta_wrapped,

                "theta_unwrapped":
                    theta_unwrapped,

                "theta_dot":
                    theta_dot,

                "action":
                    action,

                "torque":
                    float(info["torque"]),

                "reward":
                    float(reward),

                "next_state_cos_theta":
                    float(next_observation[0]),

                "next_state_sin_theta":
                    float(next_observation[1]),

                "next_state_theta_dot":
                    float(next_observation[2]),

                "next_theta_wrapped":
                    next_theta_wrapped,

                "next_theta_unwrapped":
                    next_theta_unwrapped,

                "next_theta_dot":
                    next_theta_dot,

                "terminated":
                    int(terminated),

                "truncated":
                    int(truncated),
            }
        )

        total_reward += float(reward)
        observation = next_observation

        if terminated or truncated:
            break

    env.close()

    return float(total_reward), trajectory


def summarize_trajectory(
    trajectory: List[Dict],
) -> Dict[str, float]:
    if not trajectory:
        raise ValueError(
            "Cannot summarize an empty trajectory."
        )

    minimum_abs_theta = min(
        abs(float(row["theta_wrapped"]))
        for row in trajectory
    )

    upright_steps = sum(
        1
        for row in trajectory
        if abs(float(row["theta_wrapped"])) < 0.35
    )

    stable_upright_steps = sum(
        1
        for row in trajectory
        if (
            abs(float(row["theta_wrapped"])) < 0.35
            and abs(float(row["theta_dot"])) < 1.0
        )
    )

    final_theta_wrapped = float(
        trajectory[-1]["next_theta_wrapped"]
    )

    final_theta_unwrapped = float(
        trajectory[-1]["next_theta_unwrapped"]
    )

    final_theta_dot = float(
        trajectory[-1]["next_theta_dot"]
    )

    return {
        "minimum_abs_theta":
            float(minimum_abs_theta),

        "upright_steps":
            int(upright_steps),

        "stable_upright_steps":
            int(stable_upright_steps),

        "final_theta_wrapped":
            final_theta_wrapped,

        "final_theta_unwrapped":
            final_theta_unwrapped,

        "final_theta_dot":
            final_theta_dot,
    }


class TrainingValidationCallback(BaseCallback):
    def __init__(
        self,
        validation_interval_episodes: int,
        output_directory: Path,
        verbose: int = 1,
    ):
        super().__init__(verbose)

        self.validation_interval_episodes = (
            validation_interval_episodes
        )

        self.output_directory = (
            output_directory
        )

        self.episode_number = 0

        self.training_rows = []
        self.validation_summary_rows = []
        self.validation_trajectory_rows = []

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")

        if dones is None or infos is None:
            return True

        for done, info in zip(dones, infos):
            if not done:
                continue

            episode_info = info.get("episode")

            if episode_info is None:
                continue

            self.episode_number += 1

            total_episode_reward = float(
                episode_info["r"]
            )

            episode_length = int(
                episode_info["l"]
            )

            average_reward_per_step = (
                total_episode_reward
                / max(episode_length, 1)
            )

            average_absolute_td_error = (
                self.model.consume_episode_td_error()
            )

            epsilon = float(
                self.model.exploration_rate
            )

            self.training_rows.append(
                {
                    "episode":
                        self.episode_number,

                    "episode_length":
                        episode_length,

                    "total_reward":
                        total_episode_reward,

                    "average_reward_per_step":
                        average_reward_per_step,

                    "average_absolute_td_error":
                        average_absolute_td_error,

                    "epsilon":
                        epsilon,

                    "timesteps":
                        int(self.num_timesteps),
                }
            )

            if (
                self.verbose > 0
                and (
                    self.episode_number == 1
                    or self.episode_number % 10 == 0
                )
            ):
                print(
                    f"Episode "
                    f"{self.episode_number}/"
                    f"{N_EPISODES} | "
                    f"avg reward/step="
                    f"{average_reward_per_step:.4f} | "
                    f"avg |TD|="
                    f"{average_absolute_td_error:.4f} | "
                    f"epsilon={epsilon:.4f}"
                )

            if (
                self.validation_interval_episodes > 0
                and self.episode_number
                % self.validation_interval_episodes
                == 0
            ):
                self._run_validation()

        return True

    def _run_validation(self) -> None:
        validation_index = len(
            self.validation_summary_rows
        )

        total_reward, trajectory = (
            run_fixed_episode(
                self.model,
                VALIDATION_INITIAL_STATE,
            )
        )

        summary = summarize_trajectory(
            trajectory
        )

        self.validation_summary_rows.append(
            {
                "validation_index":
                    validation_index,

                "training_episode":
                    self.episode_number,

                "initial_theta":
                    VALIDATION_INITIAL_STATE[
                        "theta"
                    ],

                "initial_theta_dot":
                    VALIDATION_INITIAL_STATE[
                        "theta_dot"
                    ],

                "total_reward":
                    total_reward,

                **summary,
            }
        )

        for row in trajectory:
            self.validation_trajectory_rows.append(
                {
                    "validation_index":
                        validation_index,

                    "training_episode":
                        self.episode_number,

                    "validation_total_reward":
                        total_reward,

                    "initial_theta":
                        VALIDATION_INITIAL_STATE[
                            "theta"
                        ],

                    "initial_theta_dot":
                        VALIDATION_INITIAL_STATE[
                            "theta_dot"
                        ],

                    **row,
                }
            )

        print(
            f"  Validation "
            f"{validation_index} at episode "
            f"{self.episode_number}: "
            f"reward={total_reward:.4f}, "
            f"min|theta|="
            f"{summary['minimum_abs_theta']:.6f}, "
            f"upright="
            f"{summary['upright_steps']}, "
            f"stable="
            f"{summary['stable_upright_steps']}"
        )

        self._save_logs()

    def _save_logs(self) -> None:
        write_csv(
            self.output_directory
            / "training_metrics.csv",

            [
                "episode",
                "episode_length",
                "total_reward",
                "average_reward_per_step",
                "average_absolute_td_error",
                "epsilon",
                "timesteps",
            ],

            self.training_rows,
        )

        write_csv(
            self.output_directory
            / "validation_summary.csv",

            [
                "validation_index",
                "training_episode",
                "initial_theta",
                "initial_theta_dot",
                "total_reward",
                "minimum_abs_theta",
                "upright_steps",
                "stable_upright_steps",
                "final_theta_wrapped",
                "final_theta_unwrapped",
                "final_theta_dot",
            ],

            self.validation_summary_rows,
        )

        write_csv(
            self.output_directory
            / "validation_trajectories.csv",

            [
                "validation_index",
                "training_episode",
                "validation_total_reward",
                "initial_theta",
                "initial_theta_dot",
                "step",
                "state_cos_theta",
                "state_sin_theta",
                "state_theta_dot",
                "theta_wrapped",
                "theta_unwrapped",
                "theta_dot",
                "action",
                "torque",
                "reward",
                "next_state_cos_theta",
                "next_state_sin_theta",
                "next_state_theta_dot",
                "next_theta_wrapped",
                "next_theta_unwrapped",
                "next_theta_dot",
                "terminated",
                "truncated",
            ],

            self.validation_trajectory_rows,
        )

    def _on_training_end(self) -> None:
        self._save_logs()


def run_final_evaluation(
    model: DQN,
    output_directory: Path,
) -> None:
    summary_rows = []
    trajectory_rows = []

    for batch_index, initial_state in enumerate(
        FINAL_EVALUATION_INITIAL_STATES
    ):
        total_reward, trajectory = (
            run_fixed_episode(
                model,
                initial_state,
            )
        )

        summary = summarize_trajectory(
            trajectory
        )

        actual_initial_theta = float(
            trajectory[0]["theta_unwrapped"]
        )

        actual_initial_theta_dot = float(
            trajectory[0]["theta_dot"]
        )

        if not np.isclose(
            actual_initial_theta,
            float(initial_state["theta"]),
            atol=1e-12,
        ):
            raise RuntimeError(
                f"Evaluation batch {batch_index} "
                "started from the wrong theta."
            )

        if not np.isclose(
            actual_initial_theta_dot,
            float(initial_state["theta_dot"]),
            atol=1e-12,
        ):
            raise RuntimeError(
                f"Evaluation batch {batch_index} "
                "started from the wrong theta_dot."
            )

        summary_rows.append(
            {
                "evaluation_batch":
                    batch_index,

                "initial_theta":
                    float(initial_state["theta"]),

                "initial_theta_dot":
                    float(initial_state["theta_dot"]),

                "actual_initial_theta":
                    actual_initial_theta,

                "actual_initial_theta_dot":
                    actual_initial_theta_dot,

                "total_reward":
                    total_reward,

                **summary,
            }
        )

        for row in trajectory:
            trajectory_rows.append(
                {
                    "evaluation_batch":
                        batch_index,

                    "evaluation_total_reward":
                        total_reward,

                    "initial_theta":
                        float(initial_state["theta"]),

                    "initial_theta_dot":
                        float(initial_state["theta_dot"]),

                    **row,
                }
            )

        print(
            f"Evaluation batch "
            f"{batch_index}: "
            f"initial=("
            f"{actual_initial_theta:.6f}, "
            f"{actual_initial_theta_dot:.6f}), "
            f"reward={total_reward:.4f}, "
            f"min|theta|="
            f"{summary['minimum_abs_theta']:.6f}, "
            f"upright="
            f"{summary['upright_steps']}, "
            f"stable="
            f"{summary['stable_upright_steps']}"
        )

    write_csv(
        output_directory
        / "evaluation_summary.csv",

        [
            "evaluation_batch",
            "initial_theta",
            "initial_theta_dot",
            "actual_initial_theta",
            "actual_initial_theta_dot",
            "total_reward",
            "minimum_abs_theta",
            "upright_steps",
            "stable_upright_steps",
            "final_theta_wrapped",
            "final_theta_unwrapped",
            "final_theta_dot",
        ],

        summary_rows,
    )

    write_csv(
        output_directory
        / "evaluation_trajectories.csv",

        [
            "evaluation_batch",
            "evaluation_total_reward",
            "initial_theta",
            "initial_theta_dot",
            "step",
            "state_cos_theta",
            "state_sin_theta",
            "state_theta_dot",
            "theta_wrapped",
            "theta_unwrapped",
            "theta_dot",
            "action",
            "torque",
            "reward",
            "next_state_cos_theta",
            "next_state_sin_theta",
            "next_state_theta_dot",
            "next_theta_wrapped",
            "next_theta_unwrapped",
            "next_theta_dot",
            "terminated",
            "truncated",
        ],

        trajectory_rows,
    )


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    training_env = Monitor(
        UprightInvertedPendulumDQNFPGAEnv()
    )

    callback = TrainingValidationCallback(
        validation_interval_episodes=(
            VALIDATION_INTERVAL_EPISODES
        ),
        output_directory=OUTPUT_DIRECTORY,
        verbose=1,
    )

    model = DQNWithTDTracking(
        "MlpPolicy",
        training_env,
        verbose=1,

        learning_rate=5e-4,

        buffer_size=10_000,
        learning_starts=1_000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1_000,

        exploration_fraction=0.30,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,

        policy_kwargs={
            "net_arch": [32],

            "optimizer_class":
                torch.optim.Adam,

            "optimizer_kwargs": {
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "weight_decay": 0.0,
            },
        },

        seed=SEED,
        device="auto",
    )

    print("=" * 72)
    print("DQN TRAINING WITH ADAM")
    print("=" * 72)
    print(
        f"Total timesteps: "
        f"{TOTAL_TIMESTEPS}"
    )
    print(
        f"Estimated episodes: "
        f"{N_EPISODES}"
    )
    print(
        "Validation every "
        f"{VALIDATION_INTERVAL_EPISODES} "
        "episodes"
    )
    print(
        f"Output directory: "
        f"{OUTPUT_DIRECTORY.resolve()}"
    )
    print("=" * 72)

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callback,
    )

    model_path = (
        OUTPUT_DIRECTORY
        / "dqn_upright_inverted_pendulum_adam"
    )

    model.save(model_path)

    print("=" * 72)
    print("FINAL SEPARATE EVALUATION")
    print("=" * 72)

    run_final_evaluation(
        model,
        OUTPUT_DIRECTORY,
    )

    training_env.close()

    print("=" * 72)
    print("Training complete.")
    print(
        f"Model saved as: "
        f"{model_path}"
    )
    print("Generated files:")

    for filename in [
        "training_metrics.csv",
        "validation_summary.csv",
        "validation_trajectories.csv",
        "evaluation_summary.csv",
        "evaluation_trajectories.csv",
    ]:
        print(
            f"  "
            f"{OUTPUT_DIRECTORY / filename}"
        )

    print("=" * 72)


if __name__ == "__main__":
    main()
