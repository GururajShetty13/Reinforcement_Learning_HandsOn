import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from upright_8000_seed import UprightInvertedPendulumDQNFPGAEnv

MULTISEED_OUTPUT_DIR = "python_dqn_multiseed_results_10"
HARDWARE_RESULTS_DIR = (
    "hardware_dqn_results"  # optional; not used during multi-seed training
)
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

MAX_EPISODES = 8000
MAX_STEPS = 250
TOTAL_TIMESTEPS = MAX_EPISODES * MAX_STEPS
VALIDATION_INTERVAL = 800

STATE_SIZE = 3
ACTION_TORQUES = np.array([-2.0, -0.5, -0.1, 0.0, 0.1, 0.5, 2.0], dtype=np.float32)

MAX_SPEED = 8.0
DT = 0.05
G = 10.0
M = 1.0
L = 1.0

UPRIGHT_THRESHOLD = 0.35
STABLE_THETA_THRESHOLD = 0.35
STABLE_THETA_DOT_THRESHOLD = 0.5
SUCCESS_CONSECUTIVE_STABLE_STEPS_THRESHOLD = 30

VALIDATION_INITIAL_STATE = {"theta": float(np.pi), "theta_dot": 0.0}

FINAL_EVALUATION_INITIAL_STATES = [
    {"theta": float(np.pi), "theta_dot": 0.0},
    {"theta": float(np.pi + 0.20), "theta_dot": -2.0},
    {"theta": 1.67, "theta_dot": 2.0},
    {"theta": 1.67, "theta_dot": -1.0},
    {"theta": float(np.pi + 1.67), "theta_dot": 0.0},
]


def angle_normalize(x: float) -> float:
    return ((x + np.pi) % (2.0 * np.pi)) - np.pi


def observation_from_state(theta: float, theta_dot: float) -> np.ndarray:
    return np.array([np.cos(theta), np.sin(theta), theta_dot], dtype=np.float32)


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def binary_metrics_from_stable_steps(
    stable_steps: int, total_steps: int
) -> Dict[str, float]:

    tp = int(stable_steps)
    fn = int(total_steps - stable_steps)
    fp = 0
    tn = 0

    accuracy = safe_div(tp + tn, tp + tn + fp + fn)
    precision = safe_div(tp, tp + fp) if tp > 0 else 0.0
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2.0 * precision * recall, precision + recall)

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


def max_consecutive_true(values) -> int:
    """Return the longest continuous run of True values."""
    max_count = 0
    current_count = 0
    for value in values:
        if bool(value):
            current_count += 1
            max_count = max(max_count, current_count)
        else:
            current_count = 0
    return int(max_count)


@dataclass
class RolloutResult:
    summary: Dict[str, float]
    trajectories: List[Dict[str, float]]
    elapsed_seconds: float


def run_policy_rollout(
    model: DQN,
    initial_state: Dict[str, float],
    batch_id: int,
    label_prefix: str,
    training_episode: int = -1,
    validation_index: int = -1,
) -> RolloutResult:
    env = UprightInvertedPendulumDQNFPGAEnv()
    obs, _ = env.reset(options=initial_state)

    total_reward = 0.0
    min_abs_theta = 999.0
    upright_steps = 0
    stable_steps = 0
    stable_flags_for_success = []
    trajectories = []

    actual_initial_theta = float(env.theta)
    actual_initial_theta_dot = float(env.theta_dot)

    start_time = time.perf_counter()

    for step in range(MAX_STEPS):
        state_cos = float(obs[0])
        state_sin = float(obs[1])
        state_theta_dot = float(obs[2])
        theta_wrapped = float(math.atan2(state_sin, state_cos))

        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        torque = float(ACTION_TORQUES[action])

        next_obs, reward, terminated, truncated, info = env.step(action)

        next_theta_wrapped = float(math.atan2(float(next_obs[1]), float(next_obs[0])))
        next_theta_dot = float(next_obs[2])

        total_reward += float(reward)
        min_abs_theta = min(min_abs_theta, abs(next_theta_wrapped))

        is_upright = abs(next_theta_wrapped) < UPRIGHT_THRESHOLD
        is_stable = (
            abs(next_theta_wrapped) < STABLE_THETA_THRESHOLD
            and abs(next_theta_dot) < STABLE_THETA_DOT_THRESHOLD
        )

        if is_upright:
            upright_steps += 1
        if is_stable:
            stable_steps += 1
        stable_flags_for_success.append(is_stable)

        row = {
            "step": step,
            "state_cos_theta": state_cos,
            "state_sin_theta": state_sin,
            "state_theta_dot": state_theta_dot,
            "theta_wrapped": theta_wrapped,
            "action": action,
            "torque": torque,
            "reward": float(reward),
            "next_state_cos_theta": float(next_obs[0]),
            "next_state_sin_theta": float(next_obs[1]),
            "next_state_theta_dot": float(next_obs[2]),
            "next_theta_wrapped": next_theta_wrapped,
            "upright": int(is_upright),
            "stable_upright": int(is_stable),
        }

        if label_prefix == "validation":
            row.update(
                {
                    "validation_index": validation_index,
                    "training_episode": training_episode,
                }
            )
        else:
            row.update(
                {
                    "evaluation_batch": batch_id,
                }
            )

        trajectories.append(row)
        obs = next_obs

        if terminated or truncated:
            break

    elapsed_seconds = time.perf_counter() - start_time

    max_consecutive_stable_steps = max_consecutive_true(stable_flags_for_success)
    reached_stable_upright = stable_steps > 0
    metrics = binary_metrics_from_stable_steps(stable_steps, MAX_STEPS)

    summary = {
        "total_reward": total_reward,
        "average_reward_per_step": total_reward / MAX_STEPS,
        "min_abs_theta": min_abs_theta,
        "upright_steps": upright_steps,
        "stable_upright_steps": stable_steps,
        "upright_accuracy": upright_steps / MAX_STEPS,
        "stable_accuracy": stable_steps / MAX_STEPS,
        "max_consecutive_stable_steps": max_consecutive_stable_steps,
        "reached_stable_upright": int(reached_stable_upright),
        "success": int(
            max_consecutive_stable_steps >= SUCCESS_CONSECUTIVE_STABLE_STEPS_THRESHOLD
        ),
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
        "tp": metrics["tp"],
        "fp": metrics["fp"],
        "tn": metrics["tn"],
        "fn": metrics["fn"],
        "inference_time_seconds": elapsed_seconds,
        "inference_steps": MAX_STEPS,
        "inference_steps_per_second": MAX_STEPS / elapsed_seconds
        if elapsed_seconds > 0
        else 0.0,
        "inference_latency_ms_per_step": 1000.0 * elapsed_seconds / MAX_STEPS
        if MAX_STEPS > 0
        else 0.0,
        "initial_theta": float(initial_state["theta"]),
        "initial_theta_dot": float(initial_state["theta_dot"]),
        "actual_initial_theta": actual_initial_theta,
        "actual_initial_theta_dot": actual_initial_theta_dot,
        "final_theta_wrapped": float(angle_normalize(env.theta)),
        "final_theta_unwrapped": float(env.theta),
        "final_theta_dot": float(env.theta_dot),
    }

    if label_prefix == "validation":
        summary.update(
            {
                "validation_index": validation_index,
                "training_episode": training_episode,
            }
        )
    else:
        summary.update(
            {
                "evaluation_batch": batch_id,
            }
        )

    return RolloutResult(
        summary=summary, trajectories=trajectories, elapsed_seconds=elapsed_seconds
    )


class TrainingValidationCallback(BaseCallback):
    def __init__(self, output_dir: str, seed: int):
        super().__init__()
        self.output_dir = output_dir
        self.seed = seed
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
        self.episode_number = 0
        self.training_rows = []
        self.validation_summary_rows = []
        self.validation_trajectory_rows = []

    def _on_step(self) -> bool:
        reward = float(self.locals["rewards"][0])
        done = bool(self.locals["dones"][0])

        self.current_episode_reward += reward
        self.current_episode_length += 1

        if done:
            self.episode_number += 1

            # SB3 DQN logs train/loss. This is a loss proxy, not exactly absolute TD error.
            loss_value = self.model.logger.name_to_value.get("train/loss", np.nan)
            epsilon_value = float(getattr(self.model, "exploration_rate", np.nan))

            self.training_rows.append(
                {
                    "seed": self.seed,
                    "episode": self.episode_number,
                    "episode_length": self.current_episode_length,
                    "total_reward": self.current_episode_reward,
                    "average_reward_per_step": self.current_episode_reward
                    / max(1, self.current_episode_length),
                    "average_abs_td_error": loss_value,
                    "epsilon": epsilon_value,
                    "timesteps": self.num_timesteps,
                }
            )

            if self.episode_number % VALIDATION_INTERVAL == 0:
                validation_index = (self.episode_number // VALIDATION_INTERVAL) - 1
                result = run_policy_rollout(
                    model=self.model,
                    initial_state=VALIDATION_INITIAL_STATE,
                    batch_id=validation_index,
                    label_prefix="validation",
                    training_episode=self.episode_number,
                    validation_index=validation_index,
                )
                result.summary["seed"] = self.seed
                for row in result.trajectories:
                    row["seed"] = self.seed

                self.validation_summary_rows.append(result.summary)
                self.validation_trajectory_rows.extend(result.trajectories)

                print(
                    f"Validation at episode {self.episode_number}: "
                    f"reward={result.summary['total_reward']:.2f}, "
                    f"stable_steps={result.summary['stable_upright_steps']}, "
                    f"max_consecutive_stable={result.summary['max_consecutive_stable_steps']}, "
                    f"success={result.summary['success']}, "
                    f"f1={result.summary['f1_score']:.4f}",
                    flush=True,
                )

            self.current_episode_reward = 0.0
            self.current_episode_length = 0

        return self.episode_number < MAX_EPISODES

    def save_validation_files(self):
        pd.DataFrame(self.training_rows).to_csv(
            os.path.join(self.output_dir, "training_metrics.csv"), index=False
        )
        pd.DataFrame(self.validation_summary_rows).to_csv(
            os.path.join(self.output_dir, "validation_summary.csv"), index=False
        )
        pd.DataFrame(self.validation_trajectory_rows).to_csv(
            os.path.join(self.output_dir, "validation_trajectories.csv"), index=False
        )


def compute_hardware_metrics(hardware_dir: str) -> Dict[str, float]:

    eval_summary_path = os.path.join(hardware_dir, "evaluation_summary.csv")
    eval_traj_path = os.path.join(hardware_dir, "evaluation_trajectories.csv")
    training_path = os.path.join(hardware_dir, "training_metrics.csv")

    if (
        not os.path.exists(eval_summary_path)
        or not os.path.exists(eval_traj_path)
        or not os.path.exists(training_path)
    ):
        return {}

    eval_summary_df = pd.read_csv(eval_summary_path)
    eval_traj_df = pd.read_csv(eval_traj_path)
    train_df = pd.read_csv(training_path)

    batch_rows = []
    for batch_id, batch_df in eval_traj_df.groupby("evaluation_batch"):
        batch_df = batch_df.sort_values("step")

        # Use next-state values when available, because the recorded upright/stable flags
        # are evaluated after applying the selected action.
        theta_col = (
            "next_theta_wrapped"
            if "next_theta_wrapped" in batch_df.columns
            else "theta_wrapped"
        )
        theta_dot_col = (
            "next_state_theta_dot"
            if "next_state_theta_dot" in batch_df.columns
            else "state_theta_dot"
        )

        upright_flags = batch_df[theta_col].abs() < UPRIGHT_THRESHOLD
        stable_flags = (batch_df[theta_col].abs() < STABLE_THETA_THRESHOLD) & (
            batch_df[theta_dot_col].abs() < STABLE_THETA_DOT_THRESHOLD
        )

        stable_steps = int(stable_flags.sum())
        upright_steps = int(upright_flags.sum())
        max_consec = max_consecutive_true(stable_flags.tolist())

        total_reward = (
            float(batch_df["reward"].sum())
            if "reward" in batch_df.columns
            else float(
                eval_summary_df.loc[
                    eval_summary_df["evaluation_batch"] == batch_id, "total_reward"
                ].iloc[0]
            )
        )

        batch_rows.append(
            {
                "evaluation_batch": int(batch_id),
                "total_reward": total_reward,
                "upright_steps": upright_steps,
                "stable_upright_steps": stable_steps,
                "upright_accuracy": upright_steps / MAX_STEPS,
                "stable_accuracy": stable_steps / MAX_STEPS,
                "max_consecutive_stable_steps": max_consec,
                "reached_stable_upright": int(stable_steps > 0),
                "success": int(
                    max_consec >= SUCCESS_CONSECUTIVE_STABLE_STEPS_THRESHOLD
                ),
            }
        )

    batch_metrics_df = pd.DataFrame(batch_rows)
    total_stable_steps = int(batch_metrics_df["stable_upright_steps"].sum())
    total_eval_steps = int(len(batch_metrics_df) * MAX_STEPS)
    metrics = binary_metrics_from_stable_steps(total_stable_steps, total_eval_steps)

    hw = {
        "source": "hardware_board_csv",
        "training_time_seconds": np.nan,  # enter board terminal execution time manually
        "training_episodes": int(train_df["episode"].max())
        if "episode" in train_df.columns
        else len(train_df),
        "training_steps": int(train_df["timesteps"].max())
        if "timesteps" in train_df.columns
        else int(len(train_df) * MAX_STEPS),
        "training_episodes_per_second": np.nan,
        "training_steps_per_second": np.nan,
        "training_time_ms_per_episode": np.nan,
        "training_time_us_per_step": np.nan,
        "evaluation_batches": int(len(batch_metrics_df)),
        "evaluation_steps": int(total_eval_steps),
        "inference_time_seconds": np.nan,
        "inference_steps_per_second": np.nan,
        "inference_latency_ms_per_step": np.nan,
        "average_evaluation_reward": float(batch_metrics_df["total_reward"].mean()),
        "average_upright_accuracy": float(batch_metrics_df["upright_accuracy"].mean()),
        "average_stable_accuracy": float(batch_metrics_df["stable_accuracy"].mean()),
        "average_max_consecutive_stable_steps": float(
            batch_metrics_df["max_consecutive_stable_steps"].mean()
        ),
        "success_rate": float(batch_metrics_df["success"].mean()),
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
        "tp": metrics["tp"],
        "fp": metrics["fp"],
        "tn": metrics["tn"],
        "fn": metrics["fn"],
        "stable_theta_threshold": STABLE_THETA_THRESHOLD,
        "stable_theta_dot_threshold": STABLE_THETA_DOT_THRESHOLD,
        "success_consecutive_stable_steps_threshold": SUCCESS_CONSECUTIVE_STABLE_STEPS_THRESHOLD,
    }

    batch_metrics_df.to_csv(
        os.path.join(hardware_dir, "hardware_evaluation_batch_metrics_strict.csv"),
        index=False,
    )
    return hw


def run_single_seed(seed: int) -> Dict[str, float]:
    """Run one complete training, validation, and evaluation experiment."""
    seed_output_dir = os.path.join(MULTISEED_OUTPUT_DIR, f"seed_{seed}")
    os.makedirs(seed_output_dir, exist_ok=True)

    np.random.seed(seed)

    print("\n========================================")
    print(f"Starting Python DQN reference training | seed={seed}")
    print("========================================")
    print(f"Output directory: {seed_output_dir}")
    print(f"Episodes: {MAX_EPISODES}")
    print(f"Steps per episode: {MAX_STEPS}")
    print(f"Total timesteps: {TOTAL_TIMESTEPS}")
    print(f"Validation interval: {VALIDATION_INTERVAL}")

    env = DummyVecEnv([lambda: Monitor(UprightInvertedPendulumDQNFPGAEnv())])
    env.seed(seed)

    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=0.0005,
        buffer_size=10000,
        learning_starts=1000,
        batch_size=64,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.30,
        exploration_final_eps=0.05,
        policy_kwargs=dict(net_arch=[32]),
        verbose=0,
        seed=seed,
    )

    callback = TrainingValidationCallback(seed_output_dir, seed)

    train_start = time.perf_counter()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)
    train_elapsed = time.perf_counter() - train_start

    callback.save_validation_files()
    model.save(os.path.join(seed_output_dir, "dqn_adam_python_model"))

    print("Training complete.")
    print(f"Seed {seed} training time: {train_elapsed:.6f} seconds")

    # Final evaluation and pure inference timing
    evaluation_summary_rows = []
    evaluation_trajectory_rows = []
    total_inference_time = 0.0

    print("Running final evaluation / inference timing...")

    for batch_id, initial_state in enumerate(FINAL_EVALUATION_INITIAL_STATES):
        result = run_policy_rollout(
            model=model,
            initial_state=initial_state,
            batch_id=batch_id,
            label_prefix="evaluation",
        )

        result.summary["seed"] = seed
        for row in result.trajectories:
            row["seed"] = seed

        evaluation_summary_rows.append(result.summary)
        evaluation_trajectory_rows.extend(result.trajectories)
        total_inference_time += result.elapsed_seconds

        print(
            f"Seed {seed} | Evaluation batch {batch_id}: "
            f"reward={result.summary['total_reward']:.2f}, "
            f"stable_steps={result.summary['stable_upright_steps']}, "
            f"max_consecutive_stable={result.summary['max_consecutive_stable_steps']}, "
            f"success={result.summary['success']}, "
            f"accuracy={result.summary['accuracy']:.4f}, "
            f"f1={result.summary['f1_score']:.4f}, "
            f"inference_time={result.elapsed_seconds:.6f}s",
            flush=True,
        )

    evaluation_summary_df = pd.DataFrame(evaluation_summary_rows)
    evaluation_trajectory_df = pd.DataFrame(evaluation_trajectory_rows)

    evaluation_summary_df.to_csv(
        os.path.join(seed_output_dir, "evaluation_summary.csv"), index=False
    )
    evaluation_trajectory_df.to_csv(
        os.path.join(seed_output_dir, "evaluation_trajectories.csv"), index=False
    )

    # Aggregate Python performance metrics for this seed
    total_stable_steps = int(evaluation_summary_df["stable_upright_steps"].sum())
    total_eval_steps = int(len(evaluation_summary_df) * MAX_STEPS)
    aggregate_metrics = binary_metrics_from_stable_steps(
        total_stable_steps, total_eval_steps
    )

    python_performance = {
        "seed": seed,
        "source": "python_reference",
        "training_time_seconds": train_elapsed,
        "training_episodes": MAX_EPISODES,
        "training_steps": TOTAL_TIMESTEPS,
        "training_episodes_per_second": MAX_EPISODES / train_elapsed
        if train_elapsed > 0
        else 0.0,
        "training_steps_per_second": TOTAL_TIMESTEPS / train_elapsed
        if train_elapsed > 0
        else 0.0,
        "training_time_ms_per_episode": 1000.0 * train_elapsed / MAX_EPISODES,
        "training_time_us_per_step": 1e6 * train_elapsed / TOTAL_TIMESTEPS,
        "evaluation_batches": len(evaluation_summary_df),
        "evaluation_steps": total_eval_steps,
        "inference_time_seconds": total_inference_time,
        "inference_steps_per_second": total_eval_steps / total_inference_time
        if total_inference_time > 0
        else 0.0,
        "inference_latency_ms_per_step": 1000.0
        * total_inference_time
        / total_eval_steps
        if total_eval_steps > 0
        else 0.0,
        "average_evaluation_reward": float(
            evaluation_summary_df["total_reward"].mean()
        ),
        "average_upright_accuracy": float(
            evaluation_summary_df["upright_accuracy"].mean()
        ),
        "average_stable_accuracy": float(
            evaluation_summary_df["stable_accuracy"].mean()
        ),
        "average_max_consecutive_stable_steps": float(
            evaluation_summary_df["max_consecutive_stable_steps"].mean()
        ),
        "success_rate": float(evaluation_summary_df["success"].mean()),
        "accuracy": aggregate_metrics["accuracy"],
        "precision": aggregate_metrics["precision"],
        "recall": aggregate_metrics["recall"],
        "f1_score": aggregate_metrics["f1_score"],
        "tp": aggregate_metrics["tp"],
        "fp": aggregate_metrics["fp"],
        "tn": aggregate_metrics["tn"],
        "fn": aggregate_metrics["fn"],
        "stable_theta_threshold": STABLE_THETA_THRESHOLD,
        "stable_theta_dot_threshold": STABLE_THETA_DOT_THRESHOLD,
        "success_consecutive_stable_steps_threshold": SUCCESS_CONSECUTIVE_STABLE_STEPS_THRESHOLD,
    }

    performance_df = pd.DataFrame([python_performance])
    performance_df.to_csv(
        os.path.join(seed_output_dir, "python_performance_metrics.csv"), index=False
    )

    print(f"\nFinal Python performance metrics for seed {seed}:")
    for key, value in python_performance.items():
        print(f"{key}: {value}")

    print("\nFiles created:")
    for name in [
        "training_metrics.csv",
        "validation_summary.csv",
        "validation_trajectories.csv",
        "evaluation_summary.csv",
        "evaluation_trajectories.csv",
        "python_performance_metrics.csv",
    ]:
        print(os.path.join(seed_output_dir, name))

    return python_performance


def write_multiseed_statistics(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Create mean/std/min/max table across seeds."""
    metric_columns = [
        "training_time_seconds",
        "training_episodes_per_second",
        "training_steps_per_second",
        "training_time_ms_per_episode",
        "training_time_us_per_step",
        "inference_time_seconds",
        "inference_steps_per_second",
        "inference_latency_ms_per_step",
        "average_evaluation_reward",
        "average_upright_accuracy",
        "average_stable_accuracy",
        "average_max_consecutive_stable_steps",
        "success_rate",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "tp",
        "fn",
    ]

    rows = []
    for metric in metric_columns:
        values = pd.to_numeric(summary_df[metric], errors="coerce")
        rows.append(
            {
                "metric": metric,
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values.dropna()) > 1 else 0.0,
                "min": float(values.min()),
                "max": float(values.max()),
            }
        )

    return pd.DataFrame(rows)


def main():
    os.makedirs(MULTISEED_OUTPUT_DIR, exist_ok=True)

    all_seed_results = []

    for seed in SEEDS:
        result = run_single_seed(seed)
        all_seed_results.append(result)

    summary_df = pd.DataFrame(all_seed_results)
    summary_path = os.path.join(MULTISEED_OUTPUT_DIR, "multiseed_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    mean_std_df = write_multiseed_statistics(summary_df)
    mean_std_path = os.path.join(MULTISEED_OUTPUT_DIR, "multiseed_mean_std.csv")
    mean_std_df.to_csv(mean_std_path, index=False)

    print("\n========================================")
    print("MULTI-SEED SUMMARY")
    print("========================================")
    print(summary_df.to_string(index=False))

    print("\n========================================")
    print("MULTI-SEED MEAN ± STD")
    print("========================================")
    print(mean_std_df.to_string(index=False))

    print("\nMulti-seed files created:")
    print(summary_path)
    print(mean_std_path)


if __name__ == "__main__":
    main()
