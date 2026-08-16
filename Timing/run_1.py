import csv
import gc
import sys
import time
from pathlib import Path

import numpy as np
from pynq import Overlay, allocate

# =============================================================================
# User configuration
# =============================================================================
BIT_PATH = "/home/xilinx/pynq/overlays/dqn_eva/dqn_eva.bit"
IP_NAME = "dqn_eva_0"

SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
FPGA_CLOCK_HZ = 100_000_000.0  # 100 MHz. Change if your HLS clock is different.
RUN_NN_INTERNAL_PROFILE = True

# Read small validation/evaluation arrays in the same run.
# If your board still gives a bus error during array reading, set this False and
# use timing only. The timing/cycle part will still work.
READ_RESULT_ARRAYS = True

# Use non-cacheable buffers and direct read by default. This avoids many PYNQ
# cache-sync problems on Zynq boards.
ALLOCATE_CACHEABLE = True
READ_WITH_SYNC_FROM_DEVICE = True

# HLS constants. Keep equal to the C++ HLS code.
STATE_SIZE = 3
MAX_EPISODES = 8000
MAX_STEPS = 250
BATCH_SIZE = 32  # Informational only. HLS must also be changed to BATCH_SIZE 32 and resynthesized.
VALIDATION_INTERVAL = 800
NUM_VALIDATIONS = MAX_EPISODES // VALIDATION_INTERVAL
EVALUATION_BATCH_SIZE = 5
INFERENCE_PROFILE_STEPS = 1000

VALIDATION_TRAJECTORY_SIZE = NUM_VALIDATIONS * MAX_STEPS
VALIDATION_STATE_BUFFER_SIZE = NUM_VALIDATIONS * MAX_STEPS * STATE_SIZE
EVALUATION_TRAJECTORY_SIZE = EVALUATION_BATCH_SIZE * MAX_STEPS
EVALUATION_STATE_BUFFER_SIZE = EVALUATION_BATCH_SIZE * MAX_STEPS * STATE_SIZE

# Padding avoids PYNQ/buffer bus errors if the PL writes slightly beyond a small array
# or if a buffer is not cache-line/page aligned on this platform. Only the first
# NUM_VALIDATIONS or EVALUATION_BATCH_SIZE values are read back.
SMALL_RESULT_PAD = 1024
VALIDATION_TRAJECTORY_PAD = 4096
EVALUATION_TRAJECTORY_PAD = 4096
VALIDATION_STATE_PAD = 8192
EVALUATION_STATE_PAD = 8192

# Output files
TXT_LOG_PATH = "combined_batch32_timing_cycles_results_multiseed_log.txt"
ALL_TIMING_CSV_PATH = "combined_batch32_timing_cycles_all_rows.csv"
SEED_SUMMARY_CSV_PATH = "combined_batch32_seed_summary.csv"
NN_SUMMARY_CSV_PATH = "combined_batch32_nn_summary.csv"
VALIDATION_CSV_PATH = "combined_batch32_validation_results.csv"
EVALUATION_CSV_PATH = "combined_batch32_evaluation_results.csv"
METRICS_CSV_PATH = "combined_batch32_hardware_performance_matrix.csv"
TRAINING_CSV_PATH = "combined_batch32_training_results.csv"
TRAINING_PROGRESS_METRICS_CSV_PATH = "combined_batch32_training_progress_f1_matrix.csv"

WORKFLOW_MODES = [
    (4, "Training + validation", 1),
    (1, "Final evaluation only", EVALUATION_BATCH_SIZE),
    (2, "One validation episode", 1),
    (3, f"Inference profile x{INFERENCE_PROFILE_STEPS}", INFERENCE_PROFILE_STEPS),
]

NN_PROFILE_MODES = [
    (6,  "forward layer 1: input -> hidden1", 10000),
    (7,  "forward layer 2: hidden1 -> hidden2", 10000),
    (8,  "forward output: hidden2 -> Q-values", 10000),
    (9,  "full forward pass + argmax", 10000),
    (10, "backprop: output -> hidden2", 10000),
    (11, "backprop: hidden2 -> hidden1", 10000),
    (12, "Adam update: W3 and B3", 10),
    (13, "Adam update: W2 and B2", 10),
    (14, "Adam update: W1 and B1", 10),
    (15, "complete train_one_sample_adam", 10),
    (16, "replay store + load", 10000),
]


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def alloc(shape, dtype):
    try:
        return allocate(shape=shape, dtype=dtype, cacheable=ALLOCATE_CACHEABLE)
    except TypeError:
        # Older PYNQ versions may not accept cacheable.
        return allocate(shape=shape, dtype=dtype)


def allocate_all_buffers():
    """Allocate all buffers required by the HLS top function."""
    bufs = {}

    # Training outputs
    bufs["training_episode"] = alloc((MAX_EPISODES,), np.int32)
    bufs["training_episode_length"] = alloc((MAX_EPISODES,), np.int32)
    bufs["training_total_reward"] = alloc((MAX_EPISODES,), np.float32)
    bufs["training_average_reward_per_step"] = alloc((MAX_EPISODES,), np.float32)
    bufs["training_average_abs_td_error"] = alloc((MAX_EPISODES,), np.float32)
    bufs["training_epsilon"] = alloc((MAX_EPISODES,), np.float32)
    bufs["training_timesteps"] = alloc((MAX_EPISODES,), np.int32)

    # Validation outputs
    bufs["validation_training_episode"] = alloc((SMALL_RESULT_PAD,), np.int32)
    bufs["validation_total_reward"] = alloc((SMALL_RESULT_PAD,), np.float32)
    bufs["validation_min_abs_theta"] = alloc((SMALL_RESULT_PAD,), np.float32)
    bufs["validation_upright_steps"] = alloc((SMALL_RESULT_PAD,), np.int32)
    bufs["validation_stable_upright_steps"] = alloc((SMALL_RESULT_PAD,), np.int32)
    bufs["validation_max_consecutive_stable_steps"] = alloc((SMALL_RESULT_PAD,), np.int32)
    bufs["validation_success"] = alloc((SMALL_RESULT_PAD,), np.int32)
    bufs["validation_final_theta_wrapped"] = alloc((SMALL_RESULT_PAD,), np.float32)
    bufs["validation_final_theta_unwrapped"] = alloc((SMALL_RESULT_PAD,), np.float32)
    bufs["validation_final_theta_dot"] = alloc((SMALL_RESULT_PAD,), np.float32)
    bufs["validation_states"] = alloc((max(VALIDATION_STATE_BUFFER_SIZE, VALIDATION_STATE_PAD),), np.float32)
    bufs["validation_actions"] = alloc((max(VALIDATION_TRAJECTORY_SIZE, VALIDATION_TRAJECTORY_PAD),), np.int32)
    bufs["validation_rewards"] = alloc((max(VALIDATION_TRAJECTORY_SIZE, VALIDATION_TRAJECTORY_PAD),), np.float32)

    # Evaluation outputs
    bufs["evaluation_initial_theta"] = alloc((SMALL_RESULT_PAD,), np.float32)
    bufs["evaluation_initial_theta_dot"] = alloc((SMALL_RESULT_PAD,), np.float32)
    bufs["evaluation_actual_initial_theta"] = alloc((SMALL_RESULT_PAD,), np.float32)
    bufs["evaluation_actual_initial_theta_dot"] = alloc((SMALL_RESULT_PAD,), np.float32)
    bufs["evaluation_total_reward"] = alloc((SMALL_RESULT_PAD,), np.float32)
    bufs["evaluation_min_abs_theta"] = alloc((SMALL_RESULT_PAD,), np.float32)
    bufs["evaluation_upright_steps"] = alloc((SMALL_RESULT_PAD,), np.int32)
    bufs["evaluation_stable_upright_steps"] = alloc((SMALL_RESULT_PAD,), np.int32)
    bufs["evaluation_max_consecutive_stable_steps"] = alloc((SMALL_RESULT_PAD,), np.int32)
    bufs["evaluation_success"] = alloc((SMALL_RESULT_PAD,), np.int32)
    bufs["evaluation_final_theta_wrapped"] = alloc((SMALL_RESULT_PAD,), np.float32)
    bufs["evaluation_final_theta_unwrapped"] = alloc((SMALL_RESULT_PAD,), np.float32)
    bufs["evaluation_final_theta_dot"] = alloc((SMALL_RESULT_PAD,), np.float32)
    bufs["evaluation_states"] = alloc((max(EVALUATION_STATE_BUFFER_SIZE, EVALUATION_STATE_PAD),), np.float32)
    bufs["evaluation_actions"] = alloc((max(EVALUATION_TRAJECTORY_SIZE, EVALUATION_TRAJECTORY_PAD),), np.int32)
    bufs["evaluation_rewards"] = alloc((max(EVALUATION_TRAJECTORY_SIZE, EVALUATION_TRAJECTORY_PAD),), np.float32)

    # Initialize buffers.
    for buf in bufs.values():
        buf[:] = 0
        try:
            buf.flush()
        except Exception:
            pass
    return bufs


def free_buffers(bufs):
    for buf in bufs.values():
        try:
            buf.freebuffer()
        except Exception:
            pass
    gc.collect()


def set_pointer(ip, name, buf):
    """Write physical address to the corresponding AXI-Lite pointer register."""
    address = int(buf.physical_address)
    low = address & 0xFFFFFFFF
    high = (address >> 32) & 0xFFFFFFFF

    reg1 = f"{name}_1"
    reg2 = f"{name}_2"

    if hasattr(ip.register_map, reg1) and hasattr(ip.register_map, reg2):
        setattr(ip.register_map, reg1, low)
        setattr(ip.register_map, reg2, high)
    elif hasattr(ip.register_map, name):
        setattr(ip.register_map, name, address)
    else:
        raise AttributeError(f"Pointer register not found for {name}")


def connect_buffers(ip, bufs):
    for name, buf in bufs.items():
        set_pointer(ip, name, buf)


def start_and_wait(ip):
    ip.write(0x00, 0x01)
    while (ip.read(0x00) & 0x4) == 0:
        pass


def read_counter(ip, name):
    try:
        return int(getattr(ip.register_map, name))
    except Exception:
        return -1


def cycles_from_seconds(seconds):
    return int(round(float(seconds) * FPGA_CLOCK_HZ))


def avg_cycles(seconds, iterations):
    if iterations <= 0:
        return 0.0
    return (float(seconds) * FPGA_CLOCK_HZ) / float(iterations)


def flush_input_buffers(bufs):
    # For non-cacheable buffers this does almost nothing, but keeping it is safe.
    for buf in bufs.values():
        try:
            buf.flush()
        except Exception:
            pass


def run_mode(ip, bufs, seed, mode, label, iterations):
    # Re-write all pointer registers before every mode. This avoids stale pointer registers.
    connect_buffers(ip, bufs)
    flush_input_buffers(bufs)

    ip.register_map.mode = int(mode)
    ip.register_map.seed_value = int(seed)

    start = time.perf_counter()
    start_and_wait(ip)
    elapsed = time.perf_counter() - start

    completed_train = read_counter(ip, "completed_training_episodes")
    completed_val = read_counter(ip, "completed_validations")
    completed_eval = read_counter(ip, "completed_evaluations")

    total_cycles = cycles_from_seconds(elapsed)
    avg_time_s = elapsed / float(iterations) if iterations > 0 else elapsed
    avg_time_us = avg_time_s * 1e6
    average_cycles = avg_cycles(elapsed, iterations)

    print(
        f"mode {mode:2d} | {label:<42} | "
        f"{elapsed:12.6f} s | cycles={total_cycles:14d} | "
        f"avg={avg_time_us:12.3f} us | avg_cycles={average_cycles:12.1f} | "
        f"train={completed_train:5d}, val={completed_val:3d}, eval/iter={completed_eval:6d}"
    )

    return {
        "seed": seed,
        "mode": mode,
        "label": label,
        "iterations": iterations,
        "time_s": elapsed,
        "total_cycles": total_cycles,
        "avg_time_s": avg_time_s,
        "avg_time_us": avg_time_us,
        "avg_cycles": average_cycles,
        "completed_training_episodes": completed_train,
        "completed_validations": completed_val,
        "completed_evaluations_or_iterations": completed_eval,
    }


def sync_if_requested(buf):
    if not READ_WITH_SYNC_FROM_DEVICE:
        return
    try:
        if hasattr(buf, "sync_from_device"):
            buf.sync_from_device()
        elif hasattr(buf, "invalidate"):
            buf.invalidate()
    except Exception as e:
        print(f"WARNING: sync_from_device failed: {e}")


def copy_small_array(buf, length, dtype):
    """Copy a small result array from PYNQ buffer to normal NumPy memory."""
    sync_if_requested(buf)
    return np.array(buf[:length], dtype=dtype, copy=True)


def collect_training_results(seed, bufs, training_rows):
    """
    Copy training output arrays produced by mode 4.

    Important:
    The current HLS top interface does not export per-episode
    training_stable_upright_steps. Therefore, true per-episode training F1
    cannot be computed from the current bitstream. This function saves all
    available training signals. The training-progress F1 is computed separately
    from validation checkpoints that are produced during training.
    """
    train_episode = copy_small_array(bufs["training_episode"], MAX_EPISODES, np.int32)
    train_length = copy_small_array(bufs["training_episode_length"], MAX_EPISODES, np.int32)
    train_reward = copy_small_array(bufs["training_total_reward"], MAX_EPISODES, np.float32)
    train_avg_reward = copy_small_array(bufs["training_average_reward_per_step"], MAX_EPISODES, np.float32)
    train_td = copy_small_array(bufs["training_average_abs_td_error"], MAX_EPISODES, np.float32)
    train_epsilon = copy_small_array(bufs["training_epsilon"], MAX_EPISODES, np.float32)
    train_timesteps = copy_small_array(bufs["training_timesteps"], MAX_EPISODES, np.int32)

    for i in range(MAX_EPISODES):
        training_rows.append({
            "seed": seed,
            "episode_index": i,
            "training_episode": int(train_episode[i]),
            "training_episode_length": int(train_length[i]),
            "training_total_reward": float(train_reward[i]),
            "training_average_reward_per_step": float(train_avg_reward[i]),
            "training_average_abs_td_error": float(train_td[i]),
            "training_epsilon": float(train_epsilon[i]),
            "training_timesteps": int(train_timesteps[i]),
            "note": "Current HLS does not export training_stable_upright_steps, so per-episode training F1 is not computed here.",
        })

    return train_episode, train_length, train_reward, train_avg_reward, train_td, train_epsilon, train_timesteps


def summarize_training_learning_rows(seed, train_reward, train_td, train_epsilon, train_timesteps):
    """Simple training learning-curve summary, not an F1 score."""
    return {
        "seed": seed,
        "phase": "training_learning_curve",
        "batch_size": BATCH_SIZE,
        "episodes": MAX_EPISODES,
        "total_environment_steps": int(train_timesteps[-1]) if len(train_timesteps) else 0,
        "mean_training_reward": float(np.mean(train_reward)) if len(train_reward) else 0.0,
        "final_100_episode_mean_reward": float(np.mean(train_reward[-100:])) if len(train_reward) >= 100 else float(np.mean(train_reward)) if len(train_reward) else 0.0,
        "final_100_episode_mean_abs_td_error": float(np.mean(train_td[-100:])) if len(train_td) >= 100 else float(np.mean(train_td)) if len(train_td) else 0.0,
        "final_epsilon": float(train_epsilon[-1]) if len(train_epsilon) else 0.0,
        "f1_score": "not_available_from_current_hls_training_outputs",
        "f1_note": "True training F1 requires training_stable_upright_steps per episode or training trajectory states. The current HLS top exports reward/TD/epsilon/timesteps only for training.",
    }



def collect_validation_results(seed, bufs, validation_rows):
    val_training_episode = copy_small_array(bufs["validation_training_episode"], NUM_VALIDATIONS, np.int32)
    val_total_reward = copy_small_array(bufs["validation_total_reward"], NUM_VALIDATIONS, np.float32)
    val_success = copy_small_array(bufs["validation_success"], NUM_VALIDATIONS, np.int32)
    val_stable_steps = copy_small_array(bufs["validation_stable_upright_steps"], NUM_VALIDATIONS, np.int32)
    val_max_consecutive = copy_small_array(bufs["validation_max_consecutive_stable_steps"], NUM_VALIDATIONS, np.int32)

    print("validation_total_reward:", val_total_reward)
    print("validation_success     :", val_success)

    for i in range(NUM_VALIDATIONS):
        validation_rows.append({
            "seed": seed,
            "validation_index": i,
            "training_episode": int(val_training_episode[i]),
            "validation_total_reward": float(val_total_reward[i]),
            "validation_success": int(val_success[i]),
            "validation_stable_upright_steps": int(val_stable_steps[i]),
            "validation_max_consecutive_stable_steps": int(val_max_consecutive[i]),
        })

    return val_total_reward, val_success, val_stable_steps, val_max_consecutive


def collect_evaluation_results(seed, bufs, evaluation_rows):
    eval_initial_theta = copy_small_array(bufs["evaluation_initial_theta"], EVALUATION_BATCH_SIZE, np.float32)
    eval_initial_theta_dot = copy_small_array(bufs["evaluation_initial_theta_dot"], EVALUATION_BATCH_SIZE, np.float32)
    eval_total_reward = copy_small_array(bufs["evaluation_total_reward"], EVALUATION_BATCH_SIZE, np.float32)
    eval_success = copy_small_array(bufs["evaluation_success"], EVALUATION_BATCH_SIZE, np.int32)
    eval_stable_steps = copy_small_array(bufs["evaluation_stable_upright_steps"], EVALUATION_BATCH_SIZE, np.int32)
    eval_max_consecutive = copy_small_array(bufs["evaluation_max_consecutive_stable_steps"], EVALUATION_BATCH_SIZE, np.int32)

    print("evaluation_total_reward:", eval_total_reward)
    print("evaluation_success     :", eval_success)

    for i in range(EVALUATION_BATCH_SIZE):
        evaluation_rows.append({
            "seed": seed,
            "evaluation_index": i,
            "initial_theta": float(eval_initial_theta[i]),
            "initial_theta_dot": float(eval_initial_theta_dot[i]),
            "evaluation_total_reward": float(eval_total_reward[i]),
            "evaluation_success": int(eval_success[i]),
            "evaluation_stable_upright_steps": int(eval_stable_steps[i]),
            "evaluation_max_consecutive_stable_steps": int(eval_max_consecutive[i]),
        })

    return eval_total_reward, eval_success, eval_stable_steps, eval_max_consecutive


def save_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)



def stable_timestep_performance_matrix(stable_steps_values, success_values, reward_values, max_consecutive_values, total_steps_per_case):
    """
    Recreate the earlier performance-matrix calculation.

    Positive class = stable-upright timestep, not validation/evaluation case success.
    Every timestep has desired label y_true = 1 because the control objective is
    to keep the pendulum stable upright. The FPGA output gives how many
    timesteps were actually stable upright.

    Therefore:
        TP = total stable-upright timesteps
        FN = total timesteps - TP
        FP = 0
        TN = 0

    The episode/case success flag is still saved as success_rate, but it is not
    used directly as the F1 label.
    """
    stable_steps_values = np.array(stable_steps_values, dtype=np.int64)
    success_values = np.array(success_values, dtype=np.int32)
    reward_values = np.array(reward_values, dtype=np.float32)
    max_consecutive_values = np.array(max_consecutive_values, dtype=np.int32)

    num_cases = int(stable_steps_values.size)
    total_steps = int(num_cases * total_steps_per_case)
    tp = int(np.sum(stable_steps_values))
    fn = int(total_steps - tp)
    fp = 0
    tn = 0

    accuracy = float((tp + tn) / (tp + tn + fp + fn)) if total_steps > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float((2.0 * precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "num_cases": num_cases,
        "total_timesteps": total_steps,
        "tp_stable_timesteps": tp,
        "fn_nonstable_timesteps": fn,
        "fp": fp,
        "tn": tn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "success_count": int(np.sum(success_values == 1)),
        "failure_count": int(np.sum(success_values == 0)),
        "success_rate": float(np.mean(success_values)) if num_cases > 0 else 0.0,
        "mean_reward": float(np.mean(reward_values)) if num_cases > 0 else 0.0,
        "min_reward": float(np.min(reward_values)) if num_cases > 0 else 0.0,
        "max_reward": float(np.max(reward_values)) if num_cases > 0 else 0.0,
        "avg_stable_upright_steps": float(np.mean(stable_steps_values)) if num_cases > 0 else 0.0,
        "avg_max_consecutive_stable_steps": float(np.mean(max_consecutive_values)) if num_cases > 0 else 0.0,
    }


def mean_float(values):
    values = list(values)
    return float(np.mean(values)) if values else 0.0


def main():
    log_file = open(TXT_LOG_PATH, "w")
    original_stdout = sys.stdout
    sys.stdout = Tee(sys.stdout, log_file)

    all_timing_rows = []
    seed_summary_rows = []
    nn_summary_rows = []
    validation_rows = []
    evaluation_rows = []
    training_rows = []
    metrics_rows = []
    training_progress_metrics_rows = []

    try:
        print("Configuration")
        print("-" * 100)
        print(f"Seeds                         : {SEEDS}")
        print(f"Run NN internal profile       : {RUN_NN_INTERNAL_PROFILE}")
        print(f"Read result arrays            : {READ_RESULT_ARRAYS}")
        print(f"Allocate cacheable buffers    : {ALLOCATE_CACHEABLE}")
        print(f"Read with sync_from_device    : {READ_WITH_SYNC_FROM_DEVICE}")
        print(f"FPGA clock for cycle estimate : {FPGA_CLOCK_HZ / 1e6:.3f} MHz")
        print(f"Batch size expected in HLS     : {BATCH_SIZE}")
        print("NOTE: mode 0 is not executed, so the full workflow is not rerun.")
        print("NOTE: Each seed runs training only once using mode 4.")
        print()

        overlay = Overlay(BIT_PATH)
        ip = getattr(overlay, IP_NAME)

        for seed in SEEDS:
            print("=" * 100)
            print(f"Running seed {seed}")
            print("=" * 100)

            bufs = allocate_all_buffers()
            connect_buffers(ip, bufs)

            seed_rows = []
            nn_rows = []
            val_total_reward = np.array([], dtype=np.float32)
            val_success = np.array([], dtype=np.int32)
            val_stable_steps = np.array([], dtype=np.int32)
            val_max_consecutive = np.array([], dtype=np.int32)
            eval_total_reward = np.array([], dtype=np.float32)
            eval_success = np.array([], dtype=np.int32)
            eval_stable_steps = np.array([], dtype=np.int32)
            eval_max_consecutive = np.array([], dtype=np.int32)

            try:
                # 1) Train + validate once for this seed.
                row_train_val = run_mode(ip, bufs, seed, 4, "Training + validation", 1)
                all_timing_rows.append(row_train_val)
                seed_rows.append(row_train_val)

                if READ_RESULT_ARRAYS:
                    print("\nTraining arrays after mode 4")
                    (train_episode, train_length, train_total_reward, train_avg_reward,
                     train_td, train_epsilon, train_timesteps) = collect_training_results(seed, bufs, training_rows)
                    print("training_total_reward first 5:", train_total_reward[:5])
                    print("training_total_reward last  5:", train_total_reward[-5:])

                    print("\nValidation arrays after mode 4")
                    val_total_reward, val_success, val_stable_steps, val_max_consecutive = collect_validation_results(seed, bufs, validation_rows)

                # 2) Evaluate the trained network from the same seed.
                row_eval = run_mode(ip, bufs, seed, 1, "Final evaluation only", EVALUATION_BATCH_SIZE)
                all_timing_rows.append(row_eval)
                seed_rows.append(row_eval)

                if READ_RESULT_ARRAYS:
                    print("\nEvaluation arrays after mode 1")
                    eval_total_reward, eval_success, eval_stable_steps, eval_max_consecutive = collect_evaluation_results(seed, bufs, evaluation_rows)

                # 3) Extra timing modes. These do not retrain.
                for mode, label, iterations in [(2, "One validation episode", 1),
                                                (3, f"Inference profile x{INFERENCE_PROFILE_STEPS}", INFERENCE_PROFILE_STEPS)]:
                    row = run_mode(ip, bufs, seed, mode, label, iterations)
                    all_timing_rows.append(row)
                    seed_rows.append(row)

                if RUN_NN_INTERNAL_PROFILE:
                    print("\nInternal neural-network profiling")
                    for mode, label, iterations in NN_PROFILE_MODES:
                        row = run_mode(ip, bufs, seed, mode, label, iterations)
                        all_timing_rows.append(row)
                        nn_rows.append(row)

                timing_by_mode = {r["mode"]: r for r in seed_rows}
                nn_by_mode = {r["mode"]: r for r in nn_rows}

                train_val_time = timing_by_mode[4]["time_s"]
                eval_time = timing_by_mode[1]["time_s"]
                one_val_time = timing_by_mode[2]["time_s"]
                inference_time = timing_by_mode[3]["time_s"]
                estimated_full_time = train_val_time + eval_time
                estimated_full_cycles = cycles_from_seconds(estimated_full_time)
                validation_overhead = one_val_time * NUM_VALIDATIONS
                avg_inference_s = inference_time / INFERENCE_PROFILE_STEPS
                avg_inference_cycles = avg_cycles(inference_time, INFERENCE_PROFILE_STEPS)

                print("\nSeed timing summary")
                print(f"Seed                                      : {seed}")
                print(f"Training + validation time               : {train_val_time:12.6f} s")
                print(f"Training + validation cycles             : {timing_by_mode[4]['total_cycles']:14d}")
                print(f"One validation episode time              : {one_val_time:12.6f} s")
                print(f"Validation overhead approx. from one val : {validation_overhead:12.6f} s")
                print(f"Final evaluation time                    : {eval_time:12.6f} s")
                print(f"Final evaluation cycles                  : {timing_by_mode[1]['total_cycles']:14d}")
                print(f"Inference profile total time             : {inference_time:12.6f} s")
                print(f"Average inference per step               : {avg_inference_s:12.9f} s")
                print(f"Average inference per step               : {avg_inference_s * 1000.0:12.6f} ms")
                print(f"Average inference per step               : {avg_inference_s * 1e6:12.3f} us")
                print(f"Average inference cycles per step        : {avg_inference_cycles:12.1f}")
                print(f"Full workflow estimated                  : {estimated_full_time:12.6f} s")
                print(f"Full workflow estimated cycles           : {estimated_full_cycles:14d}")

                if RUN_NN_INTERNAL_PROFILE:
                    print("\nNeural-network action timing")
                    for mode, label, _ in NN_PROFILE_MODES:
                        r = nn_by_mode[mode]
                        print(
                            f"mode {mode:2d} | {label:<42} | "
                            f"avg={r['avg_time_us']:12.3f} us | "
                            f"avg_cycles={r['avg_cycles']:12.1f} | "
                            f"total={r['time_s']:12.6f} s | total_cycles={r['total_cycles']:14d}"
                        )

                    print("\nSlowest NN actions by average time")
                    sorted_nn = sorted(nn_rows, key=lambda x: x["avg_time_us"], reverse=True)
                    for r in sorted_nn:
                        print(f"{r['label']:<42}: {r['avg_time_us']:12.3f} us | {r['avg_cycles']:12.1f} cycles")

                seed_summary = {
                    "seed": seed,
                    "training_validation_time_s": train_val_time,
                    "training_validation_cycles": timing_by_mode[4]["total_cycles"],
                    "one_validation_episode_time_s": one_val_time,
                    "validation_overhead_estimated_s": validation_overhead,
                    "evaluation_time_s": eval_time,
                    "evaluation_cycles": timing_by_mode[1]["total_cycles"],
                    "inference_profile_time_s": inference_time,
                    "average_inference_time_s": avg_inference_s,
                    "average_inference_time_us": avg_inference_s * 1e6,
                    "average_inference_cycles": avg_inference_cycles,
                    "estimated_full_workflow_time_s": estimated_full_time,
                    "estimated_full_workflow_cycles": estimated_full_cycles,
                }
                seed_summary_rows.append(seed_summary)

                if READ_RESULT_ARRAYS:
                    # Earlier performance matrix: based on stable-upright TIMESTEPS,
                    # not simply success/fail cases. This is the meaningful control metric.
                    val_m = stable_timestep_performance_matrix(
                        val_stable_steps, val_success, val_total_reward, val_max_consecutive, MAX_STEPS
                    )
                    eval_m = stable_timestep_performance_matrix(
                        eval_stable_steps, eval_success, eval_total_reward, eval_max_consecutive, MAX_STEPS
                    )

                    # Training-progress F1: calculated from validation checkpoints collected during training.
                    # This matches the earlier stable-timestep matrix idea, but uses the
                    # checkpoints that are actually exported by the current HLS design.
                    training_progress_m = stable_timestep_performance_matrix(
                        val_stable_steps, val_success, val_total_reward, val_max_consecutive, MAX_STEPS
                    )

                    training_progress_metrics_rows.append({
                        "seed": seed,
                        "phase": "training_progress_validation_checkpoints",
                        "batch_size": BATCH_SIZE,
                        **training_progress_m,
                    })

                    metrics_rows.append({
                        "seed": seed,
                        "phase": "training_progress_validation_checkpoints",
                        "batch_size": BATCH_SIZE,
                        **training_progress_m,
                    })

                    metrics_rows.append({
                        "seed": seed,
                        "phase": "validation_checkpoints",
                        "batch_size": BATCH_SIZE,
                        **val_m,
                    })

                    metrics_rows.append({
                        "seed": seed,
                        "phase": "evaluation",
                        "batch_size": BATCH_SIZE,
                        **eval_m,
                    })

                    print("\nSeed control-performance matrix")
                    print("Training-progress matrix from validation checkpoints:",
                          f"TP={training_progress_m['tp_stable_timesteps']}, FN={training_progress_m['fn_nonstable_timesteps']},",
                          f"F1={training_progress_m['f1_score']:.6f}, success_rate={training_progress_m['success_rate']:.3f}")
                    print("Evaluation matrix from stable timesteps:",
                          f"TP={eval_m['tp_stable_timesteps']}, FN={eval_m['fn_nonstable_timesteps']},",
                          f"F1={eval_m['f1_score']:.6f}, success_rate={eval_m['success_rate']:.3f}")

                for r in nn_rows:
                    nn_summary_rows.append({
                        "seed": seed,
                        "mode": r["mode"],
                        "label": r["label"],
                        "iterations": r["iterations"],
                        "total_time_s": r["time_s"],
                        "total_cycles": r["total_cycles"],
                        "avg_time_us": r["avg_time_us"],
                        "avg_cycles": r["avg_cycles"],
                    })

            finally:
                free_buffers(bufs)

            print()

        print("=" * 100)
        print("MULTI-SEED SUMMARY")
        print("=" * 100)
        for row in seed_summary_rows:
            print(
                f"Seed {row['seed']:>2} | "
                f"Train+Val: {row['training_validation_time_s']:10.3f} s | "
                f"Eval: {row['evaluation_time_s']:8.4f} s | "
                f"Full est.: {row['estimated_full_workflow_time_s']:10.3f} s | "
                f"Inference: {row['average_inference_time_us']:9.3f} us | "
                f"Inference cycles: {row['average_inference_cycles']:9.1f}"
            )

        if seed_summary_rows:
            train_val = np.array([r["training_validation_time_s"] for r in seed_summary_rows], dtype=np.float64)
            eval_t = np.array([r["evaluation_time_s"] for r in seed_summary_rows], dtype=np.float64)
            full_t = np.array([r["estimated_full_workflow_time_s"] for r in seed_summary_rows], dtype=np.float64)
            inf_us = np.array([r["average_inference_time_us"] for r in seed_summary_rows], dtype=np.float64)
            inf_cycles = np.array([r["average_inference_cycles"] for r in seed_summary_rows], dtype=np.float64)

            print("\nAverage over seeds")
            print(f"Training + validation mean          : {train_val.mean():12.6f} s")
            print(f"Training + validation std           : {train_val.std():12.6f} s")
            print(f"Evaluation mean                     : {eval_t.mean():12.6f} s")
            print(f"Evaluation std                      : {eval_t.std():12.6f} s")
            print(f"Estimated full workflow mean        : {full_t.mean():12.6f} s")
            print(f"Estimated full workflow std         : {full_t.std():12.6f} s")
            print(f"Average inference mean              : {inf_us.mean():12.6f} us")
            print(f"Average inference std               : {inf_us.std():12.6f} us")
            print(f"Average inference cycles mean       : {inf_cycles.mean():12.3f}")
            print(f"Average inference cycles std        : {inf_cycles.std():12.3f}")



        if READ_RESULT_ARRAYS and metrics_rows:
            print("\nControl-performance metrics over seeds")
            for phase in ["validation", "evaluation"]:
                rows = [r for r in metrics_rows if r["phase"] == phase]
                if not rows:
                    continue
                sr = np.array([r["success_rate"] for r in rows], dtype=np.float64)
                f1 = np.array([r["f1_score"] for r in rows], dtype=np.float64)
                rew = np.array([r["mean_reward"] for r in rows], dtype=np.float64)
                print(f"{phase.capitalize():<10} success rate mean/std : {sr.mean():.6f} / {sr.std():.6f}")
                print(f"{phase.capitalize():<10} F1 mean/std           : {f1.mean():.6f} / {f1.std():.6f}")
                print(f"{phase.capitalize():<10} reward mean/std       : {rew.mean():.6f} / {rew.std():.6f}")

        save_csv(ALL_TIMING_CSV_PATH, all_timing_rows)
        save_csv(SEED_SUMMARY_CSV_PATH, seed_summary_rows)
        save_csv(NN_SUMMARY_CSV_PATH, nn_summary_rows)
        save_csv(VALIDATION_CSV_PATH, validation_rows)
        save_csv(EVALUATION_CSV_PATH, evaluation_rows)
        save_csv(TRAINING_CSV_PATH, training_rows)
        save_csv(TRAINING_PROGRESS_METRICS_CSV_PATH, training_progress_metrics_rows)
        save_csv(METRICS_CSV_PATH, metrics_rows)

        print("\nSaved files")
        print(f"- {TXT_LOG_PATH}")
        print(f"- {ALL_TIMING_CSV_PATH}")
        print(f"- {SEED_SUMMARY_CSV_PATH}")
        print(f"- {NN_SUMMARY_CSV_PATH}")
        if READ_RESULT_ARRAYS:
            print(f"- {VALIDATION_CSV_PATH}")
            print(f"- {EVALUATION_CSV_PATH}")
            print(f"- {TRAINING_CSV_PATH}")
            print(f"- {TRAINING_PROGRESS_METRICS_CSV_PATH}")
            print(f"- {METRICS_CSV_PATH}")

    finally:
        sys.stdout = original_stdout
        log_file.close()


if __name__ == "__main__":
    main()