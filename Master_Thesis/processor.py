from pynq import Overlay
import math
import time
import struct
import os
import shutil
import numpy as np
import csv
from pathlib import Path


# Configuration


BITSTREAM_PATH = "/home/xilinx/pynq/overlays/dqn_eva/dqn_eva.bit"
OUTPUT_DIR = Path("/home/xilinx/dqn_agent_results")
WEIGHT_FILE = Path("/home/xilinx/trained_dqn_weights.npz")

STATE_SIZE = 3
ACTION_VALUES = [-2.0, -0.5, -0.1, 0.0, 0.1, 0.5, 2.0]
ACTION_SIZE = len(ACTION_VALUES)

MAX_EPISODES = 8000
MAX_STEPS = 250

VALIDATION_INTERVAL = 800
EVALUATION_BATCH_SIZE = 5

TARGET_UPDATE_FREQUENCY = 1000

EPS_START = 1.0
EPS_END = 0.05
EXPLORATION_FRACTION = 0.30

PI = math.pi
DT = 0.05
G = 10.0
M = 1.0
L = 1.0
MAX_SPEED = 8.0

THETA_UPRIGHT_THRESHOLD = 0.35
THETA_DOT_STABLE_THRESHOLD = 0.5
SUCCESS_CONSECUTIVE_STABLE_STEPS = 30


# AXI-Lite register map


AP_CTRL = 0x00

REG_MODE = 0x10
REG_SEED_VALUE = 0x18

REG_STATE_0 = 0x20
REG_STATE_1 = 0x28
REG_STATE_2 = 0x30

REG_ACTION_IN = 0x38
REG_REWARD_IN = 0x40

REG_NEXT_STATE_0 = 0x48
REG_NEXT_STATE_1 = 0x50
REG_NEXT_STATE_2 = 0x58

REG_DONE_IN = 0x60
REG_EPSILON = 0x68

REG_ACTION_OUT = 0x70
REG_ACTION_OUT_CTRL = 0x78

REG_TD_ERROR_OUT = 0x80
REG_TD_ERROR_OUT_CTRL = 0x88

REG_REPLAY_COUNT_OUT = 0x90
REG_REPLAY_COUNT_OUT_CTRL = 0x98

REG_GLOBAL_STEP_OUT = 0xA0
REG_GLOBAL_STEP_OUT_CTRL = 0xA8

REG_WEIGHT_LAYER = 0xB0
REG_WEIGHT_I = 0xB8
REG_WEIGHT_J = 0xC0
REG_WEIGHT_IN = 0xC8
REG_WEIGHT_OUT = 0xD0
REG_WEIGHT_OUT_CTRL = 0xD8


# Utility functions


def f32_to_u32(value):
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]

def u32_to_f32(value):
    return struct.unpack("<f", struct.pack("<I", int(value) & 0xFFFFFFFF))[0]

def clip_float(value, minimum, maximum):
    if value < minimum:
        return minimum
    if value > maximum:
        return maximum
    return value

def wrap_angle(angle):
    while angle > PI:
        angle -= 2.0 * PI
    while angle < -PI:
        angle += 2.0 * PI
    return angle

def encode_observation(theta, theta_dot):
    return [
        math.cos(theta),
        math.sin(theta),
        theta_dot,
    ]

def action_to_torque(action):
    action = int(action)
    if action < 0:
        action = 0
    if action >= ACTION_SIZE:
        action = ACTION_SIZE - 1
    return ACTION_VALUES[action]

def calculate_reward(previous_theta, new_theta, new_theta_dot, torque):
    previous_angle_error = wrap_angle(previous_theta)
    angle_error = wrap_angle(new_theta)

    upright_reward = 2.0 * math.cos(angle_error)
    velocity_penalty = 0.01 * new_theta_dot * new_theta_dot
    torque_penalty = 0.001 * torque * torque
    progress_bonus = 0.5 * (abs(previous_angle_error) - abs(angle_error))

    bonus = 0.0
    if abs(angle_error) < 0.3:
        bonus += 2.0
    if abs(angle_error) < 0.15 and abs(new_theta_dot) < 1.0:
        bonus += 4.0
    if abs(angle_error) < 0.08 and abs(new_theta_dot) < 0.5:
        bonus += 6.0

    reward = upright_reward - velocity_penalty - torque_penalty + progress_bonus + bonus
    return clip_float(reward, -10.0, 10.0)


#  Environment functions


def reset_training_environment(episode_index):
    phase = float(episode_index + 1)
    theta = PI + 0.25 * math.sin(0.173 * phase)
    theta_dot = math.sin(0.097 * phase)
    theta_dot = clip_float(theta_dot, -MAX_SPEED, MAX_SPEED)
    return theta, theta_dot, encode_observation(theta, theta_dot)

def reset_fixed_environment(theta, theta_dot):
    return theta, theta_dot, encode_observation(theta, theta_dot)

def step_environment(theta, theta_dot, action):
    torque = action_to_torque(action)
    previous_theta = theta

    new_theta_dot = theta_dot + (
        3.0 * G / (2.0 * L) * math.sin(theta)
        + 3.0 / (M * L * L) * torque
    ) * DT

    new_theta_dot = clip_float(new_theta_dot, -MAX_SPEED, MAX_SPEED)
    new_theta = theta + new_theta_dot * DT

    reward = calculate_reward(previous_theta, new_theta, new_theta_dot, torque)
    next_state = encode_observation(new_theta, new_theta_dot)

    done = 0
    return new_theta, new_theta_dot, next_state, reward, done

def is_upright(theta):
    return abs(wrap_angle(theta)) < THETA_UPRIGHT_THRESHOLD

def is_stable(theta, theta_dot):
    return (
        abs(wrap_angle(theta)) < THETA_UPRIGHT_THRESHOLD
        and abs(theta_dot) < THETA_DOT_STABLE_THRESHOLD
    )

def epsilon_for_episode(episode_index):
    progress = episode_index / (EXPLORATION_FRACTION * float(MAX_EPISODES))
    progress = min(progress, 1.0)
    epsilon = EPS_START + progress * (EPS_END - EPS_START)
    return clip_float(epsilon, EPS_END, EPS_START)

def f1_from_stable_steps(stable_steps, total_steps):
    if stable_steps <= 0 or total_steps <= 0:
        return 0.0

    precision = 1.0
    recall = stable_steps / float(total_steps)
    return (2.0 * precision * recall) / (precision + recall)


#  Direct-MMIO FPGA driver


class FastStatefulDqnIP:
    def __init__(self, ip):
        self.ip = ip
        self.reg = ip.mmio.array

        self.idx_ap_ctrl = AP_CTRL >> 2
        self.idx_mode = REG_MODE >> 2
        self.idx_seed = REG_SEED_VALUE >> 2
        self.idx_state0 = REG_STATE_0 >> 2
        self.idx_state1 = REG_STATE_1 >> 2
        self.idx_state2 = REG_STATE_2 >> 2
        self.idx_action = REG_ACTION_IN >> 2
        self.idx_reward = REG_REWARD_IN >> 2
        self.idx_next0 = REG_NEXT_STATE_0 >> 2
        self.idx_next1 = REG_NEXT_STATE_1 >> 2
        self.idx_next2 = REG_NEXT_STATE_2 >> 2
        self.idx_done = REG_DONE_IN >> 2
        self.idx_epsilon = REG_EPSILON >> 2
        self.idx_action_out = REG_ACTION_OUT >> 2
        self.idx_td_error = REG_TD_ERROR_OUT >> 2
        self.idx_replay_count = REG_REPLAY_COUNT_OUT >> 2
        self.idx_global_step = REG_GLOBAL_STEP_OUT >> 2
        self.idx_weight_layer = REG_WEIGHT_LAYER >> 2
        self.idx_weight_i = REG_WEIGHT_I >> 2
        self.idx_weight_j = REG_WEIGHT_J >> 2
        self.idx_weight_in = REG_WEIGHT_IN >> 2
        self.idx_weight_out = REG_WEIGHT_OUT >> 2

        self.full_idx = np.array([
            self.idx_mode, self.idx_seed,
            self.idx_state0, self.idx_state1, self.idx_state2,
            self.idx_action, self.idx_reward,
            self.idx_next0, self.idx_next1, self.idx_next2,
            self.idx_done, self.idx_epsilon,
        ], dtype=np.intp)
        self.full_u32 = np.zeros(12, dtype=np.uint32)
        self.full_f32 = self.full_u32.view(np.float32)

        self.setup_idx = np.array([
            self.idx_mode, self.idx_seed,
            self.idx_state0, self.idx_state1, self.idx_state2,
            self.idx_epsilon,
        ], dtype=np.intp)
        self.setup_u32 = np.zeros(6, dtype=np.uint32)
        self.setup_f32 = self.setup_u32.view(np.float32)

        self.step_idx = np.array([
            self.idx_mode,
            self.idx_reward,
            self.idx_next0, self.idx_next1, self.idx_next2,
            self.idx_done,
            self.idx_epsilon,
        ], dtype=np.intp)
        self.step_u32 = np.zeros(7, dtype=np.uint32)
        self.step_f32 = self.step_u32.view(np.float32)

    def wait_done(self):
        reg = self.reg
        idx = self.idx_ap_ctrl
        while (int(reg[idx]) & 0x2) == 0:
            pass

    def call_full(self, mode, seed, state, action_in, reward, next_state, done, epsilon):
        u = self.full_u32
        f = self.full_f32
        u[0] = np.uint32(int(mode))
        u[1] = np.uint32(int(seed))
        f[2] = np.float32(state[0])
        f[3] = np.float32(state[1])
        f[4] = np.float32(state[2])
        u[5] = np.uint32(int(action_in))
        f[6] = np.float32(reward)
        f[7] = np.float32(next_state[0])
        f[8] = np.float32(next_state[1])
        f[9] = np.float32(next_state[2])
        u[10] = np.uint32(int(done))
        f[11] = np.float32(epsilon)

        self.reg[self.full_idx] = u
        self.reg[self.idx_ap_ctrl] = np.uint32(0x01)
        self.wait_done()

        action_out = int(self.reg[self.idx_action_out])
        td_error_out = u32_to_f32(int(self.reg[self.idx_td_error]))
        replay_count_out = int(self.reg[self.idx_replay_count])
        global_step_out = int(self.reg[self.idx_global_step])
        return action_out, td_error_out, replay_count_out, global_step_out

    def call_stateful_setup_mode7(self, seed, initial_state, epsilon):
        u = self.setup_u32
        f = self.setup_f32
        u[0] = np.uint32(7)
        u[1] = np.uint32(int(seed))
        f[2] = np.float32(initial_state[0])
        f[3] = np.float32(initial_state[1])
        f[4] = np.float32(initial_state[2])
        f[5] = np.float32(epsilon)

        self.reg[self.setup_idx] = u
        self.reg[self.idx_ap_ctrl] = np.uint32(0x01)
        self.wait_done()
        return int(self.reg[self.idx_action_out])

    def call_stateful_mode6_fast(self, reward, current_state, done, epsilon):
        u = self.step_u32
        f = self.step_f32
        u[0] = np.uint32(6)
        f[1] = np.float32(reward)
        f[2] = np.float32(current_state[0])
        f[3] = np.float32(current_state[1])
        f[4] = np.float32(current_state[2])
        u[5] = np.uint32(int(done))
        f[6] = np.float32(epsilon)

        self.reg[self.step_idx] = u
        self.reg[self.idx_ap_ctrl] = np.uint32(0x01)
        self.wait_done()
        return int(self.reg[self.idx_action_out])

    def read_diagnostics(self):
        replay_count_out = int(self.reg[self.idx_replay_count])
        global_step_out = int(self.reg[self.idx_global_step])
        return replay_count_out, global_step_out

    def write_weight(self, layer, i, j, value):
        self.reg[self.idx_weight_layer] = np.uint32(int(layer))
        self.reg[self.idx_weight_i] = np.uint32(int(i))
        self.reg[self.idx_weight_j] = np.uint32(int(j))
        self.reg[self.idx_weight_in] = np.uint32(f32_to_u32(value))
        self.reg[self.idx_mode] = np.uint32(11)
        self.reg[self.idx_ap_ctrl] = np.uint32(0x01)
        self.wait_done()

    def read_weight(self, layer, i, j):
        self.reg[self.idx_weight_layer] = np.uint32(int(layer))
        self.reg[self.idx_weight_i] = np.uint32(int(i))
        self.reg[self.idx_weight_j] = np.uint32(int(j))
        self.reg[self.idx_mode] = np.uint32(10)
        self.reg[self.idx_ap_ctrl] = np.uint32(0x01)
        self.wait_done()
        return u32_to_f32(int(self.reg[self.idx_weight_out]))

def wait_done(ip):
    if isinstance(ip, FastStatefulDqnIP):
        ip.wait_done()
        return
    while (ip.read(AP_CTRL) & 0x2) == 0:
        pass

def write_common_registers(ip, mode, seed, state, action_in, reward, next_state, done, epsilon):
    ip.write(REG_MODE, int(mode))
    ip.write(REG_SEED_VALUE, int(seed))

    ip.write(REG_STATE_0, f32_to_u32(state[0]))
    ip.write(REG_STATE_1, f32_to_u32(state[1]))
    ip.write(REG_STATE_2, f32_to_u32(state[2]))

    ip.write(REG_ACTION_IN, int(action_in))
    ip.write(REG_REWARD_IN, f32_to_u32(reward))

    ip.write(REG_NEXT_STATE_0, f32_to_u32(next_state[0]))
    ip.write(REG_NEXT_STATE_1, f32_to_u32(next_state[1]))
    ip.write(REG_NEXT_STATE_2, f32_to_u32(next_state[2]))

    ip.write(REG_DONE_IN, int(done))
    ip.write(REG_EPSILON, f32_to_u32(epsilon))

def call_ip_full(ip, mode, seed, state, action_in, reward, next_state, done, epsilon):
    if isinstance(ip, FastStatefulDqnIP):
        return ip.call_full(mode, seed, state, action_in, reward, next_state, done, epsilon)

    write_common_registers(ip, mode, seed, state, action_in, reward, next_state, done, epsilon)

    ip.write(AP_CTRL, 0x01)
    wait_done(ip)

    action_out = int(ip.read(REG_ACTION_OUT))
    td_error_out = u32_to_f32(ip.read(REG_TD_ERROR_OUT))
    replay_count_out = int(ip.read(REG_REPLAY_COUNT_OUT))
    global_step_out = int(ip.read(REG_GLOBAL_STEP_OUT))

    return action_out, td_error_out, replay_count_out, global_step_out

def call_ip_stateful_setup_mode7(ip, seed, initial_state, epsilon):
    if isinstance(ip, FastStatefulDqnIP):
        return ip.call_stateful_setup_mode7(seed, initial_state, epsilon)

    ip.write(REG_MODE, 7)
    ip.write(REG_SEED_VALUE, int(seed))
    ip.write(REG_STATE_0, f32_to_u32(initial_state[0]))
    ip.write(REG_STATE_1, f32_to_u32(initial_state[1]))
    ip.write(REG_STATE_2, f32_to_u32(initial_state[2]))
    ip.write(REG_EPSILON, f32_to_u32(epsilon))

    ip.write(AP_CTRL, 0x01)
    wait_done(ip)

    return int(ip.read(REG_ACTION_OUT))

def call_ip_stateful_mode6_fast(ip, reward, current_state, done, epsilon):
    if isinstance(ip, FastStatefulDqnIP):
        return ip.call_stateful_mode6_fast(reward, current_state, done, epsilon)

    ip.write(REG_MODE, 6)
    ip.write(REG_REWARD_IN, f32_to_u32(reward))
    ip.write(REG_NEXT_STATE_0, f32_to_u32(current_state[0]))
    ip.write(REG_NEXT_STATE_1, f32_to_u32(current_state[1]))
    ip.write(REG_NEXT_STATE_2, f32_to_u32(current_state[2]))
    ip.write(REG_DONE_IN, int(done))
    ip.write(REG_EPSILON, f32_to_u32(epsilon))

    ip.write(AP_CTRL, 0x01)
    wait_done(ip)

    return int(ip.read(REG_ACTION_OUT))

def read_diagnostics(ip):
    if isinstance(ip, FastStatefulDqnIP):
        return ip.read_diagnostics()
    replay_count_out = int(ip.read(REG_REPLAY_COUNT_OUT))
    global_step_out = int(ip.read(REG_GLOBAL_STEP_OUT))
    return replay_count_out, global_step_out

def write_weight(ip, layer, i, j, value):
    if isinstance(ip, FastStatefulDqnIP):
        ip.write_weight(layer, i, j, value)
        return
    ip.write(REG_WEIGHT_LAYER, int(layer))
    ip.write(REG_WEIGHT_I, int(i))
    ip.write(REG_WEIGHT_J, int(j))
    ip.write(REG_WEIGHT_IN, f32_to_u32(value))
    ip.write(REG_MODE, 11)
    ip.write(AP_CTRL, 0x01)
    wait_done(ip)

def read_weight(ip, layer, i, j):
    if isinstance(ip, FastStatefulDqnIP):
        return ip.read_weight(layer, i, j)
    ip.write(REG_WEIGHT_LAYER, int(layer))
    ip.write(REG_WEIGHT_I, int(i))
    ip.write(REG_WEIGHT_J, int(j))
    ip.write(REG_MODE, 10)
    ip.write(AP_CTRL, 0x01)
    wait_done(ip)
    return u32_to_f32(ip.read(REG_WEIGHT_OUT))


# SD-card-safe weight persistence


def _fsync_directory(directory):
    directory = Path(directory)
    try:
        fd = os.open(str(directory), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except Exception:
        pass

def _read_meminfo_value_kb(key):
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith(key + ":"):
                    return int(line.split()[1])
    except Exception:
        pass
    return 0

def wait_until_sd_writes_finished(timeout_s=30.0):
    try:
        os.sync()
    except Exception:
        return

    start = time.perf_counter()
    stable_count = 0
    while True:
        dirty_kb = _read_meminfo_value_kb("Dirty")
        writeback_kb = _read_meminfo_value_kb("Writeback")

        if dirty_kb == 0 and writeback_kb == 0:
            stable_count += 1
            if stable_count >= 3:
                return
        else:
            stable_count = 0

        if time.perf_counter() - start > timeout_s:
            print(
                "Warning: storage sync wait timed out; "
                f"Dirty={dirty_kb} kB, Writeback={writeback_kb} kB"
            )
            return

        time.sleep(0.2)

def create_valid_weight_marker(weight_path):
    marker_path = Path(str(weight_path) + ".ok")
    with open(marker_path, "w") as f:
        f.write("valid\n")
        f.flush()
        os.fsync(f.fileno())
    _fsync_directory(marker_path.parent)
    wait_until_sd_writes_finished()

def _safe_npz_save(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.stem + ".new.npz")

    if tmp_path.exists():
        tmp_path.unlink()

    with open(tmp_path, "wb") as f:
        np.savez(f, **arrays)
        f.flush()
        os.fsync(f.fileno())

    os.replace(str(tmp_path), str(path))
    _fsync_directory(path.parent)

    wait_until_sd_writes_finished()

def save_weights_from_fpga(ip, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    w1_arr = np.zeros((STATE_SIZE, 32), dtype=np.float32)
    b1_arr = np.zeros((32,), dtype=np.float32)
    w2_arr = np.zeros((32, 32), dtype=np.float32)
    b2_arr = np.zeros((32,), dtype=np.float32)
    w3_arr = np.zeros((32, ACTION_SIZE), dtype=np.float32)
    b3_arr = np.zeros((ACTION_SIZE,), dtype=np.float32)

    for i in range(STATE_SIZE):
        for j in range(32):
            w1_arr[i, j] = read_weight(ip, 0, i, j)
    for j in range(32):
        b1_arr[j] = read_weight(ip, 1, 0, j)
    for i in range(32):
        for j in range(32):
            w2_arr[i, j] = read_weight(ip, 2, i, j)
    for j in range(32):
        b2_arr[j] = read_weight(ip, 3, 0, j)
    for i in range(32):
        for j in range(ACTION_SIZE):
            w3_arr[i, j] = read_weight(ip, 4, i, j)
    for j in range(ACTION_SIZE):
        b3_arr[j] = read_weight(ip, 5, 0, j)

    arrays = dict(w1=w1_arr, b1=b1_arr, w2=w2_arr, b2=b2_arr, w3=w3_arr, b3=b3_arr)

    backup_path = path.with_name(path.stem + "_backup.npz")
    valid_old, _ = is_valid_weight_file(path) if path.exists() else (False, "missing")
    if valid_old:
        shutil.copy2(str(path), str(backup_path))
        wait_until_sd_writes_finished()

    _safe_npz_save(path, **arrays)

    valid_new, reason = is_valid_weight_file(path)
    if not valid_new:
        raise RuntimeError(f"Weight save verification failed: {reason}")

    recovery_path = path.with_name(path.stem + "_recovery.npz")
    _safe_npz_save(recovery_path, **arrays)

    create_valid_weight_marker(path)

    print(f"Saved trained weights to: {path}")
    print(f"Saved recovery copy to: {recovery_path}")
    if valid_old:
        print(f"Previous valid weights backed up to: {backup_path}")
    print("Weight file verified and fully synced to SD card automatically.")
    print("SAFE TO POWER OFF after this message.")

def is_valid_weight_file(path):
    path = Path(path)
    if not path.exists():
        return False, "file does not exist"
    size = path.stat().st_size
    if size < 100:
        return False, f"file is too small or empty ({size} bytes)"

    try:
        data = np.load(path, allow_pickle=False)
        required = ["w1", "b1", "w2", "b2", "w3", "b3"]
        for key in required:
            if key not in data:
                return False, f"missing array {key}"

        expected_shapes = {
            "w1": (STATE_SIZE, 32),
            "b1": (32,),
            "w2": (32, 32),
            "b2": (32,),
            "w3": (32, ACTION_SIZE),
            "b3": (ACTION_SIZE,),
        }
        for key, expected in expected_shapes.items():
            if data[key].shape != expected:
                return False, f"array {key} shape is {data[key].shape}, expected {expected}"
        return True, "ok"
    except Exception as exc:
        return False, str(exc)

def load_weights_to_fpga(ip, path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Saved weight file does not exist: {path}")

    try:
        data = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise RuntimeError(
            f"Saved weight file exists but is not a valid .npz file: {path}. "
            f"It is probably empty/corrupt from an interrupted previous run. "
            f"Delete it and train again. Original error: {exc}"
        )
    required = ["w1", "b1", "w2", "b2", "w3", "b3"]
    for key in required:
        if key not in data:
            raise KeyError(f"Weight file is missing array: {key}")

    w1_arr = np.asarray(data["w1"], dtype=np.float32)
    b1_arr = np.asarray(data["b1"], dtype=np.float32)
    w2_arr = np.asarray(data["w2"], dtype=np.float32)
    b2_arr = np.asarray(data["b2"], dtype=np.float32)
    w3_arr = np.asarray(data["w3"], dtype=np.float32)
    b3_arr = np.asarray(data["b3"], dtype=np.float32)

    expected_shapes = {
        "w1": (STATE_SIZE, 32),
        "b1": (32,),
        "w2": (32, 32),
        "b2": (32,),
        "w3": (32, ACTION_SIZE),
        "b3": (ACTION_SIZE,),
    }
    actual_shapes = {
        "w1": w1_arr.shape, "b1": b1_arr.shape,
        "w2": w2_arr.shape, "b2": b2_arr.shape,
        "w3": w3_arr.shape, "b3": b3_arr.shape,
    }
    if actual_shapes != expected_shapes:
        raise ValueError(f"Weight shape mismatch. expected={expected_shapes}, actual={actual_shapes}")

    for i in range(STATE_SIZE):
        for j in range(32):
            write_weight(ip, 0, i, j, float(w1_arr[i, j]))
    for j in range(32):
        write_weight(ip, 1, 0, j, float(b1_arr[j]))
    for i in range(32):
        for j in range(32):
            write_weight(ip, 2, i, j, float(w2_arr[i, j]))
    for j in range(32):
        write_weight(ip, 3, 0, j, float(b2_arr[j]))
    for i in range(32):
        for j in range(ACTION_SIZE):
            write_weight(ip, 4, i, j, float(w3_arr[i, j]))
    for j in range(ACTION_SIZE):
        write_weight(ip, 5, 0, j, float(b3_arr[j]))

    print(f"Loaded trained weights from: {path}")

# ============================================================
# 7. Application helpers
# ============================================================

def ask_user_operation():
    print("\nSelect operation:")
    print("  1 = Train DQN and save weights")
    print("  2 = Load saved weights and run inference/evaluation")
    choice = input("Enter choice [1/2]: ").strip()
    if choice == "1":
        return "train"
    if choice == "2":
        return "infer"
    print("Invalid choice. Defaulting to inference.")
    return "infer"

def initialize_fpga_agent(ip, seed):
    zero_state = [0.0, 0.0, 0.0]
    call_ip_full(
        ip=ip,
        mode=0,
        seed=seed,
        state=zero_state,
        action_in=0,
        reward=0.0,
        next_state=zero_state,
        done=0,
        epsilon=1.0,
    )

def find_dqn_ip(overlay):
    print("Available IP blocks:")
    for name in overlay.ip_dict:
        print("  -", name)

    preferred_names = ["dqn_eva_0", "dqn_agent_ip_0"]
    for name in preferred_names:
        if hasattr(overlay, name):
            return getattr(overlay, name), name

    for name in overlay.ip_dict:
        if "dqn" in name.lower():
            return getattr(overlay, name), name

    raise RuntimeError("DQN IP not found. Check overlay.ip_dict and block design IP name.")

# ============================================================
# 8. Evaluation functions
# ============================================================

def run_greedy_episode(ip, seed, initial_theta, initial_theta_dot):
    theta, theta_dot, state = reset_fixed_environment(initial_theta, initial_theta_dot)
    zero_state = [0.0, 0.0, 0.0]

    total_reward = 0.0
    upright_steps = 0
    stable_steps = 0
    current_consecutive_stable = 0
    max_consecutive_stable = 0
    min_abs_theta = 999.0

    for _ in range(MAX_STEPS):
        theta_wrapped = wrap_angle(theta)
        min_abs_theta = min(min_abs_theta, abs(theta_wrapped))

        if is_upright(theta):
            upright_steps += 1

        if is_stable(theta, theta_dot):
            stable_steps += 1
            current_consecutive_stable += 1
            max_consecutive_stable = max(max_consecutive_stable, current_consecutive_stable)
        else:
            current_consecutive_stable = 0

        action, _, _, _ = call_ip_full(
            ip=ip,
            mode=4,
            seed=seed,
            state=state,
            action_in=0,
            reward=0.0,
            next_state=zero_state,
            done=0,
            epsilon=0.0,
        )

        theta, theta_dot, state, reward, done = step_environment(theta, theta_dot, action)
        total_reward += reward

    success = 1 if max_consecutive_stable >= SUCCESS_CONSECUTIVE_STABLE_STEPS else 0

    return {
        "total_reward": total_reward,
        "upright_steps": upright_steps,
        "stable_steps": stable_steps,
        "max_consecutive_stable": max_consecutive_stable,
        "success": success,
        "min_abs_theta": min_abs_theta,
        "final_theta_wrapped": wrap_angle(theta),
        "final_theta_unwrapped": theta,
        "final_theta_dot": theta_dot,
    }

def save_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

def run_final_evaluation_and_save(ip, seed, output_tag="stateful_mode6"):
    print("\nRunning final evaluation...")
    evaluation_initial_conditions = [
        (PI, 0.0),
        (PI + 0.20, -2.0),
        (1.67, 2.0),
        (1.67, -1.0),
        (PI + 1.67, 0.0),
    ]

    evaluation_rows = []
    for idx, (theta0, theta_dot0) in enumerate(evaluation_initial_conditions):
        row = run_greedy_episode(ip, seed, theta0, theta_dot0)
        row["case"] = idx
        row["initial_theta"] = theta0
        row["initial_theta_dot"] = theta_dot0
        evaluation_rows.append(row)

    total_eval_reward = sum(row["total_reward"] for row in evaluation_rows)
    total_stable_steps = sum(row["stable_steps"] for row in evaluation_rows)
    total_success = sum(row["success"] for row in evaluation_rows)
    total_eval_steps = EVALUATION_BATCH_SIZE * MAX_STEPS

    avg_eval_reward = total_eval_reward / float(EVALUATION_BATCH_SIZE)
    success_rate = total_success / float(EVALUATION_BATCH_SIZE)
    f1_score = f1_from_stable_steps(total_stable_steps, total_eval_steps)

    print("\nFinal evaluation metrics:")
    print(f"  Average reward = {avg_eval_reward:.6f}")
    print(f"  Stable steps   = {total_stable_steps}/{total_eval_steps}")
    print(f"  Success rate   = {success_rate:.6f}")
    print(f"  F1 score       = {f1_score:.6f}")

    for row in evaluation_rows:
        print(
            f"  Eval {row['case']} | "
            f"theta={row['initial_theta']:.4f} | "
            f"theta_dot={row['initial_theta_dot']:.4f} | "
            f"reward={row['total_reward']:.3f} | "
            f"stable={row['stable_steps']} | "
            f"maxStable={row['max_consecutive_stable']} | "
            f"success={row['success']}"
        )

    save_csv(
        OUTPUT_DIR / f"evaluation_results_{output_tag}.csv",
        evaluation_rows,
        [
            "case",
            "initial_theta",
            "initial_theta_dot",
            "total_reward",
            "upright_steps",
            "stable_steps",
            "max_consecutive_stable",
            "success",
            "min_abs_theta",
            "final_theta_wrapped",
            "final_theta_unwrapped",
            "final_theta_dot",
        ],
    )
    return evaluation_rows

# ============================================================
# 9. Main training / inference flow
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading overlay...")
    overlay = Overlay(BITSTREAM_PATH)
    ip, ip_name = find_dqn_ip(overlay)

    print(f"\nUsing IP: {ip_name}")
    print("\nRegister map:")
    print(ip.register_map)

    ip = FastStatefulDqnIP(ip)

    seed = 0

    user_operation = ask_user_operation()

    print("\nInitializing FPGA DQN agent...")
    initialize_fpga_agent(ip, seed)

    if user_operation == "infer":
        valid_weights, reason = is_valid_weight_file(WEIGHT_FILE)
        load_path = WEIGHT_FILE

        if not valid_weights:
            recovery_path = WEIGHT_FILE.with_name(WEIGHT_FILE.stem + "_recovery.npz")
            valid_recovery, recovery_reason = is_valid_weight_file(recovery_path)
            if valid_recovery:
                print(f"\nMain weight file is invalid: {WEIGHT_FILE}")
                print(f"Reason: {reason}")
                print(f"Using recovery weight file instead: {recovery_path}")
                load_path = recovery_path
                valid_weights = True
            else:
                backup_path = WEIGHT_FILE.with_name(WEIGHT_FILE.stem + "_backup.npz")
                valid_backup, backup_reason = is_valid_weight_file(backup_path)
                if valid_backup:
                    print("\nMain and recovery weight files are invalid.")
                    print(f"Using backup weight file instead: {backup_path}")
                    load_path = backup_path
                    valid_weights = True
                else:
                    print(f"\nNo valid saved weight file can be used: {WEIGHT_FILE}")
                    print(f"Main reason: {reason}")
                    print(f"Recovery reason: {recovery_reason}")
                    print(f"Backup reason: {backup_reason}")
                    if WEIGHT_FILE.exists():
                        bad_path = WEIGHT_FILE.with_suffix(WEIGHT_FILE.suffix + ".bad")
                        if bad_path.exists():
                            bad_path.unlink()
                        WEIGHT_FILE.rename(bad_path)
                        print(f"The corrupt file was renamed to: {bad_path}")
                    print("Training must be performed first. Switching automatically to training mode.")

        if valid_weights:
            print(f"\nValid saved weight file found: {load_path}")
            print("Loading weights into FPGA...")
            load_weights_to_fpga(ip, load_path)
            print("Starting inference/evaluation with loaded weights.")
            run_final_evaluation_and_save(ip, seed, output_tag="loaded_weights_inference")
            print(f"\nSaved inference CSV files to: {OUTPUT_DIR}")
            return

    training_rows = []
    validation_rows = []

    start_time = time.perf_counter()

    print("\nStarting full training...")

    for episode in range(MAX_EPISODES):
        theta, theta_dot, state = reset_training_environment(episode)
        epsilon = epsilon_for_episode(episode)

        total_reward = 0.0
        replay_count = 0
        global_step = 0
        action = call_ip_stateful_setup_mode7(
            ip=ip,
            seed=seed,
            initial_state=state,
            epsilon=epsilon,
        )

        for step in range(MAX_STEPS):
            prev_action = action

            theta, theta_dot, state, reward, done = step_environment(
                theta, theta_dot, prev_action
            )

            action = call_ip_stateful_mode6_fast(
                ip=ip,
                reward=reward,
                current_state=state,
                done=done,
                epsilon=epsilon,
            )

            total_reward += reward

        replay_count, global_step = read_diagnostics(ip)

        avg_reward_per_step = total_reward / float(MAX_STEPS)

        training_rows.append({
            "episode": episode + 1,
            "episode_length": MAX_STEPS,
            "total_reward": total_reward,
            "average_reward_per_step": avg_reward_per_step,
            "epsilon": epsilon,
            "global_step": global_step,
            "replay_count": replay_count,
        })

        if (episode + 1) % 100 == 0:
            elapsed = time.perf_counter() - start_time
            print(
                f"Episode {episode + 1:5d}/{MAX_EPISODES} | "
                f"Reward {total_reward:9.3f} | "
                f"Avg/step {avg_reward_per_step:8.4f} | "
                f"Replay {replay_count:5d} | "
                f"Global {global_step:7d} | "
                f"Elapsed {elapsed:8.1f}s"
            )

        if (episode + 1) % VALIDATION_INTERVAL == 0:
            val = run_greedy_episode(ip, seed, PI, 0.0)
            val["training_episode"] = episode + 1
            validation_rows.append(val)

            print(
                f"VALIDATION episode {episode + 1}: "
                f"reward={val['total_reward']:.3f}, "
                f"stable={val['stable_steps']}, "
                f"maxStable={val['max_consecutive_stable']}, "
                f"success={val['success']}"
            )

    total_time = time.perf_counter() - start_time
    total_steps = MAX_EPISODES * MAX_STEPS

    print("\nTraining complete.")
    print(f"Total training wall time: {total_time:.3f} s")
    print(f"Total training steps: {total_steps}")
    print(f"Average time per step: {(total_time / total_steps) * 1000.0:.6f} ms")
    print(f"Steps per second: {total_steps / total_time:.3f}")

    print("\nSaving trained weights for inference after reboot...")
    save_weights_from_fpga(ip, WEIGHT_FILE)

    evaluation_rows = run_final_evaluation_and_save(ip, seed, output_tag="trained_weights_final_eval")

    save_csv(
        OUTPUT_DIR / "training_results_stateful_mode6.csv",
        training_rows,
        [
            "episode",
            "episode_length",
            "total_reward",
            "average_reward_per_step",
            "epsilon",
            "global_step",
            "replay_count",
        ],
    )

    save_csv(
        OUTPUT_DIR / "validation_results_stateful_mode6.csv",
        validation_rows,
        [
            "training_episode",
            "total_reward",
            "upright_steps",
            "stable_steps",
            "max_consecutive_stable",
            "success",
            "min_abs_theta",
            "final_theta_wrapped",
            "final_theta_unwrapped",
            "final_theta_dot",
        ],
    )

    save_csv(
        OUTPUT_DIR / "evaluation_results_stateful_mode6.csv",
        evaluation_rows,
        [
            "case",
            "initial_theta",
            "initial_theta_dot",
            "total_reward",
            "upright_steps",
            "stable_steps",
            "max_consecutive_stable",
            "success",
            "min_abs_theta",
            "final_theta_wrapped",
            "final_theta_unwrapped",
            "final_theta_dot",
        ],
    )

    print(f"\nSaved CSV files to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
