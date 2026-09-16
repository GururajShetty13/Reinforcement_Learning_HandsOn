#include <hls_math.h>
#include <ap_int.h>

#define STATE_SIZE 3
#define HIDDEN_SIZE 32
#define ACTION_SIZE 7

#define REPLAY_SIZE 20000
#define BATCH_SIZE 64

#define LEARNING_STARTS 1000
#define TRAIN_FREQUENCY 4
#define TARGET_UPDATE_FREQUENCY 1000

#define GAMMA 0.99f
#define LEARNING_RATE 0.0001f
#define BETA1 0.9f
#define BETA2 0.999f
#define ADAM_EPS 1.0e-8f

#define PARALLEL_FACTOR 2


// Utility functions

static unsigned int g_seed = 42u;

static float rand_uniform() {
#pragma HLS INLINE
    g_seed = 1664525u * g_seed + 1013904223u;
    return ((g_seed >> 8) & 0x00FFFFFF) / 16777216.0f;
}

static int rand_int(int max_value) {
#pragma HLS INLINE
    if (max_value <= 1) return 0;
    float r = rand_uniform();
    int value = (int)(r * (float)max_value);
    if (value < 0) value = 0;
    if (value >= max_value) value = max_value - 1;
    return value;
}

static float relu(float x) {
#pragma HLS INLINE
    return (x > 0.0f) ? x : 0.0f;
}

static float relu_grad(float x) {
#pragma HLS INLINE
    return (x > 0.0f) ? 1.0f : 0.0f;
}


// Network parameters and memories

static float w1[STATE_SIZE][HIDDEN_SIZE];
static float b1[HIDDEN_SIZE];
static float w2[HIDDEN_SIZE][HIDDEN_SIZE];
static float b2[HIDDEN_SIZE];
static float w3[HIDDEN_SIZE][ACTION_SIZE];
static float b3[ACTION_SIZE];

static float target_w1[STATE_SIZE][HIDDEN_SIZE];
static float target_b1[HIDDEN_SIZE];
static float target_w2[HIDDEN_SIZE][HIDDEN_SIZE];
static float target_b2[HIDDEN_SIZE];
static float target_w3[HIDDEN_SIZE][ACTION_SIZE];
static float target_b3[ACTION_SIZE];

static float mw1[STATE_SIZE][HIDDEN_SIZE];
static float vw1[STATE_SIZE][HIDDEN_SIZE];
static float mb1[HIDDEN_SIZE];
static float vb1[HIDDEN_SIZE];

static float mw2[HIDDEN_SIZE][HIDDEN_SIZE];
static float vw2[HIDDEN_SIZE][HIDDEN_SIZE];
static float mb2[HIDDEN_SIZE];
static float vb2[HIDDEN_SIZE];

static float mw3[HIDDEN_SIZE][ACTION_SIZE];
static float vw3[HIDDEN_SIZE][ACTION_SIZE];
static float mb3[ACTION_SIZE];
static float vb3[ACTION_SIZE];

static float replay_state[REPLAY_SIZE][STATE_SIZE];
static float replay_next_state[REPLAY_SIZE][STATE_SIZE];
static float replay_reward[REPLAY_SIZE];
static int replay_action[REPLAY_SIZE];
static int replay_done[REPLAY_SIZE];

static int replay_pointer = 0;
static int replay_count = 0;
static int global_step = 0;
static float beta1_power = 1.0f;
static float beta2_power = 1.0f;
static float stateful_prev_state[STATE_SIZE];
static int stateful_prev_action = 0;
static int stateful_valid = 0;


// Forward path

static void forward_network_qonly(
    const float state[STATE_SIZE],
    const float w1_in[STATE_SIZE][HIDDEN_SIZE],
    const float b1_in[HIDDEN_SIZE],
    const float w2_in[HIDDEN_SIZE][HIDDEN_SIZE],
    const float b2_in[HIDDEN_SIZE],
    const float w3_in[HIDDEN_SIZE][ACTION_SIZE],
    const float b3_in[ACTION_SIZE],
    float q_values[ACTION_SIZE]
) {
#pragma HLS INLINE off
    float h1_local[HIDDEN_SIZE];
    float h2_local[HIDDEN_SIZE];

#pragma HLS ARRAY_PARTITION variable=state complete dim=1
#pragma HLS ARRAY_PARTITION variable=q_values cyclic factor=PARALLEL_FACTOR dim=1
#pragma HLS ARRAY_PARTITION variable=h1_local cyclic factor=PARALLEL_FACTOR dim=1
#pragma HLS ARRAY_PARTITION variable=h2_local cyclic factor=PARALLEL_FACTOR dim=1

SHARED_QONLY_L1_NEURON_LOOP:
    for (int j = 0; j < HIDDEN_SIZE; j++) {
#pragma HLS UNROLL factor=PARALLEL_FACTOR
        float sum = b1_in[j];
    SHARED_QONLY_L1_INPUT_LOOP:
        for (int i = 0; i < STATE_SIZE; i++) {
#pragma HLS PIPELINE II=1
            sum += state[i] * w1_in[i][j];
        }
        h1_local[j] = relu(sum);
    }

    float acc2_qonly[HIDDEN_SIZE];
#pragma HLS ARRAY_PARTITION variable=acc2_qonly cyclic factor=PARALLEL_FACTOR dim=1

SHARED_QONLY_L2_INIT_LOOP:
    for (int j = 0; j < HIDDEN_SIZE; j++) {
#pragma HLS UNROLL factor=PARALLEL_FACTOR
        acc2_qonly[j] = b2_in[j];
    }

SHARED_QONLY_L2_ROW_LOOP:
    for (int i = 0; i < HIDDEN_SIZE; i++) {
    SHARED_QONLY_L2_COL_LOOP:
        for (int j = 0; j < HIDDEN_SIZE; j++) {
#pragma HLS PIPELINE II=1
            acc2_qonly[j] += h1_local[i] * w2_in[i][j];
        }
    }

SHARED_QONLY_L2_ACT_LOOP:
    for (int j = 0; j < HIDDEN_SIZE; j++) {
#pragma HLS UNROLL factor=PARALLEL_FACTOR
        h2_local[j] = relu(acc2_qonly[j]);
    }

SHARED_QONLY_OUT_PAIR_LOOP:
    for (int a = 0; a < ACTION_SIZE; a += 2) {
        float sum0 = b3_in[a];
        float sum1 = 0.0f;
        if ((a + 1) < ACTION_SIZE) {
            sum1 = b3_in[a + 1];
        }
    SHARED_QONLY_OUT_PAIR_INPUT_LOOP:
        for (int i = 0; i < HIDDEN_SIZE; i++) {
#pragma HLS PIPELINE II=1
            sum0 += h2_local[i] * w3_in[i][a];
            if ((a + 1) < ACTION_SIZE) {
                sum1 += h2_local[i] * w3_in[i][a + 1];
            }
        }
        q_values[a] = sum0;
        if ((a + 1) < ACTION_SIZE) {
            q_values[a + 1] = sum1;
        }
    }
}

static void forward_main_action(const float state[STATE_SIZE], float q_values[ACTION_SIZE]) {
#pragma HLS INLINE off
    forward_network_qonly(
        state,
        w1, b1,
        w2, b2,
        w3, b3,
        q_values
    );
}

static void forward_main_train_engine(
    const float state[STATE_SIZE],
    float q_values[ACTION_SIZE],
    float h1_out[HIDDEN_SIZE],
    float h2_out[HIDDEN_SIZE],
    float z1_out[HIDDEN_SIZE],
    float z2_out[HIDDEN_SIZE]
) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=state complete dim=1
#pragma HLS ARRAY_PARTITION variable=q_values cyclic factor=PARALLEL_FACTOR dim=1
#pragma HLS ARRAY_PARTITION variable=h1_out cyclic factor=PARALLEL_FACTOR dim=1
#pragma HLS ARRAY_PARTITION variable=h2_out cyclic factor=PARALLEL_FACTOR dim=1
#pragma HLS ARRAY_PARTITION variable=z1_out cyclic factor=PARALLEL_FACTOR dim=1
#pragma HLS ARRAY_PARTITION variable=z2_out cyclic factor=PARALLEL_FACTOR dim=1

MAIN_ENGINE_L1_NEURON_LOOP:
    for (int j = 0; j < HIDDEN_SIZE; j++) {
#pragma HLS UNROLL factor=PARALLEL_FACTOR
        float sum = b1[j];
    MAIN_ENGINE_L1_INPUT_LOOP:
        for (int i = 0; i < STATE_SIZE; i++) {
#pragma HLS PIPELINE II=1
            sum += state[i] * w1[i][j];
        }
        z1_out[j] = sum;
        h1_out[j] = relu(sum);
    }

    float acc2_main[HIDDEN_SIZE];
#pragma HLS ARRAY_PARTITION variable=acc2_main cyclic factor=PARALLEL_FACTOR dim=1

MAIN_ENGINE_L2_INIT_LOOP:
    for (int j = 0; j < HIDDEN_SIZE; j++) {
#pragma HLS UNROLL factor=PARALLEL_FACTOR
        acc2_main[j] = b2[j];
    }

MAIN_ENGINE_L2_ROW_LOOP:
    for (int i = 0; i < HIDDEN_SIZE; i++) {
    MAIN_ENGINE_L2_COL_LOOP:
        for (int j = 0; j < HIDDEN_SIZE; j++) {
#pragma HLS PIPELINE II=1
            acc2_main[j] += h1_out[i] * w2[i][j];
        }
    }

MAIN_ENGINE_L2_ACT_LOOP:
    for (int j = 0; j < HIDDEN_SIZE; j++) {
#pragma HLS UNROLL factor=PARALLEL_FACTOR
        z2_out[j] = acc2_main[j];
        h2_out[j] = relu(acc2_main[j]);
    }

MAIN_ENGINE_OUT_PAIR_LOOP:
    for (int a = 0; a < ACTION_SIZE; a += 2) {
        float sum0 = b3[a];
        float sum1 = 0.0f;
        if ((a + 1) < ACTION_SIZE) {
            sum1 = b3[a + 1];
        }
    MAIN_ENGINE_OUT_PAIR_INPUT_LOOP:
        for (int i = 0; i < HIDDEN_SIZE; i++) {
#pragma HLS PIPELINE II=1
            sum0 += h2_out[i] * w3[i][a];
            if ((a + 1) < ACTION_SIZE) {
                sum1 += h2_out[i] * w3[i][a + 1];
            }
        }
        q_values[a] = sum0;
        if ((a + 1) < ACTION_SIZE) {
            q_values[a + 1] = sum1;
        }
    }
}

static void forward_target_qonly_engine(
    const float next_state[STATE_SIZE],
    float target_q_values[ACTION_SIZE]
) {
#pragma HLS INLINE off
    float h1_target[HIDDEN_SIZE];
    float h2_target[HIDDEN_SIZE];

#pragma HLS ARRAY_PARTITION variable=next_state complete dim=1
#pragma HLS ARRAY_PARTITION variable=target_q_values cyclic factor=PARALLEL_FACTOR dim=1
#pragma HLS ARRAY_PARTITION variable=h1_target cyclic factor=PARALLEL_FACTOR dim=1
#pragma HLS ARRAY_PARTITION variable=h2_target cyclic factor=PARALLEL_FACTOR dim=1

TARGET_ENGINE_L1_NEURON_LOOP:
    for (int j = 0; j < HIDDEN_SIZE; j++) {
#pragma HLS UNROLL factor=PARALLEL_FACTOR
        float sum = target_b1[j];
    TARGET_ENGINE_L1_INPUT_LOOP:
        for (int i = 0; i < STATE_SIZE; i++) {
#pragma HLS PIPELINE II=1
            sum += next_state[i] * target_w1[i][j];
        }
        h1_target[j] = relu(sum);
    }

    float acc2_target[HIDDEN_SIZE];
#pragma HLS ARRAY_PARTITION variable=acc2_target cyclic factor=PARALLEL_FACTOR dim=1

TARGET_ENGINE_L2_INIT_LOOP:
    for (int j = 0; j < HIDDEN_SIZE; j++) {
#pragma HLS UNROLL factor=PARALLEL_FACTOR
        acc2_target[j] = target_b2[j];
    }

TARGET_ENGINE_L2_ROW_LOOP:
    for (int i = 0; i < HIDDEN_SIZE; i++) {
    TARGET_ENGINE_L2_COL_LOOP:
        for (int j = 0; j < HIDDEN_SIZE; j++) {
#pragma HLS PIPELINE II=1
            acc2_target[j] += h1_target[i] * target_w2[i][j];
        }
    }

TARGET_ENGINE_L2_ACT_LOOP:
    for (int j = 0; j < HIDDEN_SIZE; j++) {
#pragma HLS UNROLL factor=PARALLEL_FACTOR
        h2_target[j] = relu(acc2_target[j]);
    }

TARGET_ENGINE_OUT_PAIR_LOOP:
    for (int a = 0; a < ACTION_SIZE; a += 2) {
        float sum0 = target_b3[a];
        float sum1 = 0.0f;
        if ((a + 1) < ACTION_SIZE) {
            sum1 = target_b3[a + 1];
        }
    TARGET_ENGINE_OUT_PAIR_INPUT_LOOP:
        for (int i = 0; i < HIDDEN_SIZE; i++) {
#pragma HLS PIPELINE II=1
            sum0 += h2_target[i] * target_w3[i][a];
            if ((a + 1) < ACTION_SIZE) {
                sum1 += h2_target[i] * target_w3[i][a + 1];
            }
        }
        target_q_values[a] = sum0;
        if ((a + 1) < ACTION_SIZE) {
            target_q_values[a + 1] = sum1;
        }
    }
}

static void forward_main_target_parallel(
    const float state[STATE_SIZE],
    const float next_state[STATE_SIZE],
    float q_values[ACTION_SIZE],
    float target_q_values[ACTION_SIZE],
    float h1_local[HIDDEN_SIZE],
    float h2_local[HIDDEN_SIZE],
    float z1_local[HIDDEN_SIZE],
    float z2_local[HIDDEN_SIZE]
) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW

    forward_main_train_engine(
        state,
        q_values,
        h1_local,
        h2_local,
        z1_local,
        z2_local
    );

    forward_target_qonly_engine(
        next_state,
        target_q_values
    );
}

static int argmax_action(const float q_values[ACTION_SIZE]) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=q_values cyclic factor=PARALLEL_FACTOR dim=1
    int best_action = 0;
    float best_value = q_values[0];
ARGMAX_LOOP:
    for (int a = 1; a < ACTION_SIZE; a++) {
#pragma HLS UNROLL factor=PARALLEL_FACTOR
        if (q_values[a] > best_value) {
            best_value = q_values[a];
            best_action = a;
        }
    }
    return best_action;
}

static int select_action(const float state[STATE_SIZE], float epsilon) {
#pragma HLS INLINE off
    float q_values[ACTION_SIZE];
#pragma HLS ARRAY_PARTITION variable=state complete dim=1
#pragma HLS ARRAY_PARTITION variable=q_values cyclic factor=PARALLEL_FACTOR dim=1

    if (rand_uniform() < epsilon) {
        return rand_int(ACTION_SIZE);
    }

    forward_main_action(state, q_values);
    return argmax_action(q_values);
}

static int select_greedy_action(const float state[STATE_SIZE]) {
#pragma HLS INLINE off
    float q_values[ACTION_SIZE];
#pragma HLS ARRAY_PARTITION variable=state complete dim=1
#pragma HLS ARRAY_PARTITION variable=q_values cyclic factor=PARALLEL_FACTOR dim=1

    forward_main_action(state, q_values);
    return argmax_action(q_values);
}


// Replay and target-network helpers

static void store_transition(const float state[STATE_SIZE], int action, float reward, const float next_state[STATE_SIZE], int done) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=state complete dim=1
#pragma HLS ARRAY_PARTITION variable=next_state complete dim=1
    int idx = replay_pointer;
STORE_STATE_LOOP:
    for (int i = 0; i < STATE_SIZE; i++) {
#pragma HLS UNROLL
        replay_state[idx][i] = state[i];
        replay_next_state[idx][i] = next_state[i];
    }
    if (action < 0) action = 0;
    if (action >= ACTION_SIZE) action = ACTION_SIZE - 1;
    replay_action[idx] = action;
    replay_reward[idx] = reward;
    replay_done[idx] = done;

    replay_pointer++;
    if (replay_pointer >= REPLAY_SIZE) replay_pointer = 0;
    if (replay_count < REPLAY_SIZE) replay_count++;
}

static void copy_main_to_target() {
#pragma HLS INLINE off
COPY_W1_I:
    for (int i = 0; i < STATE_SIZE; i++) {
    COPY_W1_J:
        for (int j = 0; j < HIDDEN_SIZE; j++) {
#pragma HLS PIPELINE II=1
            target_w1[i][j] = w1[i][j];
        }
    }
COPY_B1:
    for (int j = 0; j < HIDDEN_SIZE; j++) {
#pragma HLS PIPELINE II=1
        target_b1[j] = b1[j];
    }
COPY_W2_I:
    for (int i = 0; i < HIDDEN_SIZE; i++) {
    COPY_W2_J:
        for (int j = 0; j < HIDDEN_SIZE; j++) {
#pragma HLS PIPELINE II=1
            target_w2[i][j] = w2[i][j];
        }
    }
COPY_B2:
    for (int j = 0; j < HIDDEN_SIZE; j++) {
#pragma HLS PIPELINE II=1
        target_b2[j] = b2[j];
    }
COPY_W3_I:
    for (int i = 0; i < HIDDEN_SIZE; i++) {
    COPY_W3_A:
        for (int a = 0; a < ACTION_SIZE; a++) {
#pragma HLS PIPELINE II=1
            target_w3[i][a] = w3[i][a];
        }
    }
COPY_B3:
    for (int a = 0; a < ACTION_SIZE; a++) {
#pragma HLS PIPELINE II=1
        target_b3[a] = b3[a];
    }
}


// Training update

static void adam_update_scalar(
    float grad,
    float &param,
    float &m,
    float &v,
    float inv_bias1,
    float inv_bias2
) {
#pragma HLS INLINE
    m = BETA1 * m + (1.0f - BETA1) * grad;
    v = BETA2 * v + (1.0f - BETA2) * grad * grad;

    float m_hat = m * inv_bias1;
    float v_hat = v * inv_bias2;
    float inv_denom = 1.0f / (hls::sqrtf(v_hat) + ADAM_EPS);

    param += LEARNING_RATE * m_hat * inv_denom;
}

static float train_one_sample_adam() {
#pragma HLS INLINE off
    float state[STATE_SIZE];
    float next_state[STATE_SIZE];
    float q_values[ACTION_SIZE];
    float target_q_values[ACTION_SIZE];
    float h1_local[HIDDEN_SIZE];
    float h2_local[HIDDEN_SIZE];
    float z1_local[HIDDEN_SIZE];
    float z2_local[HIDDEN_SIZE];
    float delta_out[ACTION_SIZE];
    float delta_h2[HIDDEN_SIZE];
    float delta_h1[HIDDEN_SIZE];
#pragma HLS ARRAY_PARTITION variable=state complete dim=1
#pragma HLS ARRAY_PARTITION variable=next_state complete dim=1
#pragma HLS ARRAY_PARTITION variable=q_values cyclic factor=PARALLEL_FACTOR dim=1
#pragma HLS ARRAY_PARTITION variable=target_q_values cyclic factor=PARALLEL_FACTOR dim=1
#pragma HLS ARRAY_PARTITION variable=h1_local cyclic factor=PARALLEL_FACTOR dim=1
#pragma HLS ARRAY_PARTITION variable=h2_local cyclic factor=PARALLEL_FACTOR dim=1
#pragma HLS ARRAY_PARTITION variable=z1_local cyclic factor=PARALLEL_FACTOR dim=1
#pragma HLS ARRAY_PARTITION variable=z2_local cyclic factor=PARALLEL_FACTOR dim=1
#pragma HLS ARRAY_PARTITION variable=delta_out cyclic factor=PARALLEL_FACTOR dim=1
#pragma HLS ARRAY_PARTITION variable=delta_h2 cyclic factor=PARALLEL_FACTOR dim=1
#pragma HLS ARRAY_PARTITION variable=delta_h1 cyclic factor=PARALLEL_FACTOR dim=1

    int idx = rand_int(replay_count);
LOAD_TRANSITION:
    for (int i = 0; i < STATE_SIZE; i++) {
#pragma HLS UNROLL
        state[i] = replay_state[idx][i];
        next_state[i] = replay_next_state[idx][i];
    }
    int action = replay_action[idx];
    float reward = replay_reward[idx];
    int done = replay_done[idx];
    if (action < 0) action = 0;
    if (action >= ACTION_SIZE) action = ACTION_SIZE - 1;

    forward_main_target_parallel(
        state,
        next_state,
        q_values,
        target_q_values,
        h1_local,
        h2_local,
        z1_local,
        z2_local
    );

    float max_next_q = target_q_values[0];
MAX_NEXT_LOOP:
    for (int a = 1; a < ACTION_SIZE; a++) {
#pragma HLS UNROLL factor=PARALLEL_FACTOR
        if (target_q_values[a] > max_next_q) max_next_q = target_q_values[a];
    }

    float target_value = done ? reward : (reward + GAMMA * max_next_q);
    float q_selected = q_values[action];
    float td_error = target_value - q_selected;

INIT_DELTA_OUT:
    for (int a = 0; a < ACTION_SIZE; a++) {
#pragma HLS UNROLL factor=PARALLEL_FACTOR
        delta_out[a] = 0.0f;
    }
    delta_out[action] = td_error;

BP_H2_LOOP:
    for (int i = 0; i < HIDDEN_SIZE; i++) {
#pragma HLS UNROLL factor=PARALLEL_FACTOR
        float sum = td_error * w3[i][action];
        delta_h2[i] = sum * relu_grad(z2_local[i]);
    }

BP_H1_LOOP:
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        if (z1_local[i] > 0.0f) {
            float sum0 = 0.0f;
            float sum1 = 0.0f;
        BP_H1_H2_LOOP:
            for (int j = 0; j < HIDDEN_SIZE; j += 2) {
#pragma HLS PIPELINE II=1
                sum0 += delta_h2[j] * w2[i][j];
                sum1 += delta_h2[j + 1] * w2[i][j + 1];
            }
            delta_h1[i] = sum0 + sum1;
        }
        else {
            delta_h1[i] = 0.0f;
        }
    }

    beta1_power *= BETA1;
    beta2_power *= BETA2;

    float inv_bias1 = 1.0f / (1.0f - beta1_power);
    float inv_bias2 = 1.0f / (1.0f - beta2_power);

UPDATE_W3_I:
    for (int i = 0; i < HIDDEN_SIZE; i++) {
#pragma HLS PIPELINE II=1
        float grad = td_error * h2_local[i];
        adam_update_scalar(grad, w3[i][action], mw3[i][action], vw3[i][action], inv_bias1, inv_bias2);
    }

UPDATE_B3_SELECTED:
    adam_update_scalar(td_error, b3[action], mb3[action], vb3[action], inv_bias1, inv_bias2);

UPDATE_W2_I:
    for (int i = 0; i < HIDDEN_SIZE; i++) {
    UPDATE_W2_J_PAIR:
        for (int j = 0; j < HIDDEN_SIZE; j += 2) {
#pragma HLS PIPELINE II=1
            if (h1_local[i] != 0.0f) {
                if (delta_h2[j] != 0.0f) {
                    float grad0 = delta_h2[j] * h1_local[i];
                    adam_update_scalar(
                        grad0,
                        w2[i][j],
                        mw2[i][j],
                        vw2[i][j],
                        inv_bias1,
                        inv_bias2
                    );
                }

                if (delta_h2[j + 1] != 0.0f) {
                    float grad1 = delta_h2[j + 1] * h1_local[i];
                    adam_update_scalar(
                        grad1,
                        w2[i][j + 1],
                        mw2[i][j + 1],
                        vw2[i][j + 1],
                        inv_bias1,
                        inv_bias2
                    );
                }
            }
        }
    }
UPDATE_B2:
    for (int j = 0; j < HIDDEN_SIZE; j++) {
#pragma HLS PIPELINE II=1
        if (delta_h2[j] != 0.0f) {
            adam_update_scalar(delta_h2[j], b2[j], mb2[j], vb2[j], inv_bias1, inv_bias2);
        }
    }

UPDATE_W1_I:
    for (int i = 0; i < STATE_SIZE; i++) {
    UPDATE_W1_J:
        for (int j = 0; j < HIDDEN_SIZE; j++) {
#pragma HLS PIPELINE II=1
            if (delta_h1[j] != 0.0f && state[i] != 0.0f) {
                float grad = delta_h1[j] * state[i];
                adam_update_scalar(grad, w1[i][j], mw1[i][j], vw1[i][j], inv_bias1, inv_bias2);
            }
        }
    }
UPDATE_B1:
    for (int j = 0; j < HIDDEN_SIZE; j++) {
#pragma HLS PIPELINE II=1
        if (delta_h1[j] != 0.0f) {
            adam_update_scalar(delta_h1[j], b1[j], mb1[j], vb1[j], inv_bias1, inv_bias2);
        }
    }

    return td_error;
}

static float train_batch_if_needed() {
#pragma HLS INLINE off
    float td_sum = 0.0f;
    if (replay_count >= LEARNING_STARTS && (global_step % TRAIN_FREQUENCY) == 0) {
    BATCH_LOOP:
        for (int b = 0; b < BATCH_SIZE; b++) {
#pragma HLS LOOP_TRIPCOUNT min=32 max=32
            float td = train_one_sample_adam();
            td_sum += hls::fabsf(td);
        }
        return td_sum / (float)BATCH_SIZE;
    }
    return 0.0f;
}


// Initialization

static void initialize_agent(unsigned int seed_value) {
#pragma HLS INLINE off
    g_seed = seed_value;
    if (g_seed == 0u) g_seed = 42u;
    replay_pointer = 0;
    replay_count = 0;
    global_step = 0;
    beta1_power = 1.0f;
    beta2_power = 1.0f;

INIT_W1_I:
    for (int i = 0; i < STATE_SIZE; i++) {
    INIT_W1_J:
        for (int j = 0; j < HIDDEN_SIZE; j++) {
#pragma HLS PIPELINE II=1
            float r = rand_uniform() - 0.5f;
            w1[i][j] = 0.10f * r;
            target_w1[i][j] = w1[i][j];
            mw1[i][j] = 0.0f;
            vw1[i][j] = 0.0f;
        }
    }
INIT_B1:
    for (int j = 0; j < HIDDEN_SIZE; j++) {
#pragma HLS PIPELINE II=1
        b1[j] = 0.0f;
        target_b1[j] = 0.0f;
        mb1[j] = 0.0f;
        vb1[j] = 0.0f;
    }
INIT_W2_I:
    for (int i = 0; i < HIDDEN_SIZE; i++) {
    INIT_W2_J:
        for (int j = 0; j < HIDDEN_SIZE; j++) {
#pragma HLS PIPELINE II=1
            float r = rand_uniform() - 0.5f;
            w2[i][j] = 0.10f * r;
            target_w2[i][j] = w2[i][j];
            mw2[i][j] = 0.0f;
            vw2[i][j] = 0.0f;
        }
    }
INIT_B2:
    for (int j = 0; j < HIDDEN_SIZE; j++) {
#pragma HLS PIPELINE II=1
        b2[j] = 0.0f;
        target_b2[j] = 0.0f;
        mb2[j] = 0.0f;
        vb2[j] = 0.0f;
    }
INIT_W3_I:
    for (int i = 0; i < HIDDEN_SIZE; i++) {
    INIT_W3_A:
        for (int a = 0; a < ACTION_SIZE; a++) {
#pragma HLS PIPELINE II=1
            float r = rand_uniform() - 0.5f;
            w3[i][a] = 0.10f * r;
            target_w3[i][a] = w3[i][a];
            mw3[i][a] = 0.0f;
            vw3[i][a] = 0.0f;
        }
    }
INIT_B3:
    for (int a = 0; a < ACTION_SIZE; a++) {
#pragma HLS PIPELINE II=1
        b3[a] = 0.0f;
        target_b3[a] = 0.0f;
        mb3[a] = 0.0f;
        vb3[a] = 0.0f;
    }
}


// Weight export/import

static float read_network_parameter(int layer, int i, int j) {
#pragma HLS INLINE off
    if (layer == 0) {
        if (i >= 0 && i < STATE_SIZE && j >= 0 && j < HIDDEN_SIZE) return w1[i][j];
    }
    else if (layer == 1) {
        if (j >= 0 && j < HIDDEN_SIZE) return b1[j];
    }
    else if (layer == 2) {
        if (i >= 0 && i < HIDDEN_SIZE && j >= 0 && j < HIDDEN_SIZE) return w2[i][j];
    }
    else if (layer == 3) {
        if (j >= 0 && j < HIDDEN_SIZE) return b2[j];
    }
    else if (layer == 4) {
        if (i >= 0 && i < HIDDEN_SIZE && j >= 0 && j < ACTION_SIZE) return w3[i][j];
    }
    else if (layer == 5) {
        if (j >= 0 && j < ACTION_SIZE) return b3[j];
    }
    return 0.0f;
}

static void write_network_parameter(int layer, int i, int j, float value) {
#pragma HLS INLINE off
    if (layer == 0) {
        if (i >= 0 && i < STATE_SIZE && j >= 0 && j < HIDDEN_SIZE) {
            w1[i][j] = value;
            target_w1[i][j] = value;
        }
    }
    else if (layer == 1) {
        if (j >= 0 && j < HIDDEN_SIZE) {
            b1[j] = value;
            target_b1[j] = value;
        }
    }
    else if (layer == 2) {
        if (i >= 0 && i < HIDDEN_SIZE && j >= 0 && j < HIDDEN_SIZE) {
            w2[i][j] = value;
            target_w2[i][j] = value;
        }
    }
    else if (layer == 3) {
        if (j >= 0 && j < HIDDEN_SIZE) {
            b2[j] = value;
            target_b2[j] = value;
        }
    }
    else if (layer == 4) {
        if (i >= 0 && i < HIDDEN_SIZE && j >= 0 && j < ACTION_SIZE) {
            w3[i][j] = value;
            target_w3[i][j] = value;
        }
    }
    else if (layer == 5) {
        if (j >= 0 && j < ACTION_SIZE) {
            b3[j] = value;
            target_b3[j] = value;
        }
    }
}

// Top-level AXI-Lite function

extern "C" {
void dqn_eva(
    int mode,
    unsigned int seed_value,
    float state_0,
    float state_1,
    float state_2,
    int action_in,
    float reward_in,
    float next_state_0,
    float next_state_1,
    float next_state_2,
    int done_in,
    float epsilon,
    int &action_out,
    float &td_error_out,
    int &replay_count_out,
    int &global_step_out,
    int weight_layer,
    int weight_i,
    int weight_j,
    float weight_in,
    float &weight_out
) {
#pragma HLS INTERFACE s_axilite port=mode bundle=CTRL
#pragma HLS INTERFACE s_axilite port=seed_value bundle=CTRL
#pragma HLS INTERFACE s_axilite port=state_0 bundle=CTRL
#pragma HLS INTERFACE s_axilite port=state_1 bundle=CTRL
#pragma HLS INTERFACE s_axilite port=state_2 bundle=CTRL
#pragma HLS INTERFACE s_axilite port=action_in bundle=CTRL
#pragma HLS INTERFACE s_axilite port=reward_in bundle=CTRL
#pragma HLS INTERFACE s_axilite port=next_state_0 bundle=CTRL
#pragma HLS INTERFACE s_axilite port=next_state_1 bundle=CTRL
#pragma HLS INTERFACE s_axilite port=next_state_2 bundle=CTRL
#pragma HLS INTERFACE s_axilite port=done_in bundle=CTRL
#pragma HLS INTERFACE s_axilite port=epsilon bundle=CTRL
#pragma HLS INTERFACE s_axilite port=action_out bundle=CTRL
#pragma HLS INTERFACE s_axilite port=td_error_out bundle=CTRL
#pragma HLS INTERFACE s_axilite port=replay_count_out bundle=CTRL
#pragma HLS INTERFACE s_axilite port=global_step_out bundle=CTRL
#pragma HLS INTERFACE s_axilite port=weight_layer bundle=CTRL
#pragma HLS INTERFACE s_axilite port=weight_i bundle=CTRL
#pragma HLS INTERFACE s_axilite port=weight_j bundle=CTRL
#pragma HLS INTERFACE s_axilite port=weight_in bundle=CTRL
#pragma HLS INTERFACE s_axilite port=weight_out bundle=CTRL
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL

   
    // Storage binding and array partitioning
   
#pragma HLS BIND_STORAGE variable=w1 type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=b1 type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=w2 type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=b2 type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=w3 type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=b3 type=ram_2p impl=bram

#pragma HLS BIND_STORAGE variable=target_w1 type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=target_b1 type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=target_w2 type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=target_b2 type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=target_w3 type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=target_b3 type=ram_2p impl=bram

#pragma HLS BIND_STORAGE variable=mw1 type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=vw1 type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=mb1 type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=vb1 type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=mw2 type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=vw2 type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=mb2 type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=vb2 type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=mw3 type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=vw3 type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=mb3 type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=vb3 type=ram_2p impl=bram

#pragma HLS BIND_STORAGE variable=replay_state type=ram_2p impl=uram
#pragma HLS BIND_STORAGE variable=replay_next_state type=ram_2p impl=uram
#pragma HLS BIND_STORAGE variable=replay_reward type=ram_2p impl=uram
#pragma HLS BIND_STORAGE variable=replay_action type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=replay_done type=ram_2p impl=bram

#pragma HLS ARRAY_PARTITION variable=w1 cyclic factor=PARALLEL_FACTOR dim=2
#pragma HLS ARRAY_PARTITION variable=b1 cyclic factor=PARALLEL_FACTOR dim=1
#pragma HLS ARRAY_PARTITION variable=w2 cyclic factor=PARALLEL_FACTOR dim=2
#pragma HLS ARRAY_PARTITION variable=b2 cyclic factor=PARALLEL_FACTOR dim=1
#pragma HLS ARRAY_PARTITION variable=w3 cyclic factor=PARALLEL_FACTOR dim=2
#pragma HLS ARRAY_PARTITION variable=b3 cyclic factor=PARALLEL_FACTOR dim=1

#pragma HLS ARRAY_PARTITION variable=target_w1 cyclic factor=PARALLEL_FACTOR dim=2
#pragma HLS ARRAY_PARTITION variable=target_b1 cyclic factor=PARALLEL_FACTOR dim=1
#pragma HLS ARRAY_PARTITION variable=target_w2 cyclic factor=PARALLEL_FACTOR dim=2
#pragma HLS ARRAY_PARTITION variable=target_b2 cyclic factor=PARALLEL_FACTOR dim=1
#pragma HLS ARRAY_PARTITION variable=target_w3 cyclic factor=PARALLEL_FACTOR dim=2
#pragma HLS ARRAY_PARTITION variable=target_b3 cyclic factor=PARALLEL_FACTOR dim=1

#pragma HLS ARRAY_PARTITION variable=mw1 cyclic factor=PARALLEL_FACTOR dim=2
#pragma HLS ARRAY_PARTITION variable=vw1 cyclic factor=PARALLEL_FACTOR dim=2
#pragma HLS ARRAY_PARTITION variable=mb1 cyclic factor=PARALLEL_FACTOR dim=1
#pragma HLS ARRAY_PARTITION variable=vb1 cyclic factor=PARALLEL_FACTOR dim=1
#pragma HLS ARRAY_PARTITION variable=mw2 cyclic factor=PARALLEL_FACTOR dim=2
#pragma HLS ARRAY_PARTITION variable=vw2 cyclic factor=PARALLEL_FACTOR dim=2
#pragma HLS ARRAY_PARTITION variable=mb2 cyclic factor=PARALLEL_FACTOR dim=1
#pragma HLS ARRAY_PARTITION variable=vb2 cyclic factor=PARALLEL_FACTOR dim=1
#pragma HLS ARRAY_PARTITION variable=mw3 cyclic factor=PARALLEL_FACTOR dim=2
#pragma HLS ARRAY_PARTITION variable=vw3 cyclic factor=PARALLEL_FACTOR dim=2
#pragma HLS ARRAY_PARTITION variable=mb3 cyclic factor=PARALLEL_FACTOR dim=1
#pragma HLS ARRAY_PARTITION variable=vb3 cyclic factor=PARALLEL_FACTOR dim=1

#pragma HLS ARRAY_PARTITION variable=replay_state complete dim=2
#pragma HLS ARRAY_PARTITION variable=replay_next_state complete dim=2
#pragma HLS ARRAY_PARTITION variable=stateful_prev_state complete dim=1

    float state[STATE_SIZE];
    float next_state[STATE_SIZE];
#pragma HLS ARRAY_PARTITION variable=state complete dim=1
#pragma HLS ARRAY_PARTITION variable=next_state complete dim=1

    state[0] = state_0;
    state[1] = state_1;
    state[2] = state_2;
    next_state[0] = next_state_0;
    next_state[1] = next_state_1;
    next_state[2] = next_state_2;

    action_out = 0;
    td_error_out = 0.0f;
    weight_out = 0.0f;

    
    // Runtime mode selection
  
    if (mode == 0) {
        initialize_agent(seed_value);
        stateful_prev_state[0] = 0.0f;
        stateful_prev_state[1] = 0.0f;
        stateful_prev_state[2] = 0.0f;
        stateful_prev_action = 0;
        stateful_valid = 0;
        action_out = 0;
        td_error_out = 0.0f;
    }
    else if (mode == 4) {
        action_out = select_greedy_action(state);
        td_error_out = 0.0f;
    }
    else if (mode == 7) {
        action_out = select_action(state, epsilon);
        stateful_prev_state[0] = state[0];
        stateful_prev_state[1] = state[1];
        stateful_prev_state[2] = state[2];
        stateful_prev_action = action_out;
        stateful_valid = 1;
        td_error_out = 0.0f;
    }
    else if (mode == 6) {
        if (stateful_valid == 0) {
            action_out = select_action(next_state, epsilon);
            stateful_prev_state[0] = next_state[0];
            stateful_prev_state[1] = next_state[1];
            stateful_prev_state[2] = next_state[2];
            stateful_prev_action = action_out;
            stateful_valid = 1;
            td_error_out = 0.0f;
        }
        else {
            store_transition(stateful_prev_state, stateful_prev_action, reward_in, next_state, done_in);
            td_error_out = train_batch_if_needed();
            global_step++;

            if (global_step > 0 && (global_step % TARGET_UPDATE_FREQUENCY) == 0) {
                copy_main_to_target();
            }

            if (done_in != 0) {
                action_out = 0;
                stateful_valid = 0;
            }
            else {
                action_out = select_action(next_state, epsilon);
                stateful_prev_state[0] = next_state[0];
                stateful_prev_state[1] = next_state[1];
                stateful_prev_state[2] = next_state[2];
                stateful_prev_action = action_out;
            }
        }
    }
    else if (mode == 10) {
        weight_out = read_network_parameter(weight_layer, weight_i, weight_j);
        action_out = 0;
        td_error_out = 0.0f;
    }
    else if (mode == 11) {
        write_network_parameter(weight_layer, weight_i, weight_j, weight_in);
        action_out = 0;
        td_error_out = 0.0f;
    }
    else {
        action_out = 0;
        td_error_out = 0.0f;
    }

    replay_count_out = replay_count;
    global_step_out = global_step;
}
}
