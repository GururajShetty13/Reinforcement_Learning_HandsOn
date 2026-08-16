#include <hls_math.h>
#include <stdint.h>

// ============================================================
// Configuration
// ============================================================

#define STATE_SIZE 3
#define HIDDEN1_SIZE 32
#define HIDDEN2_SIZE 32
#define ACTION_SIZE 7
#define OUTPUT_SIZE ACTION_SIZE

#define MAX_EPISODES 8000
#define MAX_STEPS 250

#define VALIDATION_INTERVAL 800
#define NUM_VALIDATIONS (MAX_EPISODES / VALIDATION_INTERVAL)
#define EVALUATION_BATCH_SIZE 5

#define VALIDATION_TRAJECTORY_SIZE (NUM_VALIDATIONS * MAX_STEPS)
#define VALIDATION_STATE_BUFFER_SIZE (NUM_VALIDATIONS * MAX_STEPS * STATE_SIZE)
#define EVALUATION_TRAJECTORY_SIZE (EVALUATION_BATCH_SIZE * MAX_STEPS)
#define EVALUATION_STATE_BUFFER_SIZE (EVALUATION_BATCH_SIZE * MAX_STEPS * STATE_SIZE)

#define GAMMA 0.99f
#define LEARNING_RATE 0.0001f

#define ADAM_BETA1 0.9f
#define ADAM_BETA2 0.999f
#define ADAM_EPS 1.0e-8f

#define EPS_START 1.0f
#define EPS_END 0.05f
#define EXPLORATION_FRACTION 0.30f

#define REPLAY_SIZE 20000
#define BATCH_SIZE 32
#define LEARNING_STARTS 1000
#define TRAIN_FREQUENCY 4
#define TARGET_UPDATE_FREQUENCY 1000

#define PI 3.14159265358979323846f
#define DT 0.05f
#define G 10.0f
#define M 1.0f
#define L 1.0f
#define MAX_SPEED 8.0f

#define THETA_UPRIGHT_THRESHOLD 0.35f
#define THETA_DOT_STABLE_THRESHOLD 0.5f
#define SUCCESS_CONSECUTIVE_STABLE_STEPS 30

#define INFERENCE_PROFILE_STEPS 1000
#define NN_FORWARD_PROFILE_STEPS 10000
#define NN_BACKPROP_PROFILE_STEPS 10000
#define NN_UPDATE_PROFILE_STEPS 10
#define NN_TRAIN_SAMPLE_PROFILE_STEPS 10
#define REPLAY_PROFILE_STEPS 10000


struct Transition {
    float state[STATE_SIZE];
    int action;
    float reward;
    float next_state[STATE_SIZE];
    int done;
};

struct EnvState {
    float theta;
    float theta_dot;
};


static unsigned int g_seed = 42u;

static float rand_uniform() {
#pragma HLS INLINE
    g_seed = 1664525u * g_seed + 1013904223u;
    return ((g_seed >> 8) & 0x00FFFFFF) / 16777216.0f;
}

static int rand_int(int maximum) {
#pragma HLS INLINE
    int value = (int)(rand_uniform() * maximum);

    if (value < 0) {
        value = 0;
    }

    if (value >= maximum) {
        value = maximum - 1;
    }

    return value;
}


static float abs_float(float value) {
#pragma HLS INLINE
    return value < 0.0f ? -value : value;
}

static float clip_float(float value, float minimum, float maximum) {
#pragma HLS INLINE
    if (value < minimum) {
        return minimum;
    }

    if (value > maximum) {
        return maximum;
    }

    return value;
}

static float wrap_angle(float angle) {
#pragma HLS INLINE
    while (angle > PI) {
        angle -= 2.0f * PI;
    }

    while (angle < -PI) {
        angle += 2.0f * PI;
    }

    return angle;
}

static float relu(float value) {
#pragma HLS INLINE
    return value > 0.0f ? value : 0.0f;
}

static float relu_derivative(float value) {
#pragma HLS INLINE
    return value > 0.0f ? 1.0f : 0.0f;
}


static const float ACTION_VALUES[ACTION_SIZE] = {
    -2.0f, -0.5f, -0.1f, 0.0f, 0.1f, 0.5f, 2.0f
};

static float action_to_torque(int action) {
#pragma HLS INLINE
    if (action < 0) {
        action = 0;
    }

    if (action >= ACTION_SIZE) {
        action = ACTION_SIZE - 1;
    }

    return ACTION_VALUES[action];
}

static void encode_observation(
    const EnvState &environment_state,
    float observation[STATE_SIZE]
) {
#pragma HLS INLINE
    observation[0] = hls::cosf(environment_state.theta);
    observation[1] = hls::sinf(environment_state.theta);
    observation[2] = environment_state.theta_dot;
}

static void reset_training_environment(
    EnvState &environment_state,
    float observation[STATE_SIZE]
) {
#pragma HLS INLINE off
    environment_state.theta =
        PI + (rand_uniform() * 2.0f - 1.0f) * 0.25f;

    environment_state.theta_dot =
        (rand_uniform() * 2.0f - 1.0f) * 1.0f;

    environment_state.theta_dot =
        clip_float(environment_state.theta_dot, -MAX_SPEED, MAX_SPEED);

    encode_observation(environment_state, observation);
}

static void reset_fixed_environment(
    EnvState &environment_state,
    float theta,
    float theta_dot,
    float observation[STATE_SIZE]
) {
#pragma HLS INLINE off
    environment_state.theta = theta;
    environment_state.theta_dot = theta_dot;

    encode_observation(environment_state, observation);
}

static float calculate_reward(
    float previous_theta,
    float new_theta,
    float new_theta_dot,
    float torque
) {
#pragma HLS INLINE off
    float previous_angle_error = wrap_angle(previous_theta);
    float angle_error = wrap_angle(new_theta);

    float upright_reward = 2.0f * hls::cosf(angle_error);
    float velocity_penalty = 0.01f * new_theta_dot * new_theta_dot;
    float torque_penalty = 0.001f * torque * torque;

    float progress_bonus =
        0.5f * (abs_float(previous_angle_error) - abs_float(angle_error));

    float bonus = 0.0f;

    if (abs_float(angle_error) < 0.3f) {
        bonus += 2.0f;
    }

    if (abs_float(angle_error) < 0.15f && abs_float(new_theta_dot) < 1.0f) {
        bonus += 4.0f;
    }

    if (abs_float(angle_error) < 0.08f && abs_float(new_theta_dot) < 0.5f) {
        bonus += 6.0f;
    }

    float reward = upright_reward - velocity_penalty - torque_penalty + progress_bonus + bonus;

    return clip_float(reward, -10.0f, 10.0f);
}

static void step_environment(
    EnvState &environment_state,
    int action,
    float next_observation[STATE_SIZE],
    float &reward,
    int &done
) {
#pragma HLS INLINE off
    float torque = action_to_torque(action);
    float previous_theta = environment_state.theta;

    float new_theta_dot =
        environment_state.theta_dot
        + (
            3.0f * G / (2.0f * L) * hls::sinf(environment_state.theta)
            + 3.0f / (M * L * L) * torque
        ) * DT;

    new_theta_dot = clip_float(new_theta_dot, -MAX_SPEED, MAX_SPEED);

    float new_theta = environment_state.theta + new_theta_dot * DT;

    reward = calculate_reward(previous_theta, new_theta, new_theta_dot, torque);

    environment_state.theta = new_theta;
    environment_state.theta_dot = new_theta_dot;

    encode_observation(environment_state, next_observation);

    done = 0;
}

static int is_upright_state(const EnvState &environment_state) {
#pragma HLS INLINE
    float theta_wrapped = wrap_angle(environment_state.theta);
    return abs_float(theta_wrapped) < THETA_UPRIGHT_THRESHOLD ? 1 : 0;
}

static int is_stable_state(const EnvState &environment_state) {
#pragma HLS INLINE
    float theta_wrapped = wrap_angle(environment_state.theta);
    return (
        abs_float(theta_wrapped) < THETA_UPRIGHT_THRESHOLD
        && abs_float(environment_state.theta_dot) < THETA_DOT_STABLE_THRESHOLD
    ) ? 1 : 0;
}

static void initialize_network(
    float w1[STATE_SIZE][HIDDEN1_SIZE],
    float b1[HIDDEN1_SIZE],
    float w2[HIDDEN1_SIZE][HIDDEN2_SIZE],
    float b2[HIDDEN2_SIZE],
    float w3[HIDDEN2_SIZE][OUTPUT_SIZE],
    float b3[OUTPUT_SIZE]
) {
#pragma HLS INLINE off
    for (int input = 0; input < STATE_SIZE; input++) {
        for (int hidden1 = 0; hidden1 < HIDDEN1_SIZE; hidden1++) {
#pragma HLS PIPELINE II=1
            float random_value = rand_uniform() * 2.0f - 1.0f;
            w1[input][hidden1] = 0.03f * random_value;
        }
    }

    for (int hidden1 = 0; hidden1 < HIDDEN1_SIZE; hidden1++) {
#pragma HLS PIPELINE II=1
        b1[hidden1] = 0.0f;
    }

    for (int hidden1 = 0; hidden1 < HIDDEN1_SIZE; hidden1++) {
        for (int hidden2 = 0; hidden2 < HIDDEN2_SIZE; hidden2++) {
#pragma HLS PIPELINE II=1
            float random_value = rand_uniform() * 2.0f - 1.0f;
            w2[hidden1][hidden2] = 0.03f * random_value;
        }
    }

    for (int hidden2 = 0; hidden2 < HIDDEN2_SIZE; hidden2++) {
#pragma HLS PIPELINE II=1
        b2[hidden2] = 0.0f;
    }

    for (int hidden2 = 0; hidden2 < HIDDEN2_SIZE; hidden2++) {
        for (int output = 0; output < OUTPUT_SIZE; output++) {
#pragma HLS PIPELINE II=1
            float random_value = rand_uniform() * 2.0f - 1.0f;
            w3[hidden2][output] = 0.03f * random_value;
        }
    }

    for (int output = 0; output < OUTPUT_SIZE; output++) {
#pragma HLS PIPELINE II=1
        b3[output] = 0.0f;
    }
}

static void initialize_adam(
    float mw1[STATE_SIZE][HIDDEN1_SIZE],
    float vw1[STATE_SIZE][HIDDEN1_SIZE],
    float mb1[HIDDEN1_SIZE],
    float vb1[HIDDEN1_SIZE],
    float mw2[HIDDEN1_SIZE][HIDDEN2_SIZE],
    float vw2[HIDDEN1_SIZE][HIDDEN2_SIZE],
    float mb2[HIDDEN2_SIZE],
    float vb2[HIDDEN2_SIZE],
    float mw3[HIDDEN2_SIZE][OUTPUT_SIZE],
    float vw3[HIDDEN2_SIZE][OUTPUT_SIZE],
    float mb3[OUTPUT_SIZE],
    float vb3[OUTPUT_SIZE]
) {
#pragma HLS INLINE off
    for (int input = 0; input < STATE_SIZE; input++) {
        for (int hidden1 = 0; hidden1 < HIDDEN1_SIZE; hidden1++) {
#pragma HLS PIPELINE II=1
            mw1[input][hidden1] = 0.0f;
            vw1[input][hidden1] = 0.0f;
        }
    }

    for (int hidden1 = 0; hidden1 < HIDDEN1_SIZE; hidden1++) {
#pragma HLS PIPELINE II=1
        mb1[hidden1] = 0.0f;
        vb1[hidden1] = 0.0f;
    }

    for (int hidden1 = 0; hidden1 < HIDDEN1_SIZE; hidden1++) {
        for (int hidden2 = 0; hidden2 < HIDDEN2_SIZE; hidden2++) {
#pragma HLS PIPELINE II=1
            mw2[hidden1][hidden2] = 0.0f;
            vw2[hidden1][hidden2] = 0.0f;
        }
    }

    for (int hidden2 = 0; hidden2 < HIDDEN2_SIZE; hidden2++) {
#pragma HLS PIPELINE II=1
        mb2[hidden2] = 0.0f;
        vb2[hidden2] = 0.0f;
    }

    for (int hidden2 = 0; hidden2 < HIDDEN2_SIZE; hidden2++) {
        for (int output = 0; output < OUTPUT_SIZE; output++) {
#pragma HLS PIPELINE II=1
            mw3[hidden2][output] = 0.0f;
            vw3[hidden2][output] = 0.0f;
        }
    }

    for (int output = 0; output < OUTPUT_SIZE; output++) {
#pragma HLS PIPELINE II=1
        mb3[output] = 0.0f;
        vb3[output] = 0.0f;
    }
}

static void copy_network(
    float source_w1[STATE_SIZE][HIDDEN1_SIZE],
    float source_b1[HIDDEN1_SIZE],
    float source_w2[HIDDEN1_SIZE][HIDDEN2_SIZE],
    float source_b2[HIDDEN2_SIZE],
    float source_w3[HIDDEN2_SIZE][OUTPUT_SIZE],
    float source_b3[OUTPUT_SIZE],
    float destination_w1[STATE_SIZE][HIDDEN1_SIZE],
    float destination_b1[HIDDEN1_SIZE],
    float destination_w2[HIDDEN1_SIZE][HIDDEN2_SIZE],
    float destination_b2[HIDDEN2_SIZE],
    float destination_w3[HIDDEN2_SIZE][OUTPUT_SIZE],
    float destination_b3[OUTPUT_SIZE]
) {
#pragma HLS INLINE off
    for (int input = 0; input < STATE_SIZE; input++) {
        for (int hidden1 = 0; hidden1 < HIDDEN1_SIZE; hidden1++) {
#pragma HLS PIPELINE II=1
            destination_w1[input][hidden1] = source_w1[input][hidden1];
        }
    }

    for (int hidden1 = 0; hidden1 < HIDDEN1_SIZE; hidden1++) {
#pragma HLS PIPELINE II=1
        destination_b1[hidden1] = source_b1[hidden1];
    }

    for (int hidden1 = 0; hidden1 < HIDDEN1_SIZE; hidden1++) {
        for (int hidden2 = 0; hidden2 < HIDDEN2_SIZE; hidden2++) {
#pragma HLS PIPELINE II=1
            destination_w2[hidden1][hidden2] = source_w2[hidden1][hidden2];
        }
    }

    for (int hidden2 = 0; hidden2 < HIDDEN2_SIZE; hidden2++) {
#pragma HLS PIPELINE II=1
        destination_b2[hidden2] = source_b2[hidden2];
    }

    for (int hidden2 = 0; hidden2 < HIDDEN2_SIZE; hidden2++) {
        for (int output = 0; output < OUTPUT_SIZE; output++) {
#pragma HLS PIPELINE II=1
            destination_w3[hidden2][output] = source_w3[hidden2][output];
        }
    }

    for (int output = 0; output < OUTPUT_SIZE; output++) {
#pragma HLS PIPELINE II=1
        destination_b3[output] = source_b3[output];
    }
}

static void forward_layer1(
    const float observation[STATE_SIZE],
    float w1[STATE_SIZE][HIDDEN1_SIZE],
    float b1[HIDDEN1_SIZE],
    float hidden1_pre_activation[HIDDEN1_SIZE],
    float hidden1_activation[HIDDEN1_SIZE]
) {
#pragma HLS INLINE off
    for (int hidden1 = 0; hidden1 < HIDDEN1_SIZE; hidden1++) {
        float sum = b1[hidden1];

        for (int input = 0; input < STATE_SIZE; input++) {
#pragma HLS PIPELINE II=1
            sum += observation[input] * w1[input][hidden1];
        }

        hidden1_pre_activation[hidden1] = sum;
        hidden1_activation[hidden1] = relu(sum);
    }
}

static void forward_layer2(
    float hidden1_activation[HIDDEN1_SIZE],
    float w2[HIDDEN1_SIZE][HIDDEN2_SIZE],
    float b2[HIDDEN2_SIZE],
    float hidden2_pre_activation[HIDDEN2_SIZE],
    float hidden2_activation[HIDDEN2_SIZE]
) {
#pragma HLS INLINE off
    for (int hidden2 = 0; hidden2 < HIDDEN2_SIZE; hidden2++) {
        float sum = b2[hidden2];

        for (int hidden1 = 0; hidden1 < HIDDEN1_SIZE; hidden1++) {
#pragma HLS PIPELINE II=1
            sum += hidden1_activation[hidden1] * w2[hidden1][hidden2];
        }

        hidden2_pre_activation[hidden2] = sum;
        hidden2_activation[hidden2] = relu(sum);
    }
}

static void forward_output_layer(
    float hidden2_activation[HIDDEN2_SIZE],
    float w3[HIDDEN2_SIZE][OUTPUT_SIZE],
    float b3[OUTPUT_SIZE],
    float q_values[OUTPUT_SIZE]
) {
#pragma HLS INLINE off
    for (int output = 0; output < OUTPUT_SIZE; output++) {
        float sum = b3[output];

        for (int hidden2 = 0; hidden2 < HIDDEN2_SIZE; hidden2++) {
#pragma HLS PIPELINE II=1
            sum += hidden2_activation[hidden2] * w3[hidden2][output];
        }

        q_values[output] = sum;
    }
}

static void forward_network(
    const float observation[STATE_SIZE],
    float w1[STATE_SIZE][HIDDEN1_SIZE],
    float b1[HIDDEN1_SIZE],
    float w2[HIDDEN1_SIZE][HIDDEN2_SIZE],
    float b2[HIDDEN2_SIZE],
    float w3[HIDDEN2_SIZE][OUTPUT_SIZE],
    float b3[OUTPUT_SIZE],
    float hidden1_pre_activation[HIDDEN1_SIZE],
    float hidden1_activation[HIDDEN1_SIZE],
    float hidden2_pre_activation[HIDDEN2_SIZE],
    float hidden2_activation[HIDDEN2_SIZE],
    float q_values[OUTPUT_SIZE]
) {
#pragma HLS INLINE off
    forward_layer1(
        observation,
        w1,
        b1,
        hidden1_pre_activation,
        hidden1_activation
    );

    forward_layer2(
        hidden1_activation,
        w2,
        b2,
        hidden2_pre_activation,
        hidden2_activation
    );

    forward_output_layer(
        hidden2_activation,
        w3,
        b3,
        q_values
    );
}

static int argmax_q(const float q_values[OUTPUT_SIZE]) {
#pragma HLS INLINE off
    int best_action = 0;
    float best_value = q_values[0];

    for (int action = 1; action < OUTPUT_SIZE; action++) {
#pragma HLS PIPELINE II=1
        if (q_values[action] > best_value) {
            best_value = q_values[action];
            best_action = action;
        }
    }

    return best_action;
}

static int select_action(
    const float observation[STATE_SIZE],
    float epsilon,
    float w1[STATE_SIZE][HIDDEN1_SIZE],
    float b1[HIDDEN1_SIZE],
    float w2[HIDDEN1_SIZE][HIDDEN2_SIZE],
    float b2[HIDDEN2_SIZE],
    float w3[HIDDEN2_SIZE][OUTPUT_SIZE],
    float b3[OUTPUT_SIZE]
) {
#pragma HLS INLINE off
    if (rand_uniform() < epsilon) {
        return rand_int(OUTPUT_SIZE);
    }

    float hidden1_pre_activation[HIDDEN1_SIZE];
    float hidden1_activation[HIDDEN1_SIZE];
    float hidden2_pre_activation[HIDDEN2_SIZE];
    float hidden2_activation[HIDDEN2_SIZE];
    float q_values[OUTPUT_SIZE];

#pragma HLS ARRAY_PARTITION variable=q_values complete dim=1

    forward_network(
        observation,
        w1,
        b1,
        w2,
        b2,
        w3,
        b3,
        hidden1_pre_activation,
        hidden1_activation,
        hidden2_pre_activation,
        hidden2_activation,
        q_values
    );

    return argmax_q(q_values);
}

static void adam_update(
    float &parameter,
    float gradient,
    float &momentum,
    float &variance,
    float beta1_power,
    float beta2_power
) {
#pragma HLS INLINE off
    momentum =
        ADAM_BETA1 * momentum
        + (1.0f - ADAM_BETA1) * gradient;

    variance =
        ADAM_BETA2 * variance
        + (1.0f - ADAM_BETA2) * gradient * gradient;

    float momentum_hat = momentum / (1.0f - beta1_power);
    float variance_hat = variance / (1.0f - beta2_power);

    parameter +=
        LEARNING_RATE * momentum_hat / (hls::sqrtf(variance_hat) + ADAM_EPS);
}

static float train_one_sample_adam(
    const Transition &transition,
    float w1[STATE_SIZE][HIDDEN1_SIZE],
    float b1[HIDDEN1_SIZE],
    float w2[HIDDEN1_SIZE][HIDDEN2_SIZE],
    float b2[HIDDEN2_SIZE],
    float w3[HIDDEN2_SIZE][OUTPUT_SIZE],
    float b3[OUTPUT_SIZE],
    float target_w1[STATE_SIZE][HIDDEN1_SIZE],
    float target_b1[HIDDEN1_SIZE],
    float target_w2[HIDDEN1_SIZE][HIDDEN2_SIZE],
    float target_b2[HIDDEN2_SIZE],
    float target_w3[HIDDEN2_SIZE][OUTPUT_SIZE],
    float target_b3[OUTPUT_SIZE],
    float mw1[STATE_SIZE][HIDDEN1_SIZE],
    float vw1[STATE_SIZE][HIDDEN1_SIZE],
    float mb1[HIDDEN1_SIZE],
    float vb1[HIDDEN1_SIZE],
    float mw2[HIDDEN1_SIZE][HIDDEN2_SIZE],
    float vw2[HIDDEN1_SIZE][HIDDEN2_SIZE],
    float mb2[HIDDEN2_SIZE],
    float vb2[HIDDEN2_SIZE],
    float mw3[HIDDEN2_SIZE][OUTPUT_SIZE],
    float vw3[HIDDEN2_SIZE][OUTPUT_SIZE],
    float mb3[OUTPUT_SIZE],
    float vb3[OUTPUT_SIZE],
    float beta1_power,
    float beta2_power
) {
#pragma HLS INLINE off
    float hidden1_pre_activation[HIDDEN1_SIZE];
    float hidden1_activation[HIDDEN1_SIZE];
    float hidden2_pre_activation[HIDDEN2_SIZE];
    float hidden2_activation[HIDDEN2_SIZE];
    float q_values[OUTPUT_SIZE];

#pragma HLS ARRAY_PARTITION variable=q_values complete dim=1

    forward_network(
        transition.state,
        w1,
        b1,
        w2,
        b2,
        w3,
        b3,
        hidden1_pre_activation,
        hidden1_activation,
        hidden2_pre_activation,
        hidden2_activation,
        q_values
    );

    float target_hidden1_pre_activation[HIDDEN1_SIZE];
    float target_hidden1_activation[HIDDEN1_SIZE];
    float target_hidden2_pre_activation[HIDDEN2_SIZE];
    float target_hidden2_activation[HIDDEN2_SIZE];
    float target_q_values[OUTPUT_SIZE];

#pragma HLS ARRAY_PARTITION variable=target_q_values complete dim=1

    forward_network(
        transition.next_state,
        target_w1,
        target_b1,
        target_w2,
        target_b2,
        target_w3,
        target_b3,
        target_hidden1_pre_activation,
        target_hidden1_activation,
        target_hidden2_pre_activation,
        target_hidden2_activation,
        target_q_values
    );

    float maximum_next_q = target_q_values[0];

    for (int action = 1; action < OUTPUT_SIZE; action++) {
#pragma HLS PIPELINE II=1
        if (target_q_values[action] > maximum_next_q) {
            maximum_next_q = target_q_values[action];
        }
    }

    float target_q =
        transition.reward
        + (transition.done ? 0.0f : GAMMA * maximum_next_q);

    float predicted_q = q_values[transition.action];
    float td_error = target_q - predicted_q;
    float clipped_td_error = clip_float(td_error, -1.0f, 1.0f);

    float output_delta[OUTPUT_SIZE];
#pragma HLS ARRAY_PARTITION variable=output_delta complete dim=1

    for (int output = 0; output < OUTPUT_SIZE; output++) {
#pragma HLS PIPELINE II=1
        output_delta[output] = 0.0f;
    }

    output_delta[transition.action] = clipped_td_error;

    float hidden2_delta[HIDDEN2_SIZE];
    float hidden1_delta[HIDDEN1_SIZE];

    for (int hidden2 = 0; hidden2 < HIDDEN2_SIZE; hidden2++) {
        float sum = 0.0f;

        for (int output = 0; output < OUTPUT_SIZE; output++) {
#pragma HLS PIPELINE II=1
            sum += w3[hidden2][output] * output_delta[output];
        }

        hidden2_delta[hidden2] =
            clip_float(
                sum * relu_derivative(hidden2_pre_activation[hidden2]),
                -1.0f,
                1.0f
            );
    }

    for (int hidden1 = 0; hidden1 < HIDDEN1_SIZE; hidden1++) {
        float sum = 0.0f;

        for (int hidden2 = 0; hidden2 < HIDDEN2_SIZE; hidden2++) {
#pragma HLS PIPELINE II=1
            sum += w2[hidden1][hidden2] * hidden2_delta[hidden2];
        }

        hidden1_delta[hidden1] =
            clip_float(
                sum * relu_derivative(hidden1_pre_activation[hidden1]),
                -1.0f,
                1.0f
            );
    }

    for (int hidden2 = 0; hidden2 < HIDDEN2_SIZE; hidden2++) {
        for (int output = 0; output < OUTPUT_SIZE; output++) {
#pragma HLS PIPELINE II=1
            float gradient = hidden2_activation[hidden2] * output_delta[output];

            adam_update(
                w3[hidden2][output],
                gradient,
                mw3[hidden2][output],
                vw3[hidden2][output],
                beta1_power,
                beta2_power
            );
        }
    }

    for (int output = 0; output < OUTPUT_SIZE; output++) {
#pragma HLS PIPELINE II=1
        adam_update(
            b3[output],
            output_delta[output],
            mb3[output],
            vb3[output],
            beta1_power,
            beta2_power
        );
    }

    for (int hidden1 = 0; hidden1 < HIDDEN1_SIZE; hidden1++) {
        for (int hidden2 = 0; hidden2 < HIDDEN2_SIZE; hidden2++) {
#pragma HLS PIPELINE II=1
            float gradient = hidden1_activation[hidden1] * hidden2_delta[hidden2];

            adam_update(
                w2[hidden1][hidden2],
                gradient,
                mw2[hidden1][hidden2],
                vw2[hidden1][hidden2],
                beta1_power,
                beta2_power
            );
        }
    }

    for (int hidden2 = 0; hidden2 < HIDDEN2_SIZE; hidden2++) {
#pragma HLS PIPELINE II=1
        adam_update(
            b2[hidden2],
            hidden2_delta[hidden2],
            mb2[hidden2],
            vb2[hidden2],
            beta1_power,
            beta2_power
        );
    }

    for (int input = 0; input < STATE_SIZE; input++) {
        for (int hidden1 = 0; hidden1 < HIDDEN1_SIZE; hidden1++) {
#pragma HLS PIPELINE II=1
            float gradient = transition.state[input] * hidden1_delta[hidden1];

            adam_update(
                w1[input][hidden1],
                gradient,
                mw1[input][hidden1],
                vw1[input][hidden1],
                beta1_power,
                beta2_power
            );
        }
    }

    for (int hidden1 = 0; hidden1 < HIDDEN1_SIZE; hidden1++) {
#pragma HLS PIPELINE II=1
        adam_update(
            b1[hidden1],
            hidden1_delta[hidden1],
            mb1[hidden1],
            vb1[hidden1],
            beta1_power,
            beta2_power
        );
    }

    return abs_float(td_error);
}

static float epsilon_for_episode(int episode_index) {
#pragma HLS INLINE
    float progress =
        (float)episode_index
        / (EXPLORATION_FRACTION * (float)MAX_EPISODES);

    if (progress > 1.0f) {
        progress = 1.0f;
    }

    float epsilon = EPS_START + progress * (EPS_END - EPS_START);

    return clip_float(epsilon, EPS_END, EPS_START);
}

static void store_transition(
    int index,
    float replay_state[REPLAY_SIZE][STATE_SIZE],
    int replay_action[REPLAY_SIZE],
    float replay_reward[REPLAY_SIZE],
    float replay_next_state[REPLAY_SIZE][STATE_SIZE],
    int replay_done[REPLAY_SIZE],
    const float state[STATE_SIZE],
    int action,
    float reward,
    const float next_state[STATE_SIZE],
    int done
) {
#pragma HLS INLINE off
    for (int component = 0; component < STATE_SIZE; component++) {
#pragma HLS PIPELINE II=1
        replay_state[index][component] = state[component];
        replay_next_state[index][component] = next_state[component];
    }

    replay_action[index] = action;
    replay_reward[index] = reward;
    replay_done[index] = done;
}

static void load_transition(
    int index,
    float replay_state[REPLAY_SIZE][STATE_SIZE],
    int replay_action[REPLAY_SIZE],
    float replay_reward[REPLAY_SIZE],
    float replay_next_state[REPLAY_SIZE][STATE_SIZE],
    int replay_done[REPLAY_SIZE],
    Transition &transition
) {
#pragma HLS INLINE off
    for (int component = 0; component < STATE_SIZE; component++) {
#pragma HLS PIPELINE II=1
        transition.state[component] = replay_state[index][component];
        transition.next_state[component] = replay_next_state[index][component];
    }

    transition.action = replay_action[index];
    transition.reward = replay_reward[index];
    transition.done = replay_done[index];
}

static int validation_step_index(int validation_index, int step) {
#pragma HLS INLINE
    return validation_index * MAX_STEPS + step;
}

static int validation_state_index(int validation_index, int step, int component) {
#pragma HLS INLINE
    return (validation_index * MAX_STEPS + step) * STATE_SIZE + component;
}

static int evaluation_step_index(int batch_index, int step) {
#pragma HLS INLINE
    return batch_index * MAX_STEPS + step;
}

static int evaluation_state_index(int batch_index, int step, int component) {
#pragma HLS INLINE
    return (batch_index * MAX_STEPS + step) * STATE_SIZE + component;
}

static void run_validation_episode(
    int validation_index,
    int training_episode,
    float w1[STATE_SIZE][HIDDEN1_SIZE],
    float b1[HIDDEN1_SIZE],
    float w2[HIDDEN1_SIZE][HIDDEN2_SIZE],
    float b2[HIDDEN2_SIZE],
    float w3[HIDDEN2_SIZE][OUTPUT_SIZE],
    float b3[OUTPUT_SIZE],
    int *validation_training_episode,
    float *validation_total_reward,
    float *validation_min_abs_theta,
    int *validation_upright_steps,
    int *validation_stable_upright_steps,
    int *validation_max_consecutive_stable_steps,
    int *validation_success,
    float *validation_final_theta_wrapped,
    float *validation_final_theta_unwrapped,
    float *validation_final_theta_dot,
    float *validation_states,
    int *validation_actions,
    float *validation_rewards
) {
#pragma HLS INLINE off
    EnvState environment_state;
    float observation[STATE_SIZE];

#pragma HLS ARRAY_PARTITION variable=observation complete dim=1

    reset_fixed_environment(environment_state, PI, 0.0f, observation);

    float total_reward = 0.0f;
    float minimum_abs_theta = 999.0f;
    int upright_steps = 0;
    int stable_upright_steps = 0;
    int current_consecutive_stable_steps = 0;
    int max_consecutive_stable_steps = 0;

    for (int step = 0; step < MAX_STEPS; step++) {
        float theta_wrapped = wrap_angle(environment_state.theta);
        float abs_theta = abs_float(theta_wrapped);

        if (abs_theta < minimum_abs_theta) {
            minimum_abs_theta = abs_theta;
        }

        if (is_upright_state(environment_state)) {
            upright_steps++;
        }

        if (is_stable_state(environment_state)) {
            stable_upright_steps++;
            current_consecutive_stable_steps++;

            if (current_consecutive_stable_steps > max_consecutive_stable_steps) {
                max_consecutive_stable_steps = current_consecutive_stable_steps;
            }
        } else {
            current_consecutive_stable_steps = 0;
        }

        int state_base = validation_state_index(validation_index, step, 0);

        validation_states[state_base + 0] = observation[0];
        validation_states[state_base + 1] = observation[1];
        validation_states[state_base + 2] = observation[2];

        float hidden1_pre_activation[HIDDEN1_SIZE];
        float hidden1_activation[HIDDEN1_SIZE];
        float hidden2_pre_activation[HIDDEN2_SIZE];
        float hidden2_activation[HIDDEN2_SIZE];
        float q_values[OUTPUT_SIZE];

#pragma HLS ARRAY_PARTITION variable=q_values complete dim=1

        forward_network(
            observation,
            w1,
            b1,
            w2,
            b2,
            w3,
            b3,
            hidden1_pre_activation,
            hidden1_activation,
            hidden2_pre_activation,
            hidden2_activation,
            q_values
        );

        int action = argmax_q(q_values);

        float next_observation[STATE_SIZE];
#pragma HLS ARRAY_PARTITION variable=next_observation complete dim=1
        float reward;
        int done;

        step_environment(environment_state, action, next_observation, reward, done);

        int step_index = validation_step_index(validation_index, step);

        validation_actions[step_index] = action;
        validation_rewards[step_index] = reward;

        total_reward += reward;

        for (int component = 0; component < STATE_SIZE; component++) {
#pragma HLS PIPELINE II=1
            observation[component] = next_observation[component];
        }
    }

    validation_training_episode[validation_index] = training_episode;
    validation_total_reward[validation_index] = total_reward;
    validation_min_abs_theta[validation_index] = minimum_abs_theta;
    validation_upright_steps[validation_index] = upright_steps;
    validation_stable_upright_steps[validation_index] = stable_upright_steps;
    validation_max_consecutive_stable_steps[validation_index] = max_consecutive_stable_steps;
    validation_success[validation_index] =
        max_consecutive_stable_steps >= SUCCESS_CONSECUTIVE_STABLE_STEPS ? 1 : 0;
    validation_final_theta_wrapped[validation_index] = wrap_angle(environment_state.theta);
    validation_final_theta_unwrapped[validation_index] = environment_state.theta;
    validation_final_theta_dot[validation_index] = environment_state.theta_dot;
}

static void run_evaluation_episode(
    int batch_index,
    float initial_theta,
    float initial_theta_dot,
    float w1[STATE_SIZE][HIDDEN1_SIZE],
    float b1[HIDDEN1_SIZE],
    float w2[HIDDEN1_SIZE][HIDDEN2_SIZE],
    float b2[HIDDEN2_SIZE],
    float w3[HIDDEN2_SIZE][OUTPUT_SIZE],
    float b3[OUTPUT_SIZE],
    float *evaluation_initial_theta,
    float *evaluation_initial_theta_dot,
    float *evaluation_actual_initial_theta,
    float *evaluation_actual_initial_theta_dot,
    float *evaluation_total_reward,
    float *evaluation_min_abs_theta,
    int *evaluation_upright_steps,
    int *evaluation_stable_upright_steps,
    int *evaluation_max_consecutive_stable_steps,
    int *evaluation_success,
    float *evaluation_final_theta_wrapped,
    float *evaluation_final_theta_unwrapped,
    float *evaluation_final_theta_dot,
    float *evaluation_states,
    int *evaluation_actions,
    float *evaluation_rewards
) {
#pragma HLS INLINE off
    EnvState environment_state;
    float observation[STATE_SIZE];

#pragma HLS ARRAY_PARTITION variable=observation complete dim=1

    reset_fixed_environment(environment_state, initial_theta, initial_theta_dot, observation);

    evaluation_initial_theta[batch_index] = initial_theta;
    evaluation_initial_theta_dot[batch_index] = initial_theta_dot;
    evaluation_actual_initial_theta[batch_index] = environment_state.theta;
    evaluation_actual_initial_theta_dot[batch_index] = environment_state.theta_dot;

    float total_reward = 0.0f;
    float minimum_abs_theta = 999.0f;
    int upright_steps = 0;
    int stable_upright_steps = 0;
    int current_consecutive_stable_steps = 0;
    int max_consecutive_stable_steps = 0;

    for (int step = 0; step < MAX_STEPS; step++) {
        float theta_wrapped = wrap_angle(environment_state.theta);
        float abs_theta = abs_float(theta_wrapped);

        if (abs_theta < minimum_abs_theta) {
            minimum_abs_theta = abs_theta;
        }

        if (is_upright_state(environment_state)) {
            upright_steps++;
        }

        if (is_stable_state(environment_state)) {
            stable_upright_steps++;
            current_consecutive_stable_steps++;

            if (current_consecutive_stable_steps > max_consecutive_stable_steps) {
                max_consecutive_stable_steps = current_consecutive_stable_steps;
            }
        } else {
            current_consecutive_stable_steps = 0;
        }

        int state_base = evaluation_state_index(batch_index, step, 0);

        evaluation_states[state_base + 0] = observation[0];
        evaluation_states[state_base + 1] = observation[1];
        evaluation_states[state_base + 2] = observation[2];

        float hidden1_pre_activation[HIDDEN1_SIZE];
        float hidden1_activation[HIDDEN1_SIZE];
        float hidden2_pre_activation[HIDDEN2_SIZE];
        float hidden2_activation[HIDDEN2_SIZE];
        float q_values[OUTPUT_SIZE];

#pragma HLS ARRAY_PARTITION variable=q_values complete dim=1

        forward_network(
            observation,
            w1,
            b1,
            w2,
            b2,
            w3,
            b3,
            hidden1_pre_activation,
            hidden1_activation,
            hidden2_pre_activation,
            hidden2_activation,
            q_values
        );

        int action = argmax_q(q_values);

        float next_observation[STATE_SIZE];
#pragma HLS ARRAY_PARTITION variable=next_observation complete dim=1
        float reward;
        int done;

        step_environment(environment_state, action, next_observation, reward, done);

        int step_index = evaluation_step_index(batch_index, step);

        evaluation_actions[step_index] = action;
        evaluation_rewards[step_index] = reward;

        total_reward += reward;

        for (int component = 0; component < STATE_SIZE; component++) {
#pragma HLS PIPELINE II=1
            observation[component] = next_observation[component];
        }
    }

    evaluation_total_reward[batch_index] = total_reward;
    evaluation_min_abs_theta[batch_index] = minimum_abs_theta;
    evaluation_upright_steps[batch_index] = upright_steps;
    evaluation_stable_upright_steps[batch_index] = stable_upright_steps;
    evaluation_max_consecutive_stable_steps[batch_index] = max_consecutive_stable_steps;
    evaluation_success[batch_index] =
        max_consecutive_stable_steps >= SUCCESS_CONSECUTIVE_STABLE_STEPS ? 1 : 0;
    evaluation_final_theta_wrapped[batch_index] = wrap_angle(environment_state.theta);
    evaluation_final_theta_unwrapped[batch_index] = environment_state.theta;
    evaluation_final_theta_dot[batch_index] = environment_state.theta_dot;
}


static void run_inference_profile(
    float w1[STATE_SIZE][HIDDEN1_SIZE],
    float b1[HIDDEN1_SIZE],
    float w2[HIDDEN1_SIZE][HIDDEN2_SIZE],
    float b2[HIDDEN2_SIZE],
    float w3[HIDDEN2_SIZE][OUTPUT_SIZE],
    float b3[OUTPUT_SIZE],
    int &completed_inference_steps
) {
#pragma HLS INLINE off
    EnvState environment_state;
    float observation[STATE_SIZE];
    float next_observation[STATE_SIZE];

#pragma HLS ARRAY_PARTITION variable=observation complete dim=1
#pragma HLS ARRAY_PARTITION variable=next_observation complete dim=1

    reset_fixed_environment(environment_state, PI, 0.0f, observation);

    completed_inference_steps = 0;

    for (int step = 0; step < INFERENCE_PROFILE_STEPS; step++) {
        float hidden1_pre_activation[HIDDEN1_SIZE];
        float hidden1_activation[HIDDEN1_SIZE];
        float hidden2_pre_activation[HIDDEN2_SIZE];
        float hidden2_activation[HIDDEN2_SIZE];
        float q_values[OUTPUT_SIZE];

#pragma HLS ARRAY_PARTITION variable=q_values complete dim=1

        forward_network(
            observation,
            w1,
            b1,
            w2,
            b2,
            w3,
            b3,
            hidden1_pre_activation,
            hidden1_activation,
            hidden2_pre_activation,
            hidden2_activation,
            q_values
        );

        int action = argmax_q(q_values);

        float reward;
        int done;

        step_environment(environment_state, action, next_observation, reward, done);

        for (int component = 0; component < STATE_SIZE; component++) {
#pragma HLS PIPELINE II=1
            observation[component] = next_observation[component];
        }

        completed_inference_steps = step + 1;
    }
}

static void profile_forward_layer1_only(
    float w1[STATE_SIZE][HIDDEN1_SIZE],
    float b1[HIDDEN1_SIZE],
    int &completed_steps,
    int &checksum_out
) {
#pragma HLS INLINE off
    float observation[STATE_SIZE];
    float hidden1_pre_activation[HIDDEN1_SIZE];
    float hidden1_activation[HIDDEN1_SIZE];

#pragma HLS ARRAY_PARTITION variable=observation complete dim=1

    observation[0] = 1.0f;
    observation[1] = 0.0f;
    observation[2] = 0.0f;

    float checksum = 0.0f;
    completed_steps = 0;

    for (int iteration = 0; iteration < NN_FORWARD_PROFILE_STEPS; iteration++) {
        forward_layer1(
            observation,
            w1,
            b1,
            hidden1_pre_activation,
            hidden1_activation
        );

        checksum += hidden1_activation[iteration % HIDDEN1_SIZE];
        completed_steps = iteration + 1;
    }

    checksum_out = (int)(checksum * 1000.0f);
}

static void profile_forward_layer2_only(
    float w2[HIDDEN1_SIZE][HIDDEN2_SIZE],
    float b2[HIDDEN2_SIZE],
    int &completed_steps,
    int &checksum_out
) {
#pragma HLS INLINE off
    float hidden1_activation[HIDDEN1_SIZE];
    float hidden2_pre_activation[HIDDEN2_SIZE];
    float hidden2_activation[HIDDEN2_SIZE];

    for (int hidden1 = 0; hidden1 < HIDDEN1_SIZE; hidden1++) {
#pragma HLS PIPELINE II=1
        hidden1_activation[hidden1] = 0.01f * (float)(hidden1 + 1);
    }

    float checksum = 0.0f;
    completed_steps = 0;

    for (int iteration = 0; iteration < NN_FORWARD_PROFILE_STEPS; iteration++) {
        forward_layer2(
            hidden1_activation,
            w2,
            b2,
            hidden2_pre_activation,
            hidden2_activation
        );

        checksum += hidden2_activation[iteration % HIDDEN2_SIZE];
        completed_steps = iteration + 1;
    }

    checksum_out = (int)(checksum * 1000.0f);
}

static void profile_forward_output_only(
    float w3[HIDDEN2_SIZE][OUTPUT_SIZE],
    float b3[OUTPUT_SIZE],
    int &completed_steps,
    int &checksum_out
) {
#pragma HLS INLINE off
    float hidden2_activation[HIDDEN2_SIZE];
    float q_values[OUTPUT_SIZE];

#pragma HLS ARRAY_PARTITION variable=q_values complete dim=1

    for (int hidden2 = 0; hidden2 < HIDDEN2_SIZE; hidden2++) {
#pragma HLS PIPELINE II=1
        hidden2_activation[hidden2] = 0.01f * (float)(hidden2 + 1);
    }

    float checksum = 0.0f;
    completed_steps = 0;

    for (int iteration = 0; iteration < NN_FORWARD_PROFILE_STEPS; iteration++) {
        forward_output_layer(
            hidden2_activation,
            w3,
            b3,
            q_values
        );

        checksum += q_values[iteration % OUTPUT_SIZE];
        completed_steps = iteration + 1;
    }

    checksum_out = (int)(checksum * 1000.0f);
}

static void profile_full_forward_argmax(
    float w1[STATE_SIZE][HIDDEN1_SIZE],
    float b1[HIDDEN1_SIZE],
    float w2[HIDDEN1_SIZE][HIDDEN2_SIZE],
    float b2[HIDDEN2_SIZE],
    float w3[HIDDEN2_SIZE][OUTPUT_SIZE],
    float b3[OUTPUT_SIZE],
    int &completed_steps,
    int &checksum_out
) {
#pragma HLS INLINE off
    float observation[STATE_SIZE];
    float hidden1_pre_activation[HIDDEN1_SIZE];
    float hidden1_activation[HIDDEN1_SIZE];
    float hidden2_pre_activation[HIDDEN2_SIZE];
    float hidden2_activation[HIDDEN2_SIZE];
    float q_values[OUTPUT_SIZE];

#pragma HLS ARRAY_PARTITION variable=observation complete dim=1
#pragma HLS ARRAY_PARTITION variable=q_values complete dim=1

    observation[0] = 1.0f;
    observation[1] = 0.0f;
    observation[2] = 0.0f;

    int checksum = 0;
    completed_steps = 0;

    for (int iteration = 0; iteration < NN_FORWARD_PROFILE_STEPS; iteration++) {
        forward_network(
            observation,
            w1,
            b1,
            w2,
            b2,
            w3,
            b3,
            hidden1_pre_activation,
            hidden1_activation,
            hidden2_pre_activation,
            hidden2_activation,
            q_values
        );

        checksum += argmax_q(q_values);
        completed_steps = iteration + 1;
    }

    checksum_out = checksum;
}

static void profile_backprop_output_to_hidden2(
    float w3[HIDDEN2_SIZE][OUTPUT_SIZE],
    int &completed_steps,
    int &checksum_out
) {
#pragma HLS INLINE off
    float output_delta[OUTPUT_SIZE];
    float hidden2_pre_activation[HIDDEN2_SIZE];
    float hidden2_delta[HIDDEN2_SIZE];

#pragma HLS ARRAY_PARTITION variable=output_delta complete dim=1

    for (int output = 0; output < OUTPUT_SIZE; output++) {
#pragma HLS PIPELINE II=1
        output_delta[output] = 0.0f;
    }
    output_delta[0] = 1.0f;

    for (int hidden2 = 0; hidden2 < HIDDEN2_SIZE; hidden2++) {
#pragma HLS PIPELINE II=1
        hidden2_pre_activation[hidden2] = 1.0f;
    }

    float checksum = 0.0f;
    completed_steps = 0;

    for (int iteration = 0; iteration < NN_BACKPROP_PROFILE_STEPS; iteration++) {
        for (int hidden2 = 0; hidden2 < HIDDEN2_SIZE; hidden2++) {
            float sum = 0.0f;

            for (int output = 0; output < OUTPUT_SIZE; output++) {
#pragma HLS PIPELINE II=1
                sum += w3[hidden2][output] * output_delta[output];
            }

            hidden2_delta[hidden2] =
                clip_float(
                    sum * relu_derivative(hidden2_pre_activation[hidden2]),
                    -1.0f,
                    1.0f
                );
        }

        checksum += hidden2_delta[iteration % HIDDEN2_SIZE];
        completed_steps = iteration + 1;
    }

    checksum_out = (int)(checksum * 1000.0f);
}

static void profile_backprop_hidden2_to_hidden1(
    float w2[HIDDEN1_SIZE][HIDDEN2_SIZE],
    int &completed_steps,
    int &checksum_out
) {
#pragma HLS INLINE off
    float hidden2_delta[HIDDEN2_SIZE];
    float hidden1_pre_activation[HIDDEN1_SIZE];
    float hidden1_delta[HIDDEN1_SIZE];

    for (int hidden2 = 0; hidden2 < HIDDEN2_SIZE; hidden2++) {
#pragma HLS PIPELINE II=1
        hidden2_delta[hidden2] = 0.01f * (float)(hidden2 + 1);
    }

    for (int hidden1 = 0; hidden1 < HIDDEN1_SIZE; hidden1++) {
#pragma HLS PIPELINE II=1
        hidden1_pre_activation[hidden1] = 1.0f;
    }

    float checksum = 0.0f;
    completed_steps = 0;

    for (int iteration = 0; iteration < NN_BACKPROP_PROFILE_STEPS; iteration++) {
        for (int hidden1 = 0; hidden1 < HIDDEN1_SIZE; hidden1++) {
            float sum = 0.0f;

            for (int hidden2 = 0; hidden2 < HIDDEN2_SIZE; hidden2++) {
#pragma HLS PIPELINE II=1
                sum += w2[hidden1][hidden2] * hidden2_delta[hidden2];
            }

            hidden1_delta[hidden1] =
                clip_float(
                    sum * relu_derivative(hidden1_pre_activation[hidden1]),
                    -1.0f,
                    1.0f
                );
        }

        checksum += hidden1_delta[iteration % HIDDEN1_SIZE];
        completed_steps = iteration + 1;
    }

    checksum_out = (int)(checksum * 1000.0f);
}

static void profile_update_w3_b3(
    float w3[HIDDEN2_SIZE][OUTPUT_SIZE],
    float b3[OUTPUT_SIZE],
    float mw3[HIDDEN2_SIZE][OUTPUT_SIZE],
    float vw3[HIDDEN2_SIZE][OUTPUT_SIZE],
    float mb3[OUTPUT_SIZE],
    float vb3[OUTPUT_SIZE],
    int &completed_steps,
    int &checksum_out
) {
#pragma HLS INLINE off
    float hidden2_activation[HIDDEN2_SIZE];
    float output_delta[OUTPUT_SIZE];

#pragma HLS ARRAY_PARTITION variable=output_delta complete dim=1

    for (int hidden2 = 0; hidden2 < HIDDEN2_SIZE; hidden2++) {
#pragma HLS PIPELINE II=1
        hidden2_activation[hidden2] = 0.01f * (float)(hidden2 + 1);
    }

    for (int output = 0; output < OUTPUT_SIZE; output++) {
#pragma HLS PIPELINE II=1
        output_delta[output] = 0.0f;
    }
    output_delta[0] = 1.0f;

    completed_steps = 0;

    for (int iteration = 0; iteration < NN_UPDATE_PROFILE_STEPS; iteration++) {
        float beta1_power = 0.9f;
        float beta2_power = 0.999f;

        for (int hidden2 = 0; hidden2 < HIDDEN2_SIZE; hidden2++) {
            for (int output = 0; output < OUTPUT_SIZE; output++) {
#pragma HLS PIPELINE II=1
                float gradient = hidden2_activation[hidden2] * output_delta[output];
                adam_update(w3[hidden2][output], gradient, mw3[hidden2][output], vw3[hidden2][output], beta1_power, beta2_power);
            }
        }

        for (int output = 0; output < OUTPUT_SIZE; output++) {
#pragma HLS PIPELINE II=1
            adam_update(b3[output], output_delta[output], mb3[output], vb3[output], beta1_power, beta2_power);
        }

        completed_steps = iteration + 1;
    }

    checksum_out = (int)(w3[0][0] * 1000.0f);
}

static void profile_update_w2_b2(
    float w2[HIDDEN1_SIZE][HIDDEN2_SIZE],
    float b2[HIDDEN2_SIZE],
    float mw2[HIDDEN1_SIZE][HIDDEN2_SIZE],
    float vw2[HIDDEN1_SIZE][HIDDEN2_SIZE],
    float mb2[HIDDEN2_SIZE],
    float vb2[HIDDEN2_SIZE],
    int &completed_steps,
    int &checksum_out
) {
#pragma HLS INLINE off
    float hidden1_activation[HIDDEN1_SIZE];
    float hidden2_delta[HIDDEN2_SIZE];

    for (int hidden1 = 0; hidden1 < HIDDEN1_SIZE; hidden1++) {
#pragma HLS PIPELINE II=1
        hidden1_activation[hidden1] = 0.01f * (float)(hidden1 + 1);
    }

    for (int hidden2 = 0; hidden2 < HIDDEN2_SIZE; hidden2++) {
#pragma HLS PIPELINE II=1
        hidden2_delta[hidden2] = 0.01f * (float)(hidden2 + 1);
    }

    completed_steps = 0;

    for (int iteration = 0; iteration < NN_UPDATE_PROFILE_STEPS; iteration++) {
        float beta1_power = 0.9f;
        float beta2_power = 0.999f;

        for (int hidden1 = 0; hidden1 < HIDDEN1_SIZE; hidden1++) {
            for (int hidden2 = 0; hidden2 < HIDDEN2_SIZE; hidden2++) {
#pragma HLS PIPELINE II=1
                float gradient = hidden1_activation[hidden1] * hidden2_delta[hidden2];
                adam_update(w2[hidden1][hidden2], gradient, mw2[hidden1][hidden2], vw2[hidden1][hidden2], beta1_power, beta2_power);
            }
        }

        for (int hidden2 = 0; hidden2 < HIDDEN2_SIZE; hidden2++) {
#pragma HLS PIPELINE II=1
            adam_update(b2[hidden2], hidden2_delta[hidden2], mb2[hidden2], vb2[hidden2], beta1_power, beta2_power);
        }

        completed_steps = iteration + 1;
    }

    checksum_out = (int)(w2[0][0] * 1000.0f);
}

static void profile_update_w1_b1(
    float w1[STATE_SIZE][HIDDEN1_SIZE],
    float b1[HIDDEN1_SIZE],
    float mw1[STATE_SIZE][HIDDEN1_SIZE],
    float vw1[STATE_SIZE][HIDDEN1_SIZE],
    float mb1[HIDDEN1_SIZE],
    float vb1[HIDDEN1_SIZE],
    int &completed_steps,
    int &checksum_out
) {
#pragma HLS INLINE off
    float state[STATE_SIZE];
    float hidden1_delta[HIDDEN1_SIZE];

#pragma HLS ARRAY_PARTITION variable=state complete dim=1

    state[0] = 1.0f;
    state[1] = 0.0f;
    state[2] = 0.0f;

    for (int hidden1 = 0; hidden1 < HIDDEN1_SIZE; hidden1++) {
#pragma HLS PIPELINE II=1
        hidden1_delta[hidden1] = 0.01f * (float)(hidden1 + 1);
    }

    completed_steps = 0;

    for (int iteration = 0; iteration < NN_UPDATE_PROFILE_STEPS; iteration++) {
        float beta1_power = 0.9f;
        float beta2_power = 0.999f;

        for (int input = 0; input < STATE_SIZE; input++) {
            for (int hidden1 = 0; hidden1 < HIDDEN1_SIZE; hidden1++) {
#pragma HLS PIPELINE II=1
                float gradient = state[input] * hidden1_delta[hidden1];
                adam_update(w1[input][hidden1], gradient, mw1[input][hidden1], vw1[input][hidden1], beta1_power, beta2_power);
            }
        }

        for (int hidden1 = 0; hidden1 < HIDDEN1_SIZE; hidden1++) {
#pragma HLS PIPELINE II=1
            adam_update(b1[hidden1], hidden1_delta[hidden1], mb1[hidden1], vb1[hidden1], beta1_power, beta2_power);
        }

        completed_steps = iteration + 1;
    }

    checksum_out = (int)(w1[0][0] * 1000.0f);
}

static void profile_full_train_sample(
    float w1[STATE_SIZE][HIDDEN1_SIZE],
    float b1[HIDDEN1_SIZE],
    float w2[HIDDEN1_SIZE][HIDDEN2_SIZE],
    float b2[HIDDEN2_SIZE],
    float w3[HIDDEN2_SIZE][OUTPUT_SIZE],
    float b3[OUTPUT_SIZE],
    float target_w1[STATE_SIZE][HIDDEN1_SIZE],
    float target_b1[HIDDEN1_SIZE],
    float target_w2[HIDDEN1_SIZE][HIDDEN2_SIZE],
    float target_b2[HIDDEN2_SIZE],
    float target_w3[HIDDEN2_SIZE][OUTPUT_SIZE],
    float target_b3[OUTPUT_SIZE],
    float mw1[STATE_SIZE][HIDDEN1_SIZE],
    float vw1[STATE_SIZE][HIDDEN1_SIZE],
    float mb1[HIDDEN1_SIZE],
    float vb1[HIDDEN1_SIZE],
    float mw2[HIDDEN1_SIZE][HIDDEN2_SIZE],
    float vw2[HIDDEN1_SIZE][HIDDEN2_SIZE],
    float mb2[HIDDEN2_SIZE],
    float vb2[HIDDEN2_SIZE],
    float mw3[HIDDEN2_SIZE][OUTPUT_SIZE],
    float vw3[HIDDEN2_SIZE][OUTPUT_SIZE],
    float mb3[OUTPUT_SIZE],
    float vb3[OUTPUT_SIZE],
    int &completed_steps,
    int &checksum_out
) {
#pragma HLS INLINE off
    Transition transition;
#pragma HLS ARRAY_PARTITION variable=transition.state complete dim=1
#pragma HLS ARRAY_PARTITION variable=transition.next_state complete dim=1

    transition.state[0] = 1.0f;
    transition.state[1] = 0.0f;
    transition.state[2] = 0.0f;
    transition.action = 0;
    transition.reward = 1.0f;
    transition.next_state[0] = 0.99f;
    transition.next_state[1] = 0.05f;
    transition.next_state[2] = 0.1f;
    transition.done = 0;

    float beta1_power = 0.9f;
    float beta2_power = 0.999f;
    float checksum = 0.0f;
    completed_steps = 0;

    for (int iteration = 0; iteration < NN_TRAIN_SAMPLE_PROFILE_STEPS; iteration++) {
        checksum += train_one_sample_adam(
            transition,
            w1,
            b1,
            w2,
            b2,
            w3,
            b3,
            target_w1,
            target_b1,
            target_w2,
            target_b2,
            target_w3,
            target_b3,
            mw1,
            vw1,
            mb1,
            vb1,
            mw2,
            vw2,
            mb2,
            vb2,
            mw3,
            vw3,
            mb3,
            vb3,
            beta1_power,
            beta2_power
        );

        beta1_power *= ADAM_BETA1;
        beta2_power *= ADAM_BETA2;
        completed_steps = iteration + 1;
    }

    checksum_out = (int)(checksum * 1000.0f);
}

static void profile_replay_access(
    float replay_state[REPLAY_SIZE][STATE_SIZE],
    int replay_action[REPLAY_SIZE],
    float replay_reward[REPLAY_SIZE],
    float replay_next_state[REPLAY_SIZE][STATE_SIZE],
    int replay_done[REPLAY_SIZE],
    int &completed_steps,
    int &checksum_out
) {
#pragma HLS INLINE off
    Transition transition;
#pragma HLS ARRAY_PARTITION variable=transition.state complete dim=1
#pragma HLS ARRAY_PARTITION variable=transition.next_state complete dim=1

    float state[STATE_SIZE];
    float next_state[STATE_SIZE];
#pragma HLS ARRAY_PARTITION variable=state complete dim=1
#pragma HLS ARRAY_PARTITION variable=next_state complete dim=1

    state[0] = 1.0f;
    state[1] = 0.0f;
    state[2] = 0.0f;
    next_state[0] = 0.99f;
    next_state[1] = 0.05f;
    next_state[2] = 0.1f;

    int checksum = 0;
    completed_steps = 0;

    for (int iteration = 0; iteration < REPLAY_PROFILE_STEPS; iteration++) {
        int index = iteration % REPLAY_SIZE;

        store_transition(
            index,
            replay_state,
            replay_action,
            replay_reward,
            replay_next_state,
            replay_done,
            state,
            0,
            1.0f,
            next_state,
            0
        );

        load_transition(
            index,
            replay_state,
            replay_action,
            replay_reward,
            replay_next_state,
            replay_done,
            transition
        );

        checksum += transition.action + transition.done;
        completed_steps = iteration + 1;
    }

    checksum_out = checksum;
}

extern "C" {

void dqn_eva(
    int mode,
    unsigned int seed_value,

    int *training_episode,
    int *training_episode_length,
    float *training_total_reward,
    float *training_average_reward_per_step,
    float *training_average_abs_td_error,
    float *training_epsilon,
    int *training_timesteps,

    int *validation_training_episode,
    float *validation_total_reward,
    float *validation_min_abs_theta,
    int *validation_upright_steps,
    int *validation_stable_upright_steps,
    int *validation_max_consecutive_stable_steps,
    int *validation_success,
    float *validation_final_theta_wrapped,
    float *validation_final_theta_unwrapped,
    float *validation_final_theta_dot,
    float *validation_states,
    int *validation_actions,
    float *validation_rewards,

    float *evaluation_initial_theta,
    float *evaluation_initial_theta_dot,
    float *evaluation_actual_initial_theta,
    float *evaluation_actual_initial_theta_dot,
    float *evaluation_total_reward,
    float *evaluation_min_abs_theta,
    int *evaluation_upright_steps,
    int *evaluation_stable_upright_steps,
    int *evaluation_max_consecutive_stable_steps,
    int *evaluation_success,
    float *evaluation_final_theta_wrapped,
    float *evaluation_final_theta_unwrapped,
    float *evaluation_final_theta_dot,
    float *evaluation_states,
    int *evaluation_actions,
    float *evaluation_rewards,

    int &completed_training_episodes,
    int &completed_validations,
    int &completed_evaluations
) {
#pragma HLS INTERFACE m_axi port=training_episode offset=slave bundle=gmem_training depth=MAX_EPISODES
#pragma HLS INTERFACE m_axi port=training_episode_length offset=slave bundle=gmem_training depth=MAX_EPISODES
#pragma HLS INTERFACE m_axi port=training_total_reward offset=slave bundle=gmem_training depth=MAX_EPISODES
#pragma HLS INTERFACE m_axi port=training_average_reward_per_step offset=slave bundle=gmem_training depth=MAX_EPISODES
#pragma HLS INTERFACE m_axi port=training_average_abs_td_error offset=slave bundle=gmem_training depth=MAX_EPISODES
#pragma HLS INTERFACE m_axi port=training_epsilon offset=slave bundle=gmem_training depth=MAX_EPISODES
#pragma HLS INTERFACE m_axi port=training_timesteps offset=slave bundle=gmem_training depth=MAX_EPISODES

#pragma HLS INTERFACE m_axi port=validation_training_episode offset=slave bundle=gmem_validation depth=NUM_VALIDATIONS
#pragma HLS INTERFACE m_axi port=validation_total_reward offset=slave bundle=gmem_validation depth=NUM_VALIDATIONS
#pragma HLS INTERFACE m_axi port=validation_min_abs_theta offset=slave bundle=gmem_validation depth=NUM_VALIDATIONS
#pragma HLS INTERFACE m_axi port=validation_upright_steps offset=slave bundle=gmem_validation depth=NUM_VALIDATIONS
#pragma HLS INTERFACE m_axi port=validation_stable_upright_steps offset=slave bundle=gmem_validation depth=NUM_VALIDATIONS
#pragma HLS INTERFACE m_axi port=validation_max_consecutive_stable_steps offset=slave bundle=gmem_validation depth=NUM_VALIDATIONS
#pragma HLS INTERFACE m_axi port=validation_success offset=slave bundle=gmem_validation depth=NUM_VALIDATIONS
#pragma HLS INTERFACE m_axi port=validation_final_theta_wrapped offset=slave bundle=gmem_validation depth=NUM_VALIDATIONS
#pragma HLS INTERFACE m_axi port=validation_final_theta_unwrapped offset=slave bundle=gmem_validation depth=NUM_VALIDATIONS
#pragma HLS INTERFACE m_axi port=validation_final_theta_dot offset=slave bundle=gmem_validation depth=NUM_VALIDATIONS
#pragma HLS INTERFACE m_axi port=validation_states offset=slave bundle=gmem_validation depth=VALIDATION_STATE_BUFFER_SIZE
#pragma HLS INTERFACE m_axi port=validation_actions offset=slave bundle=gmem_validation depth=VALIDATION_TRAJECTORY_SIZE
#pragma HLS INTERFACE m_axi port=validation_rewards offset=slave bundle=gmem_validation depth=VALIDATION_TRAJECTORY_SIZE

#pragma HLS INTERFACE m_axi port=evaluation_initial_theta offset=slave bundle=gmem_evaluation depth=EVALUATION_BATCH_SIZE
#pragma HLS INTERFACE m_axi port=evaluation_initial_theta_dot offset=slave bundle=gmem_evaluation depth=EVALUATION_BATCH_SIZE
#pragma HLS INTERFACE m_axi port=evaluation_actual_initial_theta offset=slave bundle=gmem_evaluation depth=EVALUATION_BATCH_SIZE
#pragma HLS INTERFACE m_axi port=evaluation_actual_initial_theta_dot offset=slave bundle=gmem_evaluation depth=EVALUATION_BATCH_SIZE
#pragma HLS INTERFACE m_axi port=evaluation_total_reward offset=slave bundle=gmem_evaluation depth=EVALUATION_BATCH_SIZE
#pragma HLS INTERFACE m_axi port=evaluation_min_abs_theta offset=slave bundle=gmem_evaluation depth=EVALUATION_BATCH_SIZE
#pragma HLS INTERFACE m_axi port=evaluation_upright_steps offset=slave bundle=gmem_evaluation depth=EVALUATION_BATCH_SIZE
#pragma HLS INTERFACE m_axi port=evaluation_stable_upright_steps offset=slave bundle=gmem_evaluation depth=EVALUATION_BATCH_SIZE
#pragma HLS INTERFACE m_axi port=evaluation_max_consecutive_stable_steps offset=slave bundle=gmem_evaluation depth=EVALUATION_BATCH_SIZE
#pragma HLS INTERFACE m_axi port=evaluation_success offset=slave bundle=gmem_evaluation depth=EVALUATION_BATCH_SIZE
#pragma HLS INTERFACE m_axi port=evaluation_final_theta_wrapped offset=slave bundle=gmem_evaluation depth=EVALUATION_BATCH_SIZE
#pragma HLS INTERFACE m_axi port=evaluation_final_theta_unwrapped offset=slave bundle=gmem_evaluation depth=EVALUATION_BATCH_SIZE
#pragma HLS INTERFACE m_axi port=evaluation_final_theta_dot offset=slave bundle=gmem_evaluation depth=EVALUATION_BATCH_SIZE
#pragma HLS INTERFACE m_axi port=evaluation_states offset=slave bundle=gmem_evaluation depth=EVALUATION_STATE_BUFFER_SIZE
#pragma HLS INTERFACE m_axi port=evaluation_actions offset=slave bundle=gmem_evaluation depth=EVALUATION_TRAJECTORY_SIZE
#pragma HLS INTERFACE m_axi port=evaluation_rewards offset=slave bundle=gmem_evaluation depth=EVALUATION_TRAJECTORY_SIZE

#pragma HLS INTERFACE s_axilite port=mode bundle=CTRL
#pragma HLS INTERFACE s_axilite port=seed_value bundle=CTRL
#pragma HLS INTERFACE s_axilite port=training_episode bundle=CTRL
#pragma HLS INTERFACE s_axilite port=training_episode_length bundle=CTRL
#pragma HLS INTERFACE s_axilite port=training_total_reward bundle=CTRL
#pragma HLS INTERFACE s_axilite port=training_average_reward_per_step bundle=CTRL
#pragma HLS INTERFACE s_axilite port=training_average_abs_td_error bundle=CTRL
#pragma HLS INTERFACE s_axilite port=training_epsilon bundle=CTRL
#pragma HLS INTERFACE s_axilite port=training_timesteps bundle=CTRL

#pragma HLS INTERFACE s_axilite port=validation_training_episode bundle=CTRL
#pragma HLS INTERFACE s_axilite port=validation_total_reward bundle=CTRL
#pragma HLS INTERFACE s_axilite port=validation_min_abs_theta bundle=CTRL
#pragma HLS INTERFACE s_axilite port=validation_upright_steps bundle=CTRL
#pragma HLS INTERFACE s_axilite port=validation_stable_upright_steps bundle=CTRL
#pragma HLS INTERFACE s_axilite port=validation_max_consecutive_stable_steps bundle=CTRL
#pragma HLS INTERFACE s_axilite port=validation_success bundle=CTRL
#pragma HLS INTERFACE s_axilite port=validation_final_theta_wrapped bundle=CTRL
#pragma HLS INTERFACE s_axilite port=validation_final_theta_unwrapped bundle=CTRL
#pragma HLS INTERFACE s_axilite port=validation_final_theta_dot bundle=CTRL
#pragma HLS INTERFACE s_axilite port=validation_states bundle=CTRL
#pragma HLS INTERFACE s_axilite port=validation_actions bundle=CTRL
#pragma HLS INTERFACE s_axilite port=validation_rewards bundle=CTRL

#pragma HLS INTERFACE s_axilite port=evaluation_initial_theta bundle=CTRL
#pragma HLS INTERFACE s_axilite port=evaluation_initial_theta_dot bundle=CTRL
#pragma HLS INTERFACE s_axilite port=evaluation_actual_initial_theta bundle=CTRL
#pragma HLS INTERFACE s_axilite port=evaluation_actual_initial_theta_dot bundle=CTRL
#pragma HLS INTERFACE s_axilite port=evaluation_total_reward bundle=CTRL
#pragma HLS INTERFACE s_axilite port=evaluation_min_abs_theta bundle=CTRL
#pragma HLS INTERFACE s_axilite port=evaluation_upright_steps bundle=CTRL
#pragma HLS INTERFACE s_axilite port=evaluation_stable_upright_steps bundle=CTRL
#pragma HLS INTERFACE s_axilite port=evaluation_max_consecutive_stable_steps bundle=CTRL
#pragma HLS INTERFACE s_axilite port=evaluation_success bundle=CTRL
#pragma HLS INTERFACE s_axilite port=evaluation_final_theta_wrapped bundle=CTRL
#pragma HLS INTERFACE s_axilite port=evaluation_final_theta_unwrapped bundle=CTRL
#pragma HLS INTERFACE s_axilite port=evaluation_final_theta_dot bundle=CTRL
#pragma HLS INTERFACE s_axilite port=evaluation_states bundle=CTRL
#pragma HLS INTERFACE s_axilite port=evaluation_actions bundle=CTRL
#pragma HLS INTERFACE s_axilite port=evaluation_rewards bundle=CTRL

#pragma HLS INTERFACE s_axilite port=completed_training_episodes bundle=CTRL
#pragma HLS INTERFACE s_axilite port=completed_validations bundle=CTRL
#pragma HLS INTERFACE s_axilite port=completed_evaluations bundle=CTRL
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL


    static float w1[STATE_SIZE][HIDDEN1_SIZE];
    static float b1[HIDDEN1_SIZE];
    static float w2[HIDDEN1_SIZE][HIDDEN2_SIZE];
    static float b2[HIDDEN2_SIZE];
    static float w3[HIDDEN2_SIZE][OUTPUT_SIZE];
    static float b3[OUTPUT_SIZE];

    static float target_w1[STATE_SIZE][HIDDEN1_SIZE];
    static float target_b1[HIDDEN1_SIZE];
    static float target_w2[HIDDEN1_SIZE][HIDDEN2_SIZE];
    static float target_b2[HIDDEN2_SIZE];
    static float target_w3[HIDDEN2_SIZE][OUTPUT_SIZE];
    static float target_b3[OUTPUT_SIZE];

    static float best_w1[STATE_SIZE][HIDDEN1_SIZE];
    static float best_b1[HIDDEN1_SIZE];
    static float best_w2[HIDDEN1_SIZE][HIDDEN2_SIZE];
    static float best_b2[HIDDEN2_SIZE];
    static float best_w3[HIDDEN2_SIZE][OUTPUT_SIZE];
    static float best_b3[OUTPUT_SIZE];

    static float mw1[STATE_SIZE][HIDDEN1_SIZE];
    static float vw1[STATE_SIZE][HIDDEN1_SIZE];
    static float mb1[HIDDEN1_SIZE];
    static float vb1[HIDDEN1_SIZE];

    static float mw2[HIDDEN1_SIZE][HIDDEN2_SIZE];
    static float vw2[HIDDEN1_SIZE][HIDDEN2_SIZE];
    static float mb2[HIDDEN2_SIZE];
    static float vb2[HIDDEN2_SIZE];

    static float mw3[HIDDEN2_SIZE][OUTPUT_SIZE];
    static float vw3[HIDDEN2_SIZE][OUTPUT_SIZE];
    static float mb3[OUTPUT_SIZE];
    static float vb3[OUTPUT_SIZE];

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

#pragma HLS BIND_STORAGE variable=best_w1 type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=best_b1 type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=best_w2 type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=best_b2 type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=best_w3 type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=best_b3 type=ram_2p impl=bram

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

    static float replay_state[REPLAY_SIZE][STATE_SIZE];
    static float replay_next_state[REPLAY_SIZE][STATE_SIZE];
    static int replay_action[REPLAY_SIZE];
    static float replay_reward[REPLAY_SIZE];
    static int replay_done[REPLAY_SIZE];

#pragma HLS BIND_STORAGE variable=replay_state type=ram_2p impl=uram
#pragma HLS BIND_STORAGE variable=replay_next_state type=ram_2p impl=uram
#pragma HLS BIND_STORAGE variable=replay_reward type=ram_2p impl=uram
#pragma HLS BIND_STORAGE variable=replay_action type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=replay_done type=ram_2p impl=bram

    completed_training_episodes = 0;
    completed_validations = 0;
    completed_evaluations = 0;


    if (mode == 6) {
        profile_forward_layer1_only(w1, b1, completed_evaluations, completed_training_episodes);
        return;
    }

    if (mode == 7) {
        profile_forward_layer2_only(w2, b2, completed_evaluations, completed_training_episodes);
        return;
    }

    if (mode == 8) {
        profile_forward_output_only(w3, b3, completed_evaluations, completed_training_episodes);
        return;
    }

    if (mode == 9) {
        profile_full_forward_argmax(w1, b1, w2, b2, w3, b3, completed_evaluations, completed_training_episodes);
        return;
    }

    if (mode == 10) {
        profile_backprop_output_to_hidden2(w3, completed_evaluations, completed_training_episodes);
        return;
    }

    if (mode == 11) {
        profile_backprop_hidden2_to_hidden1(w2, completed_evaluations, completed_training_episodes);
        return;
    }

    if (mode == 12) {
        profile_update_w3_b3(w3, b3, mw3, vw3, mb3, vb3, completed_evaluations, completed_training_episodes);
        return;
    }

    if (mode == 13) {
        profile_update_w2_b2(w2, b2, mw2, vw2, mb2, vb2, completed_evaluations, completed_training_episodes);
        return;
    }

    if (mode == 14) {
        profile_update_w1_b1(w1, b1, mw1, vw1, mb1, vb1, completed_evaluations, completed_training_episodes);
        return;
    }

    if (mode == 15) {
        profile_full_train_sample(
            w1, b1, w2, b2, w3, b3,
            target_w1, target_b1, target_w2, target_b2, target_w3, target_b3,
            mw1, vw1, mb1, vb1,
            mw2, vw2, mb2, vb2,
            mw3, vw3, mb3, vb3,
            completed_evaluations,
            completed_training_episodes
        );
        return;
    }

    if (mode == 16) {
        profile_replay_access(
            replay_state,
            replay_action,
            replay_reward,
            replay_next_state,
            replay_done,
            completed_evaluations,
            completed_training_episodes
        );
        return;
    }

    if (mode == 2) {
        run_validation_episode(
            0,
            completed_training_episodes,
            w1,
            b1,
            w2,
            b2,
            w3,
            b3,
            validation_training_episode,
            validation_total_reward,
            validation_min_abs_theta,
            validation_upright_steps,
            validation_stable_upright_steps,
            validation_max_consecutive_stable_steps,
            validation_success,
            validation_final_theta_wrapped,
            validation_final_theta_unwrapped,
            validation_final_theta_dot,
            validation_states,
            validation_actions,
            validation_rewards
        );

        completed_validations = 1;
        return;
    }

    if (mode == 3) {
        run_inference_profile(
            w1,
            b1,
            w2,
            b2,
            w3,
            b3,
            completed_evaluations
        );

        return;
    }

    if (mode == 0 || mode == 4 || mode == 5) {
        g_seed = seed_value;

        initialize_network(w1, b1, w2, b2, w3, b3);
        initialize_adam(
            mw1, vw1, mb1, vb1,
            mw2, vw2, mb2, vb2,
            mw3, vw3, mb3, vb3
        );

        copy_network(
            w1, b1, w2, b2, w3, b3,
            target_w1, target_b1, target_w2, target_b2, target_w3, target_b3
        );

        copy_network(
            w1, b1, w2, b2, w3, b3,
            best_w1, best_b1, best_w2, best_b2, best_w3, best_b3
        );

        int best_success = 0;
        int best_stable_upright_steps = -1;
        float best_validation_reward = -1000000000.0f;

        int replay_pointer = 0;
        int replay_count = 0;
        int global_step = 0;
        int validation_index = 0;

        float beta1_power = 1.0f;
        float beta2_power = 1.0f;

        for (int episode = 0; episode < MAX_EPISODES; episode++) {
            float epsilon = epsilon_for_episode(episode);

            EnvState environment_state;
            float observation[STATE_SIZE];
            float next_observation[STATE_SIZE];

#pragma HLS ARRAY_PARTITION variable=observation complete dim=1
#pragma HLS ARRAY_PARTITION variable=next_observation complete dim=1

            reset_training_environment(environment_state, observation);

            float total_episode_reward = 0.0f;
            float td_error_sum = 0.0f;
            int td_update_count = 0;

            for (int step = 0; step < MAX_STEPS; step++) {
                int action = select_action(observation, epsilon, w1, b1, w2, b2, w3, b3);

                float reward;
                int done;

                step_environment(environment_state, action, next_observation, reward, done);

                store_transition(
                    replay_pointer,
                    replay_state,
                    replay_action,
                    replay_reward,
                    replay_next_state,
                    replay_done,
                    observation,
                    action,
                    reward,
                    next_observation,
                    done
                );

                replay_pointer = (replay_pointer + 1) % REPLAY_SIZE;

                if (replay_count < REPLAY_SIZE) {
                    replay_count++;
                }

                total_episode_reward += reward;

                if (
                    replay_count >= LEARNING_STARTS
                    && (global_step % TRAIN_FREQUENCY) == 0
                ) {
                    for (int batch = 0; batch < BATCH_SIZE; batch++) {
                        int replay_index = rand_int(replay_count);

                        Transition sampled_transition;
#pragma HLS ARRAY_PARTITION variable=sampled_transition.state complete dim=1
#pragma HLS ARRAY_PARTITION variable=sampled_transition.next_state complete dim=1

                        load_transition(
                            replay_index,
                            replay_state,
                            replay_action,
                            replay_reward,
                            replay_next_state,
                            replay_done,
                            sampled_transition
                        );

                        beta1_power *= ADAM_BETA1;
                        beta2_power *= ADAM_BETA2;

                        float abs_td_error = train_one_sample_adam(
                            sampled_transition,
                            w1,
                            b1,
                            w2,
                            b2,
                            w3,
                            b3,
                            target_w1,
                            target_b1,
                            target_w2,
                            target_b2,
                            target_w3,
                            target_b3,
                            mw1,
                            vw1,
                            mb1,
                            vb1,
                            mw2,
                            vw2,
                            mb2,
                            vb2,
                            mw3,
                            vw3,
                            mb3,
                            vb3,
                            beta1_power,
                            beta2_power
                        );

                        td_error_sum += abs_td_error;
                        td_update_count++;
                    }
                }

                if (
                    global_step > 0
                    && (global_step % TARGET_UPDATE_FREQUENCY) == 0
                ) {
                    copy_network(
                        w1, b1, w2, b2, w3, b3,
                        target_w1, target_b1, target_w2, target_b2, target_w3, target_b3
                    );
                }

                for (int component = 0; component < STATE_SIZE; component++) {
#pragma HLS PIPELINE II=1
                    observation[component] = next_observation[component];
                }

                global_step++;
            }

            training_episode[episode] = episode + 1;
            training_episode_length[episode] = MAX_STEPS;
            training_total_reward[episode] = total_episode_reward;
            training_average_reward_per_step[episode] = total_episode_reward / (float)MAX_STEPS;
            training_average_abs_td_error[episode] =
                td_update_count > 0 ? td_error_sum / (float)td_update_count : 0.0f;
            training_epsilon[episode] = epsilon;
            training_timesteps[episode] = global_step;

            completed_training_episodes = episode + 1;

            if (mode != 5 && ((episode + 1) % VALIDATION_INTERVAL) == 0) {
                run_validation_episode(
                    validation_index,
                    episode + 1,
                    w1,
                    b1,
                    w2,
                    b2,
                    w3,
                    b3,
                    validation_training_episode,
                    validation_total_reward,
                    validation_min_abs_theta,
                    validation_upright_steps,
                    validation_stable_upright_steps,
                    validation_max_consecutive_stable_steps,
                    validation_success,
                    validation_final_theta_wrapped,
                    validation_final_theta_unwrapped,
                    validation_final_theta_dot,
                    validation_states,
                    validation_actions,
                    validation_rewards
                );

                int current_success = validation_success[validation_index];
                int current_stable_upright_steps =
                    validation_stable_upright_steps[validation_index];
                float current_validation_reward = validation_total_reward[validation_index];

                int is_better_checkpoint = 0;

                if (current_success > best_success) {
                    is_better_checkpoint = 1;
                } else if (
                    current_success == best_success
                    && current_stable_upright_steps > best_stable_upright_steps
                ) {
                    is_better_checkpoint = 1;
                } else if (
                    current_success == best_success
                    && current_stable_upright_steps == best_stable_upright_steps
                    && current_validation_reward > best_validation_reward
                ) {
                    is_better_checkpoint = 1;
                }

                if (is_better_checkpoint) {
                    best_success = current_success;
                    best_stable_upright_steps = current_stable_upright_steps;
                    best_validation_reward = current_validation_reward;

                    copy_network(
                        w1, b1, w2, b2, w3, b3,
                        best_w1, best_b1, best_w2, best_b2, best_w3, best_b3
                    );
                }

                validation_index++;
                completed_validations = validation_index;
            }
        }

        if (mode != 5) {
            copy_network(
                best_w1, best_b1, best_w2, best_b2, best_w3, best_b3,
                w1, b1, w2, b2, w3, b3
            );
        }

        if (mode == 4 || mode == 5) {
            return;
        }
    }

    const float evaluation_theta[EVALUATION_BATCH_SIZE] = {
        PI,
        PI + 0.20f,
        1.67f,
        1.67f,
        PI + 1.67f
    };

    const float evaluation_theta_dot[EVALUATION_BATCH_SIZE] = {
        0.0f,
        -2.0f,
        2.0f,
        -1.0f,
        0.0f
    };

    for (int batch = 0; batch < EVALUATION_BATCH_SIZE; batch++) {
        run_evaluation_episode(
            batch,
            evaluation_theta[batch],
            evaluation_theta_dot[batch],
            w1,
            b1,
            w2,
            b2,
            w3,
            b3,
            evaluation_initial_theta,
            evaluation_initial_theta_dot,
            evaluation_actual_initial_theta,
            evaluation_actual_initial_theta_dot,
            evaluation_total_reward,
            evaluation_min_abs_theta,
            evaluation_upright_steps,
            evaluation_stable_upright_steps,
            evaluation_max_consecutive_stable_steps,
            evaluation_success,
            evaluation_final_theta_wrapped,
            evaluation_final_theta_unwrapped,
            evaluation_final_theta_dot,
            evaluation_states,
            evaluation_actions,
            evaluation_rewards
        );

        completed_evaluations = batch + 1;
    }
}

}