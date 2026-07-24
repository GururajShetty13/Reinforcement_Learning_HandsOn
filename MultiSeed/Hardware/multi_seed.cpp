#include <hls_math.h>
#include <stdint.h>

#define STATE_SIZE 3
#define HIDDEN_SIZE 32
#define OUTPUT_SIZE 7

#define MAX_EPISODES 8000
#define MAX_STEPS 250

#define VALIDATION_INTERVAL 800
#define NUM_VALIDATIONS (MAX_EPISODES / VALIDATION_INTERVAL)

#define EVALUATION_BATCH_SIZE 5

#define GAMMA 0.99f
#define LEARNING_RATE 0.0005f

#define ADAM_BETA1 0.9f
#define ADAM_BETA2 0.999f
#define ADAM_EPS 1.0e-8f

#define EPS_START 1.0f
#define EPS_END 0.05f

#define REPLAY_SIZE 10000
#define BATCH_SIZE 64
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
    g_seed = 1664525u * g_seed + 1013904223u;
    return ((g_seed >> 8) & 0x00FFFFFF) / 16777216.0f;
}

static int rand_int(int maximum) {
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
    return value < 0.0f ? -value : value;
}

static float clip_float(float value, float minimum, float maximum) {
    if (value < minimum) {
        return minimum;
    }

    if (value > maximum) {
        return maximum;
    }

    return value;
}

static float wrap_angle(float angle) {
    while (angle > PI) {
        angle -= 2.0f * PI;
    }

    while (angle < -PI) {
        angle += 2.0f * PI;
    }

    return angle;
}

static float relu(float value) {
    return value > 0.0f ? value : 0.0f;
}

static float relu_derivative(float value) {
    return value > 0.0f ? 1.0f : 0.0f;
}

static float action_to_torque(int action) {
    if (action == 0) return -2.0f;
    if (action == 1) return -0.5f;
    if (action == 2) return -0.1f;
    if (action == 3) return  0.0f;
    if (action == 4) return  0.1f;
    if (action == 5) return  0.5f;
    return 2.0f;
}

static void encode_observation(
    const EnvState &environment_state,
    float observation[STATE_SIZE]
) {
    observation[0] = hls::cosf(environment_state.theta);
    observation[1] = hls::sinf(environment_state.theta);
    observation[2] = environment_state.theta_dot;
}

static void reset_training_environment(
    EnvState &environment_state,
    float observation[STATE_SIZE]
) {
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
    float previous_angle_error = wrap_angle(previous_theta);
    float angle_error = wrap_angle(new_theta);

    float upright_reward =
        2.0f * hls::cosf(angle_error);

    float velocity_penalty =
        0.01f * new_theta_dot * new_theta_dot;

    float torque_penalty =
        0.001f * torque * torque;

    float progress_bonus =
        0.5f
        * (
            abs_float(previous_angle_error)
            - abs_float(angle_error)
        );

    float bonus = 0.0f;

    if (abs_float(angle_error) < 0.3f) {
        bonus += 2.0f;
    }

    if (
        abs_float(angle_error) < 0.15f
        && abs_float(new_theta_dot) < 1.0f
    ) {
        bonus += 4.0f;
    }

    if (
        abs_float(angle_error) < 0.08f
        && abs_float(new_theta_dot) < 0.5f
    ) {
        bonus += 6.0f;
    }

    float reward =
        upright_reward
        - velocity_penalty
        - torque_penalty
        + progress_bonus
        + bonus;

    return clip_float(reward, -10.0f, 10.0f);
}

static void step_environment(
    EnvState &environment_state,
    int action,
    float next_observation[STATE_SIZE],
    float &reward,
    int &done
) {
    float torque = action_to_torque(action);

    float previous_theta = environment_state.theta;

    float new_theta_dot =
        environment_state.theta_dot
        + (
            3.0f * G / (2.0f * L)
            * hls::sinf(environment_state.theta)
            + 3.0f / (M * L * L) * torque
        ) * DT;

    new_theta_dot =
        clip_float(
            new_theta_dot,
            -MAX_SPEED,
            MAX_SPEED
        );

    float new_theta =
        environment_state.theta
        + new_theta_dot * DT;

    reward =
        calculate_reward(
            previous_theta,
            new_theta,
            new_theta_dot,
            torque
        );

    environment_state.theta = new_theta;
    environment_state.theta_dot = new_theta_dot;

    encode_observation(
        environment_state,
        next_observation
    );

    done = 0;
}

static void initialize_network(
    float w1[STATE_SIZE][HIDDEN_SIZE],
    float b1[HIDDEN_SIZE],
    float w2[HIDDEN_SIZE][OUTPUT_SIZE],
    float b2[OUTPUT_SIZE]
) {
    for (int input = 0; input < STATE_SIZE; input++) {
        for (int hidden = 0; hidden < HIDDEN_SIZE; hidden++) {
            float random_value =
                rand_uniform() * 2.0f - 1.0f;

            w1[input][hidden] =
                0.03f * random_value;
        }
    }

    for (int hidden = 0; hidden < HIDDEN_SIZE; hidden++) {
        b1[hidden] = 0.0f;
    }

    for (int hidden = 0; hidden < HIDDEN_SIZE; hidden++) {
        for (int output = 0; output < OUTPUT_SIZE; output++) {
            float random_value =
                rand_uniform() * 2.0f - 1.0f;

            w2[hidden][output] =
                0.03f * random_value;
        }
    }

    for (int output = 0; output < OUTPUT_SIZE; output++) {
        b2[output] = 0.0f;
    }
}

static void initialize_adam(
    float mw1[STATE_SIZE][HIDDEN_SIZE],
    float vw1[STATE_SIZE][HIDDEN_SIZE],
    float mb1[HIDDEN_SIZE],
    float vb1[HIDDEN_SIZE],
    float mw2[HIDDEN_SIZE][OUTPUT_SIZE],
    float vw2[HIDDEN_SIZE][OUTPUT_SIZE],
    float mb2[OUTPUT_SIZE],
    float vb2[OUTPUT_SIZE]
) {
    for (int input = 0; input < STATE_SIZE; input++) {
        for (int hidden = 0; hidden < HIDDEN_SIZE; hidden++) {
            mw1[input][hidden] = 0.0f;
            vw1[input][hidden] = 0.0f;
        }
    }

    for (int hidden = 0; hidden < HIDDEN_SIZE; hidden++) {
        mb1[hidden] = 0.0f;
        vb1[hidden] = 0.0f;
    }

    for (int hidden = 0; hidden < HIDDEN_SIZE; hidden++) {
        for (int output = 0; output < OUTPUT_SIZE; output++) {
            mw2[hidden][output] = 0.0f;
            vw2[hidden][output] = 0.0f;
        }
    }

    for (int output = 0; output < OUTPUT_SIZE; output++) {
        mb2[output] = 0.0f;
        vb2[output] = 0.0f;
    }
}

static void copy_network(
    float source_w1[STATE_SIZE][HIDDEN_SIZE],
    float source_b1[HIDDEN_SIZE],
    float source_w2[HIDDEN_SIZE][OUTPUT_SIZE],
    float source_b2[OUTPUT_SIZE],
    float destination_w1[STATE_SIZE][HIDDEN_SIZE],
    float destination_b1[HIDDEN_SIZE],
    float destination_w2[HIDDEN_SIZE][OUTPUT_SIZE],
    float destination_b2[OUTPUT_SIZE]
) {
    for (int input = 0; input < STATE_SIZE; input++) {
        for (int hidden = 0; hidden < HIDDEN_SIZE; hidden++) {
            destination_w1[input][hidden] =
                source_w1[input][hidden];
        }
    }

    for (int hidden = 0; hidden < HIDDEN_SIZE; hidden++) {
        destination_b1[hidden] =
            source_b1[hidden];
    }

    for (int hidden = 0; hidden < HIDDEN_SIZE; hidden++) {
        for (int output = 0; output < OUTPUT_SIZE; output++) {
            destination_w2[hidden][output] =
                source_w2[hidden][output];
        }
    }

    for (int output = 0; output < OUTPUT_SIZE; output++) {
        destination_b2[output] =
            source_b2[output];
    }
}

static void forward_network(
    const float observation[STATE_SIZE],
    float w1[STATE_SIZE][HIDDEN_SIZE],
    float b1[HIDDEN_SIZE],
    float w2[HIDDEN_SIZE][OUTPUT_SIZE],
    float b2[OUTPUT_SIZE],
    float hidden_pre_activation[HIDDEN_SIZE],
    float hidden_activation[HIDDEN_SIZE],
    float q_values[OUTPUT_SIZE]
) {
    for (int hidden = 0; hidden < HIDDEN_SIZE; hidden++) {
        float sum = b1[hidden];

        for (int input = 0; input < STATE_SIZE; input++) {
            sum +=
                observation[input]
                * w1[input][hidden];
        }

        hidden_pre_activation[hidden] = sum;
        hidden_activation[hidden] = relu(sum);
    }

    for (int output = 0; output < OUTPUT_SIZE; output++) {
        float sum = b2[output];

        for (int hidden = 0; hidden < HIDDEN_SIZE; hidden++) {
            sum +=
                hidden_activation[hidden]
                * w2[hidden][output];
        }

        q_values[output] = sum;
    }
}

static int argmax_q(
    const float q_values[OUTPUT_SIZE]
) {
    int best_action = 0;
    float best_value = q_values[0];

    for (int action = 1; action < OUTPUT_SIZE; action++) {
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
    float w1[STATE_SIZE][HIDDEN_SIZE],
    float b1[HIDDEN_SIZE],
    float w2[HIDDEN_SIZE][OUTPUT_SIZE],
    float b2[OUTPUT_SIZE]
) {
    if (rand_uniform() < epsilon) {
        return rand_int(OUTPUT_SIZE);
    }

    float hidden_pre_activation[HIDDEN_SIZE];
    float hidden_activation[HIDDEN_SIZE];
    float q_values[OUTPUT_SIZE];

    forward_network(
        observation,
        w1,
        b1,
        w2,
        b2,
        hidden_pre_activation,
        hidden_activation,
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
    momentum =
        ADAM_BETA1 * momentum
        + (1.0f - ADAM_BETA1) * gradient;

    variance =
        ADAM_BETA2 * variance
        + (1.0f - ADAM_BETA2)
        * gradient
        * gradient;

    float momentum_hat =
        momentum / (1.0f - beta1_power);

    float variance_hat =
        variance / (1.0f - beta2_power);

    parameter +=
        LEARNING_RATE
        * momentum_hat
        / (
            hls::sqrtf(variance_hat)
            + ADAM_EPS
        );
}

static float train_one_sample_adam(
    const Transition &transition,
    float w1[STATE_SIZE][HIDDEN_SIZE],
    float b1[HIDDEN_SIZE],
    float w2[HIDDEN_SIZE][OUTPUT_SIZE],
    float b2[OUTPUT_SIZE],
    float target_w1[STATE_SIZE][HIDDEN_SIZE],
    float target_b1[HIDDEN_SIZE],
    float target_w2[HIDDEN_SIZE][OUTPUT_SIZE],
    float target_b2[OUTPUT_SIZE],
    float mw1[STATE_SIZE][HIDDEN_SIZE],
    float vw1[STATE_SIZE][HIDDEN_SIZE],
    float mb1[HIDDEN_SIZE],
    float vb1[HIDDEN_SIZE],
    float mw2[HIDDEN_SIZE][OUTPUT_SIZE],
    float vw2[HIDDEN_SIZE][OUTPUT_SIZE],
    float mb2[OUTPUT_SIZE],
    float vb2[OUTPUT_SIZE],
    float beta1_power,
    float beta2_power
) {
    float hidden_pre_activation[HIDDEN_SIZE];
    float hidden_activation[HIDDEN_SIZE];
    float q_values[OUTPUT_SIZE];

    forward_network(
        transition.state,
        w1,
        b1,
        w2,
        b2,
        hidden_pre_activation,
        hidden_activation,
        q_values
    );

    float target_hidden_pre_activation[HIDDEN_SIZE];
    float target_hidden_activation[HIDDEN_SIZE];
    float target_q_values[OUTPUT_SIZE];

    forward_network(
        transition.next_state,
        target_w1,
        target_b1,
        target_w2,
        target_b2,
        target_hidden_pre_activation,
        target_hidden_activation,
        target_q_values
    );

    float maximum_next_q = target_q_values[0];

    for (int action = 1; action < OUTPUT_SIZE; action++) {
        if (target_q_values[action] > maximum_next_q) {
            maximum_next_q = target_q_values[action];
        }
    }

    float target_q =
        transition.reward
        + (
            transition.done
            ? 0.0f
            : GAMMA * maximum_next_q
        );

    float predicted_q =
        q_values[transition.action];

    float td_error =
        target_q - predicted_q;

    float clipped_td_error =
        clip_float(td_error, -1.0f, 1.0f);

    float output_delta[OUTPUT_SIZE];

    for (int output = 0; output < OUTPUT_SIZE; output++) {
        output_delta[output] = 0.0f;
    }

    output_delta[transition.action] =
        clipped_td_error;

    float hidden_delta[HIDDEN_SIZE];

    for (int hidden = 0; hidden < HIDDEN_SIZE; hidden++) {
        float sum = 0.0f;

        for (int output = 0; output < OUTPUT_SIZE; output++) {
            sum +=
                w2[hidden][output]
                * output_delta[output];
        }

        hidden_delta[hidden] =
            clip_float(
                sum
                * relu_derivative(
                    hidden_pre_activation[hidden]
                ),
                -1.0f,
                1.0f
            );
    }

    for (int hidden = 0; hidden < HIDDEN_SIZE; hidden++) {
        for (int output = 0; output < OUTPUT_SIZE; output++) {
            float gradient =
                hidden_activation[hidden]
                * output_delta[output];

            adam_update(
                w2[hidden][output],
                gradient,
                mw2[hidden][output],
                vw2[hidden][output],
                beta1_power,
                beta2_power
            );
        }
    }

    for (int output = 0; output < OUTPUT_SIZE; output++) {
        adam_update(
            b2[output],
            output_delta[output],
            mb2[output],
            vb2[output],
            beta1_power,
            beta2_power
        );
    }

    for (int input = 0; input < STATE_SIZE; input++) {
        for (int hidden = 0; hidden < HIDDEN_SIZE; hidden++) {
            float gradient =
                transition.state[input]
                * hidden_delta[hidden];

            adam_update(
                w1[input][hidden],
                gradient,
                mw1[input][hidden],
                vw1[input][hidden],
                beta1_power,
                beta2_power
            );
        }
    }

    for (int hidden = 0; hidden < HIDDEN_SIZE; hidden++) {
        adam_update(
            b1[hidden],
            hidden_delta[hidden],
            mb1[hidden],
            vb1[hidden],
            beta1_power,
            beta2_power
        );
    }

    return abs_float(td_error);
}

static float epsilon_for_episode(int episode_index) {
    float progress =
        (float)episode_index
        / (0.30f * (float)MAX_EPISODES);

    if (progress > 1.0f) {
        progress = 1.0f;
    }

    float epsilon =
        EPS_START
        + progress
        * (EPS_END - EPS_START);

    return clip_float(
        epsilon,
        EPS_END,
        EPS_START
    );
}

static int validation_step_index(
    int validation_index,
    int step
) {
    return
        validation_index * MAX_STEPS
        + step;
}

static int validation_state_index(
    int validation_index,
    int step,
    int component
) {
    return
        (
            validation_index
            * MAX_STEPS
            + step
        ) * STATE_SIZE
        + component;
}

static int evaluation_step_index(
    int batch_index,
    int step
) {
    return
        batch_index * MAX_STEPS
        + step;
}

static int evaluation_state_index(
    int batch_index,
    int step,
    int component
) {
    return
        (
            batch_index
            * MAX_STEPS
            + step
        ) * STATE_SIZE
        + component;
}

static void run_validation_episode(
    int validation_index,
    int training_episode,
    float w1[STATE_SIZE][HIDDEN_SIZE],
    float b1[HIDDEN_SIZE],
    float w2[HIDDEN_SIZE][OUTPUT_SIZE],
    float b2[OUTPUT_SIZE],
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
    EnvState environment_state;
    float observation[STATE_SIZE];

    reset_fixed_environment(
        environment_state,
        PI,
        0.0f,
        observation
    );

    float total_reward = 0.0f;
    float minimum_abs_theta = 999.0f;
    int upright_steps = 0;
    int stable_upright_steps = 0;
    int current_consecutive_stable_steps = 0;
    int max_consecutive_stable_steps = 0;

    for (int step = 0; step < MAX_STEPS; step++) {
        float theta_wrapped =
            wrap_angle(environment_state.theta);

        float abs_theta =
            abs_float(theta_wrapped);

        if (abs_theta < minimum_abs_theta) {
            minimum_abs_theta = abs_theta;
        }

        if (abs_theta < THETA_UPRIGHT_THRESHOLD) {
            upright_steps++;
        }

        if (
            abs_theta < THETA_UPRIGHT_THRESHOLD
            && abs_float(environment_state.theta_dot)
               < THETA_DOT_STABLE_THRESHOLD
        ) {
            stable_upright_steps++;
            current_consecutive_stable_steps++;

            if (
                current_consecutive_stable_steps
                > max_consecutive_stable_steps
            ) {
                max_consecutive_stable_steps =
                    current_consecutive_stable_steps;
            }
        } else {
            current_consecutive_stable_steps = 0;
        }

        int state_base =
            validation_state_index(
                validation_index,
                step,
                0
            );

        validation_states[state_base + 0] =
            observation[0];

        validation_states[state_base + 1] =
            observation[1];

        validation_states[state_base + 2] =
            observation[2];

        float hidden_pre_activation[HIDDEN_SIZE];
        float hidden_activation[HIDDEN_SIZE];
        float q_values[OUTPUT_SIZE];

        forward_network(
            observation,
            w1,
            b1,
            w2,
            b2,
            hidden_pre_activation,
            hidden_activation,
            q_values
        );

        int action = argmax_q(q_values);

        float next_observation[STATE_SIZE];
        float reward;
        int done;

        step_environment(
            environment_state,
            action,
            next_observation,
            reward,
            done
        );

        int step_index =
            validation_step_index(
                validation_index,
                step
            );

        validation_actions[step_index] =
            action;

        validation_rewards[step_index] =
            reward;

        total_reward += reward;

        for (int component = 0; component < STATE_SIZE; component++) {
            observation[component] =
                next_observation[component];
        }
    }

    validation_training_episode[validation_index] =
        training_episode;

    validation_total_reward[validation_index] =
        total_reward;

    validation_min_abs_theta[validation_index] =
        minimum_abs_theta;

    validation_upright_steps[validation_index] =
        upright_steps;

    validation_stable_upright_steps[validation_index] =
        stable_upright_steps;

    validation_max_consecutive_stable_steps[validation_index] =
        max_consecutive_stable_steps;

    validation_success[validation_index] =
        max_consecutive_stable_steps
        >= SUCCESS_CONSECUTIVE_STABLE_STEPS
        ? 1
        : 0;

    validation_final_theta_wrapped[validation_index] =
        wrap_angle(environment_state.theta);

    validation_final_theta_unwrapped[validation_index] =
        environment_state.theta;

    validation_final_theta_dot[validation_index] =
        environment_state.theta_dot;
}

static void run_evaluation_episode(
    int batch_index,
    float initial_theta,
    float initial_theta_dot,
    float w1[STATE_SIZE][HIDDEN_SIZE],
    float b1[HIDDEN_SIZE],
    float w2[HIDDEN_SIZE][OUTPUT_SIZE],
    float b2[OUTPUT_SIZE],
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
    EnvState environment_state;
    float observation[STATE_SIZE];

    reset_fixed_environment(
        environment_state,
        initial_theta,
        initial_theta_dot,
        observation
    );

    evaluation_initial_theta[batch_index] =
        initial_theta;

    evaluation_initial_theta_dot[batch_index] =
        initial_theta_dot;

    evaluation_actual_initial_theta[batch_index] =
        environment_state.theta;

    evaluation_actual_initial_theta_dot[batch_index] =
        environment_state.theta_dot;

    float total_reward = 0.0f;
    float minimum_abs_theta = 999.0f;
    int upright_steps = 0;
    int stable_upright_steps = 0;
    int current_consecutive_stable_steps = 0;
    int max_consecutive_stable_steps = 0;

    for (int step = 0; step < MAX_STEPS; step++) {
        float theta_wrapped =
            wrap_angle(environment_state.theta);

        float abs_theta =
            abs_float(theta_wrapped);

        if (abs_theta < minimum_abs_theta) {
            minimum_abs_theta = abs_theta;
        }

        if (abs_theta < THETA_UPRIGHT_THRESHOLD) {
            upright_steps++;
        }

        if (
            abs_theta < THETA_UPRIGHT_THRESHOLD
            && abs_float(environment_state.theta_dot)
               < THETA_DOT_STABLE_THRESHOLD
        ) {
            stable_upright_steps++;
            current_consecutive_stable_steps++;

            if (
                current_consecutive_stable_steps
                > max_consecutive_stable_steps
            ) {
                max_consecutive_stable_steps =
                    current_consecutive_stable_steps;
            }
        } else {
            current_consecutive_stable_steps = 0;
        }

        int state_base =
            evaluation_state_index(
                batch_index,
                step,
                0
            );

        evaluation_states[state_base + 0] =
            observation[0];

        evaluation_states[state_base + 1] =
            observation[1];

        evaluation_states[state_base + 2] =
            observation[2];

        float hidden_pre_activation[HIDDEN_SIZE];
        float hidden_activation[HIDDEN_SIZE];
        float q_values[OUTPUT_SIZE];

        forward_network(
            observation,
            w1,
            b1,
            w2,
            b2,
            hidden_pre_activation,
            hidden_activation,
            q_values
        );

        int action = argmax_q(q_values);

        float next_observation[STATE_SIZE];
        float reward;
        int done;

        step_environment(
            environment_state,
            action,
            next_observation,
            reward,
            done
        );

        int step_index =
            evaluation_step_index(
                batch_index,
                step
            );

        evaluation_actions[step_index] =
            action;

        evaluation_rewards[step_index] =
            reward;

        total_reward += reward;

        for (int component = 0; component < STATE_SIZE; component++) {
            observation[component] =
                next_observation[component];
        }
    }

    evaluation_total_reward[batch_index] =
        total_reward;

    evaluation_min_abs_theta[batch_index] =
        minimum_abs_theta;

    evaluation_upright_steps[batch_index] =
        upright_steps;

    evaluation_stable_upright_steps[batch_index] =
        stable_upright_steps;

    evaluation_max_consecutive_stable_steps[batch_index] =
        max_consecutive_stable_steps;

    evaluation_success[batch_index] =
        max_consecutive_stable_steps
        >= SUCCESS_CONSECUTIVE_STABLE_STEPS
        ? 1
        : 0;

    evaluation_final_theta_wrapped[batch_index] =
        wrap_angle(environment_state.theta);

    evaluation_final_theta_unwrapped[batch_index] =
        environment_state.theta;

    evaluation_final_theta_dot[batch_index] =
        environment_state.theta_dot;
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
#pragma HLS INTERFACE m_axi port=training_episode offset=slave bundle=gmem0 depth=8000
#pragma HLS INTERFACE m_axi port=training_episode_length offset=slave bundle=gmem1 depth=8000
#pragma HLS INTERFACE m_axi port=training_total_reward offset=slave bundle=gmem2 depth=8000
#pragma HLS INTERFACE m_axi port=training_average_reward_per_step offset=slave bundle=gmem3 depth=8000
#pragma HLS INTERFACE m_axi port=training_average_abs_td_error offset=slave bundle=gmem4 depth=8000
#pragma HLS INTERFACE m_axi port=training_epsilon offset=slave bundle=gmem5 depth=8000
#pragma HLS INTERFACE m_axi port=training_timesteps offset=slave bundle=gmem6 depth=8000

#pragma HLS INTERFACE m_axi port=validation_training_episode offset=slave bundle=gmem7 depth=10
#pragma HLS INTERFACE m_axi port=validation_total_reward offset=slave bundle=gmem8 depth=10
#pragma HLS INTERFACE m_axi port=validation_min_abs_theta offset=slave bundle=gmem9 depth=10
#pragma HLS INTERFACE m_axi port=validation_upright_steps offset=slave bundle=gmem10 depth=10
#pragma HLS INTERFACE m_axi port=validation_stable_upright_steps offset=slave bundle=gmem11 depth=10
#pragma HLS INTERFACE m_axi port=validation_max_consecutive_stable_steps offset=slave bundle=gmem32 depth=10
#pragma HLS INTERFACE m_axi port=validation_success offset=slave bundle=gmem33 depth=10
#pragma HLS INTERFACE m_axi port=validation_final_theta_wrapped offset=slave bundle=gmem12 depth=10
#pragma HLS INTERFACE m_axi port=validation_final_theta_unwrapped offset=slave bundle=gmem13 depth=10
#pragma HLS INTERFACE m_axi port=validation_final_theta_dot offset=slave bundle=gmem14 depth=10
#pragma HLS INTERFACE m_axi port=validation_states offset=slave bundle=gmem15 depth=7500
#pragma HLS INTERFACE m_axi port=validation_actions offset=slave bundle=gmem16 depth=2500
#pragma HLS INTERFACE m_axi port=validation_rewards offset=slave bundle=gmem17 depth=2500

#pragma HLS INTERFACE m_axi port=evaluation_initial_theta offset=slave bundle=gmem18 depth=5
#pragma HLS INTERFACE m_axi port=evaluation_initial_theta_dot offset=slave bundle=gmem19 depth=5
#pragma HLS INTERFACE m_axi port=evaluation_actual_initial_theta offset=slave bundle=gmem20 depth=5
#pragma HLS INTERFACE m_axi port=evaluation_actual_initial_theta_dot offset=slave bundle=gmem21 depth=5
#pragma HLS INTERFACE m_axi port=evaluation_total_reward offset=slave bundle=gmem22 depth=5
#pragma HLS INTERFACE m_axi port=evaluation_min_abs_theta offset=slave bundle=gmem23 depth=5
#pragma HLS INTERFACE m_axi port=evaluation_upright_steps offset=slave bundle=gmem24 depth=5
#pragma HLS INTERFACE m_axi port=evaluation_stable_upright_steps offset=slave bundle=gmem25 depth=5
#pragma HLS INTERFACE m_axi port=evaluation_max_consecutive_stable_steps offset=slave bundle=gmem34 depth=5
#pragma HLS INTERFACE m_axi port=evaluation_success offset=slave bundle=gmem35 depth=5
#pragma HLS INTERFACE m_axi port=evaluation_final_theta_wrapped offset=slave bundle=gmem26 depth=5
#pragma HLS INTERFACE m_axi port=evaluation_final_theta_unwrapped offset=slave bundle=gmem27 depth=5
#pragma HLS INTERFACE m_axi port=evaluation_final_theta_dot offset=slave bundle=gmem28 depth=5
#pragma HLS INTERFACE m_axi port=evaluation_states offset=slave bundle=gmem29 depth=3750
#pragma HLS INTERFACE m_axi port=evaluation_actions offset=slave bundle=gmem30 depth=1250
#pragma HLS INTERFACE m_axi port=evaluation_rewards offset=slave bundle=gmem31 depth=1250

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

    static float w1[STATE_SIZE][HIDDEN_SIZE];
    static float b1[HIDDEN_SIZE];
    static float w2[HIDDEN_SIZE][OUTPUT_SIZE];
    static float b2[OUTPUT_SIZE];

    static float target_w1[STATE_SIZE][HIDDEN_SIZE];
    static float target_b1[HIDDEN_SIZE];
    static float target_w2[HIDDEN_SIZE][OUTPUT_SIZE];
    static float target_b2[OUTPUT_SIZE];

    static float best_w1[STATE_SIZE][HIDDEN_SIZE];
    static float best_b1[HIDDEN_SIZE];
    static float best_w2[HIDDEN_SIZE][OUTPUT_SIZE];
    static float best_b2[OUTPUT_SIZE];

    static float mw1[STATE_SIZE][HIDDEN_SIZE];
    static float vw1[STATE_SIZE][HIDDEN_SIZE];
    static float mb1[HIDDEN_SIZE];
    static float vb1[HIDDEN_SIZE];

    static float mw2[HIDDEN_SIZE][OUTPUT_SIZE];
    static float vw2[HIDDEN_SIZE][OUTPUT_SIZE];
    static float mb2[OUTPUT_SIZE];
    static float vb2[OUTPUT_SIZE];

    static Transition replay_buffer[REPLAY_SIZE];

    completed_training_episodes = 0;
    completed_validations = 0;
    completed_evaluations = 0;

    if (mode != 1) {
        g_seed = seed_value;

        initialize_network(
            w1,
            b1,
            w2,
            b2
        );

    initialize_adam(
        mw1,
        vw1,
        mb1,
        vb1,
        mw2,
        vw2,
        mb2,
        vb2
    );

    copy_network(
        w1,
        b1,
        w2,
        b2,
        target_w1,
        target_b1,
        target_w2,
        target_b2
    );

    copy_network(
        w1,
        b1,
        w2,
        b2,
        best_w1,
        best_b1,
        best_w2,
        best_b2
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
        float epsilon =
            epsilon_for_episode(episode);

        EnvState environment_state;
        float observation[STATE_SIZE];

        reset_training_environment(
            environment_state,
            observation
        );

        float total_episode_reward = 0.0f;
        float td_error_sum = 0.0f;
        int td_update_count = 0;

        for (int step = 0; step < MAX_STEPS; step++) {
            int action =
                select_action(
                    observation,
                    epsilon,
                    w1,
                    b1,
                    w2,
                    b2
                );

            float next_observation[STATE_SIZE];
            float reward;
            int done;

            step_environment(
                environment_state,
                action,
                next_observation,
                reward,
                done
            );

            replay_buffer[replay_pointer].action =
                action;

            replay_buffer[replay_pointer].reward =
                reward;

            replay_buffer[replay_pointer].done =
                done;

            for (int component = 0; component < STATE_SIZE; component++) {
                replay_buffer[replay_pointer]
                    .state[component] =
                    observation[component];

                replay_buffer[replay_pointer]
                    .next_state[component] =
                    next_observation[component];
            }

            replay_pointer =
                (replay_pointer + 1)
                % REPLAY_SIZE;

            if (replay_count < REPLAY_SIZE) {
                replay_count++;
            }

            total_episode_reward += reward;

            if (
                replay_count >= LEARNING_STARTS
                && (global_step % TRAIN_FREQUENCY) == 0
            ) {
                for (int batch = 0; batch < BATCH_SIZE; batch++) {
                    int replay_index =
                        rand_int(replay_count);

                    beta1_power *= ADAM_BETA1;
                    beta2_power *= ADAM_BETA2;

                    float abs_td_error =
                        train_one_sample_adam(
                            replay_buffer[replay_index],
                            w1,
                            b1,
                            w2,
                            b2,
                            target_w1,
                            target_b1,
                            target_w2,
                            target_b2,
                            mw1,
                            vw1,
                            mb1,
                            vb1,
                            mw2,
                            vw2,
                            mb2,
                            vb2,
                            beta1_power,
                            beta2_power
                        );

                    td_error_sum += abs_td_error;
                    td_update_count++;
                }
            }

            if (
                global_step > 0
                && (
                    global_step
                    % TARGET_UPDATE_FREQUENCY
                ) == 0
            ) {
                copy_network(
                    w1,
                    b1,
                    w2,
                    b2,
                    target_w1,
                    target_b1,
                    target_w2,
                    target_b2
                );
            }

            for (int component = 0; component < STATE_SIZE; component++) {
                observation[component] =
                    next_observation[component];
            }

            global_step++;
        }

        training_episode[episode] =
            episode + 1;

        training_episode_length[episode] =
            MAX_STEPS;

        training_total_reward[episode] =
            total_episode_reward;

        training_average_reward_per_step[episode] =
            total_episode_reward
            / (float)MAX_STEPS;

        training_average_abs_td_error[episode] =
            td_update_count > 0
            ? td_error_sum
              / (float)td_update_count
            : 0.0f;

        training_epsilon[episode] =
            epsilon;

        training_timesteps[episode] =
            global_step;

        completed_training_episodes =
            episode + 1;

        if (
            ((episode + 1) % VALIDATION_INTERVAL) == 0
        ) {
            run_validation_episode(
                validation_index,
                episode + 1,
                w1,
                b1,
                w2,
                b2,
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

            int current_success =
                validation_success[validation_index];

            int current_stable_upright_steps =
                validation_stable_upright_steps[validation_index];

            float current_validation_reward =
                validation_total_reward[validation_index];

            int is_better_checkpoint = 0;

            if (current_success > best_success) {
                is_better_checkpoint = 1;
            } else if (current_success == best_success
                       && current_stable_upright_steps
                          > best_stable_upright_steps) {
                is_better_checkpoint = 1;
            } else if (current_success == best_success
                       && current_stable_upright_steps
                          == best_stable_upright_steps
                       && current_validation_reward
                          > best_validation_reward) {
                is_better_checkpoint = 1;
            }

            if (is_better_checkpoint) {
                best_success = current_success;
                best_stable_upright_steps =
                    current_stable_upright_steps;
                best_validation_reward =
                    current_validation_reward;

                copy_network(
                    w1,
                    b1,
                    w2,
                    b2,
                    best_w1,
                    best_b1,
                    best_w2,
                    best_b2
                );
            }

            validation_index++;
            completed_validations =
                validation_index;
        }
    }

        copy_network(
            best_w1,
            best_b1,
            best_w2,
            best_b2,
            w1,
            b1,
            w2,
            b2
        );
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

        completed_evaluations =
            batch + 1;
    }
}

}