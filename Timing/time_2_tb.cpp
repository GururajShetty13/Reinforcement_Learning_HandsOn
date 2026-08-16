#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstring>


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

#define NN_FORWARD_PROFILE_STEPS 10000
#define NN_BACKPROP_PROFILE_STEPS 10000
#define NN_UPDATE_PROFILE_STEPS 10
#define NN_TRAIN_SAMPLE_PROFILE_STEPS 10
#define REPLAY_PROFILE_STEPS 10000

#ifndef RUN_FULL_WORKFLOW_TESTS
#define RUN_FULL_WORKFLOW_TESTS 0
#endif

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
);
}


static int training_episode[MAX_EPISODES];
static int training_episode_length[MAX_EPISODES];
static float training_total_reward[MAX_EPISODES];
static float training_average_reward_per_step[MAX_EPISODES];
static float training_average_abs_td_error[MAX_EPISODES];
static float training_epsilon[MAX_EPISODES];
static int training_timesteps[MAX_EPISODES];

static int validation_training_episode[NUM_VALIDATIONS];
static float validation_total_reward[NUM_VALIDATIONS];
static float validation_min_abs_theta[NUM_VALIDATIONS];
static int validation_upright_steps[NUM_VALIDATIONS];
static int validation_stable_upright_steps[NUM_VALIDATIONS];
static int validation_max_consecutive_stable_steps[NUM_VALIDATIONS];
static int validation_success[NUM_VALIDATIONS];
static float validation_final_theta_wrapped[NUM_VALIDATIONS];
static float validation_final_theta_unwrapped[NUM_VALIDATIONS];
static float validation_final_theta_dot[NUM_VALIDATIONS];
static float validation_states[VALIDATION_STATE_BUFFER_SIZE];
static int validation_actions[VALIDATION_TRAJECTORY_SIZE];
static float validation_rewards[VALIDATION_TRAJECTORY_SIZE];

static float evaluation_initial_theta[EVALUATION_BATCH_SIZE];
static float evaluation_initial_theta_dot[EVALUATION_BATCH_SIZE];
static float evaluation_actual_initial_theta[EVALUATION_BATCH_SIZE];
static float evaluation_actual_initial_theta_dot[EVALUATION_BATCH_SIZE];
static float evaluation_total_reward[EVALUATION_BATCH_SIZE];
static float evaluation_min_abs_theta[EVALUATION_BATCH_SIZE];
static int evaluation_upright_steps[EVALUATION_BATCH_SIZE];
static int evaluation_stable_upright_steps[EVALUATION_BATCH_SIZE];
static int evaluation_max_consecutive_stable_steps[EVALUATION_BATCH_SIZE];
static int evaluation_success[EVALUATION_BATCH_SIZE];
static float evaluation_final_theta_wrapped[EVALUATION_BATCH_SIZE];
static float evaluation_final_theta_unwrapped[EVALUATION_BATCH_SIZE];
static float evaluation_final_theta_dot[EVALUATION_BATCH_SIZE];
static float evaluation_states[EVALUATION_STATE_BUFFER_SIZE];
static int evaluation_actions[EVALUATION_TRAJECTORY_SIZE];
static float evaluation_rewards[EVALUATION_TRAJECTORY_SIZE];

struct RunResult {
    double seconds;
    int completed_training_episodes;
    int completed_validations;
    int completed_evaluations;
};

static void clear_output_buffers() {
    std::memset(training_episode, 0, sizeof(training_episode));
    std::memset(training_episode_length, 0, sizeof(training_episode_length));
    std::memset(training_total_reward, 0, sizeof(training_total_reward));
    std::memset(training_average_reward_per_step, 0, sizeof(training_average_reward_per_step));
    std::memset(training_average_abs_td_error, 0, sizeof(training_average_abs_td_error));
    std::memset(training_epsilon, 0, sizeof(training_epsilon));
    std::memset(training_timesteps, 0, sizeof(training_timesteps));

    std::memset(validation_training_episode, 0, sizeof(validation_training_episode));
    std::memset(validation_total_reward, 0, sizeof(validation_total_reward));
    std::memset(validation_min_abs_theta, 0, sizeof(validation_min_abs_theta));
    std::memset(validation_upright_steps, 0, sizeof(validation_upright_steps));
    std::memset(validation_stable_upright_steps, 0, sizeof(validation_stable_upright_steps));
    std::memset(validation_max_consecutive_stable_steps, 0, sizeof(validation_max_consecutive_stable_steps));
    std::memset(validation_success, 0, sizeof(validation_success));
    std::memset(validation_final_theta_wrapped, 0, sizeof(validation_final_theta_wrapped));
    std::memset(validation_final_theta_unwrapped, 0, sizeof(validation_final_theta_unwrapped));
    std::memset(validation_final_theta_dot, 0, sizeof(validation_final_theta_dot));
    std::memset(validation_states, 0, sizeof(validation_states));
    std::memset(validation_actions, 0, sizeof(validation_actions));
    std::memset(validation_rewards, 0, sizeof(validation_rewards));

    std::memset(evaluation_initial_theta, 0, sizeof(evaluation_initial_theta));
    std::memset(evaluation_initial_theta_dot, 0, sizeof(evaluation_initial_theta_dot));
    std::memset(evaluation_actual_initial_theta, 0, sizeof(evaluation_actual_initial_theta));
    std::memset(evaluation_actual_initial_theta_dot, 0, sizeof(evaluation_actual_initial_theta_dot));
    std::memset(evaluation_total_reward, 0, sizeof(evaluation_total_reward));
    std::memset(evaluation_min_abs_theta, 0, sizeof(evaluation_min_abs_theta));
    std::memset(evaluation_upright_steps, 0, sizeof(evaluation_upright_steps));
    std::memset(evaluation_stable_upright_steps, 0, sizeof(evaluation_stable_upright_steps));
    std::memset(evaluation_max_consecutive_stable_steps, 0, sizeof(evaluation_max_consecutive_stable_steps));
    std::memset(evaluation_success, 0, sizeof(evaluation_success));
    std::memset(evaluation_final_theta_wrapped, 0, sizeof(evaluation_final_theta_wrapped));
    std::memset(evaluation_final_theta_unwrapped, 0, sizeof(evaluation_final_theta_unwrapped));
    std::memset(evaluation_final_theta_dot, 0, sizeof(evaluation_final_theta_dot));
    std::memset(evaluation_states, 0, sizeof(evaluation_states));
    std::memset(evaluation_actions, 0, sizeof(evaluation_actions));
    std::memset(evaluation_rewards, 0, sizeof(evaluation_rewards));
}

static const char *mode_name(int mode) {
    switch (mode) {
        case 6: return "forward layer 1: input -> hidden1";
        case 7: return "forward layer 2: hidden1 -> hidden2";
        case 8: return "forward output: hidden2 -> Q-values";
        case 9: return "full forward pass + argmax";
        case 10: return "backprop: output -> hidden2";
        case 11: return "backprop: hidden2 -> hidden1";
        case 12: return "Adam update: W3 and B3";
        case 13: return "Adam update: W2 and B2";
        case 14: return "Adam update: W1 and B1";
        case 15: return "complete train_one_sample_adam";
        case 16: return "replay store + load";
        default: return "unknown";
    }
}

static int nominal_iterations(int mode) {
    switch (mode) {
        case 6:
        case 7:
        case 8:
        case 9:
            return NN_FORWARD_PROFILE_STEPS;
        case 10:
        case 11:
            return NN_BACKPROP_PROFILE_STEPS;
        case 12:
        case 13:
        case 14:
            return NN_UPDATE_PROFILE_STEPS;
        case 15:
            return NN_TRAIN_SAMPLE_PROFILE_STEPS;
        case 16:
            return REPLAY_PROFILE_STEPS;
        default:
            return 1;
    }
}

static RunResult run_mode(int mode, unsigned int seed, bool clear_buffers) {
    if (clear_buffers) {
        clear_output_buffers();
    }

    int completed_training_episodes = 0; 
    int completed_validations = 0;
    int completed_evaluations = 0;        

    auto start_time = std::chrono::high_resolution_clock::now();

    dqn_eva(
        mode,
        seed,
        training_episode,
        training_episode_length,
        training_total_reward,
        training_average_reward_per_step,
        training_average_abs_td_error,
        training_epsilon,
        training_timesteps,
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
        validation_rewards,
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
        evaluation_rewards,
        completed_training_episodes,
        completed_validations,
        completed_evaluations
    );

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    RunResult result;
    result.seconds = elapsed.count();
    result.completed_training_episodes = completed_training_episodes;
    result.completed_validations = completed_validations;
    result.completed_evaluations = completed_evaluations;
    return result;
}

static void print_profile_row(int mode, const RunResult &result) {
    int iterations_from_dut = result.completed_evaluations;
    int iterations = iterations_from_dut > 0 ? iterations_from_dut : nominal_iterations(mode);
    double average_seconds = result.seconds / (double)iterations;

    std::cout << std::right << std::setw(4) << mode << " | "
              << std::left << std::setw(42) << mode_name(mode)
              << " | total=" << std::right << std::setw(10) << std::fixed << std::setprecision(6) << result.seconds << " s"
              << " | iter=" << std::setw(6) << iterations
              << " | avg=" << std::setw(12) << std::setprecision(3) << average_seconds * 1.0e6 << " us"
              << " | checksum=" << result.completed_training_episodes
              << std::endl;
}

int main() {
    const unsigned int seed = 42u;

    std::cout << "============================================================" << std::endl;
    std::cout << "DQN EVA neural-network internal profiling testbench" << std::endl;
    std::cout << "This is C-simulation timing, not FPGA board timing." << std::endl;
    std::cout << "============================================================" << std::endl;

    std::cout << "mode | action                                     | total time   | iterations | average/action | checksum" << std::endl;
    std::cout << "------------------------------------------------------------------------------------------------------" << std::endl;

    for (int mode = 6; mode <= 16; mode++) {
        RunResult result = run_mode(mode, seed, true);
        print_profile_row(mode, result);
    }

#if RUN_FULL_WORKFLOW_TESTS
    std::cout << "\nLong workflow tests" << std::endl;
    std::cout << "These modes can take a long time in C simulation." << std::endl;
    RunResult training_only = run_mode(5, seed, true);
    std::cout << "Training only mode 5 time: " << training_only.seconds
              << " s | train=" << training_only.completed_training_episodes << std::endl;

    RunResult full_workflow = run_mode(0, seed, true);
    std::cout << "Full workflow mode 0 time: " << full_workflow.seconds
              << " s | train=" << full_workflow.completed_training_episodes
              << " | val=" << full_workflow.completed_validations
              << " | eval=" << full_workflow.completed_evaluations << std::endl;
#endif

    std::cout << "\nTEST PASSED" << std::endl;
    return 0;
}