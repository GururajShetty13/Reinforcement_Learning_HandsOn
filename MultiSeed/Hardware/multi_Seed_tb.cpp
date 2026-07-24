#include <stdio.h>
#include <math.h>
#include <string.h>

#define STATE_SIZE 3
#define MAX_EPISODES 8000
#define MAX_STEPS 250
#define VALIDATION_INTERVAL 800
#define NUM_VALIDATIONS (MAX_EPISODES / VALIDATION_INTERVAL)
#define EVALUATION_BATCH_SIZE 5
#define NUM_SEEDS 5

#define PI 3.14159265358979323846f
#define THETA_UPRIGHT_THRESHOLD 0.35f
#define THETA_DOT_STABLE_THRESHOLD 0.5f
#define SUCCESS_CONSECUTIVE_STABLE_STEPS 30

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

static float validation_states[
    NUM_VALIDATIONS
    * MAX_STEPS
    * STATE_SIZE
];

static int validation_actions[
    NUM_VALIDATIONS
    * MAX_STEPS
];

static float validation_rewards[
    NUM_VALIDATIONS
    * MAX_STEPS
];

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

static float evaluation_states[
    EVALUATION_BATCH_SIZE
    * MAX_STEPS
    * STATE_SIZE
];

static int evaluation_actions[
    EVALUATION_BATCH_SIZE
    * MAX_STEPS
];

static float evaluation_rewards[
    EVALUATION_BATCH_SIZE
    * MAX_STEPS
];

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


static unsigned int current_seed_value = 0u;

static FILE *open_seed_csv(const char *base_name) {
    char filename[256];
    snprintf(
        filename,
        sizeof(filename),
        "seed_%u_%s",
        current_seed_value,
        base_name
    );
    return fopen(filename, "w");
}

static float safe_div_float(float numerator, float denominator) {
    if (denominator == 0.0f) {
        return 0.0f;
    }
    return numerator / denominator;
}

static float compute_f1_from_stable_steps(int stable_steps, int total_steps) {
    if (stable_steps <= 0 || total_steps <= 0) {
        return 0.0f;
    }

    float precision = 1.0f;
    float recall = (float)stable_steps / (float)total_steps;
    return safe_div_float(2.0f * precision * recall, precision + recall);
}

static int write_training_csv() {
    FILE *file =
        open_seed_csv("training_metrics.csv");

    if (file == NULL) {
        return 0;
    }

    fprintf(
        file,
        "episode,episode_length,total_reward,"
        "average_reward_per_step,"
        "average_absolute_td_error,"
        "epsilon,timesteps\n"
    );

    for (int episode = 0; episode < MAX_EPISODES; episode++) {
        fprintf(
            file,
            "%d,%d,%.9f,%.9f,%.9f,%.9f,%d\n",
            training_episode[episode],
            training_episode_length[episode],
            training_total_reward[episode],
            training_average_reward_per_step[episode],
            training_average_abs_td_error[episode],
            training_epsilon[episode],
            training_timesteps[episode]
        );
    }

    fclose(file);
    return 1;
}

static int write_validation_summary_csv() {
    FILE *file =
        open_seed_csv("validation_summary.csv");

    if (file == NULL) {
        return 0;
    }

    fprintf(
        file,
        "validation_index,training_episode,"
        "initial_theta,initial_theta_dot,"
        "total_reward,minimum_abs_theta,"
        "upright_steps,stable_upright_steps,"
        "max_consecutive_stable_steps,success,"
        "stable_theta_threshold,stable_theta_dot_threshold,"
        "success_consecutive_stable_steps_threshold,"
        "final_theta_wrapped,"
        "final_theta_unwrapped,"
        "final_theta_dot\n"
    );

    for (
        int validation = 0;
        validation < NUM_VALIDATIONS;
        validation++
    ) {
        fprintf(
            file,
            "%d,%d,%.9f,%.9f,%.9f,%.9f,"
            "%d,%d,%d,%d,%.9f,%.9f,%d,"
            "%.9f,%.9f,%.9f\n",
            validation,
            validation_training_episode[validation],
            3.14159265358979323846f,
            0.0f,
            validation_total_reward[validation],
            validation_min_abs_theta[validation],
            validation_upright_steps[validation],
            validation_stable_upright_steps[validation],
            validation_max_consecutive_stable_steps[validation],
            validation_success[validation],
            THETA_UPRIGHT_THRESHOLD,
            THETA_DOT_STABLE_THRESHOLD,
            SUCCESS_CONSECUTIVE_STABLE_STEPS,
            validation_final_theta_wrapped[validation],
            validation_final_theta_unwrapped[validation],
            validation_final_theta_dot[validation]
        );
    }

    fclose(file);
    return 1;
}

static int write_validation_trajectory_csv() {
    FILE *file =
        open_seed_csv("validation_trajectories.csv");

    if (file == NULL) {
        return 0;
    }

    fprintf(
        file,
        "validation_index,training_episode,"
        "step,state_cos_theta,state_sin_theta,"
        "state_theta_dot,theta_wrapped,"
        "upright,stable_upright,action,torque,reward,"
        "validation_total_reward\n"
    );

    for (
        int validation = 0;
        validation < NUM_VALIDATIONS;
        validation++
    ) {
        for (int step = 0; step < MAX_STEPS; step++) {
            int state_base =
                validation_state_index(
                    validation,
                    step,
                    0
                );

            int step_index =
                validation_step_index(
                    validation,
                    step
                );

            int action =
                validation_actions[step_index];

            float torque;

            if (action == 0) torque = -2.0f;
            else if (action == 1) torque = -0.5f;
            else if (action == 2) torque = -0.1f;
            else if (action == 3) torque = 0.0f;
            else if (action == 4) torque = 0.1f;
            else if (action == 5) torque = 0.5f;
            else torque = 2.0f;

            float theta_wrapped = atan2f(
                validation_states[state_base + 1],
                validation_states[state_base + 0]
            );

            int upright =
                fabsf(theta_wrapped) < THETA_UPRIGHT_THRESHOLD
                ? 1
                : 0;

            int stable_upright =
                (
                    fabsf(theta_wrapped) < THETA_UPRIGHT_THRESHOLD
                    && fabsf(validation_states[state_base + 2])
                       < THETA_DOT_STABLE_THRESHOLD
                )
                ? 1
                : 0;

            fprintf(
                file,
                "%d,%d,%d,%.9f,%.9f,%.9f,%.9f,"
                "%d,%d,%d,%.9f,%.9f,%.9f\n",
                validation,
                validation_training_episode[validation],
                step,
                validation_states[state_base + 0],
                validation_states[state_base + 1],
                validation_states[state_base + 2],
                theta_wrapped,
                upright,
                stable_upright,
                action,
                torque,
                validation_rewards[step_index],
                validation_total_reward[validation]
            );
        }
    }

    fclose(file);
    return 1;
}

static int write_evaluation_summary_csv() {
    FILE *file =
        open_seed_csv("evaluation_summary.csv");

    if (file == NULL) {
        return 0;
    }

    fprintf(
        file,
        "evaluation_batch,initial_theta,"
        "initial_theta_dot,"
        "actual_initial_theta,"
        "actual_initial_theta_dot,"
        "total_reward,minimum_abs_theta,"
        "upright_steps,stable_upright_steps,"
        "max_consecutive_stable_steps,success,"
        "stable_theta_threshold,stable_theta_dot_threshold,"
        "success_consecutive_stable_steps_threshold,"
        "final_theta_wrapped,"
        "final_theta_unwrapped,"
        "final_theta_dot\n"
    );

    for (
        int batch = 0;
        batch < EVALUATION_BATCH_SIZE;
        batch++
    ) {
        fprintf(
            file,
            "%d,%.9f,%.9f,%.9f,%.9f,"
            "%.9f,%.9f,%d,%d,%d,%d,"
            "%.9f,%.9f,%d,%.9f,%.9f,%.9f\n",
            batch,
            evaluation_initial_theta[batch],
            evaluation_initial_theta_dot[batch],
            evaluation_actual_initial_theta[batch],
            evaluation_actual_initial_theta_dot[batch],
            evaluation_total_reward[batch],
            evaluation_min_abs_theta[batch],
            evaluation_upright_steps[batch],
            evaluation_stable_upright_steps[batch],
            evaluation_max_consecutive_stable_steps[batch],
            evaluation_success[batch],
            THETA_UPRIGHT_THRESHOLD,
            THETA_DOT_STABLE_THRESHOLD,
            SUCCESS_CONSECUTIVE_STABLE_STEPS,
            evaluation_final_theta_wrapped[batch],
            evaluation_final_theta_unwrapped[batch],
            evaluation_final_theta_dot[batch]
        );
    }

    fclose(file);
    return 1;
}

static int write_evaluation_trajectory_csv() {
    FILE *file =
        open_seed_csv("evaluation_trajectories.csv");

    if (file == NULL) {
        return 0;
    }

    fprintf(
        file,
        "evaluation_batch,initial_theta,"
        "initial_theta_dot,step,"
        "state_cos_theta,state_sin_theta,"
        "state_theta_dot,theta_wrapped,"
        "upright,stable_upright,action,torque,reward,"
        "evaluation_total_reward\n"
    );

    for (
        int batch = 0;
        batch < EVALUATION_BATCH_SIZE;
        batch++
    ) {
        for (int step = 0; step < MAX_STEPS; step++) {
            int state_base =
                evaluation_state_index(
                    batch,
                    step,
                    0
                );

            int step_index =
                evaluation_step_index(
                    batch,
                    step
                );

            int action =
                evaluation_actions[step_index];

            float torque;

            if (action == 0) torque = -2.0f;
            else if (action == 1) torque = -0.5f;
            else if (action == 2) torque = -0.1f;
            else if (action == 3) torque = 0.0f;
            else if (action == 4) torque = 0.1f;
            else if (action == 5) torque = 0.5f;
            else torque = 2.0f;

            float theta_wrapped = atan2f(
                evaluation_states[state_base + 1],
                evaluation_states[state_base + 0]
            );

            int upright =
                fabsf(theta_wrapped) < THETA_UPRIGHT_THRESHOLD
                ? 1
                : 0;

            int stable_upright =
                (
                    fabsf(theta_wrapped) < THETA_UPRIGHT_THRESHOLD
                    && fabsf(evaluation_states[state_base + 2])
                       < THETA_DOT_STABLE_THRESHOLD
                )
                ? 1
                : 0;

            fprintf(
                file,
                "%d,%.9f,%.9f,%d,"
                "%.9f,%.9f,%.9f,%.9f,"
                "%d,%d,%d,%.9f,%.9f,%.9f\n",
                batch,
                evaluation_initial_theta[batch],
                evaluation_initial_theta_dot[batch],
                step,
                evaluation_states[state_base + 0],
                evaluation_states[state_base + 1],
                evaluation_states[state_base + 2],
                theta_wrapped,
                upright,
                stable_upright,
                action,
                torque,
                evaluation_rewards[step_index],
                evaluation_total_reward[batch]
            );
        }
    }

    fclose(file);
    return 1;
}


static int write_multiseed_summary_csv(
    const unsigned int seeds[NUM_SEEDS],
    const float training_time_seconds[NUM_SEEDS],
    const float average_evaluation_reward[NUM_SEEDS],
    const float average_upright_accuracy[NUM_SEEDS],
    const float average_stable_accuracy[NUM_SEEDS],
    const float average_max_consecutive_stable_steps[NUM_SEEDS],
    const float success_rate[NUM_SEEDS],
    const float f1_score[NUM_SEEDS]
) {
    FILE *file = fopen("multiseed_summary.csv", "w");
    if (file == NULL) {
        return 0;
    }

    fprintf(
        file,
        "seed,training_episodes,training_steps,final_training_reward_per_step,"
        "average_evaluation_reward,average_upright_accuracy,"
        "average_stable_accuracy,average_max_consecutive_stable_steps,"
        "success_rate,f1_score\n"
    );

    for (int i = 0; i < NUM_SEEDS; i++) {
        fprintf(
            file,
            "%u,%d,%d,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f\n",
            seeds[i],
            MAX_EPISODES,
            MAX_EPISODES * MAX_STEPS,
            training_time_seconds[i],
            average_evaluation_reward[i],
            average_upright_accuracy[i],
            average_stable_accuracy[i],
            average_max_consecutive_stable_steps[i],
            success_rate[i],
            f1_score[i]
        );
    }

    fclose(file);
    return 1;
}

static void mean_std(
    const float values[NUM_SEEDS],
    float &mean,
    float &std_value,
    float &minimum,
    float &maximum
) {
    mean = 0.0f;
    minimum = values[0];
    maximum = values[0];

    for (int i = 0; i < NUM_SEEDS; i++) {
        mean += values[i];
        if (values[i] < minimum) minimum = values[i];
        if (values[i] > maximum) maximum = values[i];
    }
    mean /= (float)NUM_SEEDS;

    float variance = 0.0f;
    for (int i = 0; i < NUM_SEEDS; i++) {
        float diff = values[i] - mean;
        variance += diff * diff;
    }

    if (NUM_SEEDS > 1) {
        variance /= (float)(NUM_SEEDS - 1);
    }

    std_value = sqrtf(variance);
}

static int write_multiseed_mean_std_csv(
    const float average_evaluation_reward[NUM_SEEDS],
    const float average_upright_accuracy[NUM_SEEDS],
    const float average_stable_accuracy[NUM_SEEDS],
    const float average_max_consecutive_stable_steps[NUM_SEEDS],
    const float success_rate[NUM_SEEDS],
    const float f1_score[NUM_SEEDS]
) {
    FILE *file = fopen("multiseed_mean_std.csv", "w");
    if (file == NULL) {
        return 0;
    }

    fprintf(file, "metric,mean,std,min,max\n");

    const char *names[6] = {
        "average_evaluation_reward",
        "average_upright_accuracy",
        "average_stable_accuracy",
        "average_max_consecutive_stable_steps",
        "success_rate",
        "f1_score"
    };

    const float *metrics[6] = {
        average_evaluation_reward,
        average_upright_accuracy,
        average_stable_accuracy,
        average_max_consecutive_stable_steps,
        success_rate,
        f1_score
    };

    for (int metric = 0; metric < 6; metric++) {
        float mean;
        float std_value;
        float minimum;
        float maximum;

        mean_std(metrics[metric], mean, std_value, minimum, maximum);

        fprintf(
            file,
            "%s,%.9f,%.9f,%.9f,%.9f\n",
            names[metric],
            mean,
            std_value,
            minimum,
            maximum
        );
    }

    fclose(file);
    return 1;
}

int main() {
    const unsigned int seeds[NUM_SEEDS] = {0u, 1u, 2u, 3u, 4u};

    float final_training_reward_per_step[NUM_SEEDS];
    float average_evaluation_reward[NUM_SEEDS];
    float average_upright_accuracy[NUM_SEEDS];
    float average_stable_accuracy[NUM_SEEDS];
    float average_max_consecutive_stable_steps[NUM_SEEDS];
    float success_rate[NUM_SEEDS];
    float f1_score[NUM_SEEDS];

    int overall_pass = 1;

    printf("========================================\n");
    printf("DQN ADAM VITIS MULTI-SEED TESTBENCH\n");
    printf("========================================\n");

    for (int seed_index = 0; seed_index < NUM_SEEDS; seed_index++) {
        current_seed_value = seeds[seed_index];

        int completed_training_episodes = 0;
        int completed_validations = 0;
        int completed_evaluations = 0;

        printf("\n========================================\n");
        printf("Starting seed %u\n", current_seed_value);
        printf("========================================\n");

        dqn_eva(
            0,
            current_seed_value,

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

        printf(
            "Completed training episodes = %d / %d\n",
            completed_training_episodes,
            MAX_EPISODES
        );
        printf(
            "Completed validations       = %d / %d\n",
            completed_validations,
            NUM_VALIDATIONS
        );
        printf(
            "Completed evaluations       = %d / %d\n",
            completed_evaluations,
            EVALUATION_BATCH_SIZE
        );

        printf("----------------------------------------\n");
        printf(
            "Final training reward/step  = %f\n",
            training_average_reward_per_step[MAX_EPISODES - 1]
        );
        printf(
            "Final average abs TD error  = %f\n",
            training_average_abs_td_error[MAX_EPISODES - 1]
        );
        printf(
            "Final epsilon               = %f\n",
            training_epsilon[MAX_EPISODES - 1]
        );

        printf("----------------------------------------\n");
        for (int validation = 0; validation < NUM_VALIDATIONS; validation++) {
            printf(
                "Validation %d | Episode %d | "
                "Reward=%f | Upright=%d | Stable=%d | MaxStable=%d | Success=%d\n",
                validation,
                validation_training_episode[validation],
                validation_total_reward[validation],
                validation_upright_steps[validation],
                validation_stable_upright_steps[validation],
                validation_max_consecutive_stable_steps[validation],
                validation_success[validation]
            );
        }

        printf("----------------------------------------\n");

        int initial_state_check_pass = 1;
        int total_upright_steps = 0;
        int total_stable_steps = 0;
        int total_successes = 0;
        int total_max_consecutive = 0;
        float total_evaluation_reward = 0.0f;

        for (int batch = 0; batch < EVALUATION_BATCH_SIZE; batch++) {
            float theta_error =
                fabsf(
                    evaluation_initial_theta[batch]
                    - evaluation_actual_initial_theta[batch]
                );

            float theta_dot_error =
                fabsf(
                    evaluation_initial_theta_dot[batch]
                    - evaluation_actual_initial_theta_dot[batch]
                );

            if (theta_error > 1.0e-6f || theta_dot_error > 1.0e-6f) {
                initial_state_check_pass = 0;
            }

            total_evaluation_reward += evaluation_total_reward[batch];
            total_upright_steps += evaluation_upright_steps[batch];
            total_stable_steps += evaluation_stable_upright_steps[batch];
            total_successes += evaluation_success[batch];
            total_max_consecutive += evaluation_max_consecutive_stable_steps[batch];

            printf(
                "Evaluation %d | theta=%f | theta_dot=%f | "
                "reward=%f | upright=%d | stable=%d | MaxStable=%d | Success=%d\n",
                batch,
                evaluation_actual_initial_theta[batch],
                evaluation_actual_initial_theta_dot[batch],
                evaluation_total_reward[batch],
                evaluation_upright_steps[batch],
                evaluation_stable_upright_steps[batch],
                evaluation_max_consecutive_stable_steps[batch],
                evaluation_success[batch]
            );
        }

        int total_eval_steps = EVALUATION_BATCH_SIZE * MAX_STEPS;

        final_training_reward_per_step[seed_index] =
            training_average_reward_per_step[MAX_EPISODES - 1];

        average_evaluation_reward[seed_index] =
            total_evaluation_reward / (float)EVALUATION_BATCH_SIZE;

        average_upright_accuracy[seed_index] =
            (float)total_upright_steps / (float)total_eval_steps;

        average_stable_accuracy[seed_index] =
            (float)total_stable_steps / (float)total_eval_steps;

        average_max_consecutive_stable_steps[seed_index] =
            (float)total_max_consecutive / (float)EVALUATION_BATCH_SIZE;

        success_rate[seed_index] =
            (float)total_successes / (float)EVALUATION_BATCH_SIZE;

        f1_score[seed_index] =
            compute_f1_from_stable_steps(total_stable_steps, total_eval_steps);

        int csv_pass =
            write_training_csv()
            && write_validation_summary_csv()
            && write_validation_trajectory_csv()
            && write_evaluation_summary_csv()
            && write_evaluation_trajectory_csv();

        int seed_pass =
            completed_training_episodes == MAX_EPISODES
            && completed_validations == NUM_VALIDATIONS
            && completed_evaluations == EVALUATION_BATCH_SIZE
            && initial_state_check_pass
            && csv_pass;

        if (!seed_pass) {
            overall_pass = 0;
        }

        printf("----------------------------------------\n");
        printf(
            "Seed %u summary | AvgReward=%f | StableAcc=%f | SuccessRate=%f | F1=%f\n",
            current_seed_value,
            average_evaluation_reward[seed_index],
            average_stable_accuracy[seed_index],
            success_rate[seed_index],
            f1_score[seed_index]
        );
        printf(seed_pass ? "SEED PASS\n" : "SEED FAIL\n");
    }

    int summary_pass =
        write_multiseed_summary_csv(
            seeds,
            final_training_reward_per_step,
            average_evaluation_reward,
            average_upright_accuracy,
            average_stable_accuracy,
            average_max_consecutive_stable_steps,
            success_rate,
            f1_score
        )
        && write_multiseed_mean_std_csv(
            average_evaluation_reward,
            average_upright_accuracy,
            average_stable_accuracy,
            average_max_consecutive_stable_steps,
            success_rate,
            f1_score
        );

    printf("\n========================================\n");
    if (overall_pass && summary_pass) {
        printf("MULTI-SEED TESTBENCH PASS\n");
        return 0;
    }

    printf("MULTI-SEED TESTBENCH FAIL\n");
    return 1;
}