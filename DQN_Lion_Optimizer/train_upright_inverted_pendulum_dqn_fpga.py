import random
import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from upright_inverted_pendulum_dqn_fpga import UprightInvertedPendulumDQNFPGAEnv


# -----------------------------
# Lion optimizer
# -----------------------------
class Lion(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                # Optional decoupled weight decay
                if weight_decay != 0.0:
                    p.data.mul_(1.0 - lr * weight_decay)

                state = self.state[p]

                # Momentum buffer
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)

                m = state["exp_avg"]

                # Lion update:
                # update = beta1 * m + (1 - beta1) * grad
                update = m * beta1 + grad * (1.0 - beta1)

                # w = w - lr * sign(update)
                p.add_(torch.sign(update), alpha=-lr)

                # m = beta2 * m + (1 - beta2) * grad
                m.mul_(beta2).add_(grad, alpha=1.0 - beta2)

        return loss


# -----------------------------
# Reproducibility
# -----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# -----------------------------
# Environment
# -----------------------------
env = Monitor(UprightInvertedPendulumDQNFPGAEnv())


# -----------------------------
# DQN with Lion optimizer
# -----------------------------
model = DQN(
    "MlpPolicy",
    env,
    verbose=1,

    # Use smaller LR for Lion than Adam
    learning_rate=1e-4,

    buffer_size=10000,
    learning_starts=1000,
    batch_size=64,
    gamma=0.99,
    train_freq=4,
    gradient_steps=1,
    target_update_interval=1000,

    exploration_fraction=0.30,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,

    policy_kwargs=dict(
        net_arch=[32],

        # Replaces Adam with Lion
        optimizer_class=Lion,
        optimizer_kwargs=dict(
            betas=(0.9, 0.99),
            weight_decay=0.0,
        ),
    ),

    seed=SEED,
)


# -----------------------------
# Train and save
# -----------------------------
model.learn(total_timesteps=800000)

model.save("dqn_upright_inverted_pendulum_fpga_lion")

env.close()

print("Training complete.")
print("Saved model as: dqn_upright_inverted_pendulum_fpga_lion")