"""
train.py
Actor-Critic DRL training loop for FieldDispatchEnv (DRL + GNN).
Evaluation: Regret Analysis + Bellman Consistency (as per report).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register

register(
    id="FieldDispatch-v0",
    entry_point="env:FieldDispatchEnv",
)

# ── Hyperparameters ────────────────────────────────────────────────────────
NUM_TECHS   = 5
NUM_JOBS    = 3
EPISODES    = 1000
GAMMA       = 0.99
LR_ACTOR    = 0.001
LR_CRITIC   = 0.001
EVAL_EVERY  = 100     # episodes between evaluations
EVAL_RUNS   = 20      # runs for regret / Bellman consistency averaging

env = gym.make("FieldDispatch-v0",
               num_techs=NUM_TECHS, num_jobs=NUM_JOBS)

INPUT_SIZE  = env.observation_space.shape[0]
OUTPUT_SIZE = env.action_space.n

print(f"Observation size : {INPUT_SIZE}")
print(f"Action size      : {OUTPUT_SIZE}")


# ═══════════════════════════════════════════════════════════════════════════
# Actor Network (Policy)
# ═══════════════════════════════════════════════════════════════════════════
class Actor(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(x), dim=-1)


# ═══════════════════════════════════════════════════════════════════════════
# Critic Network (Value Function)
# ═══════════════════════════════════════════════════════════════════════════
class Critic(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


actor  = Actor(INPUT_SIZE, OUTPUT_SIZE)
critic = Critic(INPUT_SIZE)

actor_optimizer  = optim.Adam(actor.parameters(),  lr=LR_ACTOR)
critic_optimizer = optim.Adam(critic.parameters(), lr=LR_CRITIC)


# ── Action selection ───────────────────────────────────────────────────────
def select_action(state: np.ndarray):
    state_t     = torch.tensor(state, dtype=torch.float32)
    action_probs = actor(state_t)
    dist         = torch.distributions.Categorical(action_probs)
    action       = dist.sample()
    return action.item(), dist.log_prob(action)


# ── Loss calculation ───────────────────────────────────────────────────────
def calculate_losses(log_prob, reward, state, next_state, done):
    s  = torch.tensor(state,      dtype=torch.float32)
    s_ = torch.tensor(next_state, dtype=torch.float32)
    r  = torch.tensor(reward,     dtype=torch.float32)
    d  = torch.tensor(done,       dtype=torch.float32)

    value = critic(s)

    with torch.no_grad():
        next_value = critic(s_)

    td_target = r + (1 - d) * GAMMA * next_value
    td_error  = td_target - value   # advantage

    actor_loss  = -log_prob * td_error.detach()
    critic_loss = td_error.pow(2)

    return actor_loss, critic_loss


# ═══════════════════════════════════════════════════════════════════════════
# Baseline policy — Random (for Regret Analysis)
# ═══════════════════════════════════════════════════════════════════════════
def run_baseline(n_runs: int = EVAL_RUNS) -> float:
    """Returns mean cumulative reward under a random policy."""
    rewards = []
    for _ in range(n_runs):
        obs, _ = env.reset()
        done = False
        total = 0.0
        while not done:
            action = env.action_space.sample()
            obs, r, term, trunc, _ = env.step(action)
            total += r
            done = term or trunc
        rewards.append(total)
    return float(np.mean(rewards))


# ═══════════════════════════════════════════════════════════════════════════
# Bellman Consistency Evaluation  (Eq 3.8 from report)
# δt = rt + γ·V(st+1) − V(st)
# ═══════════════════════════════════════════════════════════════════════════
def bellman_consistency(n_runs: int = EVAL_RUNS) -> float:
    """Returns mean absolute TD error (Bellman residual) over n_runs episodes."""
    td_errors = []
    actor.eval()
    critic.eval()
    with torch.no_grad():
        for _ in range(n_runs):
            state, _ = env.reset()
            done = False
            while not done:
                action, _ = select_action(state)
                next_state, reward, term, trunc, _ = env.step(action)
                done = term or trunc

                s  = torch.tensor(state,      dtype=torch.float32)
                s_ = torch.tensor(next_state, dtype=torch.float32)
                r  = torch.tensor(reward,     dtype=torch.float32)
                d  = torch.tensor(float(done), dtype=torch.float32)

                v      = critic(s)
                v_next = critic(s_)
                delta  = r + (1 - d) * GAMMA * v_next - v
                td_errors.append(abs(delta.item()))
                state = next_state
    actor.train()
    critic.train()
    return float(np.mean(td_errors))


# ═══════════════════════════════════════════════════════════════════════════
# Regret Analysis  (Eq 3.7 from report)
# R(π) = J(π_ref) − J(π)
# Negative = learned policy outperforms baseline
# ═══════════════════════════════════════════════════════════════════════════
def regret_analysis(n_runs: int = EVAL_RUNS) -> tuple[float, float, float]:
    """Returns (J_baseline, J_learned, regret)."""
    J_ref = run_baseline(n_runs)

    learned_rewards = []
    actor.eval()
    with torch.no_grad():
        for _ in range(n_runs):
            state, _ = env.reset()
            done = False
            total = 0.0
            while not done:
                action, _ = select_action(state)
                state, r, term, trunc, _ = env.step(action)
                total += r
                done = term or trunc
            learned_rewards.append(total)
    actor.train()

    J_learned = float(np.mean(learned_rewards))
    regret    = J_ref - J_learned
    return J_ref, J_learned, regret


# ═══════════════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════════════
episode_rewards = []
bellman_log     = []
regret_log      = []

print("\n── Training ────────────────────────────────────")

for episode in range(1, EPISODES + 1):
    state, _ = env.reset()
    done      = False
    ep_reward = 0.0

    while not done:
        action, log_prob = select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        actor_loss, critic_loss = calculate_losses(
            log_prob, reward, state, next_state, done
        )

        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        state      = next_state
        ep_reward += reward

    episode_rewards.append(ep_reward)

    if episode % 50 == 0:
        avg = np.mean(episode_rewards[-50:])
        print(f"Episode {episode:4d} | Avg Reward (last 50): {avg:8.2f}")

    # ── Periodic evaluation ────────────────────────────────────────────────
    if episode % EVAL_EVERY == 0:
        bc = bellman_consistency()
        J_ref, J_pi, regret = regret_analysis()
        bellman_log.append((episode, bc))
        regret_log.append((episode, J_ref, J_pi, regret))
        print(f"\n  📊 Evaluation @ Episode {episode}")
        print(f"     Bellman Residual (mean |δ|) : {bc:.4f}")
        print(f"     Baseline reward (random)    : {J_ref:.2f}")
        print(f"     Learned policy reward       : {J_pi:.2f}")
        print(f"     Regret R(π) = J_ref − J(π)  : {regret:.2f}"
              f"  {'✅ outperforms baseline' if regret < 0 else '⚠️  below baseline'}\n")

# ═══════════════════════════════════════════════════════════════════════════
# Save models
# ═══════════════════════════════════════════════════════════════════════════
torch.save(actor.state_dict(),  "actor.pth")
torch.save(critic.state_dict(), "critic.pth")
print("\n✅ Models saved: actor.pth, critic.pth")

# Final evaluation summary
bc_final            = bellman_consistency(50)
J_ref, J_pi, regret = regret_analysis(50)

print("\n── Final Evaluation Summary ────────────────────")
print(f"  Bellman Residual (mean |δ|) : {bc_final:.4f}")
print(f"  Baseline (random) reward    : {J_ref:.2f}")
print(f"  Learned policy reward       : {J_pi:.2f}")
print(f"  Final Regret                : {regret:.2f}")
print(f"  Result: {'✅ Outperforms baseline' if regret < 0 else '⚠️ Does not yet outperform baseline'}")
