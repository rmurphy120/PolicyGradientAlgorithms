import game_manager
import models
import helper

import torch
import torch.nn as nn

from tqdm import tqdm
import matplotlib.pyplot as plt

import numpy as np
import random

# Training constants
GAMMA = .7     # Discount factor
LEARNING_RATE = 2 ** -13
EPSILON = .0000001  # Convergence bound
LENGTH_TO_CHECK_CONVERGENCE = 70

# Constants specifically for SAC
TARGET_ENTROPY = 0.98 * -np.log(1 / len(game_manager.ActionSpace))
ALPHA_INITIAL = 1
TARGET_UPDATE_LENGTH = 5


def find_losses(id: int, policy_nets: list[nn.Module], value_nets: list[nn.Module], target_value_nets: list[nn.Module],
                log_alphas: list[torch.tensor], device: str):
    """
    Calculate and return the loss tensors of policy_nets, value_nets, and log_alphas
    """

    # Get state and policies
    state = game_manager.SoccerState.ALL_STATES[id]
    state_values = torch.stack([vn(state.get_state_tensor(device)) for vn in value_nets])
    action_probs = [policy_nets[j](state.get_state_tensor(device)) for j in range(game_manager.SoccerState.NUM_AGENTS)]

    y = torch.zeros(game_manager.SoccerState.NUM_AGENTS, device=device)
    avg_policy_losses = torch.zeros(game_manager.SoccerState.NUM_AGENTS, device=device)

    # Don't sample, loop over all action pairs
    for actions in game_manager.ActionSpace.calculate_all_actions():
        next_states = state.transition(actions)

        # Find expected Qs
        Q = torch.tensor(state.reward, dtype=torch.float).repeat_interleave(2)
        for each in next_states:
            if any(each.is_off_board):
                val = torch.tensor(each.reward, dtype=torch.float).repeat_interleave(2)
            else:
                val = torch.stack([vn(each.get_state_tensor(device)) for vn in target_value_nets])

            Q = Q + next_states[each] * GAMMA * val

        # Define losses for policies and target for value
        probs = torch.stack([
            action_probs[a][actions[a].value]
            for a in range(game_manager.SoccerState.NUM_AGENTS)
        ])
        joint_prob = probs.prod()
        entropy = -torch.stack(log_alphas).exp() * probs.log()

        # Converts Q from a vector of size 2*n to a matrix of shape (n, 2) and computes the minimums of each
        # column to output a vector of size n
        t = Q.view(-1, 2)
        min_Q, _ = t.min(dim=1)

        y = y + (min_Q + entropy) * joint_prob.item()
        policy_losses = (min_Q - entropy) * joint_prob

        # Store gradients wrt policy parameters
        for a in range(game_manager.SoccerState.NUM_AGENTS):
            models.gradients_wrt_params(policy_nets[a], policy_losses[a])

        avg_policy_losses = avg_policy_losses + policy_losses.detach()

    avg_policy_losses = avg_policy_losses / len(game_manager.ActionSpace.calculate_all_actions())

    # Expands y to cover all Q nets
    y_expanded = y.repeat_interleave(2)

    value_losses = 0.5 * (y_expanded - state_values).pow(2)

    alphas_loss = [
        action_probs[a] @ (-log_alphas[a].exp() * action_probs[a].log() + TARGET_ENTROPY)
        for a in range(game_manager.SoccerState.NUM_AGENTS)
    ]

    return avg_policy_losses, value_losses, alphas_loss


def train(device: str):
    # Temperatures
    log_alphas = [torch.tensor(np.log(ALPHA_INITIAL), dtype=torch.float, requires_grad=True)] * game_manager.SoccerState.NUM_AGENTS
    alphas_optimiser = [torch.optim.Adam([log_alphas[a]], lr=LEARNING_RATE) for a in range(game_manager.SoccerState.NUM_AGENTS)]

    # Models
    policy_nets = [
        models.MarkovianPolicyNet(2 * game_manager.SoccerState.NUM_AGENTS + 1, len(game_manager.ActionSpace)).to(device)
        for _ in range(game_manager.SoccerState.NUM_AGENTS)
    ]
    value_nets = [
        models.ValueNet(2 * game_manager.SoccerState.NUM_AGENTS + 1, 1).to(device)
        for _ in range(2 * game_manager.SoccerState.NUM_AGENTS)
    ]
    target_value_nets = [
        models.ValueNet(2 * game_manager.SoccerState.NUM_AGENTS + 1, 1).to(device)
        for _ in range(2 * game_manager.SoccerState.NUM_AGENTS)
    ]

    # Loss histories to be returned
    policy_losses_over_time = [[] for _ in range(game_manager.SoccerState.NUM_AGENTS)]
    value_losses_over_time = [[] for _ in range(game_manager.SoccerState.NUM_AGENTS)]

    # Used to average losses when checking for convergence
    sum_policy_losses = [0] * game_manager.SoccerState.NUM_AGENTS
    sum_value_losses = [0] * game_manager.SoccerState.NUM_AGENTS

    # Training loop
    pbar = tqdm(desc="Training")
    shuffled_state_ids = list(range(len(game_manager.SoccerState.ALL_STATES)))
    has_converged = False
    try:
        while not has_converged:
            random.shuffle(shuffled_state_ids)

            for id in shuffled_state_ids:
                # Zeros out gradients
                for policy_net in policy_nets:
                    models.zero_gradients(policy_net)
                for value_net in value_nets:
                    models.zero_gradients(value_net)

                # Update target nets
                if pbar.n % TARGET_UPDATE_LENGTH == 0:
                    for i in range(len(value_nets)):
                        target_value_nets[i].load_state_dict(value_nets[i].state_dict())

                avg_policy_losses, value_losses, alphas_loss = find_losses(id, policy_nets, value_nets, target_value_nets,
                                                                           log_alphas, device)

                # Update policy, value, and temp parameters
                for a in range(game_manager.SoccerState.NUM_AGENTS):
                    models.update_params(policy_nets[a], LEARNING_RATE)

                    sum_policy_losses[a] += avg_policy_losses[a].item()

                    models.gradients_wrt_params(value_nets[2*a], value_losses[2*a])
                    models.update_params(value_nets[2*a], LEARNING_RATE)
                    models.gradients_wrt_params(value_nets[2*a+1], value_losses[2*a+1])
                    models.update_params(value_nets[2*a+1], LEARNING_RATE)

                    sum_value_losses[a] += value_losses[2*a].item() + value_losses[2*a+1].item()

                    alphas_loss[a].backward()
                    alphas_optimiser[a].step()

                # Convergence check (Checks if both agents' Q networks converged)
                if pbar.n % LENGTH_TO_CHECK_CONVERGENCE == 0:
                    has_converged = pbar.n > 2 * LENGTH_TO_CHECK_CONVERGENCE

                    for a in range(game_manager.SoccerState.NUM_AGENTS):
                        policy_losses_over_time[a].append(sum_policy_losses[a] / LENGTH_TO_CHECK_CONVERGENCE)
                        value_losses_over_time[a].append(sum_value_losses[a] / (2 * LENGTH_TO_CHECK_CONVERGENCE))

                        if (pbar.n > 2 * LENGTH_TO_CHECK_CONVERGENCE and
                                abs(value_losses_over_time[a][-1] - value_losses_over_time[a][-2]) >= EPSILON):
                            has_converged = False

                        sum_policy_losses[a] = 0
                        sum_value_losses[a] = 0

                    if has_converged:
                        break

                    pbar.set_postfix(value_losses=f"{value_losses_over_time[0][-1]:.4f}, {value_losses_over_time[1][-1]:.4f}",
                                     temps=f"{log_alphas[0].exp():.3f}, {log_alphas[1].exp():.3f}")

                pbar.update(1)
    except KeyboardInterrupt:
        pass

    pbar.close()

    return policy_nets, value_nets, policy_losses_over_time, value_losses_over_time


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    p_nets, v_nets, p_losses, v_losses = train(device=dev)

    """policy_filename = 'C:\\Users\\rynom\\OneDrive - UW-Madison\\Desktop\\Java Projects\\NashEquilibriumChecker\\nash_equilibrium.csv'
    value_filename = 'converged_values.csv'

    helper.save_policies_to_csv(policy_filename, p_nets, game_manager.SoccerState.ALL_STATES, dev)
    helper.save_values_to_csv(value_filename, v_nets, game_manager.SoccerState.ALL_STATES, dev)"""

    iterations = np.arange(1, len(p_losses[0]) + 1)
    fig, axs = plt.subplots(2, 1, figsize=(8, 5))

    # Plot policy losses over time
    for each in p_losses:
        axs[0].plot(iterations, each)
    axs[0].set_xlabel('Iteration')
    axs[0].set_ylabel('Loss')
    axs[0].tick_params(axis='y')
    axs[0].set_title('Policy Losses')
    axs[0].grid(True, linestyle='--', alpha=0.5)

    # Plot value loss over time
    for each in v_losses:
        axs[1].plot(iterations, each)
    axs[1].set_xlabel('Iteration')
    axs[1].set_ylabel('Loss')
    axs[1].tick_params(axis='y')
    axs[1].set_title('Value Losses')
    axs[1].grid(True, linestyle='--', alpha=0.5)

    fig.tight_layout()
    plt.show()
