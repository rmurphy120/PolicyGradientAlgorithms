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
GAMMA = 0.7
LEARNING_RATE = 2 ** -12
EPSILON = .0001
LENGTH_TO_CHECK_CONVERGENCE = 70


def next_state_circle(position: list[int]) -> game_manager.ActionSpace:
    """
    Given a position `state = (row, col)` on a board of size height×width,
    returns the adjacent state according to:
      - If state is on the perimeter, move to the next perimeter cell
        in clockwise order.
      - Otherwise, move one step down (row + 1).
    """
    x, y = position

    # If not on the edge: just go down
    if 0 < y < game_manager.CarState.HEIGHT - 1 and 0 < x < game_manager.CarState.HEIGHT - 1:
        return game_manager.ActionSpace.DOWN

    if x == 0:
        if y == 0:
            return game_manager.ActionSpace.RIGHT
        return game_manager.ActionSpace.UP
    if y == game_manager.CarState.HEIGHT - 1:
        return game_manager.ActionSpace.LEFT
    if x == game_manager.CarState.WIDTH - 1:
        return game_manager.ActionSpace.DOWN
    if y == 0:
        return game_manager.ActionSpace.RIGHT


def find_losses(id: int, policy_nets: list[nn.Module], value_net: nn.Module, device: str):
    # Get state and policies
    state = game_manager.CarState.ALL_STATES[id]
    state_value = value_net(state.get_state_tensor(device))
    action_probs = [policy_nets[j](state.get_state_tensor(device)) for j in range(
        game_manager.CarState.NUM_AGENTS)]
    y = 0
    avg_policy_losses = torch.zeros(game_manager.CarState.NUM_AGENTS, device=device)

    # Don't sample, loop over all action pairs
    for actions in game_manager.ActionSpace.calculate_all_actions():
        next_states = state.transition(actions)

        # Find expected Qs
        Q = state.reward[0]
        for each in next_states:
            if any(each.is_off_board):
                Q = Q + next_states[each] * GAMMA * each.reward[0]
            else:
                Q = Q + next_states[each] * GAMMA * value_net(each.get_state_tensor(device)).item()

        # Define losses for policies and target for value
        probs = torch.stack([
            action_probs[a][actions[a].value]
            for a in range(game_manager.CarState.NUM_AGENTS)
        ])
        joint_prob = probs.prod()

        advantage = (Q - state_value) * joint_prob.item()
        advantages = torch.stack([-advantage, advantage]).squeeze()
        policy_losses = probs.log() * advantages

        y += Q * joint_prob.item()

        # Store gradients
        for j in range(game_manager.CarState.NUM_AGENTS):
            models.gradients_wrt_params(policy_nets[j], policy_losses[j])

        avg_policy_losses += policy_losses.detach()

    avg_policy_losses = avg_policy_losses / len(game_manager.ActionSpace.calculate_all_actions())

    # Define value loss
    value_loss = 0.5 * (y - state_value).pow(2)

    return avg_policy_losses, value_loss


def train(device: str):
    # Models
    policy_nets = [
        models.MarkovianPolicyNet(2 * game_manager.CarState.NUM_AGENTS, len(game_manager.ActionSpace)).to(device)
        for _ in range(game_manager.CarState.NUM_AGENTS)
    ]
    value_net = models.ValueNet(2 * game_manager.CarState.NUM_AGENTS, 1).to(device)

    # Loss history to be returned
    policy_losses_over_time = [[], []]
    value_loss_over_time = []

    # Used to average losses when checking for convergence
    sum_policy_losses = [0] * game_manager.CarState.NUM_AGENTS
    sum_value_losses = 0

    # Training loop
    pbar = tqdm(desc="Training")
    shuffled_state_ids = list(range(len(game_manager.CarState.ALL_STATES)))
    has_converged = False
    try:
        while not has_converged:
            random.shuffle(shuffled_state_ids)

            for id in shuffled_state_ids:
                # Zeros out gradients
                for policy_net in policy_nets:
                    models.zero_gradients(policy_net)

                models.zero_gradients(value_net)

                avg_policy_losses, value_loss = find_losses(id, policy_nets, value_net, device)

                # Update policy and value parameters
                for a in range(game_manager.CarState.NUM_AGENTS):
                    models.update_params(policy_nets[a], LEARNING_RATE)

                models.gradients_wrt_params(value_net, value_loss)
                models.update_params(value_net, LEARNING_RATE)

                for a in range(game_manager.CarState.NUM_AGENTS):
                    sum_policy_losses[a] += avg_policy_losses[a].item()
                sum_value_losses += value_loss.item()

                # Convergence check (Checks if both agents' Q networks converged)
                if pbar.n % LENGTH_TO_CHECK_CONVERGENCE == 0:
                    has_converged = pbar.n > 2 * LENGTH_TO_CHECK_CONVERGENCE

                    for a in range(game_manager.CarState.NUM_AGENTS):
                        policy_losses_over_time[a].append(sum_policy_losses[a] / LENGTH_TO_CHECK_CONVERGENCE)
                        sum_policy_losses[a] = 0

                    value_loss_over_time.append(sum_value_losses / LENGTH_TO_CHECK_CONVERGENCE)

                    if (pbar.n > 2 * LENGTH_TO_CHECK_CONVERGENCE and
                            abs(value_loss_over_time[-1] - value_loss_over_time[-2]) >= EPSILON):
                        has_converged = False

                    sum_value_losses = 0

                    if has_converged:
                        break

                    pbar.set_postfix(value_loss=f"{value_loss_over_time[-1]:.4f}")

                pbar.update(1)
    except KeyboardInterrupt:
        pass

    pbar.close()

    return policy_nets, value_net, policy_losses_over_time, value_loss_over_time


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    p_nets, v_net, p_losses, v_losses = train(device=dev)

    """policy_filename = 'C:\\Users\\rynom\\OneDrive - UW-Madison\\Desktop\\Java Projects\\NashEquilibriumChecker\\nash_equilibrium.csv'
    value_filename = 'converged_values.csv'

    helper.save_policies_to_csv(policy_filename, p_nets, game_manager.CarState.ALL_STATES, dev)
    helper.save_values_to_csv(value_filename, [v_net], game_manager.CarState.ALL_STATES, dev)"""

    iterations = np.arange(1, len(v_losses) + 1)
    fig, axs = plt.subplots(2, 1, figsize=(8, 5))

    # Plot policy losses over time
    axs[0].plot(iterations, p_losses[0], color='tab:red')
    axs[0].plot(iterations, p_losses[1], color='tab:blue')
    axs[0].set_xlabel('Iteration')
    axs[0].set_ylabel('Loss', color='tab:blue')
    axs[0].tick_params(axis='y', labelcolor='tab:blue')
    axs[0].set_title('Policy Losses')
    axs[0].grid(True, linestyle='--', alpha=0.5)

    # Plot value loss over time
    axs[1].plot(iterations, v_losses, color='tab:blue')
    axs[1].set_xlabel('Iteration')
    axs[1].set_ylabel('Loss', color='tab:blue')
    axs[1].tick_params(axis='y', labelcolor='tab:blue')
    axs[1].set_title('Value Losses')
    axs[1].grid(True, linestyle='--', alpha=0.5)

    fig.tight_layout()
    plt.show()
