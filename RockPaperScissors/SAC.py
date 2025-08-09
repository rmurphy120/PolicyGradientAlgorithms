import game_manager

import torch
import torch.nn as nn
from torch.autograd import grad

from tqdm import tqdm
import matplotlib.pyplot as plt
import csv

import numpy as np
import random

# Training constants
GAMMA = .99     # Discount factor
LEARNING_RATE = 2 ** -12
EPSILON = .000001  # Convergence bound
LENGTH_TO_CHECK_CONVERGENCE = 50

# Constants specifically for SAC
TARGET_ENTROPY = 0.98 * -np.log(1 / len(game_manager.ActionSpace))
ALPHA_INITIAL = 1
TARGET_UPDATE_LENGTH = 3


class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(1, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, len(game_manager.ActionSpace))

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.softmax(self.fc3(x), dim=1)
        return x.squeeze()


class ValueNet(nn.Module):
    def __init__(self):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(1, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze()


def save_policies_to_csv(filename: str, list_policy_net: list[nn.Module], device: str):
    header = [
        "X1", "Y1", "X2", "Y2",
        "U1", "D1", "L1", "R1", "U2", "D2", "L2", "R2"
    ]

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for each in game_manager.RPSState.ALL_STATES:
            row = [each.state]
            state_tensor = each.get_state_tensor(device)
            for policy_net in list_policy_net:
                row += policy_net(state_tensor).squeeze().tolist()
            writer.writerow(row)

    print(f'Successfully printed to {filename}')


def save_values_to_csv(filename: str, value_nets: list[nn.Module], device: str):
    """
    Save the values of each state computed by the value net to a CSV file
    """
    header = ["X1", "Y1", "X2", "Y2", "V1", "V2"]

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for each in game_manager.RPSState.ALL_STATES:
            row = [each.state]
            state_tensor = each.get_state_tensor(device)
            for value_net in value_nets:
                row.append(value_net(state_tensor).squeeze().item())
            writer.writerow(row)

    print(f'Successfully printed to {filename}')


def gradients_wrt_params(net: nn.Module, loss_tensor: torch.Tensor):
    """
    Dictionary to store gradients for each parameter
    Compute gradients with respect to each parameter
    """

    for name, param in net.named_parameters():
        g = grad(loss_tensor, param, retain_graph=True)[0]
        param.grad = param.grad + g


def update_params(net: nn.Module, lr: float):
    """
    Update parameters for the network
    """

    for name, param in net.named_parameters():
        param.data -= lr * param.grad


def zero_gradients(net: nn.Module):
    """
    Zero out stored gradients in the network
    """
    for p in net.parameters():
        if p.grad is None:
            p.grad = torch.zeros_like(p.data)
        else:
            p.grad.detach_()
            p.grad.zero_()


def find_losses(id: int, policy_nets: list[nn.Module], value_nets: list[nn.Module], target_value_nets: list[nn.Module],
                log_alphas: list[torch.tensor], device: str):
    """
    Calculate and return the loss tensors of policy_nets, value_nets, and log_alphas
    """

    # Get state and policies
    state = game_manager.RPSState.ALL_STATES[id]
    state_values = torch.stack([vn(state.get_state_tensor(device)) for vn in value_nets])
    action_probs = [policy_nets[j](state.get_state_tensor(device)) for j in range(game_manager.RPSState.NUM_AGENTS)]

    y = torch.zeros(game_manager.RPSState.NUM_AGENTS, device=device)
    avg_policy_losses = torch.zeros(game_manager.RPSState.NUM_AGENTS, device=device)

    # Don't sample, loop over all action pairs
    for actions in game_manager.ActionSpace.calculate_all_actions():
        next_states = state.transition(actions)

        # Find expected Qs
        Q = torch.tensor(state.reward, dtype=torch.float).repeat_interleave(2)
        for each in next_states:
            val = torch.stack([vn(each.get_state_tensor(device)) for vn in target_value_nets])
            Q = Q + next_states[each] * GAMMA * val

        # Define losses for policies and target for value
        probs = torch.stack([
            action_probs[a][actions[a].value]
            for a in range(game_manager.RPSState.NUM_AGENTS)
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
        for a in range(game_manager.RPSState.NUM_AGENTS):
            gradients_wrt_params(policy_nets[a], policy_losses[a])

        avg_policy_losses = avg_policy_losses + policy_losses.detach()

    avg_policy_losses = avg_policy_losses / len(game_manager.ActionSpace.calculate_all_actions())

    # Expands y to cover all Q nets
    y_expanded = y.repeat_interleave(2)

    value_losses = 0.5 * (y_expanded - state_values).pow(2)

    alphas_loss = [
        action_probs[a] @ (-log_alphas[a].exp() * action_probs[a].log() + TARGET_ENTROPY)
        for a in range(game_manager.RPSState.NUM_AGENTS)
    ]

    return avg_policy_losses, value_losses, alphas_loss


def train(device: str):
    # Temperatures
    log_alphas = [torch.tensor(np.log(ALPHA_INITIAL), dtype=torch.float, requires_grad=True)] * game_manager.RPSState.NUM_AGENTS
    alphas_optimiser = [torch.optim.Adam([log_alphas[a]], lr=LEARNING_RATE) for a in range(game_manager.RPSState.NUM_AGENTS)]

    # Models
    policy_nets = [PolicyNet().to(device) for _ in range(game_manager.RPSState.NUM_AGENTS)]
    value_nets = [ValueNet().to(device) for _ in range(2 * game_manager.RPSState.NUM_AGENTS)]
    target_value_nets = [ValueNet().to(device) for _ in range(2 * game_manager.RPSState.NUM_AGENTS)]

    # Policy histories to be returned
    policies_over_time = [[[], [], []], [[], [], []]]

    prev_value_losses = [-1] * game_manager.RPSState.NUM_AGENTS

    # Used to average losses
    sum_value_losses = [0] * game_manager.RPSState.NUM_AGENTS

    # Training loop
    pbar = tqdm(desc="Training")
    shuffled_state_ids = list(range(len(game_manager.RPSState.ALL_STATES)))
    has_converged = False
    while not has_converged:
        random.shuffle(shuffled_state_ids)

        for id in shuffled_state_ids:
            # Zeros out gradients
            for policy_net in policy_nets:
                zero_gradients(policy_net)
            for value_net in value_nets:
                zero_gradients(value_net)

            # Update target nets
            if pbar.n % TARGET_UPDATE_LENGTH == 0:
                for i in range(len(value_nets)):
                    target_value_nets[i].load_state_dict(value_nets[i].state_dict())

            avg_policy_losses, value_losses, alphas_loss = find_losses(id, policy_nets, value_nets, target_value_nets,
                                                                       log_alphas, device)

            # Update policy, value, and temp parameters
            for a in range(game_manager.RPSState.NUM_AGENTS):
                update_params(policy_nets[a], LEARNING_RATE)

                gradients_wrt_params(value_nets[2*a], value_losses[2*a])
                update_params(value_nets[2*a], LEARNING_RATE)
                gradients_wrt_params(value_nets[2*a+1], value_losses[2*a+1])
                update_params(value_nets[2*a+1], LEARNING_RATE)

                sum_value_losses[a] += value_losses[2*a].item() + value_losses[2*a+1].item()

                alphas_loss[a].backward()
                alphas_optimiser[a].step()

            if id == 0:
                state = game_manager.RPSState.ALL_STATES[id]
                action_probs = [policy_nets[j](state.get_state_tensor(device)).squeeze() for j in range(
                    game_manager.RPSState.NUM_AGENTS)]
                for a in range(game_manager.RPSState.NUM_AGENTS):
                    for action in range(len(game_manager.ActionSpace)):
                        policies_over_time[a][action].append(action_probs[a][action].item())

            # Convergence check (Checks if both agents' Q networks converged)
            if pbar.n % LENGTH_TO_CHECK_CONVERGENCE == 0:
                has_converged = pbar.n > 2 * LENGTH_TO_CHECK_CONVERGENCE

                for a in range(game_manager.RPSState.NUM_AGENTS):
                    avg_value_loss = sum_value_losses[a] / (2 * LENGTH_TO_CHECK_CONVERGENCE)

                    if (pbar.n > 2 * LENGTH_TO_CHECK_CONVERGENCE and
                            abs(prev_value_losses[a] - avg_value_loss) >= EPSILON):
                        has_converged = False

                    prev_value_losses[a] = avg_value_loss
                    sum_value_losses[a] = 0

                if has_converged:
                    has_converged = True
                    break

                pbar.set_postfix(values_loss=f"{prev_value_losses[0]:.4f}, {prev_value_losses[1]:.4f}",
                                 temps=f"{log_alphas[0].exp():.3f}, {log_alphas[1].exp():.3f}")

            pbar.update(1)

    pbar.close()

    return policy_nets, value_nets, policies_over_time


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    p_nets, v_nets, p_over_time = train(device=dev)

    policy_filename = 'C:\\Users\\rynom\\OneDrive - UW-Madison\\Desktop\\Java Projects\\NashEquilibriumChecker\\nash_equilibrium.csv'
    value_filename = 'converged_values.csv'

    save_policies_to_csv(policy_filename, p_nets, dev)
    save_values_to_csv(value_filename, v_nets, dev)

    iterations = np.arange(1, len(p_over_time[0][0]) + 1)
    fig, axs = plt.subplots(2, 1, figsize=(8, 5))

    # Plot each agent's policy over time
    for a in range(game_manager.RPSState.NUM_AGENTS):
        for action in game_manager.ActionSpace:
            axs[a].plot(iterations, p_over_time[a][action.value], label=action.name)
        axs[a].set_xlabel('Iteration')
        axs[a].set_ylabel('Action probability')
        axs[a].tick_params(axis='y')
        axs[a].set_title(f'Agent {a} action probability')
        axs[a].legend()
        axs[a].grid(True, linestyle='--', alpha=0.5)

    fig.tight_layout()
    plt.show()
