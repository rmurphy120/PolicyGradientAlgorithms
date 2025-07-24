import game_manager

import torch
import torch.nn as nn
from torch.autograd import grad

from tqdm import tqdm
import matplotlib.pyplot as plt
import csv

import numpy as np
import random


class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(2 * game_manager.SoccerState.NUM_AGENTS + 1, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, len(game_manager.ActionSpace))

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.softmax(self.fc3(x), dim=1)
        return x.squeeze(0)


class ValueNet(nn.Module):
    def __init__(self):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(2 * game_manager.SoccerState.NUM_AGENTS + 1, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze(-1)


def save_policies_to_csv(filename: str, list_policy_net: list[nn.Module], device: str):
    header = [
        "Ball", "X1", "Y1", "X2", "Y2",
        "U1", "D1", "L1", "R1", "U2", "D2", "L2", "R2"
    ]

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for each in game_manager.SoccerState.ALL_STATES:
            row = list(each.state)
            state_tensor = each.get_state_tensor(device)
            for policy_net in list_policy_net:
                row += policy_net(state_tensor).squeeze().tolist()
            writer.writerow(row)

    print(f'Successfully printed to {filename}')


def save_values_to_csv(filename: str, value_net: nn.Module, device: str):
    """
    Save the values of each state computed by the value net to a CSV file
    """
    header = ["X1", "Y1", "X2", "Y2", "V"]

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for each in game_manager.SoccerState.ALL_STATES:
            row = list(each.state)
            state_tensor = each.get_state_tensor(device)
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


def train(device: str):
    # Training constants
    gamma = 0.9
    base_lr = 2 ** -12
    epsilon = .0001

    # Models
    policy_nets = [PolicyNet().to(device) for _ in range(game_manager.SoccerState.NUM_AGENTS)]
    value_net = ValueNet().to(device)

    # Loss histories to be returned
    policies_losses = [[], []]
    value_losses = []

    # Training loop
    pbar = tqdm(desc="Training")
    shuffled_state_ids = list(range(len(game_manager.SoccerState.ALL_STATES)))
    has_converged = False
    while not has_converged:
        has_converged = True
        avg_value_loss = 0
        avg_policy_loss = [0] * game_manager.SoccerState.NUM_AGENTS

        random.shuffle(shuffled_state_ids)

        for id in shuffled_state_ids:
            # Zeros out gradients
            for policy_net in policy_nets:
                for p in policy_net.parameters():
                    if p.grad is None:
                        p.grad = torch.zeros_like(p.data)
                    else:
                        p.grad.detach_()
                        p.grad.zero_()

            for p in value_net.parameters():
                if p.grad is None:
                    p.grad = torch.zeros_like(p.data)
                else:
                    p.grad.detach_()
                    p.grad.zero_()

            # Get state and policies
            state = game_manager.SoccerState.ALL_STATES[id]
            state_value = value_net(state.get_state_tensor(device)).squeeze()
            action_probs = [policy_nets[j](state.get_state_tensor(device)).squeeze() for j in range(
                game_manager.SoccerState.NUM_AGENTS)]
            y = 0

            # Don't sample, loop over all action pairs
            for actions in game_manager.ActionSpace.calculate_all_actions():
                next_states = state.transition(actions)

                # Find expected Qs
                Q = state.reward[0]
                for each in next_states:
                    if any(each.is_off_board):
                        Q = Q + next_states[each] * gamma * each.reward[0]
                    else:
                        Q = Q + next_states[each] * gamma * value_net(each.get_state_tensor(device)).squeeze().item()

                # Define losses for policies and target for value
                y_for_actions = Q
                advantage = Q - state_value
                policies_loss = []

                for j in range(game_manager.SoccerState.NUM_AGENTS):
                    prob = action_probs[j][actions[j].value]
                    advantage *= prob.item()
                    y_for_actions = y_for_actions * prob.item()
                    policies_loss.append(torch.log(prob))

                y += y_for_actions

                policies_loss[0] = -advantage * policies_loss[0]
                policies_loss[1] = advantage * policies_loss[1]

                # Store gradients
                for j in range(game_manager.SoccerState.NUM_AGENTS):
                    gradients_wrt_params(policy_nets[j], policies_loss[j])
                    avg_policy_loss[j] += policies_loss[j].item()

            # Define value loss
            value_loss = 0.5 * (y - state_value).pow(2)

            # Update policy and value parameters
            for j in range(game_manager.SoccerState.NUM_AGENTS):
                update_params(policy_nets[j], base_lr)

            gradients_wrt_params(value_net, value_loss)
            update_params(value_net, base_lr)

            curr_loss = value_loss.item()

            avg_value_loss += curr_loss

        avg_value_loss /= len(game_manager.SoccerState.ALL_STATES)

        # Convergence check
        if len(value_losses) == 0 or abs(value_losses[-1] - avg_value_loss) >= epsilon:
            has_converged = False

        for j in range(game_manager.SoccerState.NUM_AGENTS):
            policies_losses[j].append(avg_policy_loss[j] / len(game_manager.ActionSpace.calculate_all_actions())
                                      / len(game_manager.SoccerState.ALL_STATES))
        value_losses.append(avg_value_loss)

        pbar.update(1)
        pbar.set_postfix(value_loss=f"{avg_value_loss:.4f}")
    pbar.close()

    return policy_nets, value_net, policies_losses, value_losses


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    policy_nets, value_net, policies_losses, value_losses = train(device=dev)

    policy_filename = 'C:\\Users\\rynom\\OneDrive - UW-Madison\\Desktop\\Java Projects\\NashEquilibriumChecker\\nash_equilibrium.csv'
    value_filename = 'converged_values.csv'

    save_policies_to_csv(policy_filename, policy_nets, dev)
    save_values_to_csv(value_filename, value_net, dev)

    iterations = np.arange(1, len(value_losses) + 1)
    fig, axs = plt.subplots(2, 1, figsize=(8, 5))

    # Plot for the length of the trajectories
    axs[0].plot(iterations, policies_losses[0], color='tab:red')
    axs[0].plot(iterations, policies_losses[1], color='tab:blue')
    axs[0].set_xlabel('Iteration')
    axs[0].set_ylabel('Loss', color='tab:blue')
    axs[0].tick_params(axis='y', labelcolor='tab:blue')
    axs[0].set_title('Pursuer Policy Loss')
    axs[0].grid(True, linestyle='--', alpha=0.5)

    # Plot for each agent's expected reward at the start of each trajectory
    axs[1].plot(iterations, value_losses, color='tab:blue')
    axs[1].set_xlabel('Iteration')
    axs[1].set_ylabel('Loss', color='tab:blue')
    axs[1].tick_params(axis='y', labelcolor='tab:blue')
    axs[1].set_title('Value Losses')
    axs[1].grid(True, linestyle='--', alpha=0.5)

    fig.tight_layout()
    plt.show()
