import game_manager

import torch
import torch.nn as nn
from torch.autograd import grad

from tqdm import tqdm
import matplotlib.pyplot as plt
import csv

import numpy as np
import math


class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(2 * game_manager.CarState.NUM_AGENTS, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, len(game_manager.ActionSpace))

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.softmax(self.fc3(x), dim=1)
        return x


def save_policies_to_csv(filename: str, list_policy_net: list[nn.Module], device: str):
    header = [
        "X1", "Y1", "X2", "Y2",
        "U1", "D1", "L1", "R1",
        "U2", "D2", "L2", "R2"
    ]

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for each in game_manager.CarState.ALL_STATES:
            row = list(each.state)
            for policy_net in list_policy_net:
                row += policy_net(each.get_state_tensor(device)).squeeze().tolist()
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
        param.data += lr * param.grad


def compare_models(model1: nn.Module, model2: nn.Module, device: str):
    """
    Computes the average different between models of all outputs over all states
    """
    mean_difference = 0
    states = game_manager.CarState.ALL_STATES
    for state in states:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        action_probs1 = model1(state_tensor).squeeze()
        action_probs2 = model2(state_tensor).squeeze()
        mean_difference += (action_probs2 - action_probs1).abs().mean().item()

    return mean_difference / len(states)


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
    if 0 < y < game_manager.CarState.N - 1 and 0 < x < game_manager.CarState.N - 1:
        return game_manager.ActionSpace.DOWN

    if x == 0:
        if y == 0:
            return game_manager.ActionSpace.RIGHT
        return game_manager.ActionSpace.UP
    if y == game_manager.CarState.N - 1:
        return game_manager.ActionSpace.LEFT
    if x == game_manager.CarState.M - 1:
        return game_manager.ActionSpace.DOWN
    if y == 0:
        return game_manager.ActionSpace.RIGHT


def generate_trajectory(state: game_manager.CarState, list_policy_net: list[nn.Module], device: str,
                        max_episode_len=15):
    state_at_timestep = []
    action_at_timestep = []
    reward_at_timestep = []
    log_probs_at_timestep = []

    for ep in range(max_episode_len):
        log_probs = []
        actions = []
        # Get the policy and choose an action for each agent
        for policy_net in list_policy_net:
            action_probs = policy_net(state.get_state_tensor(device)).squeeze()
            log_probs.append(torch.log(action_probs))
            cpu_action_probs = action_probs.detach().cpu().numpy()
            actions.append(game_manager.ActionSpace(np.random.choice(
                np.arange(2 * game_manager.CarState.NUM_AGENTS), p=cpu_action_probs)))

        state_at_timestep.append(state.get_state_tensor(device))
        log_probs_at_timestep.append(log_probs)
        action_at_timestep.append(actions)
        reward_at_timestep.append(state.reward)

        # Take the action and get the new state and reward
        state = list(state.transition(actions).keys())[0]

        # Stops the trajectory if a car is in an invalid position
        if any(state.is_off_board):
            break

    # Adds the final state and reward to their lists
    state_at_timestep.append(state.get_state_tensor(device))
    reward_at_timestep.append(state.reward)

    return state_at_timestep, action_at_timestep, reward_at_timestep, log_probs_at_timestep


def train(device: str, num_trajectories: int):
    # Training constants
    gamma = 0.5
    base_lr = 2 ** -13

    policy_nets = [PolicyNet().to(device) for _ in range(2)]

    lengths, returns = [], []

    # Training loop
    for i in tqdm(range(num_trajectories), desc="Training"):
        # Zeros out gradients
        for policy_net in policy_nets:
            for p in policy_net.parameters():
                if p.grad is None:
                    p.grad = torch.zeros_like(p.data)
                else:
                    p.grad.detach_()
                    p.grad.zero_()

        # Generate the trajectory
        state_at_timestep, action_at_timestep, reward_at_timestep, log_probs_at_timestep = \
            (generate_trajectory(game_manager.CarState.get_random_state(), policy_nets, device=device))
        lengths.append(len(log_probs_at_timestep))

        # REINFORCE update per‑step
        for t in range(len(log_probs_at_timestep)):
            # Prints out the policy for the last 20 trajectories
            if i > num_trajectories - 20:
                print(state_at_timestep[t])
                for j in range(len(policy_nets)):
                    print([math.exp(lp) for lp in log_probs_at_timestep[t][j]])

            # Calculate Q estimate
            discounts = (gamma ** (torch.arange(t + 1, len(state_at_timestep), device=device) - t - 1))
            rewards_tensor = torch.tensor(
                reward_at_timestep[t + 1:],
                device=device,
                dtype=discounts.dtype
            )
            Q = rewards_tensor.transpose(0, 1) @ discounts
            if t == 0:
                returns.append(Q.tolist())

            # ascent on log π(a|s) for each model
            for j in range(len(policy_nets)):
                loss = log_probs_at_timestep[t][j][action_at_timestep[t][j].value] * Q[j]
                gradients_wrt_params(policy_nets[j], loss)

        # Update the models parameters
        for policy_net in policy_nets:
            update_params(policy_net, base_lr)

        # Formatting for the last 20 trajectories
        if len(lengths) > num_trajectories - 20:
            print('')

    return policy_nets, lengths, returns


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    num_trajectories = 5000
    nets, lens, rets = train(device=dev, num_trajectories=num_trajectories)

    filename = 'C:\\Users\\rynom\\OneDrive - UW-Madison\\Desktop\\Java Projects\\NashEquilibriumChecker\\nash_equilibrium.csv'

    save_policies_to_csv(filename, nets, dev)

    trajectories = np.arange(1, num_trajectories + 1)
    fig, axs = plt.subplots(2, 1, figsize=(8, 5))

    # Plot for the length of the trajectories
    axs[0].plot(trajectories, lens, color='tab:blue', label='Length')
    axs[0].set_xlabel('Trajectory')
    axs[0].set_ylabel('Length', color='tab:blue')
    axs[0].tick_params(axis='y', labelcolor='tab:blue')
    axs[0].set_title('Trajectory Length')
    axs[0].grid(True, linestyle='--', alpha=0.5)

    # Plot for each agent's expected reward at the start of each trajectory
    axs[1].plot(trajectories, [ret[0] for ret in rets], color='tab:blue', label='Pursuer')
    axs[1].plot(trajectories, [ret[1] for ret in rets], color='tab:red', label='Evader')
    axs[1].set_xlabel('Trajectory')
    axs[1].set_ylabel('Expected Reward', color='tab:blue')
    axs[1].tick_params(axis='y', labelcolor='tab:blue')
    axs[1].set_title('Expected Rewards')
    axs[1].grid(True, linestyle='--', alpha=0.5)

    fig.tight_layout()
    plt.show()
