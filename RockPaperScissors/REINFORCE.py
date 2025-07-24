import game_manager

import torch
import torch.nn as nn
from torch.autograd import grad

from tqdm import tqdm
import matplotlib.pyplot as plt

import numpy as np


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
        return x.squeeze(0)


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


def generate_trajectory(state: game_manager.RPSState, list_policy_net: list[nn.Module], device: str,
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
            action_probs = policy_net(state.get_state_tensor(device))
            log_probs.append(torch.log(action_probs))
            cpu_action_probs = action_probs.detach().cpu().numpy()
            actions.append(game_manager.ActionSpace(np.random.choice(
                np.arange(3), p=cpu_action_probs)))

        state_at_timestep.append(state.get_state_tensor(device))
        log_probs_at_timestep.append(log_probs)
        action_at_timestep.append(actions)
        reward_at_timestep.append(state.reward)

        # Take the action and get the new state and reward
        state = list(state.transition(actions).keys())[0]

    # Adds the final state and reward to their lists
    state_at_timestep.append(state.get_state_tensor(device))
    reward_at_timestep.append(state.reward)

    return state_at_timestep, action_at_timestep, reward_at_timestep, log_probs_at_timestep


def train(device: str, num_trajectories: int):
    # Training constants
    gamma = 0.5
    base_lr = 2 ** -13

    policy_nets = [PolicyNet().to(device) for _ in range(2)]

    returns = []
    policies_over_time = [[[], [], []], [[], [], []]]

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
            (generate_trajectory(game_manager.RPSState.get_random_state(), policy_nets, device=device))

        # REINFORCE update per‑step
        for t in range(len(log_probs_at_timestep)):
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

        if state_at_timestep[0] == 0:
            for j in range(game_manager.RPSState.NUM_AGENTS):
                for action in range(len(game_manager.ActionSpace)):
                    policies_over_time[j][action].append(policy_nets[j](state_at_timestep[0])[action].item())

    return policy_nets, returns, policies_over_time


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    num_trajectories = 50
    nets, rets, policies_over_time = train(device=dev, num_trajectories=num_trajectories)

    iterations = np.arange(1, len(policies_over_time[0][0]) + 1)
    fig, axs = plt.subplots(2, 1, figsize=(8, 5))

    for a in range(game_manager.RPSState.NUM_AGENTS):
        # Plot for the length of the trajectories
        for action in game_manager.ActionSpace:
            axs[a].plot(iterations, policies_over_time[a][action.value], label=action.name)
        axs[a].set_xlabel('Iteration')
        axs[a].set_ylabel('Action probability')
        axs[a].tick_params(axis='y')
        axs[a].set_title(f'Agent {a} action probability')
        axs[a].legend()
        axs[a].grid(True, linestyle='--', alpha=0.5)

    fig.tight_layout()
    plt.show()
