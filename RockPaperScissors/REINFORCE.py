import game_manager
import models

import torch
import torch.nn as nn
from torch.autograd import grad

from tqdm import tqdm
import matplotlib.pyplot as plt

import numpy as np


# Training constants
GAMMA = 0.5
LEARNING_RATE = 2 ** -13
NUM_TRAJECTORIES = 3 * 20000
MAX_EPISODE_LENGTH = 2


def generate_trajectory(state: game_manager.RPSState, list_policy_net: list[nn.Module], device: str):
    state_at_timestep = []
    action_at_timestep = []
    reward_at_timestep = []
    log_probs_at_timestep = []

    for ep in range(MAX_EPISODE_LENGTH):
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
    policy_nets = [
        models.MarkovianPolicyNet(1, len(game_manager.ActionSpace)).to(device)
        for _ in range(game_manager.RPSState.NUM_AGENTS)
    ]

    # Expected returns
    returns = []
    # Policy history to be returned
    policies_over_time = [[[], [], []], [[], [], []]]

    # Training loop
    try:
        for _ in tqdm(range(NUM_TRAJECTORIES), desc="Training"):
            # Zeros out gradients
            for policy_net in policy_nets:
                models.zero_gradients(policy_net)

            # Generate the trajectory
            state_at_timestep, action_at_timestep, reward_at_timestep, log_probs_at_timestep = \
                (generate_trajectory(game_manager.RPSState.get_random_state(), policy_nets, device=device))

            # REINFORCE update per‑step
            for t in range(len(log_probs_at_timestep)):
                # Calculate Q estimate
                discounts = (GAMMA ** (torch.arange(t + 1, len(state_at_timestep), device=device) - t - 1))
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
                    models.gradients_wrt_params(policy_nets[j], loss)

            # Update the models' parameters
            for policy_net in policy_nets:
                models.update_params(policy_net, LEARNING_RATE)

            if state_at_timestep[0] == 0:
                for j in range(game_manager.RPSState.NUM_AGENTS):
                    for action in range(len(game_manager.ActionSpace)):
                        policies_over_time[j][action].append(policy_nets[j](state_at_timestep[0])[action].item())
    except KeyboardInterrupt:
        pass

    return policy_nets, returns, policies_over_time


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    nets, rets, policies_over_time = train(device=dev, num_trajectories=NUM_TRAJECTORIES)

    iterations = np.arange(1, len(policies_over_time[0][0]) + 1)
    fig, axs = plt.subplots(2, 1, figsize=(8, 5))

    # Plot each agent's policy over time
    for a in range(game_manager.RPSState.NUM_AGENTS):
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
