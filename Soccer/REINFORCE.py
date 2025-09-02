import game_manager
import models

import torch
import torch.nn as nn

from tqdm import tqdm
import matplotlib.pyplot as plt

import numpy as np
import math


# Training constants
GAMMA = 0.5
LEARNING_RATE = 2 ** -13
NUM_TRAJECTORIES = 10000
MAX_EPISODE_LENGTH = 15

# Data collecting constants
AVERAGING_LENGTH_FOR_DATA_COLLECTION = 25
NUM_EX_TRAJECTORIES = 20


def generate_trajectory(state: game_manager.SoccerState, list_policy_net: list[nn.Module], device: str):
    state_at_timestep = []
    action_at_timestep = []
    reward_at_timestep = []
    log_probs_at_timestep = []

    for ep in range(MAX_EPISODE_LENGTH):
        log_probs = []
        actions = []
        # Get the policy and choose an action for each agent
        for policy_net in list_policy_net:
            action_probs = policy_net(state.get_state_tensor(device)).squeeze()
            log_probs.append(torch.log(action_probs))
            cpu_action_probs = action_probs.detach().cpu().numpy()
            actions.append(game_manager.ActionSpace(np.random.choice(
                np.arange(len(game_manager.ActionSpace)), p=cpu_action_probs)))

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


def train(device: str):
    policy_nets = [
        models.MarkovianPolicyNet(2 * game_manager.SoccerState.NUM_AGENTS + 1, len(game_manager.ActionSpace)).to(device)
        for _ in range(game_manager.SoccerState.NUM_AGENTS)
    ]

    lengths, returns = [], []
    avg_lengths, avg_returns = 0, 0

    last_log_probs = []
    last_states = []

    # Training loop
    try:
        for i in tqdm(range(NUM_TRAJECTORIES), desc="Training"):
            # Zeros out gradients
            for policy_net in policy_nets:
                models.zero_gradients(policy_net)

            # Generate the trajectory
            state_at_timestep, action_at_timestep, reward_at_timestep, log_probs_at_timestep = \
                (generate_trajectory(game_manager.SoccerState.get_random_state(), policy_nets, device=device))

            # Keep track of avg lengths of trajectories over a set period
            avg_lengths += len(log_probs_at_timestep)
            if i % AVERAGING_LENGTH_FOR_DATA_COLLECTION == 0:
                lengths.append(avg_lengths / AVERAGING_LENGTH_FOR_DATA_COLLECTION)
                avg_lengths = 0

            # Store log probs from the last 20 trajectories to output at the end
            last_log_probs.append(log_probs_at_timestep)
            last_states.append(state_at_timestep)
            if len(last_log_probs) > NUM_EX_TRAJECTORIES:
                last_log_probs.pop(0)
                last_states.pop(0)

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

                # Keep track of avg-expected reward of trajectories over a set period
                if t == 0:
                    avg_returns += Q.tolist()[0]
                    if i % AVERAGING_LENGTH_FOR_DATA_COLLECTION == 0:
                        returns.append(avg_returns / AVERAGING_LENGTH_FOR_DATA_COLLECTION)
                        avg_returns = 0

                # ascent on log π(a|s) for each model
                for j in range(len(policy_nets)):
                    loss = log_probs_at_timestep[t][j][action_at_timestep[t][j].value] * Q[j]
                    models.gradients_wrt_params(policy_nets[j], loss)

            # Update the models' parameters
            for policy_net in policy_nets:
                models.update_params(policy_net, LEARNING_RATE)
    except KeyboardInterrupt:
        pass

    # Prints out the policy for the last 20 trajectories
    for i in range(len(last_log_probs)):
        for t in range(len(last_log_probs[i])):
            print(last_states[i][t])
            for agent_probs in last_log_probs[i][t]:
                print([math.exp(lp) for lp in agent_probs])

        print('')

    return policy_nets, lengths, returns


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    nets, lens, rets = train(device=dev)

    trajectories = np.arange(1, len(lens) + 1)
    fig, axs = plt.subplots(2, 1, figsize=(8, 5))

    # Plot for the length of the trajectories
    axs[0].plot(trajectories, lens)
    axs[0].set_xlabel('Trajectory')
    axs[0].set_ylabel('Length')
    axs[0].tick_params(axis='y')
    axs[0].set_title('Trajectory Length')
    axs[0].grid(True, linestyle='--', alpha=0.5)

    # Plot for each agent's expected reward at the start of each trajectory
    axs[1].plot(trajectories, rets)
    axs[1].set_xlabel('Trajectory')
    axs[1].set_ylabel('Expected Reward')
    axs[1].tick_params(axis='y')
    axs[1].set_title('Expected Rewards')
    axs[1].grid(True, linestyle='--', alpha=0.5)

    fig.tight_layout()
    plt.show()
