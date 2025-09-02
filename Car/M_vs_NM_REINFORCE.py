import game_manager
import models
import helper

import torch
import torch.nn as nn

from tqdm import tqdm
import matplotlib.pyplot as plt

import numpy as np
import math
from typing import Dict

# Training constants
GAMMA = 0.5
LEARNING_RATE = 2 ** -13
NUM_TRAJECTORIES = 15000
MAX_EPISODE_LENGTH = 30

# Data collecting constants
AVERAGING_LENGTH_FOR_DATA_COLLECTION = 25
NUM_EX_TRAJECTORIES = 20


def get_action_probs(
        state: tuple[int],
        policy_map: Dict[tuple[int], list[float]]
) -> list[float]:
    """
    Returns action probabilities for a given state.
    - If agent is None: returns all 8 probs [U1,D1,L1,R1,U2,D2,L2,R2]
    - If agent == 0: returns [U1,D1,L1,R1]
    - If agent == 1: returns [U2,D2,L2,R2]
    """
    key = tuple(state)
    if key not in policy_map:
        raise KeyError(f"State {key} not found in policy map.")
    return policy_map[key]


def next_state_circle(position: tuple[int]) -> game_manager.ActionSpace:
    """
    Given a position `state = (row, col)` on a board of size height×width,
    returns the adjacent state according to:
      - If state is on the perimeter, move to the next perimeter cell
        in clockwise order.
      - Otherwise, move one step down (row + 1).
    """
    _, _, x, y = position

    # If not on the edge: just go down
    if 0 < y < game_manager.CarState.HEIGHT - 1 and 0 < x < game_manager.CarState.WIDTH - 1:
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


def generate_trajectory(state: game_manager.CarState, policy_nets: list[nn.Module], device: str):
    state_at_timestep = []
    action_at_timestep = []
    reward_at_timestep = []
    log_probs_at_timestep = []
    h0 = [None] * game_manager.CarState.NUM_AGENTS
    c0 = [None] * game_manager.CarState.NUM_AGENTS

    for ep in range(MAX_EPISODE_LENGTH):
        log_probs = []
        actions = []
        # Get the policy and choose an action for each agent
        for i in range(game_manager.CarState.NUM_AGENTS):
            if policy_nets[i].__class__ == models.NonMarkovianPolicyNet:
                action_probs, h0[i], c0[i] = policy_nets[i](state.get_state_tensor(device), h0[i], c0[i])
            else:
                action_probs = policy_nets[i](state.get_state_tensor(device))
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


def train(device: str, markovian: bool):
    policy_nets = [models.MarkovianPolicyNet(2 * game_manager.CarState.NUM_AGENTS,
                                             len(game_manager.ActionSpace)).to(device) if markovian else
                   models.NonMarkovianPolicyNet(input_dim=2 * game_manager.CarState.NUM_AGENTS,
                                                hidden_dim=100,
                                                layer_dim=1,
                                                output_dim=len(game_manager.ActionSpace)).to(device),
                   helper.load_policy_net_pickle('saved_net.pkl', device)]

    lengths, returns = [], []
    avg_lengths, avg_returns = 0, 0

    last_log_probs = []
    last_states = []

    # Training loop
    try:
        for i in tqdm(range(NUM_TRAJECTORIES), desc="Training"):
            # Zeros out gradients
            models.zero_gradients(policy_nets[0])

            # Generate the trajectory
            state_at_timestep, action_at_timestep, reward_at_timestep, log_probs_at_timestep = \
                (generate_trajectory(game_manager.CarState.get_random_state(), policy_nets, device=device))

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
                loss = log_probs_at_timestep[t][0][action_at_timestep[t][0].value] * Q[0]
                models.gradients_wrt_params(policy_nets[0], loss)

            # Update the model parameters
            models.update_params(policy_nets[0], LEARNING_RATE)
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

    m_nets, m_lens, m_rets = train(device=dev, markovian=True)
    nm_nets, nm_lens, nm_rets = train(device=dev, markovian=False)

    m_trajectories = np.arange(1, len(m_lens) + 1)
    nm_trajectories = np.arange(1, len(nm_lens) + 1)
    fig, axs = plt.subplots(2, 1, figsize=(8, 5))

    # Plot for the length of the trajectories
    axs[0].plot(m_trajectories, m_lens, label='Markovian')
    axs[0].plot(nm_trajectories, nm_lens, label='Non-Markovian')
    axs[0].set_xlabel('Trajectory')
    axs[0].set_ylabel('Length')
    axs[0].tick_params(axis='y')
    axs[0].set_title('Trajectory Lengths vs Circle Policy')
    axs[0].grid(True, linestyle='--', alpha=0.5)
    axs[0].legend()

    # Plot for each agent's expected reward at the start of each trajectory
    axs[1].plot(m_trajectories, m_rets, label='Markovian')
    axs[1].plot(nm_trajectories, nm_rets, label='Non-Markovian')
    axs[1].set_xlabel('Trajectory')
    axs[1].set_ylabel('Expected Reward')
    axs[1].tick_params(axis='y')
    axs[1].set_title('Expected Rewards vs Circle Policy')
    axs[1].grid(True, linestyle='--', alpha=0.5)
    axs[1].legend()

    fig.tight_layout()
    plt.show()
