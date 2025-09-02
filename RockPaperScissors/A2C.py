import game_manager
import models
import helper

import torch

from tqdm import tqdm
import matplotlib.pyplot as plt

import numpy as np
import random


# Training constants
GAMMA = .99
LEARNING_RATE = 2 ** -12
EPSILON = .0000001


def train(device: str):
    # Models
    policy_nets = [models.MarkovianPolicyNet(1, len(game_manager.ActionSpace)).to(device) for _ in range(game_manager.RPSState.NUM_AGENTS)]
    value_net = models.ValueNet(1, 1).to(device)

    # Policy history to be returned
    policies_over_time = [[[], [], []], [[], [], []]]
    prev_value_loss = -1

    # Training loop
    pbar = tqdm(desc="Training")
    shuffled_state_ids = list(range(len(game_manager.RPSState.ALL_STATES)))
    has_converged = False
    try:
        while pbar.n < 20000:
            avg_value_loss = 0

            random.shuffle(shuffled_state_ids)

            for id in shuffled_state_ids:
                # Zeros out gradients
                for policy_net in policy_nets:
                    models.zero_gradients(policy_net)

                models.zero_gradients(value_net)

                # Get state and policies
                state = game_manager.RPSState.ALL_STATES[id]
                state_value = value_net(state.get_state_tensor(device)).squeeze()
                action_probs = [policy_nets[j](state.get_state_tensor(device)).squeeze() for j in range(
                    game_manager.RPSState.NUM_AGENTS)]
                y = 0

                # Don't sample, loop over all action pairs
                for actions in game_manager.ActionSpace.calculate_all_actions():
                    next_states = state.transition(actions)

                    # Find expected Qs
                    Q = state.reward[0]
                    for each in next_states:
                        Q = Q + next_states[each] * GAMMA * value_net(each.get_state_tensor(device)).squeeze().item()

                    # Define losses for policies and target for value
                    y_for_actions = Q
                    advantage = Q - state_value
                    policies_loss = []

                    for j in range(game_manager.RPSState.NUM_AGENTS):
                        prob = action_probs[j][actions[j].value]
                        advantage *= prob.item()
                        y_for_actions = y_for_actions * prob.item()
                        policies_loss.append(torch.log(prob))

                    y += y_for_actions

                    policies_loss[0] = -advantage * policies_loss[0]
                    policies_loss[1] = advantage * policies_loss[1]

                    # Store gradients
                    for j in range(game_manager.RPSState.NUM_AGENTS):
                        models.gradients_wrt_params(policy_nets[j], policies_loss[j])

                # Define value loss
                value_loss = 0.5 * (y - state_value).pow(2)

                # Update policy and value parameters
                for j in range(game_manager.RPSState.NUM_AGENTS):
                    models.update_params(policy_nets[j], LEARNING_RATE)

                if not has_converged:
                    models.gradients_wrt_params(value_net, value_loss)
                    models.update_params(value_net, LEARNING_RATE)

                avg_value_loss += value_loss.item()

                if id == 0:
                    for j in range(game_manager.RPSState.NUM_AGENTS):
                        for action in range(len(game_manager.ActionSpace)):
                            policies_over_time[j][action].append(action_probs[j][action].item())

            avg_value_loss /= len(game_manager.RPSState.ALL_STATES)

            # Convergence check
            if not has_converged and abs(prev_value_loss - avg_value_loss) < EPSILON:
                print('Value function converged')
                has_converged = True

            prev_value_loss = avg_value_loss

            pbar.update(1)
            pbar.set_postfix(value_loss=f"{avg_value_loss:.4f}")
    except KeyboardInterrupt:
        pass

    pbar.close()

    return policy_nets, value_net, policies_over_time


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    p_nets, v_net, policies_over_time = train(device=dev)

    """policy_filename = 'C:\\Users\\rynom\\OneDrive - UW-Madison\\Desktop\\Java Projects\\NashEquilibriumChecker\\nash_equilibrium.csv'
    value_filename = 'converged_values.csv'

    helper.save_policies_to_csv(policy_filename, p_nets, game_manager.RPSState.ALL_STATES, dev)
    helper.save_values_to_csv(value_filename, [v_net], game_manager.RPSState.ALL_STATES, dev)"""

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
