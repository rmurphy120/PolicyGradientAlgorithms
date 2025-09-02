import models

import torch
import torch.nn as nn

import pickle
import csv
import numpy as np
from typing import Dict


def save_policy_net_pickle(model: nn.Module, path: str):
    """
    Saves the entire model object with pickle.
    """
    model.eval()
    model_cpu = model.to("cpu")
    with open(path, "wb") as f:
        pickle.dump(model_cpu, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_policy_net_pickle(path: str, device: str) -> nn.Module:
    """
    Loads a pickled nn.Module object.
    """
    with open(path, "rb") as f:
        model: nn.Module = pickle.load(f)
    return model.to(device)

def save_policies_to_csv(filename: str, list_policy_net: list[models.MarkovianPolicyNet], all_states, device: str):
    """
    Saves the policy of a Markovian model to a CSV file
    """
    header = [
        "X1", "Y1", "X2", "Y2",
        "U1", "D1", "L1", "R1", "U2", "D2", "L2", "R2"
    ]

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for each in all_states:
            row = list(each.state)
            state_tensor = each.get_state_tensor(device)
            for policy_net in list_policy_net:
                row += policy_net(state_tensor).tolist()
            writer.writerow(row)

    print(f'Successfully printed to {filename}')

def load_policies_from_csv(filename: str) -> Dict[tuple[int], list[float]]:
    """
    Loads the policy of a Markovian model from a CSV file and saves it into a dict:
      (X1, Y1, X2, Y2) -> [U1, D1, L1, R1]
    """
    key_cols = ["X1", "Y1", "X2", "Y2"]
    prob_cols = ["U2", "D2", "L2", "R2"]

    mapping: Dict[tuple[int], list[float]] = {}
    with open(filename, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            state = tuple(int(row[c]) for c in key_cols)  # type: ignore
            probs = [float(row[c]) for c in prob_cols]

            probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
            probs[probs < 0] = 0.0
            s = probs.sum()
            probs = (probs / s) if s > 0 else np.ones_like(probs) / len(probs)

            mapping[state] = probs
    return mapping

def save_grads(net: nn.Module, suffix: int, device: str):
    """
    Save the gradients of a model
    """
    output_path = f"gradients{suffix}.txt"
    with open(output_path, "w") as f:
        for name, param in net.named_parameters():
            if param.grad is None:
                f.write(f"{name}: no gradient\n")
            else:
                f.write(f"{name}.grad shape={param.grad.shape}\n")
                # Convert the tensor to a Python string
                f.write(param.grad.to(device).numpy().tolist().__str__() + "\n\n")

def save_values_to_csv(filename: str, value_nets: list[nn.Module], all_states, device: str):
    """
    Save the values of each state computed by the value net to a CSV file
    """
    header = ["X1", "Y1", "X2", "Y2", "V1", "V2"]

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for each in all_states:
            row = list(each.state)
            state_tensor = each.get_state_tensor(device)
            for value_net in value_nets:
                row.append(value_net(state_tensor).item())
            writer.writerow(row)

    print(f'Successfully printed to {filename}')


def compare_models(model1: nn.Module, model2: nn.Module, device: str, all_state):
    """
    Computes the average different between models of all outputs over all states
    """
    mean_difference = 0
    for state in all_state:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        action_probs1 = model1(state_tensor).squeeze()
        action_probs2 = model2(state_tensor).squeeze()
        mean_difference += (action_probs2 - action_probs1).abs().mean().item()

    return mean_difference / len(all_state)
