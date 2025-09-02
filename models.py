import torch
import torch.nn as nn
from torch.autograd import grad


class NonMarkovianPolicyNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(NonMarkovianPolicyNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, h0=None, c0=None):
        if h0 is None or c0 is None:
            h0 = torch.zeros(self.layer_dim, self.hidden_dim).to(x.device)
            c0 = torch.zeros(self.layer_dim, self.hidden_dim).to(x.device)

        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = nn.functional.softmax(self.fc(out), dim=1).squeeze()

        return out, hn, cn


class MarkovianPolicyNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MarkovianPolicyNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, output_dim)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.softmax(self.fc3(x), dim=1).squeeze()
        return x


class ValueNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, output_dim)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze()


def gradients_wrt_params(net: nn.Module, loss_tensor: torch.Tensor):
    """
    Dictionary to store gradients for each parameter
    Compute gradients with respect to each parameter
    """

    for name, param in net.named_parameters():
        g = grad(loss_tensor, param, retain_graph=True)[0]
        param.grad = param.grad + g


def update_params(net: nn.Module, learning_rate: float):
    """
    Update parameters for the network
    """

    for name, param in net.named_parameters():
        param.data += learning_rate * param.grad


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
