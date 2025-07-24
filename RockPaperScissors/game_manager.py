import random
from functools import lru_cache

from enum import Enum
import torch


class ActionSpace(Enum):
    ROCK = 0
    PAPER = 1
    SCISSORS = 2

    @classmethod
    @lru_cache(maxsize=None)
    def calculate_all_actions(cls) -> list[list["ActionSpace"]]:
        values = list(cls)
        num_vals = len(values)
        total = num_vals ** RPSState.NUM_AGENTS

        all_actions = []
        for idx in range(total):
            combo = []
            rem = idx
            for _ in range(RPSState.NUM_AGENTS):
                combo.append(values[rem % num_vals])
                rem //= num_vals
            all_actions.append(combo)

        return all_actions


class RPSState:
    NUM_AGENTS = 2

    ALL_STATES = []

    @staticmethod
    def get_random_state() -> "RPSState":
        state = RPSState.ALL_STATES[RPSState.shuffledStateIds[RPSState.nextId]]
        RPSState.nextId += 1

        if RPSState.nextId == len(RPSState.ALL_STATES):
            RPSState.nextId = 0
            random.shuffle(RPSState.shuffledStateIds)

        return state

    def __init__(self, state: int):
        self.state = state
        self.reward = self.calculate_reward()

    @staticmethod
    def transition(actions: list[ActionSpace]) -> dict["RPSState", float]:
        next_states: dict[RPSState, float] = {}

        ac_val1 = actions[0].value
        ac_val2 = actions[1].value

        if ac_val1 == ac_val2:
            next_state = 1
        elif (ac_val1 + 3 + 1) % 3 == ac_val2:
            next_state = 0
        elif (ac_val1 + 3 - 1) % 3 == ac_val2:
            next_state = 2

        next_states[RPSState.ALL_STATES[next_state]] = 1.0

        return next_states

    def calculate_reward(self) -> list[int]:
        reward = [0] * self.NUM_AGENTS

        reward[0] = self.state - 1

        # 0 sum game
        reward[1] = -reward[0]

        return reward

    @lru_cache(maxsize=None)
    def get_state_tensor(self, device: str) -> torch.Tensor:
        return torch.FloatTensor([self.state]).unsqueeze(0).to(device)


total_states = 3
RPSState.ALL_STATES = [RPSState(i) for i in range(total_states)]

# For getting random states
RPSState.shuffledStateIds = list(range(len(RPSState.ALL_STATES)))
random.shuffle(RPSState.shuffledStateIds)
RPSState.nextId = 0
