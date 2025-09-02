import random
from functools import lru_cache

from enum import Enum
import torch


class ActionSpace(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    @classmethod
    @lru_cache(maxsize=None)
    def calculate_all_actions(cls) -> list[list["ActionSpace"]]:
        values = list(cls)
        num_vals = len(values)
        total = num_vals ** CarState.NUM_AGENTS

        all_actions = []
        for idx in range(total):
            combo = []
            rem = idx
            for _ in range(CarState.NUM_AGENTS):
                combo.append(values[rem % num_vals])
                rem //= num_vals
            all_actions.append(combo)

        return all_actions


class CarState:
    NUM_AGENTS = 2
    WIDTH = 3
    HEIGHT = 3

    ALL_STATES = []

    @staticmethod
    def index_to_state(index: int) -> "CarState":
        state = [0] * (2 * CarState.NUM_AGENTS)
        rem = index

        for i in range(len(state)):
            if i % 2 == 0:
                state[i] = rem % 3
                rem //= 3
            else:
                state[i] = rem % 3
                rem //= 3

        return CarState(tuple(state), (False, False))

    @staticmethod
    def get_state_index(state: list[int]) -> int:
        index = 0
        base = 1

        for i in range(len(state)):
            index += state[i] * base

            if i % 2 == 0:
                base *= CarState.WIDTH
            else:
                base *= CarState.HEIGHT

        return index

    @staticmethod
    def get_random_state() -> "CarState":
        state = None

        while state is None or (abs(state.state[2] - state.state[0]) - abs(state.state[3] - state.state[1])) % 2 == 1:
            state = CarState.ALL_STATES[CarState.shuffledStateIds[CarState.nextId]]
            CarState.nextId += 1

            if CarState.nextId == len(CarState.ALL_STATES):
                CarState.nextId = 0
                random.shuffle(CarState.shuffledStateIds)

        return state

    @staticmethod
    def off_board(state: list[int]) -> tuple[bool, bool]:
        """
        Returns a tuple of booleans representing if each player is in a valid state (In this case, are
        they on the board)
        """
        x1, y1, x2, y2 = state
        return (x1 < 0 or x1 >= CarState.WIDTH or y1 < 0 or y1 >= CarState.HEIGHT,
                x2 < 0 or x2 >= CarState.WIDTH or y2 < 0 or y2 >= CarState.HEIGHT)

    def __init__(self, state: tuple[int], off_board: tuple[bool, bool]):
        self.state = state
        self.is_off_board = off_board
        self.reward = self.calculate_reward()

    def transition(self, actions: list[ActionSpace]) -> dict["CarState", float]:
        new_state = list(self.state)
        next_states: dict[CarState, float] = {}

        for a, act in enumerate(actions):
            x_i = 2 * a
            y_i = x_i + 1
            if act == ActionSpace.LEFT:
                new_state[x_i] -= 1
            elif act == ActionSpace.RIGHT:
                new_state[x_i] += 1
            elif act == ActionSpace.UP:
                new_state[y_i] -= 1
            elif act == ActionSpace.DOWN:
                new_state[y_i] += 1

        off_board = self.off_board(new_state)
        if any(off_board):
            next_states[CarState(tuple(new_state), off_board)] = 1.0
        else:
            next_states[CarState.ALL_STATES[self.get_state_index(new_state)]] = 1.0

        return next_states

    def calculate_reward(self) -> list[int]:
        reward = [0] * self.NUM_AGENTS
        cx = self.WIDTH // 2
        cy = self.HEIGHT // 2

        """# Off board penalty
        if any(self.is_off_board):
            reward[0] = -25
            return reward

        # Board reward
        reward[0] = (cx + cy - abs(cx - self.state[0]) - abs(cy - self.state[1]))"""

        x1, y1, x2, y2 = self.state

        # Off board penalty
        if any(self.is_off_board):
            if all(self.is_off_board):
                reward = [-15, -15]
            elif self.is_off_board[0]:
                reward = [-15, 15]
            elif self.is_off_board[1]:
                reward = [15, -15]
            return reward

        # Board reward
        # reward[0] = abs(cx - x1) + abs(cy - y1) - abs(cx - x2) - abs(cy - y2)

        # Collision reward
        if (x1, y1) == (x2, y2):
            reward[0] += 5

        # 0 sum game
        reward[1] = -reward[0]

        return reward

    @lru_cache(maxsize=None)
    def get_state_tensor(self, device: str) -> torch.Tensor:
        return torch.FloatTensor(self.state).unsqueeze(0).to(device)


total_states = (CarState.WIDTH * CarState.HEIGHT) ** CarState.NUM_AGENTS
CarState.ALL_STATES = [CarState.index_to_state(i) for i in range(total_states)]

# For getting random states
CarState.shuffledStateIds = list(range(len(CarState.ALL_STATES)))
random.shuffle(CarState.shuffledStateIds)
CarState.nextId = 0
