import random
from enum import Enum
from functools import lru_cache

import torch


class ActionSpace(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    STOP = 4

    @classmethod
    @lru_cache(maxsize=None)
    def calculate_all_actions(cls):
        values = list(cls)
        num_vals = len(values)
        total = num_vals ** SoccerState.NUM_AGENTS

        all_actions = []
        for idx in range(total):
            combo = []
            rem = idx
            for _ in range(SoccerState.NUM_AGENTS):
                combo.append(values[rem % num_vals])
                rem //= num_vals
            all_actions.append(combo)
        return all_actions


class SoccerState:
    NUM_AGENTS = 2
    WIDTH = 7
    HEIGHT = 4

    ALL_STATES = []

    @staticmethod
    def index_to_state(index: int) -> "SoccerState":
        state = [0] * (2 * SoccerState.NUM_AGENTS + 1)

        state[0] = index % 2
        index //= 2

        for i in range(1, len(state)):
            if i % 2 == 0:
                state[i] = index % SoccerState.WIDTH
                index //= SoccerState.WIDTH
            else:
                state[i] = index % SoccerState.HEIGHT
                index //= SoccerState.HEIGHT

        off_board = SoccerState.off_board(state)
        return SoccerState(tuple(state), off_board)

    @staticmethod
    def get_state_index(state: list[int]) -> int:
        index = 0
        base = 1

        index += state[0]
        base *= 2

        for i in range(1, len(state)):
            index += state[i] * base

            if i % 2 == 1:
                base *= SoccerState.WIDTH
            else:
                base *= SoccerState.HEIGHT

        return index

    @staticmethod
    def get_random_state() -> "SoccerState":
        state = -1
        while state == -1 or any(state.is_off_board) or SoccerState.in_goal(state.state[1], state.state[2]) or \
                SoccerState.in_goal(state.state[3], state.state[4]):
            state = SoccerState.ALL_STATES[SoccerState.shuffledStateIds[SoccerState.nextId]]
            SoccerState.nextId += 1

            if SoccerState.nextId == len(SoccerState.ALL_STATES):
                SoccerState.nextId = 0
                random.shuffle(SoccerState.shuffledStateIds)

        return state

    @staticmethod
    def off_board(state: list[int]) -> list[bool]:
        """
        Returns a list of booleans representing if each player is in a valid state (In this case, are
        they on the board)
        """

        invalid_agents = []
        for a in range(SoccerState.NUM_AGENTS):
            invalid_agents.append(not SoccerState.in_goal(state[2 * a + 1], state[2 * a + 2]) and
                                  (state[2 * a + 1] < 1 or state[2 * a + 1] >= SoccerState.WIDTH - 1 or state[2 * a + 2] < 0 or
                                  state[2 * a + 2] >= SoccerState.HEIGHT))
        return invalid_agents

    @staticmethod
    def in_goal(x: int, y: int) -> bool:
        goal_y_pos = [SoccerState.HEIGHT / 2 - 1, SoccerState.HEIGHT / 2]
        return (x == 0 or x == SoccerState.WIDTH - 1) and (y == goal_y_pos[0] or y == goal_y_pos[1])

    @staticmethod
    def same_position(x1, y1, x2, y2):
        return x1 == x2 and y1 == y2

    def __init__(self, state: tuple[int], off_board: list[bool]):
        self.state = state
        self.is_off_board = off_board
        self.reward = self.calculate_reward()

    def __eq__(self, other):
        return isinstance(other, SoccerState) and self.state == other.state

    def __hash__(self):
        return hash(self.state)

    def transition(self, actions: list[ActionSpace]) -> dict["SoccerState", float]:
        next_states: dict[SoccerState, float] = {}

        # All move orders
        move_orders = [[0, 1], [1, 0]]

        for move_order in move_orders:
            new_state = list(self.state)
            for a in range(self.NUM_AGENTS):
                if actions[move_order[a]] == ActionSpace.UP:
                    new_state[2 * move_order[a] + 2] -= 1
                elif actions[move_order[a]] == ActionSpace.DOWN:
                    new_state[2 * move_order[a] + 2] += 1
                elif actions[move_order[a]] == ActionSpace.LEFT:
                    new_state[2 * move_order[a] + 1] -= 1
                elif actions[move_order[a]] == ActionSpace.RIGHT:
                    new_state[2 * move_order[a] + 1] += 1

                # Check if stealing the ball
                if self.same_position(new_state[1], new_state[2], new_state[3], new_state[4]):
                    if actions[move_order[a]] == ActionSpace.UP:
                        new_state[2 * move_order[a] + 2] += 1
                    elif actions[move_order[a]] == ActionSpace.DOWN:
                        new_state[2 * move_order[a] + 2] -= 1
                    elif actions[move_order[a]] == ActionSpace.LEFT:
                        new_state[2 * move_order[a] + 1] += 1
                    elif actions[move_order[a]] == ActionSpace.RIGHT:
                        new_state[2 * move_order[a] + 1] -= 1

                    new_state[0] = 0 if move_order[a] == 1 else 1

            off_board = self.off_board(new_state)

            if any(off_board):
                s = SoccerState(tuple(new_state), off_board)
            else:
                s = SoccerState.ALL_STATES[self.get_state_index(new_state)]

            next_states[s] = next_states.get(s, 0) + 1 / len(move_orders)

        return next_states

    def calculate_reward(self) -> list[int]:
        reward = [0] * self.NUM_AGENTS
        ball_player, x1, y1, x2, y2 = self.state

        # Check if off board
        if any(self.is_off_board):
            if all(self.is_off_board):
                reward[0] = -25
                reward[1] = -25
            elif self.is_off_board[0]:
                reward[0] = 25
                reward[1] = -25
            elif self.is_off_board[1]:
                reward[0] = -25
                reward[1] = 25

        # Check if goal
        elif self.state[0] == 0 and self.in_goal(x1, y1):
            reward[0] = -24 if x1 == 0 else 24
            reward[1] = 24 if x1 == 0 else -24
        elif self.state[0] == 1 and self.in_goal(x2, y2):
            reward[0] = -24 if x2 == 0 else 24
            reward[1] = 24 if x2 == 0 else -24

        # Board reward
        if ball_player == 0:
            reward[0] += .25 * (x1 - 1)
        else:
            reward[1] += .25 * (self.WIDTH - 2 - x2)

        return reward

    @lru_cache(maxsize=None)
    def get_state_tensor(self, device: str) -> torch.Tensor:
        return torch.FloatTensor(self.state).unsqueeze(0).to(device)


total_states = 2 * ((SoccerState.WIDTH * SoccerState.HEIGHT) ** SoccerState.NUM_AGENTS)
SoccerState.ALL_STATES = [SoccerState.index_to_state(i) for i in range(total_states)]

# For getting random states
SoccerState.shuffledStateIds = list(range(len(SoccerState.ALL_STATES)))
random.shuffle(SoccerState.shuffledStateIds)
SoccerState.nextId = 0
