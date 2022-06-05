import itertools
import random
from tkinter.messagebox import NO
from typing import Optional, Tuple

import numpy as np
from pyrsistent import optional


class QPlayer:
    def __init__(
        self,
        player: str,
        alpha: float = 0.05,
        gamma=0.99,
        epsilon_min: float = 0.1,
        epsilon_max: float = 0.8,
        n_star=10000,
    ) -> None:

        self.player = player  # 'X' or 'O'
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.n_star = n_star

        self.player2value = {"X": 1, "O": -1}

        self.q_table = {}
        for state in itertools.product([0, 1, -1], repeat=9):
            # when q_learning player first play, the sum is always 0 if this is its turn
            # when it is the second player, sum of grid is 1 if this is its turn
            if (np.sum(state) == 0) or (np.sum(state) == 1):
                dict_state_temp = {}
                for action in range(9):
                    if state[action] == 0:
                        dict_state_temp.update({action: 0})
                self.q_table.update({state: dict_state_temp})

        self.prev_state: Optional[Tuple[int]] = None
        self.prev_action: Optional[int] = None

    def empty(self, grid: np.ndarray) -> "list[Tuple]":
        """return all empty positions"""
        avail = []
        for i in range(9):
            pos = (int(i / 3), i % 3)
            if grid[pos] == 0:
                avail.append(pos)
        return avail

    def randomMove(self, grid: np.ndarray) -> "Tuple[int]":
        """Chose a random move from the available options."""
        avail = self.empty(grid)

        return avail[random.randint(0, len(avail) - 1)]

    def get_q_value(self, grid: np.ndarray, move: "Tuple[int]") -> float:
        # for debug

        return self.q_table[self.grid_to_state(grid)][move[0] * 3 + move[1]]

    def update_q(
        self,
        reward: float,
        grid: np.ndarray,
        is_end: bool = False,
    ) -> None:
        if is_end:
            # if game ends, no need to find max Q value in next state

            q_next_max = 0
        else:
            current_state = self.grid_to_state(grid)
            next_action_rank = sorted(
                zip(
                    self.q_table[current_state].values(),
                    self.q_table[current_state].keys(),
                )
            )

            q_next_max = next_action_rank[-1][0]

        self.q_table[self.prev_state][self.prev_action] = self.q_table[self.prev_state][
            self.prev_action
        ] + self.alpha * (
            reward
            + self.gamma * q_next_max
            - self.q_table[self.prev_state][self.prev_action]
        )

    def act(self, grid: np.ndarray, n=0) -> "Tuple[int]":
        # whether move in random or not
        epsilon = max(self.epsilon_min, self.epsilon_max * (1 - n / self.n_star))

        current_state = self.grid_to_state(grid)

        if random.random() < epsilon:
            move = self.randomMove(grid)
        else:

            next_action_rank = sorted(
                zip(
                    self.q_table[current_state].values(),
                    self.q_table[current_state].keys(),
                )
            )

            best_action = next_action_rank[-1][1]
            move = (best_action // 3, best_action % 3)

        self.prev_state = current_state
        self.prev_action = move[0] * 3 + move[1]

        return move

    @staticmethod
    def grid_to_state(grid: np.ndarray) -> "Tuple[int]":

        return tuple(grid.reshape(-1))
