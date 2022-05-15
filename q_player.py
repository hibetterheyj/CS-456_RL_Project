import itertools
import random
from tkinter.messagebox import NO
from typing import Tuple

import numpy as np


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
            for action in range(9):
                if state[action] == 0:
                    if (np.sum(state) == 0) or (np.sum(state) == -1):
                        self.q_table.update({state + (action,): 0})

    def empty(self, grid: np.ndarray) -> list[Tuple]:
        """return all empty positions"""
        avail = []
        for i in range(9):
            pos = (int(i / 3), i % 3)
            if grid[pos] == 0:
                avail.append(pos)
        return avail

    def randomMove(self, grid: np.ndarray) -> Tuple[int]:
        """Chose a random move from the available options."""
        avail = self.empty(grid)

        return avail[random.randint(0, len(avail) - 1)]

    def get_index(self, grid: np.ndarray, move: Tuple[int]) -> Tuple[int]:
        grid = grid * self.player2value[self.player]

        state = tuple(grid.reshape(-1))
        action = move[0] * 3 + move[1]

        return state + (action,)

    def get_q_value(self, grid: np.ndarray, move: Tuple[int]) -> float:

        return self.q_table[self.get_index(grid, move)]

    def find_best_action(self, grid: np.ndarray) -> Tuple[int]:
        avails = self.empty(grid)

        current_max_q = -float("inf")
        best_actions = []
        for avail in avails:
            current_q = self.get_q_value(grid, avail)
            if current_q > current_max_q:
                best_actions = [
                    avail,
                ]
            elif current_q == current_max_q:
                best_actions.append(avail)

        return best_actions[random.randint(0, len(best_actions) - 1)]

    def update_q(
        self,
        reward: float,
        grid: np.ndarray,
        grid_prev: np.ndarray,
        move_prev: Tuple[int],
        is_end: bool = False,
    ) -> None:

        tab_idx = self.get_index(grid_prev, move_prev)

        if is_end:
            # if game ends, no need to find max Q value in next state

            self.q_table[tab_idx] = self.q_table[tab_idx] + self.alpha * (
                reward - self.q_table[tab_idx]
            )
        else:
            # if game doesn't end, reward is always zero

            best_move_next = self.find_best_action(grid)
            q_next_max = self.q_table[self.get_index(grid, best_move_next)]

            self.q_table[tab_idx] = self.q_table[tab_idx] + self.alpha * (
                self.gamma * q_next_max - self.q_table[tab_idx]
            )

    def act(self, grid: np.ndarray, n=0) -> Tuple[int]:
        # whether move in random or not
        epsilon = max(self.epsilon_min, self.epsilon_max * (1 - n / self.n_star))
        if random.random() < epsilon:
            return self.randomMove(grid)

        return self.find_best_action(grid)
