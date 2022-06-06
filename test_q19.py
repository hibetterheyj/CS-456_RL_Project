"""
_summary_: In self-practice, visualize the q table
"""

# std
import os
from typing import List, Dict, Tuple

# imported
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# customized
from tic_env import TictactoeEnv, OptimalPlayer
from dqn_player import DQNPlayer
from dqn_utils import *
from viz_utils import reward_loss_plots, metrics_plots, mul_metrics_plots

os.makedirs('plot', exist_ok=True)
os.makedirs('res', exist_ok=True)
save_prefix = 'question19'
save_single = True
save_all = True


def prepare_test_states():
    ## test states
    state1 = (0, 0, 0, 0, 0, 0, 0, 0, 0)
    env = TictactoeEnv()
    grid1, _, _ = env.observe()

    state2 = (1, 1, 0, -1, -1, 0, 0, 0, 0)
    env.reset()
    moves2 = [(0, 0), (1, 0), (0, 1), (1, 1)]
    for move in moves2:
        grid2, _, __ = env.step(move)

    state3 = (1, -1, 0, 1, 0, 0, 0, 0, 0)
    env.reset()
    moves3 = [(1, 1), (0, 1), (0, 0)]
    for move in moves3:
        grid3, _, __ = env.step(move)

    states = [state1, state2, state3]
    players = ['X', 'X', 'O']
    grids = [grid1, grid2, grid3]
    states_2d = [grid2state(grids[idx], players[idx]) for idx in range(3)]
    return states, states_2d


def main(policy_path='./policy.pth'):
    states, states_2d = prepare_test_states()
    dqn_player = DQNPlayer(verbose=False)
    dqn_player.load(policy_path)
    # q_tables: 3x3x3
    q_tables = [
        dqn_player.policy(state).view((3, 3)).detach().numpy() for state in states_2d
    ]

    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    value2player = {0: ' -', 1: 'X', -1: 'O'}

    for idx, q_table in enumerate(q_tables):
        im = axes[idx].imshow(q_table)
        axes[idx].get_xaxis().set_visible(False)
        axes[idx].get_yaxis().set_visible(False)
        state_symbol = []
        for value in list(states[idx]):
            state_symbol.append(value2player[value])
        axes[idx].set_title(
            '|{0} {1} {2}|\n|{3} {4} {5}|\n|{6} {7} {8}|'.format(*state_symbol),
            fontsize=30,
        )
        for i in range(3):
            for j in range(3):
                text = axes[idx].text(
                    j,
                    i,
                    round(q_table[i, j], 3),
                    ha="center",
                    va="center",
                    color="w",
                    fontsize=30,
                )
    fig.tight_layout()
    plt.savefig("plot/question19.pdf", dpi=300)
    plt.show()


if __name__ == "__main__":
    # TODO: save results from q17 and load to dqn!!!
    main(policy_path='./policy.pth')
