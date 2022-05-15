# std
import logging
import random
from typing import Dict, List

import matplotlib.pyplot as plt

# imported
import numpy as np
from tqdm import tqdm

from q_player import QPlayer

# customized
from tic_env import OptimalPlayer, TictactoeEnv

env = TictactoeEnv()

Turns_odd = np.array(["X", "O"])
Turns_even = np.array(["O", "X"])
epsilon = 0.5
player_qlearning = QPlayer(player="X", epsilon_min=epsilon, epsilon_max=epsilon)

record_inter = 2000
q_tables_record = {}
reward_record = []
avg_reward_record = {}

for i in range(20000):
    env.reset()
    grid, end, winner = env.observe()
    if i % 2 == 0:
        Turns = Turns_odd
    else:
        Turns = Turns_even
    player_opt = OptimalPlayer(epsilon=0.5, player=Turns[0])
    player_qlearning.player = Turns[1]

    if i % 2 == 0:
        move = player_opt.act(grid)
        grid, end, winner = env.step(move, print_grid=False)

    while not end:

        grid_prev = grid.copy()

        move = player_qlearning.act(grid)
        move_prev = move
        grid, end, winner = env.step(move, print_grid=False)

        if not end:
            move = player_opt.act(grid)
            grid, end, winner = env.step(move, print_grid=False)

        print("Optimal player 1 = " + Turns[0])
        print("Qlearning player 2 = " + Turns[1])
        env.render()

        if not end:
            player_qlearning.update_q(
                reward=0, grid=grid, grid_prev=grid_prev, move_prev=move_prev
            )
        else:
            player_qlearning.update_q(
                reward=env.reward(player=player_qlearning.player),
                grid=grid,
                grid_prev=grid_prev,
                move_prev=move_prev,
                is_end=True,
            )

        if end:
            # print('-------------------------------------------')
            # print('Game end, winner is player ' + str(winner))
            # print('Optimal player 1 = ' +  Turns[0])
            # print('Qlearning player 2 = ' +  Turns[1])
            # env.render()

            reward_record.append(env.reward(player=player_qlearning.player))

            env.reset()

            if ((i + 1) % record_inter == 0) & (i != 0):
                q_tables_record.update({i: player_qlearning.q_table})
                avg_reward_record.update({i: np.mean(reward_record)})
                # reward_record = []
                print(
                    "avg reward between {} to {} is: {}".format(
                        int(i - record_inter + 1), int(i + 1), avg_reward_record[i]
                    )
                )

                player_rand = OptimalPlayer(epsilon=1, player=Turns[0])
                player_qlearning.epsilon_min = 0
                player_qlearning.epsilon_max = 0
                res_info = eval(player_qlearning, player_rand)
                player_qlearning.epsilon_min = epsilon
                player_qlearning.epsilon_max = epsilon

                print("# Eval with Opt({})".format(0))
                print(
                    "M{} = {}, Draw rate = {}".format(
                        "rand", res_info["metric"], res_info["draw_rate"]
                    )
                )
