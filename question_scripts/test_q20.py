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
from dqn_utils import *
from dqn_player import DQNPlayer, setup_seed
from viz_utils import (
    reward_loss_plots,
    metrics_plots,
    mul_metrics_plots,
    mul_reward_loss_plots,
)

os.makedirs('plot', exist_ok=True)
os.makedirs('res', exist_ok=True)

# setup seed for random, numpy, and torch
setup_seed(2022)

save_prefix = 'question20'
val_interval = 250
save_single = True
save_all = True

losses_dict = {}
rewards_dict = {}
metrics_dict = {"M_opt": {}, "M_rand": {}}

explore = True
n_star = 10000


def single_run():
    agent = DQNPlayer(explore=explore, n_star=10000, verbose=False)
    expert = OptimalPlayer(0.5)
    rewards, losses = agent.train(expert, nr_episodes=20000, val_interval=val_interval)
    np.savez_compressed(
        os.path.join('res', save_prefix + '_expert'),
        rewards=rewards,
        losses=losses,
        m_opts=agent.m_opts,
        m_rands=agent.m_rands,
    )


def single_run_self():
    agent = DQNPlayer(explore=explore, n_star=10000, verbose=False)
    rewards, losses = agent.self_train(nr_episodes=20000, val_interval=val_interval)
    np.savez_compressed(
        os.path.join('res', save_prefix + '_self'),
        rewards=rewards,
        losses=losses,
        m_opts=agent.m_opts,
        m_rands=agent.m_rands,
    )


def find_m_rand_step_max(m_rands):
    max_m_rand = max(m_rands)
    m_rand_thresh = 0.8 * max_m_rand
    ge_m_rand = [m_rand > m_rand_thresh for m_rand in m_rands]
    return ge_m_rand.index(True) + 1, max(m_rands)


def find_m_opt_step_max(m_opts):
    max_m_opt = max(m_opts)
    m_opt_thresh = 0.8 * (max_m_opt + 1) - 1
    ge_m_rand = [m_rand > m_opt_thresh for m_rand in m_opts]
    return ge_m_rand.index(True) + 1, max(m_opts)


def test_from_saves():
    for suffix in ['_self', '_expert']:
        print(suffix[1:])
        res = np.load(os.path.join('res', save_prefix + suffix + '.npz'))
        m_opts = res['m_opts']
        m_rands = res['m_rands']
        opt_step, opt_max = find_m_opt_step_max(m_opts)
        rand_step, rand_max = find_m_rand_step_max(m_rands)
        print(
            f'Achieve 80% step-rand_step: {rand_step*val_interval}; opt_step: {opt_step*val_interval}'
        )
        print(f'Max-rand_max: {rand_max}; opt_max: {opt_max}')


if __name__ == "__main__":
    # single_run()
    # single_run_self()
    test_from_saves()
