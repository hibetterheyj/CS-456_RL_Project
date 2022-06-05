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
from viz_utils import reward_loss_plots, metrics_plots

os.makedirs('plot', exist_ok=True)
os.makedirs('res', exist_ok=True)

val_interval = 250
agent = DQNPlayer(epsilon=0.01, buffer_sz=1, batch_sz=1)
expert = OptimalPlayer(0.5)
rewards, losses = agent.train(
    expert, nr_episodes=20000, val_interval=val_interval(10, 6)
)

## viz
save_prefix = 'question12'
reward_loss_plots(rewards, losses, save_dir='plot', save_fn=save_prefix)
metrics_plots(
    agent.m_opts,
    agent.m_rands,
    val_interval=val_interval,
    save_dir='plot',
    save_fn=save_prefix,
)

np.savez_compressed(
    os.path.join('res', save_prefix),
    rewards=rewards,
    losses=losses,
    m_opts=agent.m_opts,
    m_rands=agent.m_rands,
)
# res = np.load(os.path.join('res', save_prefix + '.npz'))
