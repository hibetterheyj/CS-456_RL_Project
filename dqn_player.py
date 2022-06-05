import copy
import random
from typing import List, Dict, Tuple
from collections import namedtuple, deque
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tic_env import TictactoeEnv, OptimalPlayer
from dqn_utils import grid2state, decreasing_exploration, position2move


# default parameters
R_UNAV = -1
BATCH_SIZE = 64
BUFFER_SIZE = 10000
GAMMA = 0.99  # discount factor
TARGET_UPDATE = 500  # update interval
VAL_INTERVAL = 250
START_LR = 5e-4
EPS_MAX = 0.8
EPS_MIN = 0.1
N_STAR = 20000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
METRIC_DICT = {'opt': 0.0, 'rand': 1.0}


# ref: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#replay-memory
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))


class ReplayMemory(object):
    def __init__(self, capacity: int):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)

    def last(self):
        return self.memory[-1]

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, n_inputs: int = 18, n_outputs: int = 9, lr=5e-4) -> None:
        """Deep-Q network based on fully-connected network

        Args:
            n_inputs (int, optional): _description_. Defaults to 18.
            n_outputs (int, optional): _description_. Defaults to 9.
        """
        super(DQN, self).__init__()
        self.i2h = nn.Linear(n_inputs, 128)
        self.hid = nn.Linear(128, 128)
        self.h2o = nn.Linear(128, n_outputs)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss()

    # Called with either one element to determine next action, or a batch
    # during optimization.
    def forward(self, x):
        x = x.float()
        x = F.relu(self.i2h(x))
        x = F.relu(self.hid(x))
        x = self.h2o(x)
        return x

    def act(self, state):
        with torch.no_grad():
            return self.forward(state).max(1)[1].view(1, 1)


class DQNPlayer:
    def __init__(
        self,
        player='X',
        epsilon=0.2,
        gamma=GAMMA,
        buffer_sz=BUFFER_SIZE,
        batch_sz=BATCH_SIZE,
        target_update=TARGET_UPDATE,
        explore=False,
        eps_min=0.1,
        eps_max=0.8,
        n_star=N_STAR,
        device: torch.device = None,
        verbose: bool = True,
        **kwargs,
    ):
        # init variables
        self.player = player  # 'X' or 'O'
        self.gamma = gamma  # discount factor
        self.buffer_sz = buffer_sz
        self.batch_sz = batch_sz
        self.eps = epsilon
        self.target_update = target_update
        self.explore = explore
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.n_star = n_star
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.verbose = verbose

        # internal variables
        self.counter = 0
        self.policy = DQN()
        self.target = DQN()
        self.memory = ReplayMemory(self.buffer_sz)
        self.last_state = None
        self.last_action = None
        self.losses = []
        self.m_opts = []
        self.m_rands = []

    def copy(self):
        return copy.deepcopy(self)

    def empty(self, grid: np.array) -> List[int]:
        """Return all available positions in the grid

        Args:
            grid (np.array): current grid map

        Returns:
            List[int]: all available positions
        """
        avail = []
        for i in range(9):
            pos = (int(i / 3), i % 3)
            if grid[pos] == 0:
                avail.append(pos)
        return avail

    def random_move(self, grid: np.array) -> int:
        """Select a random move from the available positions in the grid

        Args:
            grid (np.array): current grid map

        Returns:
            int: randomly chosen move
        """
        avail = self.empty(grid)
        return avail[random.randint(0, len(avail) - 1)]

    def select_action(self, grid: np.array) -> int:
        """Select a move based on Epsilon greedy strategy

        Args:
            grid (np.array): current grid map

        Returns:
            int: final chosen move
        """
        if self.explore:
            self.eps = decreasing_exploration(
                self.counter,
                self.n_star,
                self.eps_max,
                self.eps_min,
            )
        if random.random() < self.eps:
            action = self.random_move(grid)
            self.last_action = position2move(action)
        else:
            with torch.no_grad():
                action = (
                    self.policy(grid2state(grid.copy(), self.player)).argmax().item()
                )
                self.last_action = action

        self.last_state = grid2state(grid.copy(), self.player)
        return action

    def one_game_expert(self, agent, env, episode, train=True) -> str:
        # init
        grid, end, __ = env.observe()
        if episode % 2 == 0:
            self.player = 'X'
            agent.player = 'O'
        else:
            self.player = 'O'
            agent.player = 'X'

        # simulation
        invalid_move = False
        while not end:
            if env.current_player == self.player:
                move = self.select_action(grid)
                # print(type(move))
                if env.check_valid(move):
                    grid, end, winner = env.step(move, print_grid=False)
                else:
                    # temp_state = grid2state(grid.copy(), self.player)
                    # state = temp_state
                    end, winner = True, agent.player
                    invalid_move = True
            else:
                move = agent.act(grid)
                grid, end, winner = env.step(move, print_grid=False)
                if train and not end:
                    reward = env.reward(self.player)
                    self.memory.push(
                        self.last_state,
                        self.last_action,
                        reward,
                        grid2state(grid.copy(), self.player),
                    )
                    self.optimize()
        if train:
            if invalid_move:
                reward = R_UNAV
            else:
                reward = env.reward(self.player)
            self.memory.push(
                self.last_state,
                self.last_action,
                reward,
                grid2state(grid.copy(), self.player),
            )
            self.optimize()
        return winner

    def multi_games_expert(self, expert, env, episodes):
        win = 0
        los = 0
        for j in range(episodes):
            env.reset()
            winner = self.one_game_expert(expert, env, 0, train=False)

            if winner == self.player:
                win += 1
            elif winner == expert.player:
                los += 1

        return (win - los) / episodes

    def compute_metrics(self, episodes: int = 500) -> None:
        """Compute m_opt and m_rand from 500 games by measureing
        the performance of policy against the optimal and random policy

        Args:
            episodes (int, optional): number of test runs. Defaults to 500.
        """
        env = TictactoeEnv()

        # set exploration to 0 in test environment
        eps_ = self.eps
        explore_ = self.explore
        self.eps = 0
        self.explore = False

        m_opt = self.multi_games_expert(OptimalPlayer(0), env, episodes)
        m_rand = self.multi_games_expert(OptimalPlayer(1), env, episodes)
        self.m_opts.append(m_opt)
        self.m_rands.append(m_rand)
        if self.verbose:
            print(f'{self.counter}-m_opt({m_opt})/m_rand({m_rand})')

        self.eps = eps_
        self.explore = explore_

    def one_game_self(self, env, self_copy) -> str:
        """self practice

        Args:
            env: gym-like environment
            self_copy: copy of self agent

        Returns:
            str: winner of self learning
        """
        grid, end, __ = env.observe()
        first_move = True
        self.player = 'X'
        self_copy.player = 'O'
        while not end:
            self_copy.policy.load_state_dict(self.policy.state_dict())
            self.optimize()
            if env.current_player == 'X':
                move = self.select_action(grid)
                grid, end, winner = env.step(move, print_grid=False)
                if not first_move and winner != 'O':
                    reward = env.reward('O')
                    self.memory.push(
                        self_copy.last_state,
                        self_copy.last_action,
                        reward,
                        grid2state(grid.copy(), 'O'),
                    )
                first_move = False
            else:
                move = self_copy.select_action(grid)
                grid, end, winner = env.step(move, print_grid=False)
                if winner != 'X':
                    reward = env.reward('X')
                    self.memory.push(
                        self.last_state,
                        self.last_action,
                        reward,
                        grid2state(grid.copy(), 'X'),
                    )

        reward = env.reward('X')
        self.memory.push(
            self.last_state, self.last_action, reward, grid2state(grid.copy(), 'X')
        )
        reward = env.reward('O')
        self.memory.push(
            self_copy.last_state,
            self_copy.last_action,
            reward,
            grid2state(grid.copy(), 'O'),
        )
        self_copy.policy.load_state_dict(self.policy.state_dict())
        self.optimize()
        return winner

    def optimize(self):
        if len(self.memory) >= self.batch_sz:
            # sample mini-batch from memory
            # state, action, reward, next_state = self.memory.sample(self.batch_sz)
            transitions = self.memory.sample(self.batch_sz)
            batch = Transition(*zip(*transitions))
            state_batch = torch.cat(batch.state)
            action_batch = torch.Tensor(batch.action)  # torch.cat(batch.action)
            reward_batch = torch.Tensor(batch.reward)  # torch.cat()
            # non_final_next_state_batch
            next_state_batch = torch.cat([s for s in batch.next_state if s is not None])

            self.policy.optimizer.zero_grad()
            # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
            # columns of actions taken. These are the actions which would've been taken
            # for each batch state according to policy_net
            state_action_values = (
                self.policy(state_batch)
                .gather(1, action_batch.long().view((self.batch_sz, 1)))
                .view(-1)
            )

            # calculate "target" Q-values from Q-Learning update rule
            non_final_mask = (~reward_batch.abs().bool()).int()
            # Compute the expected Q values
            expected_state_action_values = (
                reward_batch
                + self.gamma
                * non_final_mask
                * self.target(next_state_batch).max(dim=1)[0].detach()
            )

            # forward + backward + optimize
            loss = self.policy.criterion(
                state_action_values, expected_state_action_values
            )
            loss.backward()
            for param in self.policy.parameters():
                param.grad.data.clamp_(-1, 1)
            self.policy.optimizer.step()
            self.losses.append(loss.detach().numpy())

    def train(
        self,
        agent=None,
        nr_episodes: int = 20000,
        val_interval: int = 250,
        self_practice: bool = False,
        save_model: bool = False,
        ckpt_name: str = None,
    ) -> List[float]:
        """Training pipeline for DQNPlayer

        Args:
            agent (optional): agent to play against for policy learning
            nr_episodes (int, optional): number of learning episodes. Defaults to 20000.
            val_interval (int, optional): _description_. Defaults to 250.
            self_practice (bool, optional): _description_. Defaults to False.
            save_model (bool, optional): _description_. Defaults to False.
            save_avg_losses (bool, optional): _description_. Defaults to True.

        Returns:
            List[float]: rewards along the learning procedures
        """
        rewards = []
        avg_losses = []
        env = TictactoeEnv()
        self_copy = self.copy()
        for episode in tqdm(range(nr_episodes)):
            env.reset()
            self.counter += 1
            if self_practice:
                winner = self.one_game_self(env, self_copy)

                if episode % self.target_update == 0:
                    self.target.load_state_dict(self.policy.state_dict())
                    self_copy.target_network.load_state_dict(self.policy.state_dict())

            else:
                winner = self.one_game_expert(agent, env, episode, train=True)

                if episode % self.target_update == 0:
                    self.target.load_state_dict(self.policy.state_dict())

            if winner == self.player:
                rewards.append(1)
            elif winner == agent.player:
                rewards.append(-1)
            else:
                rewards.append(0)

            if episode % val_interval == 0:
                self.compute_metrics()

            if episode + 1 >= val_interval and (episode + 1) % val_interval == 0:
                avg_loss = np.mean(np.array(self.losses))
                self.losses = []
                avg_losses.append(avg_loss)
                if self.verbose:
                    avg_reward = np.mean(rewards[-250:])
                    print(
                        f'# {self.counter}:\navg_loss: {avg_loss}\navg_reward: {avg_reward}'
                    )

        if save_model:
            torch.save(self.policy.state_dict(), f'{ckpt_name}.pth')
            print(f"Save policy as {ckpt_name}.pth!")

        return rewards, avg_losses
