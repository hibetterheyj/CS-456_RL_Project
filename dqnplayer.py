# std
import random
import logging
from re import S
from typing import List, Dict, Tuple
from collections import namedtuple, deque

# imported
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# custom
from tic_env import TictactoeEnv, OptimalPlayer
from dqn_utils import grid_to_state, decreasing_exploration

# ref: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#replay-memory
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

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


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def last(self):
        return self.memory[-1]

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, n_inputs: int = 18, n_outputs: int = 9) -> None:
        """_summary_

        Args:
            n_inputs (int, optional): _description_. Defaults to 18.
            n_outputs (int, optional): _description_. Defaults to 9.
        """
        super(DQN, self).__init__()
        self.i2h = nn.Linear(n_inputs, 128)
        self.hid = nn.Linear(128, 128)
        self.h2o = nn.Linear(128, n_outputs)

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
        device: torch.device,
        eps: float = 2,
        player: str = 'X',
        memory_capacity: int = 10000,
        explore=False,
    ) -> None:
        self.player = player  # 'X' or 'O'
        self.device = device
        self.memory = ReplayMemory(memory_capacity)
        self.losses = []
        self.policy = DQN(n_inputs=18, n_outputs=9).to(self.device)
        self.target = DQN(n_inputs=18, n_outputs=9).to(self.device)
        self.epsilon = eps
        self.explore = explore
        self.val_metrics = {'opt': [], 'rand': []}
        self.counter = 0
        self.last_state = None
        self.last_action = None
        self.last_reward = None

    def empty(self, grid):
        '''return all empty positions'''
        avail = []
        for i in range(9):
            pos = (int(i / 3), i % 3)
            if grid[pos] == 0:
                avail.append(pos)
        return avail

    def random_move(self, grid):
        """Chose a random move from the available options."""
        avail = self.empty(grid)
        random_pos = avail[random.randint(0, len(avail) - 1)]
        return random_pos[0] * 3 + random_pos[1]

    def load_model(self, path: str):
        model_state_dict = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(model_state_dict)
        self.policy.eval()

    def act(self, grid):

        if self.player == 'X':
            switch = False
        else:
            switch = True
        state = grid_to_state(grid, switch=switch, vec=True).to(self.device)

        if self.explore:
            self.epsilon = self.decreasing_exploration(
                self.e_min, self.e_max, self.n_star, self.counter
            )

        # epsilon greedy policy w.r.t. Q-values
        if random.random() < self.epsilon:
            action = self.random_move(grid)
        else:
            with torch.no_grad():
                pred = F.softmax(self.policy.forward(state), dim=0).cpu().numpy()
                action = pred.argmax()

        self.last_action = action
        self.last_state = state

        return action

    def act_valid(self, grid: np.array):
        if self.player == 'X':
            switch = False
        else:
            switch = True
        state = grid_to_state(grid, switch=switch, vec=True).to(self.device)
        with torch.no_grad():
            pred = F.softmax(self.policy.forward(state), dim=0).cpu().numpy()
            valid_moves = (
                state.cpu().numpy().reshape(3, 3, 2).sum(axis=2).reshape(-1) == 0
            )
            move = (valid_moves * pred).argmax()
            return int(move)

    # TODO:
    def play_game(self, agent, env, episode_idx, train=True):
        grid, end, __ = env.observe()
        if episode_idx % 2 == 0:
            self.player = 'X'
            agent.player = 'O'
        else:
            self.player = 'O'
            agent.player = 'X'
        while end == False:
            if env.current_player == self.player:
                move = self.act(grid)
                if env.check_valid(move.item()):
                    grid, end, winner = env.step(move, print_grid=False)
                else:
                    end = True
                    env.winner = agent.player
                    winner = agent.player
            else:
                move = agent.act(grid)
                grid, end, winner = env.step(move, print_grid=False)
                if train and not end:
                    reward = env.reward(self.player)
                    self.memory.update(
                        self.last_state, self.last_action, reward, grid.copy()
                    )
        if train:
            reward = env.reward(self.player)
            self.memory.push(
                self.last_state,
                self.last_action,
                next_state,
                torch.tensor([reward], device=self.device),
            )
        return winner

    def play_n_games(self, agent, env, episodes, train=True):
        # set exploration to 0 in test environment
        if not train:
            temp_eps = self.epsilon
            temp_explore = self.explore
            self.epsilon = 0
            self.explore = False

        n_win, n_los, n_draw = 0, 0, 0
        for j in range(episodes):
            env.reset()
            winner = self.play_game(agent, env, j, train=False)

            if winner == self.player:
                n_win += 1
            elif winner == agent.player:
                n_los += 1
            else:
                n_draw += 1

        # set exploration back to
        self.epsilon = temp_eps
        self.explore = temp_explore

        res_info = {
            'win': n_win,
            'los': n_los,
            'draw': n_draw,
            'measure': (n_win - n_los) / episodes,
        }

        return res_info

    def select_action(
        self,
        eps: float,
        state: torch.Tensor,
        nr_action: int = 9,
    ) -> Tuple[torch.tensor, bool]:
        """select action given current state

        Args:
            eps (float): _description_
            policy (nn.Module): _description_
            state (torch.Tensor): _description_
            device (torch.device): _description_
            nr_action (int, optional): Possible action in the game. Defaults to 9.

        Returns:
            Tuple[torch.tensor, bool]: _description_
        """

        if random.random() > eps:
            return self.policy.act(state), False
        else:
            return (
                torch.tensor(
                    [[random.randrange(nr_action)]],
                    device=self.device,
                ),
                True,
            )

    def train_per_game(
        self,
        n_episode: int = 20000,
        batch_size: int = BATCH_SIZE,
        gamma: float = GAMMA,
        target_update: int = TARGET_UPDATE,
        val_interval: int = VAL_INTERVAL,
        logging_size=2000,
        save_res: bool = True,
        ckpt_name: str = 'policy.pth',
        expert=OptimalPlayer(epsilon=0.5),
        debug_mode=True,
    ) -> List:

        logging.info("Beginning training on: {}".format(self.device))

        # history
        history = []

        # Env
        env = TictactoeEnv()

        # Adam optimizer
        optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Huber loss (delte=1 in SmoothL1Loss)
        criterion = nn.SmoothL1Loss()

        for i in tqdm(range(n_episode)):
            env.reset()
            winner = self.play_game(expert, env, i)

            if self.memory.memory_used > self.batch:
                # train phase

                # sample mini-batch from memory
                state, action, reward, next_state = self.memory.sample(self.batch)

                # calculate "prediction" Q values
                output = self.DQN.forward(state)
                y_pred = output.gather(1, action.long().view((self.batch, 1))).view(-1)

                # calculate "target" Q-values from Q-Learning update rule
                reward_indicator = (
                    ~reward.abs().bool()
                ).int()  # invert rewards {0 -> 1, {-1,1} -> 0}
                y_target = self.gamma * (
                    reward
                    + reward_indicator
                    * self.target_network(next_state).max(dim=1).values
                )

                # forward + backward + optimize
                loss = self.DQN.criterion(y_pred, y_target)
                self.DQN.optimizer.zero_grad()
                loss.backward()
                self.DQN.optimizer.step()
                self.losses.append(loss.detach().numpy())

            # update target network
            if i % self.update_target == 0:
                self.target_network.load_state_dict(self.DQN.state_dict())

            if i % val_interval == 0:
                # METRIC_DICT = {'opt': 0.0, 'rand': 1.0}
                for (mode, epsilon) in METRIC_DICT.items():
                    player = OptimalPlayer(epsilon=epsilon)
                    res_info = self.play_n_games(player, env, episodes=500, train=False)
                    self.val_metrics[mode].append(res_info['measure'])
                    if debug_mode:
                        logging.debug("# Eval with Opt({})".format(epsilon))
                        print(
                            'M{} = {}, Win/Loss/Draw = {}'.format(
                                mode,
                                res_info['measure'],
                                res_info['win'] / 500,
                                res_info['los'],
                                res_info['draw'],
                            )
                        )

            # save results
            if winner == self.player:
                history.append(1)
            elif winner == expert.player:
                history.append(-1)
            else:
                history.append(0)

        logging.info("Complete")
        if save_res:
            torch.save(self.policy.state_dict(), 'policy.pth')
            logging.info("Save policy as {}!".format(ckpt_name))

        return history

    def optimize(
        self,
        batch_size: int,
        gamma: float,
        optimizer: optim.Optimizer,
        criterion: torch.nn,
        replay: bool = True,
    ):
        """Model optimization step, borrow from the Torch DQN tutorial.

        Arguments:
            batch_size {int} -- Number of observations to use per batch step
            gamma {float} -- Reward discount factor
            optimizer {torch.optim.Optimizer} -- Optimizer
            criterion {torch.nn} -- Loss
            policy {nn.Module} -- Policy net
            target {nn.Module} -- Target net
            replay {bool} -- Use replay buffer for not
        """
        if replay:
            if len(self.memory) < batch_size:
                return
            transitions = self.memory.sample(batch_size)
            # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
            # detailed explanation). This converts batch-array of Transitions
            # to Transition of batch-arrays.
        else:
            # update the network by using only the latest transition
            transitions = [self.memory.last()]
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool,
        )
        non_final_next_states_ = [s for s in batch.next_state if s is not None]
        # TODO: check correct or not with batch_size = 1
        if len(non_final_next_states_) == 0:
            return
        non_final_next_states = torch.cat(non_final_next_states_)
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy
        state_action_values = self.policy(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(batch_size, device=self.device)
        next_state_values[non_final_mask] = (
            self.target(non_final_next_states).max(1)[0].detach()
        )
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * gamma) + reward_batch

        # Compute Huber loss
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        for param in self.policy.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()
        self.losses.append(loss.detach().numpy())

    def train_per_step(
        self,
        n_episode: int = 20000,
        batch_size: int = BATCH_SIZE,
        gamma: float = GAMMA,
        target_update: int = TARGET_UPDATE,
        explore: bool = False,
        replay: bool = True,
        n_star: int = N_STAR,
        e_max: float = EPS_MAX,
        e_min: float = EPS_MIN,
        lr: float = START_LR,
        logging_size=2000,
        save_res: bool = True,
        ckpt_name: str = 'policy.pth',
        expert=OptimalPlayer(epsilon=0.5),
        seed=None,
    ) -> List:

        logging.info("Beginning training on: {}".format(self.device))

        # Env
        env = TictactoeEnv()

        # Network
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()

        # Adam optimizer
        optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Huber loss (delte=1 in SmoothL1Loss)
        criterion = nn.SmoothL1Loss()

        # state = torch.tensor([], dtype=torch.float).to(self.device)
        info = {
            "total": 0,
            "illegals": 0,
            "tie_g": 0,
            "win_g": 0,
            "los_g": 0,
            "win_rate": 0.0,
            "eps": 0.0,
            "loss": [],
        }
        summaries = []

        for episode in tqdm(range(n_episode)):
            switch = episode % 2
            if switch:
                player1, player2 = 'O', 'X'
            else:
                player1, player2 = 'X', 'O'
            expert.player = player2

            grid, end, _ = env.reset()

            if explore:
                eps = decreasing_exploration(episode, n_star, e_max, e_min)
            else:
                eps = e_max

            # Take first step by expert
            if env.num_step == 0 and switch:
                move = expert.act(grid)
                grid, end, winner = env.step(move, print_grid=False)

            while not end:
                # Select and perform an action from DQN
                state = grid_to_state(grid, switch=switch, vec=True).to(self.device)
                action, _ = self.select_action(eps, state)
                if env.check_valid(action.item()):
                    grid, end, winner = env.step(action.item(), print_grid=False)
                    reward = env.reward(player=player1)
                else:
                    reward = R_UNAV  # -1
                    end = True
                    info["illegals"] += 1
                    winner = player2
                    # print(episode, "Illegal moves")

                if not end:
                    move = expert.act(grid)
                    grid, end, winner = env.step(move, print_grid=False)
                    # Observe new state
                    next_state = grid_to_state(grid, switch=switch, vec=True).to(
                        self.device
                    )
                else:
                    next_state = None

                self.memory.push(
                    state,
                    action,
                    next_state,
                    torch.tensor([reward], device=self.device),
                )

                self.optimize(
                    batch_size=batch_size,
                    gamma=gamma,
                    optimizer=optimizer,
                    criterion=criterion,
                    replay=replay,
                )

            # update summary
            info['total'] = episode + 1
            if winner == None:
                info['tie_g'] += 1
            elif winner == player1:
                info['win_g'] += 1
            else:
                info['los_g'] += 1
            info['win_rate'] = info['win_g'] / info['total']
            info['eps'] = eps
            # info['loss'] = loss
            summaries.append(info)

            if episode and ((episode + 1) % target_update) == 0:
                self.target.load_state_dict(self.policy.state_dict())
            if episode and ((episode + 1) % logging_size) == 0:
                # print(info)
                [print(key, ':', value) for key, value in info.items()]

        logging.info("Complete")
        if save_res:
            torch.save(self.policy.state_dict(), 'policy.pth')
            logging.info("Save policy as {}!".format(ckpt_name))

        return summaries
