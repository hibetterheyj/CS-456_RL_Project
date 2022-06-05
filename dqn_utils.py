import matplotlib.pyplot as plt
from pathlib import Path
import typing
import numpy as np
import torch
import yaml


def position2move(action):
    """Converts the tuple encoding of a board position to integer value."""
    return action[0] * 3 + action[1]


def load_yaml(path: typing.Union[str, Path]):
    with open(path) as file:
        res = yaml.load(file, Loader=yaml.FullLoader)
    return res


def grid2state(
    grid: np.array,
    player: bool = False,
    vec: bool = True,
) -> torch.tensor:
    """If not switched, 1 and -1 on the grid means X and O, and vice versa

    Args:
        grid (np.array): _description_
        player (bool, optional): _description_. Defaults to False.
        vec (bool, optional): _description_. Defaults to True.

    Returns:
        np.array: _description_
    """

    state = np.zeros((3, 3, 2))

    if player == 'X':
        state[:, :, 0] = (grid == 1).astype(float)
        state[:, :, 1] = (grid == -1).astype(float)
    else:
        state[:, :, 0] = (grid == -1).astype(float)
        state[:, :, 1] = (grid == 1).astype(float)

    if vec:
        state = state.reshape(1, -1)

    return torch.tensor(state, dtype=torch.float)


def grid_to_state(
    grid: np.array,
    switch: bool = False,
    vec: bool = True,
) -> torch.tensor:
    """If not switched, 1 and -1 on the grid means X and O, and vice versa

    Args:
        grid (np.array): _description_
        switch (bool, optional): _description_. Defaults to False.
        vec (bool, optional): _description_. Defaults to True.

    Returns:
        np.array: _description_
    """

    state = np.zeros((3, 3, 2))

    if not switch:
        state[:, :, 0] = (grid == 1).astype(float)
        state[:, :, 1] = (grid == -1).astype(float)
    else:
        state[:, :, 0] = (grid == -1).astype(float)
        state[:, :, 1] = (grid == 1).astype(float)

    if vec:
        state = state.reshape(1, -1)

    return torch.tensor(state, dtype=torch.float)


def decreasing_exploration(
    n_step: int,
    n_star: int = 20000,
    e_max: float = 0.8,
    e_min: float = 0.1,
):
    return max(e_min, e_max * (1 - n_step / n_star))


def window_avg_plot(axes, var: np.array, window_sz: int = 250) -> np.array:
    """Computes the average over successive windows of an array and plot correspoding plot

    Args:
        axes (_type_): _description_
        var (np.array): _description_
        window_sz (int, optional): _description_. Defaults to 250.

    Returns:
        np.array: _description_
    """

    var_np = np.array(var)
    avg_var = np.empty((int(var_np.shape[0] / window_sz),))
    for i in range(0, int(var_np.shape[0] / window_sz)):
        avg_var[i] = var_np[window_sz * i : window_sz * (i + 1) - 1].sum() / window_sz
    episodes = np.linspace(0, avg_var.shape[0], avg_var.shape[0]) * window_sz
    axes.plot(episodes, avg_var)


#%% test cases for given function
def test_grid_to_state():
    from tic_env import TictactoeEnv

    env = TictactoeEnv()
    grid, _, _ = env.reset()
    grid, _, _ = env.step(2)
    grid, _, _ = env.step(3)
    grid, _, _ = env.step(5)
    print(grid)
    test_state1 = grid_to_state(grid, vec=False)
    # print(test_state1[:,:,0], test_state1[:,:,1])
    test_state2 = grid_to_state(grid, switch=True, vec=False)
    # print(test_state2[:,:,0], test_state2[:,:,1])
    assert np.array_equal(test_state1[:, :, 0], test_state2[:, :, 1])
    assert np.array_equal(test_state2[:, :, 0], test_state1[:, :, 1])


def test_window_avg_plot():
    import numpy as np
    import matplotlib.pyplot as plt

    _, ax = plt.subplots(figsize=(10, 4))
    rewards_example = np.linspace(-1, 1, 20000) + np.random.rand(20000)
    window_avg_plot(axes=ax, var=rewards_example, window_sz=20)
