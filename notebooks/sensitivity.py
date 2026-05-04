
from typing import Any, Callable
from SALib.sample import morris as morris_sampler
from SALib.analyze import morris as morris_analyzer
from jax import vmap
from jax import numpy as jnp
import numpy as np

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


def global_sensitivity(
    map_model: Callable[[jnp.ndarray], jnp.ndarray],
    problem_space: dict[str, Any],
    n_trajectories: int
):
    """
    Calculate Morris sensitivies for a given model and parameter space.

    For Morris, mean and standard deviation of elementary effects are returned
    at each time point

    Parameters
    ----------
    map_model : Callable[[float], float]
        Callable for mapping input params to model output
    problem_space : dict[str, Any]
        Dictionary defining the problem space with the following minimum keys:
            'num_vars' : int,
            'names': list[str],
            'bounds': list[list[ScalarLike]]

    n_trajectories : int
        Number of trajectories
    """

    param_vectors = morris_sampler.sample(problem_space, n_trajectories)
    #y = vmap(map_model)(param_vectors)
    y = np.array([map_model(p) for p in param_vectors])

    if len(y.shape) > 2:
        y = y.squeeze(-1)

    sensitivity = [morris_analyzer.analyze(problem_space, param_vectors, Y, scaled=True) for Y in y.T]

    return y, sensitivity

def plot_mu(time: jnp.ndarray, mus: jnp.ndarray, confs: jnp.ndarray, labels: list[str]) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots()

    for index, label in enumerate(labels):
        plt.plot(time, mus[:, index], label=label)
        plt.fill_between(
            time,
            mus[:,index] - confs[:,index],
            mus[:,index] + confs[:,index],
            alpha=0.1
        )

    return (fig, ax)

def plot_sigma(time: jnp.ndarray, sigmas: jnp.ndarray, labels: list[str]) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots()

    for index, label in enumerate(labels):
        plt.plot(time, sigmas[:, index], label=label)

    return (fig, ax)