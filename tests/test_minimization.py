import pytest
import pygmo as pg
import numpy as np

from gibbs.minimization import PygmoOptimizationProblemWrapper, PygmoSelfAdaptativeDESettings


def f_rosenbrock(x):
    """
    Define the benchmark Rosenbrock function.
    :param numpy.ndarray x:
        The function's argument array.
    :return:
        The evaluated function at the given input array.
    :rtype: numpy.float64
    """
    dim = len(x)
    f = 0.0
    for i in range(dim-1):
        left_term = 100. * (x[i + 1] - x[i] * x[i]) * (x[i + 1] - x[i] * x[i])
        right_term = (1. - x[i]) * (1. - x[i])
        f += left_term + right_term
    return f


@pytest.mark.parametrize("problem_dimension", range(2, 8))
def test_pygmo_rosenbrock_minimization(problem_dimension):
    bounds = problem_dimension * [[-6, 6]]
    problem_wrapper = PygmoOptimizationProblemWrapper(
        objective_function=f_rosenbrock,
        bounds=bounds
    )
    algo = pg.algorithm(pg.sade(gen=1000, xtol=1e-8))
    prob = pg.problem(problem_wrapper)
    pop = pg.population(prob, 20)
    solution = algo.evolve(pop)

    sol_x = solution.champion_x

    assert pytest.approx(np.ones(problem_dimension), rel=1e-3) == sol_x
