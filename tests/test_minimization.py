import pytest
import pygmo as pg

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


@pytest.mark.parametrize("problem_dimension", [2, 3, 4])
def test_pygmo_rosenbrock_minimization(problem_dimension):
    bounds = problem_dimension * [[-6, 6]]
    problem_wrapper = PygmoOptimizationProblemWrapper(
        objective_function=f_rosenbrock,
        bounds=bounds
    )
    algo = pg.algorithm(pg.sade(gen=500))
    prob = pg.problem(problem_wrapper)
    pop = pg.population(prob, 20)
    pop = algo.evolve(pop)

    uda = algo.extract(pg.sade)
    print(uda.get_log())
    assert True
