import pytest
import numpy as np

from gibbs.minimization import PygmoSelfAdaptiveDESettings, OptimizationProblem
from gibbs.minimization import OptimizationMethod, ScipyDifferentialEvolutionSettings


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


@pytest.mark.parametrize("problem_dimension", range(2, 5))
def test_pygmo_sade_rosenbrock_minimization(problem_dimension):
    bounds = problem_dimension * [[-6, 6]]
    solver_settings = PygmoSelfAdaptiveDESettings(
        gen=1000,
        popsize=60
    )

    problem = OptimizationProblem(
        objective_function=f_rosenbrock,
        bounds=bounds,
        optimization_method=OptimizationMethod.PYGMO_SADE,
        solver_args=solver_settings
    )

    solution = problem.solve_minimization()

    assert pytest.approx(np.ones(problem_dimension), rel=1e-3) == solution.x


@pytest.mark.parametrize("problem_dimension", range(2, 5))
def test_scipy_de_rosenbrock_minimization(problem_dimension):
    bounds = problem_dimension * [[-6, 6]]
    solver_settings = ScipyDifferentialEvolutionSettings(
        number_of_decision_variables=problem_dimension
    )

    problem = OptimizationProblem(
        objective_function=f_rosenbrock,
        bounds=bounds,
        optimization_method=OptimizationMethod.SCIPY_DE,
        solver_args=solver_settings
    )

    solution = problem.solve_minimization()

    assert pytest.approx(np.ones(problem_dimension), rel=1e-3) == solution.x
