import numpy as np
import attr

from gibbs.minimization import OptimizationProblem, OptimizationMethod, ScipyDifferentialEvolutionSettings


@attr.s(auto_attribs=True)
class StabilityResult:
    phase_split: bool
    x: np.ndarray
    reduced_tpd: float


def stability_test(
        model, P, T, z, strategy='best1bin', popsize=35, recombination=0.95, mutation=0.6,
        tol=1e-2, rtol=1e-3, seed=np.random.RandomState(), workers=1, monitor=False, polish=True
):
    if popsize <= 0:
        raise ValueError('Number of individuals must be greater than 0.')
    if type(popsize) != int:
        raise TypeError('Population size must be an integer number.')
    if not 0 < recombination <= 1:
        raise ValueError('Recombination must be a value between 0 and 1.')
    if type(mutation) == tuple:
        mutation_dithering_array = np.array(mutation)
        if len(mutation) > 2:
            raise ValueError('Mutation can be a tuple with two numbers, not more.')
        if mutation_dithering_array.min() < 0 or mutation_dithering_array.max() > 2:
            raise ValueError('Mutation must be floats between 0 and 2.')
        elif mutation_dithering_array.min() == mutation_dithering_array.max():
            raise ValueError("Values for mutation dithering can't be equal.")
    else:
        if type(mutation) != int and type(mutation) != float:
            raise TypeError('When mutation is provided as a single number, it must be a float or an int.')
        if not 0 < mutation < 2:
            raise ValueError('Mutation must be a number between 0 and 2.')
    if tol < 0 or rtol < 0:
        raise ValueError('Tolerance must be a positive float.')

    n_components = model.number_of_components
    search_space = [(0, 1)] * n_components
    f_z = model.fugacity(P, T, z)

    scipy_differential_evolution_optional_args = ScipyDifferentialEvolutionSettings(
        number_of_decision_variables=n_components,
        strategy=strategy,
        popsize=popsize,
        recombination=recombination,
        mutation=mutation,
        tol=tol,
        disp=monitor,
        polish=polish,
        seed=seed,
        workers=workers
    )

    optimization_problem = OptimizationProblem(
        objective_function=_reduced_tpd,
        bounds=search_space,
        args=[model, P, T, f_z],
        optimization_method=OptimizationMethod.SCIPY_DE,
        solver_args=scipy_differential_evolution_optional_args
    )

    result = optimization_problem.solve_minimization()
    x = result.x / result.x.sum()
    reduced_tpd = result.fun

    if np.allclose(x, z, rtol=rtol):  # This criterion could be separated in a function itself
        # with a norm
        phase_split = False
    elif np.abs(reduced_tpd) < 1e-10:
        phase_split = False
    elif reduced_tpd < 0.:
        phase_split = True
    else:
        phase_split = False

    stability_test_result = StabilityResult(
        phase_split=phase_split,
        x=x,
        reduced_tpd=reduced_tpd
    )

    return stability_test_result


def _reduced_tpd(n, model, P, T, f_z):
    try:
        x = n / n.sum()
        f_x = model.fugacity(P, T, x)
        tpd = np.sum(x * (np.log(f_x / f_z)))
    except TypeError:
        tpd = np.inf

    return tpd
