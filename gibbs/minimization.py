import attr
import types
from typing import Union
from enum import Enum
import numpy as np
from scipy.optimize import differential_evolution
import pygmo as pg


class OptimizationMethod(Enum):
    """
    Available optimization solvers.
    """
    SCIPY_DE = 1
    PYGMO_DE1220 = 2


@attr.s(auto_attribs=True)
class ScipyDifferentialEvolutionSettings:
    """
    Optional arguments to pass for SciPy's differential evolution caller.

    Members
    ----------------

    :ivar str strategy:
        The differential evolution strategy to use. Should be one of: - 'best1bin' - 'best1exp' - 'rand1exp' -
        'randtobest1exp' - 'currenttobest1exp' - 'best2exp' - 'rand2exp' - 'randtobest1bin' - 'currenttobest1bin' -
        'best2bin' - 'rand2bin' - 'rand1bin' The default is 'best1bin'.

    :ivar float recombination:
        The recombination constant, should be in the range [0, 1]. In the literature this is also known as the crossover
        probability, being denoted by CR. Increasing this value allows a larger number of mutants to progress into the
        next generation, but at the risk of population stability.

    :ivar float mutation:
        The mutation constant. In the literature this is also known as differential weight, being denoted by F. If
        specified as a float it should be in the range [0, 2].

    :ivar float tol:
        Relative tolerance for convergence, the solving stops when `np.std(pop) = atol + tol * np.abs(np.mean(population_energies))`,
        where and `atol` and `tol` are the absolute and relative tolerance respectively.

    :ivar int|numpy.random.RandomState seed:
        If `seed` is not specified the `np.RandomState` singleton is used. If `seed` is an int, a new
        `np.random.RandomState` instance is used, seeded with seed. If `seed` is already a `np.random.RandomState instance`,
        then that `np.random.RandomState` instance is used. Specify `seed` for repeatable minimizations.

    :ivar int workers:
        If `workers` is an int the population is subdivided into `workers` sections and evaluated in parallel
        (uses `multiprocessing.Pool`). Supply -1 to use all available CPU cores. Alternatively supply a map-like
        callable, such as `multiprocessing.Pool.map` for evaluating the population in parallel. This evaluation is
        carried out as `workers(func, iterable)`.

    :ivar bool disp:
        Display status messages during optimization iterations.

    :ivar polish:
        If True (default), then `scipy.optimize.minimize` with the `L-BFGS-B` method is used to polish the best
        population member at the end, which can improve the minimization slightly.
    """
    number_of_decision_variables: int
    strategy: str = 'best1bin'
    recombination: float = 0.3
    mutation: float = 0.6
    tol: float = 1e-5
    seed: Union[np.random.RandomState, int] = np.random.RandomState()
    workers: int = 1
    disp: bool = False
    polish: bool = True
    popsize: int = None
    population_size_for_each_variable: int = 15
    total_population_size_limit: int = 100

    def __attrs_post_init__(self):
        if self.popsize is None:
            self.popsize = self._estimate_population_size()
        elif self.popsize <= 0:
            raise ValueError('Number of individuals must be greater than 0.')
        if type(self.popsize) != int:
            raise TypeError('Population size must be an integer number.')
        if not 0 < self.recombination <= 1:
            raise ValueError('Recombination must be a value between 0 and 1.')
        if type(self.mutation) == tuple:
            mutation_dithering_array = np.array(self.mutation)
            if len(self.mutation) > 2:
                raise ValueError('Mutation can be a tuple with two numbers, not more.')
            if mutation_dithering_array.min() < 0 or mutation_dithering_array.max() > 2:
                raise ValueError('Mutation must be floats between 0 and 2.')
            elif mutation_dithering_array.min() == mutation_dithering_array.max():
                raise ValueError("Values for mutation dithering can't be equal.")
        else:
            if type(self.mutation) != int and type(self.mutation) != float:
                raise TypeError('When mutation is provided as a single number, it must be a float or an int.')
            if not 0 < self.mutation < 2:
                raise ValueError('Mutation must be a number between 0 and 2.')
        if self.tol < 0:
            raise ValueError('Tolerance must be a positive float.')

    def _estimate_population_size(self):
        population_size = self.population_size_for_each_variable * self.number_of_decision_variables
        if population_size > self.total_population_size_limit:
            population_size = self.total_population_size_limit

        return population_size


@attr.s(auto_attribs=True)
class PygmoSelfAdaptiveDESettings:
    # TODO: docs and validations

    gen: int
    popsize: int
    allowed_variants: list = [2, 6, 7]
    variant_adptv: int = 2
    ftol: float = 1e-5
    xtol: float = 1e-5
    memory: bool = True
    seed: int = int(np.random.randint(0, 2000))
    polish: bool = True
    polish_method: str = 'lbfgs'
    parallel_execution: bool = False
    number_of_islands: int = 2
    archipelago_gen: int = 50


@attr.s(auto_attribs=True)
class PygmoOptimizationProblemWrapper:
    # TODO: docs and validations

    objective_function: types.FunctionType
    bounds: list
    args: list = []

    def fitness(self, x):
        return [self.objective_function(x, *self.args)]

    def get_bounds(self):
        return self._transform_bounds_to_pygmo_standard

    def gradient(self, x):
        return pg.estimate_gradient(lambda x: self.fitness(x), x)

    @property
    def _transform_bounds_to_pygmo_standard(self):
        bounds_numpy = np.array(self.bounds, dtype=np.float64)
        lower_bounds = list(bounds_numpy[:, 0])
        upper_bounds = list(bounds_numpy[:, 1])
        return lower_bounds, upper_bounds


@attr.s(auto_attribs=True)
class PygmoSolutionWrapperSerial:
    # TODO: docs and validations

    solution: pg.core.population

    @property
    def fun(self):
        return self.solution.champion_f

    @property
    def x(self):
        return self.solution.champion_x


@attr.s(auto_attribs=True)
class PygmoSolutionWrapperParallel:
    # TODO: docs and validations

    champion_x: np.ndarray
    champion_f: Union[float, np.float64, np.ndarray]

    @property
    def fun(self):
        return self.champion_f

    @property
    def x(self):
        return self.champion_x


@attr.s(auto_attribs=True)
class OptimizationProblem:
    """
    This class stores and solve optimization problems with the available solvers.
    """
    # TODO: docs and validations
    objective_function: types.FunctionType
    bounds: list
    optimization_method: OptimizationMethod
    solver_args: Union[ScipyDifferentialEvolutionSettings, PygmoSelfAdaptiveDESettings]
    args: list = []

    def __attrs_post_init__(self):
        if self.optimization_method == OptimizationMethod.SCIPY_DE and self.solver_args is None:
            self.solver_args = ScipyDifferentialEvolutionSettings(self._number_of_decision_variables)

    @property
    def _number_of_decision_variables(self):
        return len(self.bounds)

    def solve_minimization(self):
        if self.optimization_method == OptimizationMethod.SCIPY_DE:
            result = differential_evolution(
                self.objective_function,
                bounds=self.bounds,
                args=self.args,
                strategy=self.solver_args.strategy,
                popsize=self.solver_args.popsize,
                recombination=self.solver_args.recombination,
                mutation=self.solver_args.mutation,
                tol=self.solver_args.tol,
                disp=self.solver_args.disp,
                polish=self.solver_args.polish,
                seed=self.solver_args.seed,
                workers=self.solver_args.workers
            )
            return result

        elif self.optimization_method == OptimizationMethod.PYGMO_DE1220:
            problem_wrapper = PygmoOptimizationProblemWrapper(
                objective_function=self.objective_function,
                bounds=self.bounds,
                args=self.args
            )
            pygmo_algorithm = pg.algorithm(
                pg.de1220(
                    gen=self.solver_args.gen,
                    # allowed_variants=self.solver_args.allowed_variants,
                    variant_adptv=self.solver_args.variant_adptv,
                    ftol=self.solver_args.ftol,
                    xtol=self.solver_args.xtol,
                    memory=self.solver_args.memory,
                    seed=self.solver_args.seed
                )
            )
            pygmo_problem = pg.problem(problem_wrapper)

            if self.solver_args.parallel_execution:
                solution_wrapper = self._run_pygmo_parallel(
                    pygmo_algorithm,
                    pygmo_problem,
                    number_of_islands=self.solver_args.number_of_islands,
                    archipelago_gen=self.solver_args.archipelago_gen
                )
            else:
                pygmo_solution = self._run_pygmo_serial(pygmo_algorithm, pygmo_problem)
                if self.solver_args.polish:
                    pygmo_solution = self._polish_pygmo_population(pygmo_solution)

                solution_wrapper = PygmoSolutionWrapperSerial(pygmo_solution)

            return solution_wrapper

        else:
            raise NotImplementedError('Unavailable optimization method.')

    @staticmethod
    def _select_best_pygmo_archipelago_solution(champions_x, champions_f):
        best_index = np.argmin(champions_f)
        return champions_x[best_index], champions_f[best_index]

    def _run_pygmo_parallel(self, algorithm, problem, number_of_islands=2, archipelago_gen=50):
        pygmo_archipelago = pg.archipelago(
            n=number_of_islands,
            algo=algorithm,
            prob=problem,
            pop_size=self.solver_args.popsize,
            seed=self.solver_args.seed
        )
        pygmo_archipelago.evolve(n=archipelago_gen)
        pygmo_archipelago.wait()
        champions_x = pygmo_archipelago.get_champions_x()
        champions_f = pygmo_archipelago.get_champions_f()
        champion_x, champion_f = self._select_best_pygmo_archipelago_solution(champions_x, champions_f)
        return PygmoSolutionWrapperParallel(champion_x=champion_x, champion_f=champion_f)

    def _run_pygmo_serial(self, algorithm, problem):
        population = pg.population(
            prob=problem,
            size=self.solver_args.popsize,
            seed=self.solver_args.seed
        )
        solution = algorithm.evolve(population)
        return solution

    def _polish_pygmo_population(self, population):
        pygmo_nlopt_wrapper = pg.nlopt(self.solver_args.polish_method)
        nlopt_algorithm = pg.algorithm(pygmo_nlopt_wrapper)
        solution_wrapper = nlopt_algorithm.evolve(population)
        return solution_wrapper
