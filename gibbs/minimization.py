import attr
import types
from enum import Enum


class OptimizationSolver(Enum):
    SCIPY_DE = 1


@attr.s(auto_attribs=True)
class OptimizationProblem:
    objective_function: types.FunctionType
    bounds: list
    args: list
    solver: OptimizationSolver
