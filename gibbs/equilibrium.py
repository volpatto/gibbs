import attr
import numpy as np

from gibbs.mixture import Mixture
from gibbs.models.ceos import CEOS, PengRobinson78


@attr.s
class ResultEquilibrium:
    """
    Class for storage the results from equilibrium calculations.

    :param F:
        Phase molar fractions.
    :type F: numpy.ndarray

    :param X:
        Component molar fractions in each phase.
    :type X: numpy.ndarray

    :param num_of_phases:
        Number of present phases.
    :type num_of_phases: float
    """
    F = attr.ib(type=np.array)
    X = attr.ib(type=np.array)
    num_of_phases = float


def calculate_equilibrium(mix, eos, P, T):
    """
    Given a mixture modeled by an EoS at a known PT-conditions, calculate the thermodynamical equilibrium.

    :param Mixture mix:
        A mixture.
    :param CEOS|PengRobinson78 eos:
        Equation of State.
    :param float P:
        Pressure value.
    :param float T:
        Temperature value.

    :return:
        The equilibrium result, providing the phase molar fractions, compositions in each phase and number of
        phases.
    :rtype: ResultEquilibrium
    """
    raise NotImplementedError('Equilibrium calculation is not implemented yet.')
