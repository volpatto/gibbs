import attr
import numpy as np

from gibbs.mixture import Mixture
from gibbs.models.ceos import CEOS, PengRobinson78, SoaveRedlichKwong


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
    :param CEOS|PengRobinson78|SoaveRedlichKwong eos:
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


def gibbs_free_energy_reduced(N, number_of_components, number_of_phases, mix, eos, P, T):
    n_ij = _transform_molar_vector_data_in_matrix(N, number_of_components, number_of_phases)
    reduced_gibbs_energy = 0.0
    # for j in range(number_of_phases):
    #     reduced_gibbs_energy +=
    return


def _transform_molar_vector_data_in_matrix(N, number_of_components, number_of_phases):
    """
    Given a vector N containing ordered component molar amount for each component, arranged in terms of phases,
    return a matrix where its lines represents phases and columns are the components. In more details,
    N = (n_1, n_2, ..., n_np)^T, where n_j here is the molar composition vector for each component in the phase j.

    :param numpy.ndarray N:
        Vector ordered component molar amount for each component, arranged in terms of phases.

    :param int number_of_components:
        Number of components in the mixture.

    :param int number_of_phases:
        Expected number of phases in the mixture.

    :return:
        A matrix where its lines represents phases and columns are the components.
    :rtype: numpy.ndarray
    """
    if N.size != number_of_phases * number_of_components:
        raise ValueError('Unexpected amount of storage for amount of mols in each phase.')

    n_ij = N.reshape((number_of_phases, number_of_components))
    return n_ij
