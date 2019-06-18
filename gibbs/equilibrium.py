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
        A mixture setup.

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


def _calculate_gibbs_free_energy_reduced(N, number_of_components, number_of_phases, eos, P, T):
    """
    Calculates the reduced Gibbs free energy. This function is used as objective function in the minimization procedure
    for equilibrium calculations.

    :param numpy.ndarray N:
        Vector ordered component molar amount for each component, arranged in terms of phases. This array must
        be indexed as N[number_of_phases, number_of_components].

    :param int number_of_components:
        Number of components in the mixture.

    :param int number_of_phases:
        Number of trial phases.

    :param CEOS|PengRobinson78|SoaveRedlichKwong eos:
        Equation of State or model.

    :param float P:
        Pressure value.

    :param float T:
        Temperature value.

    :return:
        The evaluation of reduced Gibbs free energy.
    :rtype: float
    """
    if N.size != number_of_phases * number_of_components:
        raise ValueError('Unexpected amount of storage for amount of mols in each phase.')

    n_ij = _transform_molar_vector_data_in_matrix(N, number_of_components, number_of_phases)
    x_ij = _normalize_phase_molar_amount_to_fractions_and_transform_to_matrix(n_ij, number_of_phases)
    fugacities_matrix = _assemble_fugacity_matrix(x_ij, number_of_phases, eos, P, T)
    ln_f_matrix = np.log(fugacities_matrix)

    return np.tensordot(n_ij, ln_f_matrix)


def _transform_molar_vector_data_in_matrix(N, number_of_components, number_of_phases):
    """
    Given a vector N containing ordered component molar amount for each component, arranged in terms of phases,
    return a matrix where its lines represents phases and columns are the components. In more details,
    N = (n_1, n_2, ..., n_np)^T, where n_j here is the molar composition vector for each component in the phase j.

    :param numpy.ndarray N:
        :func:`See here <gibbs.equilibrium.calculate_gibbs_free_energy_reduced>`.

    :param int number_of_components:
        :func:`See here <gibbs.equilibrium.calculate_gibbs_free_energy_reduced>`.

    :param int number_of_phases:
        :func:`See here <gibbs.equilibrium.calculate_gibbs_free_energy_reduced>`.

    :return:
        A matrix where its lines represents phases and columns are the components.
    :rtype: numpy.ndarray
    """
    n_ij = N.reshape((number_of_phases, number_of_components))
    return n_ij


def _assemble_fugacity_matrix(X, number_of_phases, eos, P, T):
    """
    Assembles the fugacity matrix f_ij.

    :param numpy.ndarray X:
        Vector ordered component molar fraction for each component, arranged in terms of phases. This array must
        be indexed as X[number_of_phases, number_of_components].

    :param int number_of_phases:
        :func:`See here <gibbs.equilibrium.calculate_gibbs_free_energy_reduced>`.

    :param CEOS|PengRobinson78|SoaveRedlichKwong eos:
        :func:`See here <gibbs.equilibrium.calculate_gibbs_free_energy_reduced>`.

    :param float P:
        :func:`See here <gibbs.equilibrium.calculate_gibbs_free_energy_reduced>`.

    :param float T:
        :func:`See here <gibbs.equilibrium.calculate_gibbs_free_energy_reduced>`.

    :return:
        The matrix f_ij containing all the fugacities for each component in each phase.
    :rtype: numpy.ndarray
    """
    fugacity_matrix = np.zeros(X.shape)
    for phase in range(number_of_phases):  # Unfortunately, this calculation can not be vectorized
        phase_composition = X[phase, :]
        phase_compressibility = eos.calculate_Z_minimal_energy(P, T, phase_composition)
        phase_fugacities = eos.calculate_fugacity(P, T, phase_composition, phase_compressibility)
        fugacity_matrix[phase, :] = phase_fugacities

    return fugacity_matrix


def _normalize_phase_molar_amount_to_fractions_and_transform_to_matrix(N, number_of_phases):
    """
    Converts the component molar amount matrix entries to molar fractions.

    :param numpy.ndarray N:
        :func:`See here <gibbs.equilibrium.calculate_gibbs_free_energy_reduced>`.

    :param int number_of_phases:
        :func:`See here <gibbs.equilibrium.calculate_gibbs_free_energy_reduced>`.

    :return:
        The matrix X_ij containing all the mole fractions for each component in each phase.
    :rtype: numpy.ndarray
    """
    X = np.zeros(N.shape)
    for phase in range(number_of_phases):  # This loop could be optimized
        phase_composition = N[phase, :] / N[phase, :].sum()
        X[phase, :] = phase_composition

    return X
