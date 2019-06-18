import attr
import numpy as np

from scipy.optimize import differential_evolution as de


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


def calculate_equilibrium(model, P, T, number_of_trial_phases, strategy='best1bin', popsize=35,
    recombination=0.95, mutation=0.6, tol=1e-2, rtol=1e-3, seed=np.random.RandomState(),
    workers=1, monitor=False, polish=True):
    """
    Given a mixture modeled by an EoS at a known PT-conditions, calculate the thermodynamical equilibrium.

    :param model:
    :param P:
    :param T:
    :param number_of_trial_phases:
    :param strategy:
    :param popsize:
    :param recombination:
    :param mutation:
    :param tol:
    :param rtol:
    :param seed:
    :param workers:
    :param monitor:
    :param polish:

    :return:
        The equilibrium result, providing the phase molar fractions, compositions in each phase and number of
        phases.
    :rtype: ResultEquilibrium
    """
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
    search_space = [(0, 1)] * n_components * number_of_trial_phases

    result = de(
        _calculate_gibbs_free_energy_reduced,
        bounds=search_space,
        args=[model, n_components, number_of_trial_phases, model, P, T],
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

    reduced_gibbs_free_energy = result.fun
    molar_composition_solution = result.x
    n_ij_solution = _transform_molar_vector_data_in_matrix(
        molar_composition_solution,
        n_components,
        number_of_trial_phases
    )
    x_ij_solution = _normalize_phase_molar_amount_to_fractions_and_transform_to_matrix(
        n_ij_solution,
        number_of_trial_phases
    )

    return x_ij_solution, reduced_gibbs_free_energy


def _calculate_gibbs_free_energy_reduced(N, number_of_components, number_of_phases, model, P, T):
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

    :param model:
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
    fugacities_matrix = _assemble_fugacity_matrix(x_ij, number_of_phases, model, P, T)
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


def _assemble_fugacity_matrix(X, number_of_phases, model, P, T):
    """
    Assembles the fugacity matrix f_ij.

    :param numpy.ndarray X:
        Vector ordered component molar fraction for each component, arranged in terms of phases. This array must
        be indexed as X[number_of_phases, number_of_components].

    :param int number_of_phases:
        :func:`See here <gibbs.equilibrium.calculate_gibbs_free_energy_reduced>`.

    :param model:
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
        phase_compressibility = model.calculate_Z_minimal_energy(P, T, phase_composition)
        phase_fugacities = model.calculate_fugacity(P, T, phase_composition, phase_compressibility)
        fugacity_matrix[phase, :] = phase_fugacities

    return fugacity_matrix


def _normalize_phase_molar_amount_to_fractions_and_transform_to_matrix(
    N, number_of_phases, tol=1e-5):
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
        N_phase_sum = N[phase, :].sum()
        if N_phase_sum > tol:
            phase_composition = N[phase, :] / N_phase_sum
            X[phase, :] = phase_composition
        else:
            X[phase, :] = np.zeros(N.shape[1])

    return X
