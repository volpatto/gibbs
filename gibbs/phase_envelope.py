import attr

from gibbs.mixture import Mixture
from gibbs.models.ceos import CEOS, PengRobinson78


@attr.s
class PhaseEnvelope:
    """
    Storage phase envelope data.
    """


def calculate_phase_envelope_grid(mix, eos, P_min, P_max, T_min, T_max, n_points_P=30, n_points_T=30):
    """
    Calculate phase envelope, in a grid, for given ranges for pressure and temperature.

    :param Mixture mix:
        A mixture.

    :param CEOS|PengRobinson78 eos:
        Equation of State.

    :param float P_min:
        Minimum pressure value to generate the envelope.

    :param float P_max:
        Maximum pressure value to generate the envelope.

    :param float T_min:
        Minimum temperature value to generate the envelope.

    :param float T_max:
        Maximum temperature value to generate the envelope.

    :param int n_points_P:
        Number of points for the pressure interval.

    :param int n_points_T:
        Number of points for the temperature interval.

    :return:
        Phase envelope data containing the information about the generated points and which phases are
        present in each PT-point.
    :rtype: PhaseEnvelope
    """
    raise NotImplementedError('Equilibrium calculation is not implemented yet.')
