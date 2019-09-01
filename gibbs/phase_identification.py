import attr

from gibbs._cpp_wrapper import Mixture
from gibbs._cpp_wrapper import PengRobinson78


@attr.s
class ResultPhaseIdentification:
    """
    Class which stores the results for phase identification procedures.

    :param num_of_phases:
        Amount of phases in equilibrium.
    :type num_of_phases: float

    :param phases:
        The phase which are present. Can be vapor, liquid or solid and combinations.
    :type num_of_phases: dict
    """
    num_of_phases = attr.ib(type=float)
    phases = attr.ib(type=dict)


def estimate_phases(mix, eos, P, T):
    """
    Given a mixture modeled by an EoS at a known PT-conditions, estimate which phases are present.

    :param Mixture mix:
        An input mixture.
    :param CEOS|PengRobinson78 eos:
        Equation of State.
    :param float P:
        Pressure value.
    :param float T:
        Temperature value.

    :return:
        Phases in equilibrium and the amount of phases..
    :rtype: ResultPhaseIdentification
    """
    raise NotImplementedError('Phase identification is not implemented yet.')
