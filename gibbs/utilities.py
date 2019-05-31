import numpy as np


def convert_psi_to_Pa(P):
    """
    Convert pressures in psi to Pa.

    :param float|numpy.ndarray P:
        Pressure values in psi.

    :return:
        Pressure values in Pa.
    :rtype: float|numpy.ndarray
    """
    return 6894.7572931783 * P


def convert_F_to_K(T):
    """
    Convert temperatures in F to K.

    :param float|numpy.ndarray T:
        Temperature values in F.

    :return:
        Temperature values in K.
    :rtype: float|numpy.ndarray
    """
    return (T + 459.67) * (5. / 9.)


def convert_bar_to_Pa(P):
    """
    Convert pressure in Pa units to bar.

    :param float|numpy.ndarray P:
        Pressure value(s) in Pa.

    :return:
        Pressure value(s) in bar.
    :rtype float|numpy.ndarray
    """
    return P * 1e5
