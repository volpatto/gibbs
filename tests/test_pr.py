import pytest
import numpy as np

from gibbs.models.ceos import PengRobinson78
from gibbs.mixture import Mixture


@pytest.fixture
def mixture_whitson():
    z = np.array([0.5, 0.42, 0.08])
    omegas = np.array([0.0115, 0.1928, 0.4902])
    Tcs = np.array([190.556, 425.16667, 617.666667])
    Pcs = np.array([4604318.9, 3796942.8, 2.096e6])
    return Mixture(z=z, Tc=Tcs, Pc=Pcs, acentric_factor=omegas)


@pytest.fixture
def eos_whitson(mixture_whitson):
    kijs = np.zeros((3, 3))
    return PengRobinson78(mixture=mixture_whitson, bip=kijs)


def test_eos_parameters_whitson_ex18(eos_whitson):
    # Temperature and Pressure in Pa and K, respectively
    P = 3.447e6
    T = 410.928
    expected_Ais = np.array([0.04906, 0.4544, 2.435])
    expected_Bis = np.array([0.02701, 0.07308, 0.1922])
    assert eos_whitson.A_i(P=P, T=T) == pytest.approx(expected_Ais, rel=1e-3)
    assert eos_whitson.B_i(P=P, T=T) == pytest.approx(expected_Bis, rel=1e-3)


def test_Z_real_root_liquid_whitson_ex18(eos_whitson):
    P = 3.447e6
    T = 410.928
    z = np.array([0.02370, 0.46695, 0.50935])
    expected_Z_l = 0.1812
    assert eos_whitson.calculate_Z_factor(P=P, T=T, z=z) == pytest.approx(
        expected_Z_l, rel=1e-3
    )


def test_Z_real_root_vapor_whitson_ex18(eos_whitson):
    P = 3.447e6
    T = 410.928
    z = np.array([0.58262, 0.41186, 0.00553])
    expected_Z_v = 0.8785
    assert eos_whitson.calculate_Z_factor(P=P, T=T, z=z) == pytest.approx(
        expected_Z_v, rel=1e-3
    )


def test_liquid_fugacity_first_flash_iteration_whitson_ex18(eos_whitson):
    P = 3.447e6
    T = 410.928
    z = np.array([0.02370, 0.46695, 0.50935])
    Z_factor = eos_whitson.calculate_Z_factor(P=P, T=T, z=z)
    liquid_fugacities = eos_whitson.calculate_fugacity(P, T, z, Z_factor)
    expected_liquid_fugacities = np.array(
        [588706.781925616, 1043762.8308328, 23135.6338433312]
    )
    assert liquid_fugacities == pytest.approx(
        expected_liquid_fugacities, rel=1e-3
    )


def test_vapor_fugacity_first_flash_iteration_whitson_ex18(eos_whitson):
    P = 3.447e6
    T = 410.928
    z = np.array([0.58262, 0.41186, 0.00553])
    Z_factor = eos_whitson.calculate_Z_factor(P=P, T=T, z=z)
    vapor_fugacities = eos_whitson.calculate_fugacity(P, T, z, Z_factor)
    expected_vapor_fugacities = np.array(
        [2054796.24885744, 1030945.47704928, 7362.0839284384]
    )
    assert vapor_fugacities == pytest.approx(
        expected_vapor_fugacities, rel=1e-3
    )
