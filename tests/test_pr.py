import pytest
import numpy as np

from gibbs.ceos import PengRobinson78


@pytest.fixture
def eos_whitson():
    z = np.array([0.5, 0.42, 0.08])
    omegas = np.array([0.0115, 0.1928, 0.4902])
    Tcs = np.array([190.556, 425.16667, 617.666667])
    Pcs = np.array([4604318.9, 3796942.8, 2.096e6])
    kijs = np.zeros((3, 3))
    return PengRobinson78(z=z, Tc=Tcs, Pc=Pcs, acentric_factor=omegas, bip=kijs)


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
    assert eos_whitson.calculate_Z(P=P, T=T, z=z) == pytest.approx(
        expected_Z_l, rel=1e-3
    )


def test_Z_real_root_vapor_whitson_ex18(eos_whitson):
    P = 3.447e6
    T = 410.928
    z = np.array([0.58262, 0.41186, 0.00553])
    expected_Z_v = 0.8785
    assert eos_whitson.calculate_Z(P=P, T=T, z=z) == pytest.approx(
        expected_Z_v, rel=1e-3
    )
