import pytest
import numpy as np

from gibbs.mixture import Mixture
from gibbs.models.ceos import PengRobinson78
from gibbs.equilibrium import calculate_equilibrium
from gibbs.utilities import convert_F_to_K, convert_psi_to_Pa


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


@pytest.mark.xfail(reason='To be implemented')
def test_equilibrium_whitson_example_18(mixture_whitson, eos_whitson):
    T = convert_F_to_K(280)
    P = convert_psi_to_Pa(500)
    expected_x = np.array([0.330082, 0.513307, 0.156611])
    expected_y = np.array([0.629843, 0.348699, 0.021457])
    expected_F = np.array([0.853401, 1 - 0.853401])
    expect_n_phases = 2

    result = calculate_equilibrium(mixture_whitson, eos_whitson, P, T)

    assert result.num_of_phases == expect_n_phases
    assert any(np.allclose(composition, expected_x, rtol=1e-2) for composition in result.X)
    assert any(np.allclose(composition, expected_y, rtol=1e-2) for composition in result.X)
    assert result.F == pytest.approx(expected_F, rel=1e-2)
