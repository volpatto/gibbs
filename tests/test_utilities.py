import pytest
from gibbs.utilities import convert_psi_to_Pa, convert_F_to_K


@pytest.mark.parametrize('P_in_psi, P_in_Pa', [
    [5.00, 34473.8],
    [10.00, 68947.6],
    [20.00, 137895],
    [100.00, 689476],
    [200.00, 1.379e6]
])
def test_psi_to_Pa(P_in_psi, P_in_Pa):
    assert convert_psi_to_Pa(P_in_psi) == pytest.approx(P_in_Pa, rel=1e-3)


@pytest.mark.parametrize('T_in_F, T_in_K', [
    [-459.67, 0],
    [0, 255.37],
    [10, 260.93],
    [20, 266.48],
    [80, 299.82],
    [200, 366.48]
])
def test_F_to_K(T_in_F, T_in_K):
    assert convert_F_to_K(T_in_F) == pytest.approx(T_in_K, rel=1e-3)
