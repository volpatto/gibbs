import pytest
import numpy as np

from gibbs.cpp_wrapper import _cubic_cardano_real_roots, _cubic_cardano_real_positive_roots


@pytest.mark.parametrize("coeffs", [
    np.array([1, 3.2, 2, 1]),
    np.array([1, 1, 1, 1]),
    np.array([1, 1, 0, 1]),
    np.array([1, 0, 1, 1]),
    np.array([1, 0, 0, 1]),
    np.array([1, -0.5, 0.5, 1])
])
def test_polynomial_roots(coeffs):
    numpy_result = np.roots(coeffs)
    numpy_real_roots = numpy_result.real[np.abs(numpy_result.imag) < 1e-5]
    real_roots_eos = _cubic_cardano_real_roots(coeffs[::-1])  # the order is reversed compared with numpy
    assert pytest.approx(numpy_real_roots.max()) == real_roots_eos.max()
    assert pytest.approx(numpy_real_roots.min()) == real_roots_eos.min()


@pytest.mark.parametrize("coeffs, real_roots", [
    [np.array([1, 3, 2, -3]), 0.671699881],
    [np.array([1, 0, -6, -9]), 3.0],
    [np.array([1, 0, -3, -2]), 2.0],
    [np.array([1, 0, -1, -1]), 1.324718]
])
def test_known_real_positive_roots(coeffs, real_roots):
    real_roots_eos = _cubic_cardano_real_positive_roots(coeffs[::-1])
    assert real_roots_eos == pytest.approx(real_roots)
