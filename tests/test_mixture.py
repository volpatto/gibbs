import pytest
import numpy as np

from gibbs.mixture import Mixture


def test_invalid_input_composition():
    z = np.array([0.5, 0.42, 0.05])
    omegas = np.array([0.0115, 0.1928, 0.4902])
    Tcs = np.array([190.556, 425.16667, 617.666667])
    Pcs = np.array([4604318.9, 3796942.8, 2.096e6])

    with pytest.raises(ValueError) as message:
        Mixture(z=z, Tc=Tcs, Pc=Pcs, acentric_factor=omegas)

    assert str(message.value) == 'Overall composition must has summation equal 1.'


def test_invalid_input_omegas():
    z = np.array([0.5, 0.42, 0.08])
    omegas = np.array([0.0115, 0.1928, 0.4902, 0.1000])
    Tcs = np.array([190.556, 425.16667, 617.666667])
    Pcs = np.array([4604318.9, 3796942.8, 2.096e6])

    with pytest.raises(ValueError) as message:
        Mixture(z=z, Tc=Tcs, Pc=Pcs, acentric_factor=omegas)

    assert str(message.value) == 'Input values have incompatible dimensions.'


def test_invalid_input_Tc():
    z = np.array([0.5, 0.42, 0.08])
    omegas = np.array([0.0115, 0.1928, 0.4902])
    Tcs = np.array([190.556, 425.16667])
    Pcs = np.array([4604318.9, 3796942.8, 2.096e6])

    with pytest.raises(ValueError) as message:
        Mixture(z=z, Tc=Tcs, Pc=Pcs, acentric_factor=omegas)

    assert str(message.value) == 'Input values have incompatible dimensions.'


def test_invalid_input_Pc():
    z = np.array([0.5, 0.42, 0.08])
    omegas = np.array([0.0115, 0.1928, 0.4902])
    Tcs = np.array([190.556, 425.16667, 617.666667])
    Pcs = np.array([4604318.9, 3796942.8])

    with pytest.raises(ValueError) as message:
        Mixture(z=z, Tc=Tcs, Pc=Pcs, acentric_factor=omegas)

    assert str(message.value) == 'Input values have incompatible dimensions.'


def test_non_physical_input_Tc():
    z = np.array([0.5, 0.42, 0.08])
    omegas = np.array([0.0115, 0.1928, 0.4902])
    Tcs = np.array([190.556, -425.16667, 617.666667])
    Pcs = np.array([4604318.9, 3796942.8, 2.096e6])

    with pytest.raises(ValueError) as message:
        Mixture(z=z, Tc=Tcs, Pc=Pcs, acentric_factor=omegas)

    assert str(message.value) == 'Temperature must be greater than zero.'


def test_non_physical_input_Pc():
    z = np.array([0.5, 0.42, 0.08])
    omegas = np.array([0.0115, 0.1928, 0.4902])
    Tcs = np.array([190.556, 425.16667, 617.666667])
    Pcs = np.array([-4604318.9, 3796942.8, 2.096e6])

    with pytest.raises(ValueError) as message:
        Mixture(z=z, Tc=Tcs, Pc=Pcs, acentric_factor=omegas)

    assert str(message.value) == 'Pressure must be greater than zero.'


def test_mixture_whitson_ex18():
    z = np.array([0.5, 0.42, 0.08])
    omegas = np.array([0.0115, 0.1928, 0.4902])
    Tcs = np.array([190.556, 425.16667, 617.666667])
    Pcs = np.array([4604318.9, 3796942.8, 2.096e6])
    whiston_mixture = Mixture(z=z, Tc=Tcs, Pc=Pcs, acentric_factor=omegas)

    assert whiston_mixture.z.all() == z.all()
    assert whiston_mixture.acentric_factor.all() == omegas.all()
    assert whiston_mixture.Tc.all() == Tcs.all()
    assert whiston_mixture.Pc.all() == Pcs.all()
