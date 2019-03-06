import pytest
import attr
import numpy as np

from gibbs.models.ceos import PengRobinson78
from gibbs.stability_analysis import stability_test


@attr.s(auto_attribs=True)
class InputModel:
    z: np.ndarray
    P: float
    T: float
    Tc: np.ndarray
    Pc: np.ndarray
    acentric_factor: np.ndarray
    bip: np.ndarray

    @property
    def model(self):
        return PengRobinson78(
            z=self.z, 
            Tc=self.Tc, 
            Pc=self.Pc, 
            acentric_factor=self.acentric_factor, 
            bip=self.bip
        )

    @property
    def number_of_components(self):
        return len(self.z)

    def fugacity(self, P, T, z):
        Z_factor = self.calculate_Z(P, T, z)
        return self.model.calculate_fugacity(P, T, z, Z_factor)

    def calculate_Z(self, P, T, z):
        Z_factor = self.model.calculate_Z_minimal_energy(P, T, z)
        return Z_factor



@pytest.fixture
def sample_model():
    z = np.array([0.5, 0.42, 0.08])
    omegas = np.array([0.0115, 0.1928, 0.4902])
    Tcs = np.array([190.556, 425.16667, 617.666667])
    Pcs = np.array([4604318.9, 3796942.8, 2.096e6])
    kijs = np.zeros((3, 3))
    P = 3.447e6
    T = 410.928
    model = InputModel(
        z=z, 
        P=P, 
        T=T,
        Tc=Tcs,
        Pc=Pcs,
        acentric_factor=omegas, 
        bip=kijs
    )

    return model


def test_stochastic_consistency(sample_model):
    num_of_realizations = 10
    P = sample_model.P
    T = sample_model.T
    z = sample_model.z
    for run in range(num_of_realizations):
        result = stability_test(sample_model, P, T, z)
        assert result.x.sum() == pytest.approx(1.0, rel=1e-5)
