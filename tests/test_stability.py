import pytest
import attr
import numpy as np
from thermo import Chemical
from pytest_lazyfixture import lazy_fixture

from gibbs.mixture import Mixture
from gibbs.models.ceos import PengRobinson78
from gibbs.models.ceos import SoaveRedlichKwong
from gibbs.minimization import PygmoSelfAdaptiveDESettings
from gibbs.stability_analysis import stability_test
from gibbs.utilities import convert_bar_to_Pa

seed = 1234


@pytest.fixture
def methane():
    return Chemical('methane')


@pytest.fixture
def ethane():
    return Chemical('ethane')


@pytest.fixture
def propane():
    return Chemical('propane')


@pytest.fixture
def hydrogen_sulfide():
    return Chemical('H2S')


@pytest.fixture
def nitrogen():
    return Chemical('N2')


@pytest.fixture
def carbon_dioxide():
    return Chemical('CO2')


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
    def input_mixture(self):
        return Mixture(
            z=self.z,
            Tc=self.Tc,
            Pc=self.Pc,
            omega=self.acentric_factor
        )

    @property
    def model(self):
        return PengRobinson78(
            mixture=self.input_mixture,
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


@attr.s(auto_attribs=True)
class NichitaPR:
    mixture: Mixture
    bip: np.ndarray

    @property
    def model(self):
        return PengRobinson78(
            mixture=self.mixture,
            bip=self.bip
        )

    @property
    def number_of_components(self):
        return len(self.mixture.z)

    def fugacity(self, P, T, z):
        Z_factor = self.calculate_Z(P, T, z)
        return self.model.calculate_fugacity(P, T, z, Z_factor)

    def calculate_Z(self, P, T, z):
        Z_factor = self.model.calculate_Z_minimal_energy(P, T, z)
        return Z_factor


@attr.s(auto_attribs=True)
class NichitaSRK:
    mixture: Mixture
    bip: np.ndarray

    @property
    def model(self):
        return SoaveRedlichKwong(
            mixture=self.mixture,
            bip=self.bip
        )

    @property
    def number_of_components(self):
        return len(self.mixture.z)

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


@pytest.fixture
def model_problem_1_1(methane, hydrogen_sulfide):
    z = np.array([0.5, 0.5])
    omegas = np.array([methane.omega, hydrogen_sulfide.omega])
    Tcs = np.array([methane.Tc, hydrogen_sulfide.Tc])
    Pcs = np.array([methane.Pc, hydrogen_sulfide.Pc])
    mixture = Mixture(z, Tcs, Pcs, omegas)
    kijs = np.array([
        [0.000, 0.080],
        [0.080, 0.000]
    ])
    return NichitaSRK(
        mixture=mixture,
        bip=kijs
    )


@pytest.fixture
def model_problem_1_2(methane, hydrogen_sulfide):
    z = np.array([0.9885, 0.0115])
    omegas = np.array([methane.omega, hydrogen_sulfide.omega])
    Tcs = np.array([methane.Tc, hydrogen_sulfide.Tc])
    Pcs = np.array([methane.Pc, hydrogen_sulfide.Pc])
    mixture = Mixture(z, Tcs, Pcs, omegas)
    kijs = np.array([
        [0.000, 0.080],
        [0.080, 0.000]
    ])
    return NichitaSRK(
        mixture=mixture,
        bip=kijs
    )


@pytest.fixture
def model_problem_1_3(methane, hydrogen_sulfide):
    z = np.array([0.9813, 0.0187])
    omegas = np.array([methane.omega, hydrogen_sulfide.omega])
    Tcs = np.array([methane.Tc, hydrogen_sulfide.Tc])
    Pcs = np.array([methane.Pc, hydrogen_sulfide.Pc])
    mixture = Mixture(z, Tcs, Pcs, omegas)
    kijs = np.array([
        [0.000, 0.080],
        [0.080, 0.000]
    ])
    return NichitaSRK(
        mixture=mixture,
        bip=kijs
    )


@pytest.fixture
def model_problem_1_4(methane, hydrogen_sulfide):
    z = np.array([0.112, 0.888])
    omegas = np.array([methane.omega, hydrogen_sulfide.omega])
    Tcs = np.array([methane.Tc, hydrogen_sulfide.Tc])
    Pcs = np.array([methane.Pc, hydrogen_sulfide.Pc])
    mixture = Mixture(z, Tcs, Pcs, omegas)
    kijs = np.array([
        [0.000, 0.080],
        [0.080, 0.000]
    ])
    return NichitaSRK(
        mixture=mixture,
        bip=kijs
    )


@pytest.fixture
def model_problem_1_5(methane, hydrogen_sulfide):
    z = np.array([0.11, 0.89])
    omegas = np.array([methane.omega, hydrogen_sulfide.omega])
    Tcs = np.array([methane.Tc, hydrogen_sulfide.Tc])
    Pcs = np.array([methane.Pc, hydrogen_sulfide.Pc])
    mixture = Mixture(z, Tcs, Pcs, omegas)
    kijs = np.array([
        [0.000, 0.080],
        [0.080, 0.000]
    ])
    return NichitaSRK(
        mixture=mixture,
        bip=kijs
    )


@pytest.fixture
def model_problem_2_1_50_bar(methane, propane):
    z = np.array([0.10, 0.90])
    omegas = np.array([methane.omega, propane.omega])
    Tcs = np.array([methane.Tc, propane.Tc])
    Pcs = np.array([methane.Pc, propane.Pc])
    mixture = Mixture(z, Tcs, Pcs, omegas)
    kijs = np.array([
        [0.000, 0.029],
        [0.029, 0.000]
    ])
    return NichitaSRK(
        mixture=mixture,
        bip=kijs
    )


@pytest.fixture
def model_problem_2_2_50_bar(methane, propane):
    z = np.array([0.40, 0.60])
    omegas = np.array([methane.omega, propane.omega])
    Tcs = np.array([methane.Tc, propane.Tc])
    Pcs = np.array([methane.Pc, propane.Pc])
    mixture = Mixture(z, Tcs, Pcs, omegas)
    kijs = np.array([
        [0.000, 0.029],
        [0.029, 0.000]
    ])
    return NichitaSRK(
        mixture=mixture,
        bip=kijs
    )


@pytest.fixture
def model_problem_2_3_50_bar(methane, propane):
    z = np.array([0.60, 0.40])
    omegas = np.array([methane.omega, propane.omega])
    Tcs = np.array([methane.Tc, propane.Tc])
    Pcs = np.array([methane.Pc, propane.Pc])
    mixture = Mixture(z, Tcs, Pcs, omegas)
    kijs = np.array([
        [0.000, 0.029],
        [0.029, 0.000]
    ])
    return NichitaSRK(
        mixture=mixture,
        bip=kijs
    )


@pytest.fixture
def model_problem_2_4_50_bar(methane, propane):
    z = np.array([0.90, 0.10])
    omegas = np.array([methane.omega, propane.omega])
    Tcs = np.array([methane.Tc, propane.Tc])
    Pcs = np.array([methane.Pc, propane.Pc])
    mixture = Mixture(z, Tcs, Pcs, omegas)
    kijs = np.array([
        [0.000, 0.029],
        [0.029, 0.000]
    ])
    return NichitaSRK(
        mixture=mixture,
        bip=kijs
    )


@pytest.fixture
def model_problem_2_1_100_bar(methane, propane):
    z = np.array([0.40, 0.60])
    omegas = np.array([methane.omega, propane.omega])
    Tcs = np.array([methane.Tc, propane.Tc])
    Pcs = np.array([methane.Pc, propane.Pc])
    mixture = Mixture(z, Tcs, Pcs, omegas)
    kijs = np.array([
        [0.000, 0.029],
        [0.029, 0.000]
    ])
    return NichitaSRK(
        mixture=mixture,
        bip=kijs
    )


@pytest.fixture
def model_problem_2_2_100_bar(methane, propane):
    z = np.array([0.40, 0.60])
    omegas = np.array([methane.omega, propane.omega])
    Tcs = np.array([methane.Tc, propane.Tc])
    Pcs = np.array([methane.Pc, propane.Pc])
    mixture = Mixture(z, Tcs, Pcs, omegas)
    kijs = np.array([
        [0.000, 0.029],
        [0.029, 0.000]
    ])
    return NichitaSRK(
        mixture=mixture,
        bip=kijs
    )


@pytest.fixture
def model_problem_2_2_100_bar(methane, propane):
    z = np.array([0.68, 0.32])
    omegas = np.array([methane.omega, propane.omega])
    Tcs = np.array([methane.Tc, propane.Tc])
    Pcs = np.array([methane.Pc, propane.Pc])
    mixture = Mixture(z, Tcs, Pcs, omegas)
    kijs = np.array([
        [0.000, 0.029],
        [0.029, 0.000]
    ])
    return NichitaSRK(
        mixture=mixture,
        bip=kijs
    )


@pytest.fixture
def model_problem_2_3_100_bar(methane, propane):
    z = np.array([0.73, 0.27])
    omegas = np.array([methane.omega, propane.omega])
    Tcs = np.array([methane.Tc, propane.Tc])
    Pcs = np.array([methane.Pc, propane.Pc])
    mixture = Mixture(z, Tcs, Pcs, omegas)
    kijs = np.array([
        [0.000, 0.029],
        [0.029, 0.000]
    ])
    return NichitaSRK(
        mixture=mixture,
        bip=kijs
    )


@pytest.fixture
def model_problem_2_4_100_bar(methane, propane):
    z = np.array([0.90, 0.10])
    omegas = np.array([methane.omega, propane.omega])
    Tcs = np.array([methane.Tc, propane.Tc])
    Pcs = np.array([methane.Pc, propane.Pc])
    mixture = Mixture(z, Tcs, Pcs, omegas)
    kijs = np.array([
        [0.000, 0.029],
        [0.029, 0.000]
    ])
    return NichitaSRK(
        mixture=mixture,
        bip=kijs
    )


@pytest.fixture
def model_problem_3_1(ethane, nitrogen):
    z = np.array([0.90, 0.10])
    omegas = np.array([ethane.omega, nitrogen.omega])
    Tcs = np.array([ethane.Tc, nitrogen.Tc])
    Pcs = np.array([ethane.Pc, nitrogen.Pc])
    mixture = Mixture(z, Tcs, Pcs, omegas)
    kijs = np.array([
        [0.000, 0.080],
        [0.080, 0.000]
    ])
    return NichitaPR(
        mixture=mixture,
        bip=kijs
    )


@pytest.fixture
def model_problem_3_2(ethane, nitrogen):
    z = np.array([0.82, 0.18])
    omegas = np.array([ethane.omega, nitrogen.omega])
    Tcs = np.array([ethane.Tc, nitrogen.Tc])
    Pcs = np.array([ethane.Pc, nitrogen.Pc])
    mixture = Mixture(z, Tcs, Pcs, omegas)
    kijs = np.array([
        [0.000, 0.080],
        [0.080, 0.000]
    ])
    return NichitaPR(
        mixture=mixture,
        bip=kijs
    )


@pytest.fixture
def model_problem_3_3(ethane, nitrogen):
    z = np.array([0.70, 0.30])
    omegas = np.array([ethane.omega, nitrogen.omega])
    Tcs = np.array([ethane.Tc, nitrogen.Tc])
    Pcs = np.array([ethane.Pc, nitrogen.Pc])
    mixture = Mixture(z, Tcs, Pcs, omegas)
    kijs = np.array([
        [0.000, 0.080],
        [0.080, 0.000]
    ])
    return NichitaPR(
        mixture=mixture,
        bip=kijs
    )


@pytest.fixture
def model_problem_3_4(ethane, nitrogen):
    z = np.array([0.56, 0.44])
    omegas = np.array([ethane.omega, nitrogen.omega])
    Tcs = np.array([ethane.Tc, nitrogen.Tc])
    Pcs = np.array([ethane.Pc, nitrogen.Pc])
    mixture = Mixture(z, Tcs, Pcs, omegas)
    kijs = np.array([
        [0.000, 0.080],
        [0.080, 0.000]
    ])
    return NichitaPR(
        mixture=mixture,
        bip=kijs
    )


@pytest.fixture
def model_problem_3_5(ethane, nitrogen):
    z = np.array([0.40, 0.60])
    omegas = np.array([ethane.omega, nitrogen.omega])
    Tcs = np.array([ethane.Tc, nitrogen.Tc])
    Pcs = np.array([ethane.Pc, nitrogen.Pc])
    mixture = Mixture(z, Tcs, Pcs, omegas)
    kijs = np.array([
        [0.000, 0.080],
        [0.080, 0.000]
    ])
    return NichitaPR(
        mixture=mixture,
        bip=kijs
    )


@pytest.fixture
def model_problem_4_1(methane, carbon_dioxide):
    z = np.array([0.90, 0.10])
    omegas = np.array([methane.omega, carbon_dioxide.omega])
    Tcs = np.array([methane.Tc, carbon_dioxide.Tc])
    Pcs = np.array([methane.Pc, carbon_dioxide.Pc])
    mixture = Mixture(z, Tcs, Pcs, omegas)
    kijs = np.array([
        [0.000, 0.095],
        [0.095, 0.000]
    ])
    return NichitaPR(
        mixture=mixture,
        bip=kijs
    )


@pytest.fixture
def model_problem_4_2(methane, carbon_dioxide):
    z = np.array([0.80, 0.20])
    omegas = np.array([methane.omega, carbon_dioxide.omega])
    Tcs = np.array([methane.Tc, carbon_dioxide.Tc])
    Pcs = np.array([methane.Pc, carbon_dioxide.Pc])
    mixture = Mixture(z, Tcs, Pcs, omegas)
    kijs = np.array([
        [0.000, 0.095],
        [0.095, 0.000]
    ])
    return NichitaPR(
        mixture=mixture,
        bip=kijs
    )


@pytest.fixture
def model_problem_4_3(methane, carbon_dioxide):
    z = np.array([0.70, 0.30])
    omegas = np.array([methane.omega, carbon_dioxide.omega])
    Tcs = np.array([methane.Tc, carbon_dioxide.Tc])
    Pcs = np.array([methane.Pc, carbon_dioxide.Pc])
    mixture = Mixture(z, Tcs, Pcs, omegas)
    kijs = np.array([
        [0.000, 0.095],
        [0.095, 0.000]
    ])
    return NichitaPR(
        mixture=mixture,
        bip=kijs
    )


@pytest.fixture
def model_problem_4_4(methane, carbon_dioxide):
    z = np.array([0.57, 0.43])
    omegas = np.array([methane.omega, carbon_dioxide.omega])
    Tcs = np.array([methane.Tc, carbon_dioxide.Tc])
    Pcs = np.array([methane.Pc, carbon_dioxide.Pc])
    mixture = Mixture(z, Tcs, Pcs, omegas)
    kijs = np.array([
        [0.000, 0.095],
        [0.095, 0.000]
    ])
    return NichitaPR(
        mixture=mixture,
        bip=kijs
    )


@pytest.fixture
def model_problem_4_5(methane, carbon_dioxide):
    z = np.array([0.40, 0.60])
    omegas = np.array([methane.omega, carbon_dioxide.omega])
    Tcs = np.array([methane.Tc, carbon_dioxide.Tc])
    Pcs = np.array([methane.Pc, carbon_dioxide.Pc])
    mixture = Mixture(z, Tcs, Pcs, omegas)
    kijs = np.array([
        [0.000, 0.095],
        [0.095, 0.000]
    ])
    return NichitaPR(
        mixture=mixture,
        bip=kijs
    )


@pytest.fixture
def model_problem_5_1(methane, ethane, nitrogen):
    z = np.array([0.10, 0.60, 0.30])
    omegas = np.array([methane.omega, ethane.omega, nitrogen.omega])
    Tcs = np.array([methane.Tc, ethane.Tc, nitrogen.Tc])
    Pcs = np.array([methane.Pc, ethane.Pc, nitrogen.Pc])
    mixture = Mixture(z, Tcs, Pcs, omegas)
    kijs = np.array([
        [0.000, 0.021, 0.038],
        [0.021, 0.000, 0.080],
        [0.038, 0.080, 0.000]
    ])
    return NichitaPR(
        mixture=mixture,
        bip=kijs
    )


@pytest.fixture
def model_problem_5_2(methane, ethane, nitrogen):
    z = np.array([0.30, 0.55, 0.15])
    omegas = np.array([methane.omega, ethane.omega, nitrogen.omega])
    Tcs = np.array([methane.Tc, ethane.Tc, nitrogen.Tc])
    Pcs = np.array([methane.Pc, ethane.Pc, nitrogen.Pc])
    mixture = Mixture(z, Tcs, Pcs, omegas)
    kijs = np.array([
        [0.000, 0.021, 0.038],
        [0.021, 0.000, 0.080],
        [0.038, 0.080, 0.000]
    ])
    return NichitaPR(
        mixture=mixture,
        bip=kijs
    )


@pytest.fixture
def model_problem_5_3(methane, ethane, nitrogen):
    z = np.array([0.38, 0.54, 0.08])
    omegas = np.array([methane.omega, ethane.omega, nitrogen.omega])
    Tcs = np.array([methane.Tc, ethane.Tc, nitrogen.Tc])
    Pcs = np.array([methane.Pc, ethane.Pc, nitrogen.Pc])
    mixture = Mixture(z, Tcs, Pcs, omegas)
    kijs = np.array([
        [0.000, 0.021, 0.038],
        [0.021, 0.000, 0.080],
        [0.038, 0.080, 0.000]
    ])
    return NichitaPR(
        mixture=mixture,
        bip=kijs
    )


@pytest.fixture
def model_problem_5_4(methane, ethane, nitrogen):
    z = np.array([0.05, 0.90, 0.05])
    omegas = np.array([methane.omega, ethane.omega, nitrogen.omega])
    Tcs = np.array([methane.Tc, ethane.Tc, nitrogen.Tc])
    Pcs = np.array([methane.Pc, ethane.Pc, nitrogen.Pc])
    mixture = Mixture(z, Tcs, Pcs, omegas)
    kijs = np.array([
        [0.000, 0.021, 0.038],
        [0.021, 0.000, 0.080],
        [0.038, 0.080, 0.000]
    ])
    return NichitaPR(
        mixture=mixture,
        bip=kijs
    )


def test_stochastic_consistency(sample_model):
    num_of_realizations = 10
    P = sample_model.P
    T = sample_model.T
    z = sample_model.z
    for run in range(num_of_realizations):
        result = stability_test(sample_model, P, T, z)
        assert result.x.sum() == pytest.approx(1.0, rel=1e-5)


@pytest.mark.parametrize('model, expected_phase_split', [
    [lazy_fixture('model_problem_1_1'), True],
    [lazy_fixture('model_problem_1_2'), False],
    [lazy_fixture('model_problem_1_3'), True],
    [lazy_fixture('model_problem_1_4'), True],
    [lazy_fixture('model_problem_1_5'), False]
])
def test_nichita_problem_1(model, expected_phase_split):
    P = convert_bar_to_Pa(40.53)
    T = 190

    result = stability_test(model, P, T, model.mixture.z, solver_args=PygmoSelfAdaptiveDESettings(20, 800, seed=seed))

    assert result.phase_split == expected_phase_split


@pytest.mark.parametrize('model, pressure, expected_phase_split', [
    [lazy_fixture('model_problem_2_1_50_bar'), 50, False],
    [lazy_fixture('model_problem_2_2_50_bar'), 50, True],
    [lazy_fixture('model_problem_2_3_50_bar'), 50, True],
    [lazy_fixture('model_problem_2_4_50_bar'), 50, False],
    [lazy_fixture('model_problem_2_1_100_bar'), 100, False],
    [lazy_fixture('model_problem_2_2_100_bar'), 100, True],
    [lazy_fixture('model_problem_2_3_100_bar'), 100, True],
    [lazy_fixture('model_problem_2_4_100_bar'), 100, False],
])
def test_nichita_problem_2(model, pressure, expected_phase_split):
    P = convert_bar_to_Pa(pressure)
    T = 277.6

    result = stability_test(model, P, T, model.mixture.z, solver_args=PygmoSelfAdaptiveDESettings(20, 500, seed=seed))

    assert result.phase_split == expected_phase_split


@pytest.mark.parametrize('model, expected_phase_split', [
    [lazy_fixture('model_problem_3_1'), False],
    [lazy_fixture('model_problem_3_2'), True],
    [lazy_fixture('model_problem_3_3'), True],
    [lazy_fixture('model_problem_3_4'), True],
    [lazy_fixture('model_problem_3_5'), False]
])
def test_nichita_problem_3(model, expected_phase_split):
    P = convert_bar_to_Pa(76)
    T = 270

    result = stability_test(model, P, T, model.mixture.z, solver_args=PygmoSelfAdaptiveDESettings(20, 500, seed=seed))

    assert result.phase_split == expected_phase_split


@pytest.mark.parametrize('model, expected_phase_split', [
    [lazy_fixture('model_problem_4_1'), False],
    [lazy_fixture('model_problem_4_2'), True],
    [lazy_fixture('model_problem_4_3'), True],
    [lazy_fixture('model_problem_4_4'), True],
    [lazy_fixture('model_problem_4_5'), False]
])
def test_nichita_problem_4(model, expected_phase_split):
    P = convert_bar_to_Pa(60.8)
    T = 220

    result = stability_test(model, P, T, model.mixture.z, solver_args=PygmoSelfAdaptiveDESettings(20, 500, seed=seed))

    assert result.phase_split == expected_phase_split


@pytest.mark.parametrize('model, expected_phase_split', [
    [lazy_fixture('model_problem_5_1'), True],
    pytest.param(lazy_fixture('model_problem_5_2'), True, marks=pytest.mark.xfail(reason='Differing result from '
                                                                                         'paper')),
    [lazy_fixture('model_problem_5_3'), False],
    [lazy_fixture('model_problem_5_4'), False]
])
def test_nichita_problem_5(model, expected_phase_split):
    P = convert_bar_to_Pa(76)
    T = 270

    result = stability_test(model, P, T, model.mixture.z, solver_args=PygmoSelfAdaptiveDESettings(20, 500, seed=seed))

    assert result.phase_split == expected_phase_split
