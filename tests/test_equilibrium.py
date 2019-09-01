import pytest
import numpy as np
import numpy.linalg as la
import attr
from thermo import Chemical

from gibbs.cpp_wrapper import Mixture
from gibbs.cpp_wrapper import PengRobinson78, PengRobinson
from gibbs.equilibrium import calculate_equilibrium
from gibbs.minimization import PygmoSelfAdaptiveDESettings
from gibbs.utilities import convert_F_to_K, convert_psi_to_Pa, convert_bar_to_Pa

seed = 1234


@attr.s(auto_attribs=True)
class ModelPR78:
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
        # return len(self.mixture.z())
        return len(self.mixture.z)

    def fugacity(self, P, T, z):
        Z_factor = self.calculate_Z(P, T, z)
        return self.model.calculate_fugacity(P, T, z, Z_factor)

    def calculate_Z(self, P, T, z):
        Z_factor = self.model.calculate_Z_minimal_energy(P, T, z)
        return Z_factor


@attr.s(auto_attribs=True)
class ModelPR:
    mixture: Mixture
    bip: np.ndarray

    @property
    def model(self):
        return PengRobinson(
            mixture=self.mixture,
            bip=self.bip
        )

    @property
    def number_of_components(self):
        # return len(self.mixture.z())
        return len(self.mixture.z)

    def fugacity(self, P, T, z):
        Z_factor = self.calculate_Z(P, T, z)
        return self.model.calculate_fugacity(P, T, z, Z_factor)

    def calculate_Z(self, P, T, z):
        Z_factor = self.model.calculate_Z_minimal_energy(P, T, z)
        return Z_factor


@pytest.fixture
def mixture_whitson():
    z = np.array([0.5, 0.42, 0.08])
    omegas = np.array([0.0115, 0.1928, 0.4902])
    Tcs = np.array([190.556, 425.16667, 617.666667])
    Pcs = np.array([4604318.9, 3796942.8, 2.096e6])
    return Mixture(z=z, Tc=Tcs, Pc=Pcs, omega=omegas)
    # return Mixture(z=z, Tc=Tcs, Pc=Pcs, omega=omegas)


@pytest.fixture
def model_whitson(mixture_whitson):
    kijs = np.zeros((3, 3))
    model = ModelPR78(
        mixture=mixture_whitson,
        bip=kijs
    )

    return model


@pytest.fixture
def mixture_nichita_ternary():
    methane = Chemical('methane')
    nhexadecane = Chemical('n-hexadecane')
    carbon_dioxide = Chemical('carbon-dioxide')

    z = np.array([0.05, 0.05, 0.90])
    omegas = np.array([methane.omega, nhexadecane.omega, carbon_dioxide.omega])
    Tcs = np.array([methane.Tc, nhexadecane.Tc, carbon_dioxide.Tc])
    Pcs = np.array([methane.Pc, nhexadecane.Pc, carbon_dioxide.Pc])
    return Mixture(z, Tcs, Pcs, omegas)


@pytest.fixture
def model_nichita_ternary(mixture_nichita_ternary):
    kijs = np.array([
        [0.000, 0.078, 0.100],
        [0.078, 0.000, 0.125],
        [0.100, 0.125, 0.000]
    ])
    model = ModelPR(
        mixture=mixture_nichita_ternary,
        bip=kijs
    )

    return model


def test_equilibrium_whitson_example_18(mixture_whitson, model_whitson):
    T = convert_F_to_K(280)
    P = convert_psi_to_Pa(500)
    z = mixture_whitson.z
    expected_F = np.array([0.853401, 1 - 0.853401])
    expected_n_phases = 2

    result = calculate_equilibrium(
        model_whitson,
        P,
        T,
        z,
        number_of_trial_phases=expected_n_phases,
        molar_base=1,
        solver_args=PygmoSelfAdaptiveDESettings(20, 2000, seed=seed, polish=True, variant_adptv=2,
                                                allowed_variants=[2, 7])
    )

    assert np.sort(result.F) == pytest.approx(np.sort(expected_F), rel=1e-1)


@pytest.mark.xfail(reason="Result is not the same from the paper. Investigations are demanded.")
def test_equilibrium_nichita_ternary_mixture_composition(mixture_nichita_ternary, model_nichita_ternary):
    T = 294.3
    P = convert_bar_to_Pa(67)
    z = mixture_nichita_ternary.z
    expected_y = np.array([0.078112, 0.000069, 0.921819])
    expected_x1 = np.array([0.036181, 0.340224, 0.623595])
    expected_x2 = np.array([0.038707, 0.004609, 0.956683])

    result = calculate_equilibrium(
        model_nichita_ternary,
        P,
        T,
        z,
        number_of_trial_phases=3,
        compare_trial_phases=False,
        solver_args=PygmoSelfAdaptiveDESettings(30, 2000, seed=seed)
    )

    for expected_composition in [expected_y, expected_x1, expected_x2]:
        expected_norm = la.norm(expected_composition)
        assert any(la.norm(composition - expected_composition) / expected_norm < 5e-2 for composition in result.X)


def test_equilibrium_nichita_ternary_mixture_phase_fractions(mixture_nichita_ternary, model_nichita_ternary):
    T = 294.3
    P = convert_bar_to_Pa(67)
    z = mixture_nichita_ternary.z
    expected_F = np.sort(np.array([0.5645, 0.2962, 0.1393]))

    result = calculate_equilibrium(
        model_nichita_ternary,
        P,
        T,
        z,
        number_of_trial_phases=3,
        compare_trial_phases=False,
        solver_args=PygmoSelfAdaptiveDESettings(20, 500, seed=seed)
    )

    phase_fraction_sorted = np.sort(result.F)

    relative_l2_norm = la.norm(phase_fraction_sorted - expected_F) / la.norm(expected_F)
    assert relative_l2_norm < 1.5e-1
