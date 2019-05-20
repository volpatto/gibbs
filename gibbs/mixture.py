import attr
import numpy as np


def check_input_dimensions(instance, attribute, value):
    acentric_factor_not_eq_Tc = len(value) != len(instance.Tc)
    acentric_factor_not_eq_Pc = len(value) != len(instance.Pc)
    Pc_not_eq_Tc = len(instance.Tc) != len(instance.Pc)
    Pc_not_eq_z = len(instance.Pc) != len(instance.z)
    if acentric_factor_not_eq_Tc or acentric_factor_not_eq_Pc or Pc_not_eq_Tc or Pc_not_eq_z:
        raise ValueError("Input values have incompatible dimensions.")


@attr.s
class Mixture:
    z: np.ndarray
    Tc: np.ndarray
    Pc: np.ndarray
    acentric_factor: np.ndarray

    @z.validator
    def check_overall_composition(self, attribute, value):
        tol = 1e-5
        if not 1 - tol <= np.sum(value) <= 1 + tol:
            raise ValueError('Overall composition must has summation equal 1.')

    @Tc.validator
    def validate_Tc(self, attribute, value):
        if np.any(value < 0):
            raise ValueError('Temperature must be greater than zero.')

    @Pc.validator
    def validate_Pc(self, attribute, value):
        if np.any(value < 0):
            raise ValueError('Pressure must be greater than zero.')
