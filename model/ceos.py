import numpy as np
import attr


def check_input_dimensions(instance, attribute, value):
    accentric_factor_not_eq_Tc = len(value) != len(instance.Tc)
    accentric_factor_not_eq_Pc = len(value) != len(instance.Pc)
    Pc_not_eq_Tc = len(instance.Tc) != len(instance.Pc)
    if accentric_factor_not_eq_Tc or accentric_factor_not_eq_Pc or Pc_not_eq_Tc:
        raise ValueError("Inputed values have incompatible dimensions.")


def check_bip(instance, attribute, value):
        if value.shape[0] != value.shape[1]:
            raise ValueError("BIP's must be a 2-dim symmetric array.")
        if value.shape[0] != len(instance.Tc):
            raise ValueError("BIP's have incompatible dimension with input data such as critical temperature.")


@attr.s
class CEOS(object):
    """
    docstring here
        :param object: 
    """
    z = attr.ib(type=np.ndarray)
    Tc = attr.ib(type=np.ndarray)
    Pc = attr.ib(type=np.ndarray)
    acentric_factor = attr.ib(type=np.ndarray, validator=[check_input_dimensions])
    bip = attr.ib(type=np.ndarray, validator=[check_bip])

    @z.validator
    def check_overall_composition(self, attribute, value):
        tol = 1e-5
        if not 1 - tol <= np.sum(value) <= 1 + tol:
            raise ValueError('Overall composition must have summation equal 1.')


    def Tr(self, T):
        if T < 0:
            raise ValueError('Temperature must be greater than zero.')
        return T / self.Tc

    def Pr(self, P):
        if P < 0:
            raise ValueError('Pressure must be greater than zero.')
        return P / self.Pc


@attr.s
class PengRobinson76(CEOS):
    """
    docstring here
        :param CEOS: 
    """
    _Z_c = attr.ib(default=0.3074)
    _Omega_a = attr.ib(default=0.45724)
    _Omega_b = attr.ib(default=0.07780)

    @property
    def m(self):
        return 0.37464 + 1.54226 * self.acentric_factor - 0.26992 * self.acentric_factor * self.acentric_factor

    def alpha(self, T):
        return (1 + self.m * (1 - np.sqrt(self.Tr(T)))) * (1 + self.m * (1 - np.sqrt(self.Tr(T))))

    def A_i(self, P, T):
        return self._Omega_a * (self.Pr(P) / (self.Tr(T) * self.Tr(T))) * self.alpha(T)

    def B_i(self, P, T):
        return self._Omega_b * self.Pr(P) / self.Tr(T)

    def A_mix(self, P, T, z):
        # raise NotImplementedError('To be implemented.')
        return np.dot(z, np.dot(z, self.A_ij(P, T)))

    def B_mix(self, P, T, z):
        # raise NotImplementedError('To be implemented.')
        return np.dot(z, self.B_i(P, T))

    def A_ij(self, P, T):
        return (1 - self.bip) * np.sqrt(np.outer(self.A_i(P, T), self.A_i(P, T)))


z = np.array([0.5, 0.5])
Tcs = np.array([126.1, 190.6])
Pcs = np.array([33.94E5, 46.04E5])
omegas = np.array([0.04, 0.011])
kijs = np.array([[0, 0], [0, 0]])

eos = PengRobinson76(z=z, Tc=Tcs, Pc=Pcs, acentric_factor=omegas, bip=kijs)

print(eos.A_mix(P=1e6, T=115, z=np.array([0.5, 0.5])))
print(eos.A_ij(P=1e6, T=115))
print(eos.B_i(P=1e6, T=115))

