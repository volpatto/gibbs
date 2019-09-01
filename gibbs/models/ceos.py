import numpy as np
import attr

from gibbs.mixture import Mixture

R = 8.3144598


def check_bip(instance, attribute, value):
    if value.shape[0] != value.shape[1]:
        raise ValueError("BIP's must be a 2-dim symmetric array.")
    if value.shape[0] != len(instance.mixture.Tc):
        raise ValueError("BIP's have incompatible dimension with input data such as critical temperature.")


@attr.s
class CEOS(object):
    """
    docstring here
        :param object: 
    """
    mixture = attr.ib(type=Mixture)
    bip = attr.ib(type=np.ndarray, validator=[check_bip])
    _Z_c = attr.ib(type=float)
    _Omega_a = attr.ib(type=float)
    _Omega_b = attr.ib(type=float)
    _m = attr.ib(type=np.ndarray, default=None)
    _b_i = attr.ib(type=np.ndarray, default=None)

    def __attrs_post_init__(self):
        # Used like caches, to prevent several evaluations
        self._m = self.m
        self._b_i = self.b_i

    @property
    def n_components(self):
        return len(self.mixture.Pc)

    def Tr(self, T):
        if T < 0:
            raise ValueError('Temperature must be greater than zero.')
        return T / self.mixture.Tc

    def Pr(self, P):
        if P < 0:
            raise ValueError('Pressure must be greater than zero.')
        return P / self.mixture.Pc

    @property
    def m(self):
        return NotImplementedError('Abstract method')

    def a_i(self, T):
        return self._Omega_a * R ** 2.0 * self.mixture.Tc * self.mixture.Tc / self.mixture.Pc * self.alpha(T)

    @property
    def b_i(self):
        return self._Omega_b * R * self.mixture.Tc / self.mixture.Pc

    def alpha(self, T):
        return NotImplementedError('Abstract method')

    def A_i(self, P, T):
        return self.a_i(T) * P / ((R * T) ** 2.0)

    def B_i(self, P, T):
        return self._b_i * P / (R * T)

    def A_mix(self, P, T, z):
        return np.dot(z, np.dot(z, self.A_ij(P, T)))

    def B_mix(self, P, T, z):
        return np.dot(z, self.B_i(P, T))

    def A_ij(self, P, T):
        return (1 - self.bip) * np.sqrt(np.outer(self.A_i(P, T), self.A_i(P, T)))

    def calculate_Z_factor(self, P, T, z):
        return NotImplementedError('Abstract method')

    def calculate_Z_minimal_energy(self, P, T, z):
        Z_roots = self.calculate_Z_factor(P, T, z)
        previous_normalized_gibbs_energy = np.inf
        Z_min = None
        if len(Z_roots) == 1:
            Z_min = Z_roots[0]
        else:
            for Z in Z_roots:
                fugacity = self.calculate_fugacity(P, T, z, Z)
                ln_f = np.log(fugacity)
                current_normalized_gibbs = np.dot(z, ln_f)
                if current_normalized_gibbs <= previous_normalized_gibbs_energy:
                    Z_min = Z
        return Z_min

    def calculate_fugacity_coefficient(self, P, T, z, Z_factor):
        return NotImplementedError('Abstract method')

    def calculate_fugacity(self, P, T, z, Z_factor):
        return z * P * self.calculate_fugacity_coefficient(P, T, z, Z_factor)


@attr.s
class PengRobinson(CEOS):
    """
    docstring here
        :param CEOS: 
    """
    _Z_c = attr.ib(default=0.3074)
    _Omega_a = attr.ib(default=0.45724)
    _Omega_b = attr.ib(default=0.07780)

    @property
    def m(self):
        omega = self.mixture.omega
        m = 0.37464 + 1.54226 * omega - 0.26992 * omega * omega
        return m

    def alpha(self, T):
        return (1 + self._m * (1 - np.sqrt(self.Tr(T)))) * (1 + self._m * (1 - np.sqrt(self.Tr(T))))

    def calculate_Z_factor(self, P, T, z):
        A = self.A_mix(P, T, z)
        B = self.B_mix(P, T, z)

        # Obtaining the roots
        coeff0 = -(A * B - B ** 2.0 - B ** 3.0)
        coeff1 = A - 3 * B ** 2.0 - 2 * B
        coeff2 = -(1 - B)
        coeff3 = 1
        coefficients = np.array([coeff3, coeff2, coeff1, coeff0])
        Z_roots = np.roots(coefficients)

        # A threshold is applied in imaginary part, since it can be 
        # numerically spurious.
        Z_real_roots = Z_roots.real[np.abs(Z_roots.imag) < 1e-5]

        # Filtering non-physical roots
        Z_real_roots = Z_real_roots[Z_real_roots >= 0.0]

        if len(Z_real_roots) == 3:
            Z_real_roots = np.append(Z_real_roots[0], Z_real_roots[-1])

        return Z_real_roots

    def calculate_fugacity_coefficient(self, P, T, z, Z_factor):
        # Avoiding multiple function calls
        A = self.A_mix(P, T, z)
        B = self.B_mix(P, T, z)
        Bi_B = self.B_i(P, T) / B
        A_ij = self.A_ij(P, T)
        sqrted_2 = np.sqrt(2.)

        # Computing fugacity coefficients by parts
        first_term = Bi_B * (Z_factor - 1)
        second_term = np.log(Z_factor - B)
        third_term = A / (2 * sqrted_2 * B) * (Bi_B - 2 / A * np.dot(z, A_ij))
        fourth_term = np.log(
            (Z_factor + (1 + sqrted_2) * B) / (Z_factor + (1 - sqrted_2) * B)
        )
        ln_phi = first_term - second_term + third_term * fourth_term

        return np.exp(ln_phi)


@attr.s
class PengRobinson78(PengRobinson):
    """
    docstring here
        :param CEOS:
    """

    @property
    def m(self):
        omega = self.mixture.omega
        m_low = 0.37464 + 1.54226 * omega - 0.26992 * omega * omega
        m_high = 0.3796 + 1.485 * omega - 0.1644 * omega * omega \
            + 0.01667 * omega * omega * omega
        m_value = np.where(self.mixture.omega > 0.49, m_high, m_low)
        return m_value


@attr.s
class SoaveRedlichKwong(CEOS):
    """
    docstring here
        :param CEOS:
    """
    _Z_c = attr.ib(default=1/3)
    _Omega_a = attr.ib(default=0.42748)
    _Omega_b = attr.ib(default=0.08664)

    @property
    def m(self):
        omega = self.mixture.omega
        m_value = 0.480 + 1.574 * omega - 0.176 * omega * omega
        return m_value

    def alpha(self, T):
        return (1 + self._m * (1 - np.sqrt(self.Tr(T)))) * (1 + self._m * (1 - np.sqrt(self.Tr(T))))

    def calculate_Z_factor(self, P, T, z):
        A = self.A_mix(P, T, z)
        B = self.B_mix(P, T, z)

        # Obtaining the roots
        coeff0 = -A * B
        coeff1 = A - B - B ** 2.0
        coeff2 = -1
        coeff3 = 1
        coefficients = np.array([coeff3, coeff2, coeff1, coeff0])
        Z_roots = np.roots(coefficients)

        # A threshold is applied in imaginary part, since it can be
        # numerically spurious.
        Z_real_roots = Z_roots.real[np.abs(Z_roots.imag) < 1e-5]

        # Filtering non-physical roots
        Z_real_roots = Z_real_roots[Z_real_roots >= 0.0]

        if len(Z_real_roots) == 3:
            Z_real_roots = np.append(Z_real_roots[0], Z_real_roots[-1])

        return Z_real_roots

    def calculate_fugacity_coefficient(self, P, T, z, Z_factor):
        # Avoiding multiple function calls
        A = self.A_mix(P, T, z)
        B = self.B_mix(P, T, z)
        Bi_B = self.B_i(P, T) / B
        A_ij = self.A_ij(P, T)

        # Computing fugacity coefficients by parts
        first_term = Bi_B * (Z_factor - 1)
        second_term = np.log(Z_factor - B)
        third_term = A / B * (Bi_B - 2 / A * np.dot(z, A_ij))
        fourth_term = np.log(1 + B / Z_factor)
        ln_phi = first_term - second_term + third_term * fourth_term

        return np.exp(ln_phi)
