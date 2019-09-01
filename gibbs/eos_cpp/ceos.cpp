#include <variant>
#include <optional>
#include <stdexcept>
#include <cmath>
#include <vector>
#include <Eigen/Dense>
#include "eos_cpp/mixture.hpp"
#include "eos_cpp/ceos.hpp"
#include "eos_cpp/constants.hpp"
#include "eos_cpp/solvers.hpp"

using namespace Eigen;
using namespace eos;
using namespace mixture;

/*
 * Abstract Cubic EoS member functions definitions.
 * */
namespace eos {

    Mixture CubicEos::get_mixture() const { return this->mixture; }

    ArrayXXd CubicEos::get_bip() const { return this->bip; }

    ArrayXd CubicEos::Tr(const double &T) const {
        if (T < 0) {
            throw std::invalid_argument("Temperature must be greater than zero.");
        }
        return T / this->mixture.Tc();
    }

    ArrayXd CubicEos::Pr(const double &P) const {
        if (P < 0) {
            throw std::invalid_argument("Pressure must be greater than zero.");
        }
        return P / this->mixture.Pc();
    }

    ArrayXd CubicEos::calculate_a(const double &T) const {
        ArrayXd Tc_squared = this->mixture.Tc().square();
        ArrayXd numerator = (this->_Omega_a * constants::R * constants::R * Tc_squared * this->alpha(T));
        ArrayXd denominator = this->mixture.Pc();
        return (numerator / denominator);
    }

    ArrayXd CubicEos::calculate_b() const {
        return (this->_Omega_b * constants::R * this->mixture.Tc() / this->mixture.Pc());
    }

    ArrayXd CubicEos::calculate_A(const double &P, const double &T) const {
        return (this->calculate_a(T) * P / (constants::R * T * constants::R * T));
    }

    ArrayXd CubicEos::calculate_B(const double &P, const double &T) const {
        return (this->calculate_b() * P / (constants::R * T));
    }

    ArrayXXd CubicEos::calculate_A_ij(const double &P, const double &T) const {
        ArrayXd A = this->calculate_A(P, T);
        ArrayXXd bips = this->bip;
        auto size_A = A.size();
        ArrayXXd A_ij(size_A, size_A);

        for (int i = 0; i < size_A; ++i)
            for (int j = 0; j < size_A; ++j)
                A_ij(i, j) = (1 - bips(i, j)) * std::sqrt(A(i) * A(j));

        return A_ij;
    }

    double CubicEos::calculate_A_mix(const double &P, const double &T, const ArrayXd &z) const {
        VectorXd z_matrix = z.matrix();
        MatrixXd A_ij = this->calculate_A_ij(P, T).matrix();
        VectorXd z_dot_A_ij = (z_matrix.transpose() * A_ij);
        return z_matrix.dot(z_dot_A_ij);
    }

    double CubicEos::calculate_B_mix(const double &P, const double &T, const ArrayXd &z) const {
        VectorXd z_matrix = z.matrix();
        VectorXd B_i = this->calculate_B(P, T).matrix();
        return z_matrix.dot(B_i);
    }

    ArrayXd CubicEos::calculate_Z_factor(const double &P, const double &T, const ArrayXd &z) const {
        Vector4d coefficients = this->calculate_Z_cubic_polynomial_coeffs(P, T, z);

        auto result_roots = eos::solvers::cubic_polynomial_real_positive_roots(coefficients);

        Array2d empty_array;
        if (std::holds_alternative<ArrayXd>(result_roots)) {
            auto Z_roots_full = std::get<ArrayXd>(result_roots);
            if (Z_roots_full.size() < 1)
                return empty_array;

            auto smallest_root = Z_roots_full.minCoeff();
            auto greatest_root = Z_roots_full.maxCoeff();
            auto Z_positive_roots = eos::solvers::select_polynomial_real_positive_roots(
                    smallest_root,
                    greatest_root);

            if (std::holds_alternative<ArrayXd>(Z_positive_roots)) {
                return std::get<ArrayXd>(Z_positive_roots);
            }

            return empty_array;

        }

        return empty_array;

    }

    std::variant<double, std::nullopt_t> CubicEos::calculate_Z_minimal_energy(const double &P, const double &T, const ArrayXd &z) const {
        ArrayXd Z_roots = this->calculate_Z_factor(P, T, z);
        auto previous_normalized_gibbs_energy = std::numeric_limits<double>::infinity();
        std::variant<double, std::nullopt_t> Z_min;
        Z_min = std::nullopt;

        if (Z_roots.size() == 1)
            return Z_roots(0);
        else {
            for (int i = 0; i < Z_roots.size(); ++i) {
                ArrayXd fugacity = this->calculate_fugacity(P, T, z, Z_roots(i));
                ArrayXd ln_f = fugacity.log();
                double current_normalized_gibbs = z.matrix().dot(ln_f.matrix());
                if (current_normalized_gibbs <= previous_normalized_gibbs_energy)
                    Z_min = Z_roots(i);
            }
        }
        return Z_min;
    }

    ArrayXd CubicEos::calculate_fugacity(const double &P, const double &T, const ArrayXd &z, const double &Z_factor) const {
        return z * P * this->calculate_fugacity_coefficients(P, T, z, Z_factor);
    }

}

/*
 * Peng-Robinson member functions definitions.
 * */
namespace eos {

    ArrayXd PengRobinson::m() const {
        ArrayXd omega = this->mixture.omega();
        return (0.37464 + 1.54226 * omega - 0.26992 * omega * omega);
    }

    ArrayXd PengRobinson::alpha(const double &T) const {
        return ((1 + this->m() * (1 - this->Tr(T).sqrt())) * (1 + this->m() * (1 - this->Tr(T).sqrt())));
    }

    Vector4d PengRobinson::calculate_Z_cubic_polynomial_coeffs(const double &P, const double &T, const ArrayXd &z)
    const {
        double A = this->calculate_A_mix(P, T, z);
        double B = this->calculate_B_mix(P, T, z);

        // Obtaining the roots
        Vector4d coefficients;
        coefficients(0) = - (A * B - B * B - B * B * B);
        coefficients(1) = A - 3 * B * B - 2 * B;
        coefficients(2) = B - 1;
        coefficients(3) = 1;

        return coefficients;

    }

    ArrayXd PengRobinson::calculate_fugacity_coefficients(const double &P, const double &T, const ArrayXd &z,
                                                          const double &Z_factor) const {
        // Avoiding multiple function calls
        double A = this->calculate_A_mix(P, T, z);
        double B = this->calculate_B_mix(P, T, z);
        ArrayXd Bi_B = this->calculate_B(P, T) / B;
        ArrayXXd A_ij = this->calculate_A_ij(P, T);
        double sqrted_2 = std::sqrt(double{2.0});

        auto N = this->n_components();
        ArrayXd ln_phi(N);
        for (int i = 0; i < N; ++i) {
            // Computing fugacity coefficients by parts
            auto first_term = (Bi_B(i) * (Z_factor - 1.0));
            auto second_term = std::log(Z_factor - B);
            auto z_dot_A_ij = 0.0;
            for (int j = 0; j < N; ++j)
                z_dot_A_ij += z(j) * A_ij(i, j);
            auto third_term = A / (2.0 * sqrted_2 * B) * (Bi_B(i) - 2.0 / A * z_dot_A_ij);
            auto fourth_term = std::log(
                    (Z_factor + (1.0 + sqrted_2) * B) / (Z_factor + (1.0 - sqrted_2) * B)
                    );
            ln_phi(i) = first_term - second_term + third_term * fourth_term;
        }

        return ln_phi.exp();

    }
}

/*
 * Peng-Robinson 78 member functions definitions.
 * */
namespace eos {

    ArrayXd PengRobinson78::m() const {
        auto omega = this->mixture.omega();
        std::vector<double> m_vector;
        for (int i = 0; i < omega.size(); ++i) {
            double m_value;
            if (omega(i) <= 0.49)
                m_value = 0.37464 + 1.54226 * omega(i) - 0.26992 * omega(i) * omega(i);
            else
                m_value = 0.3796 + 1.485 * omega(i) - 0.1644 * omega(i) * omega(i) + 0.01667 * omega(i) * omega(i) *
                        omega(i);
            m_vector.push_back(m_value);
        }

        return Map<ArrayXd>(m_vector.data(), m_vector.size());
    }
}