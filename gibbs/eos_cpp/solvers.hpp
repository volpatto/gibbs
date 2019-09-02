#ifndef EOS_SOLVERS_HPP
#define EOS_SOLVERS_HPP

#include <variant>
#include <any>
#include <optional>
#include <unsupported/Eigen/Polynomials>
#include <Eigen/Dense>

namespace eos::solvers {

    using namespace Eigen;

    /*
     * Computes the roots of a polynomial function p(x) = 0.
     * */
    std::variant<ArrayXd, std::nullopt_t> cubic_polynomial_real_positive_roots(const Vector4d &coefficients);

    /*
     * Solves p(x) = coeff0 + coeff1 * x + coeff2 * x^2 + coeff3 * x^3 = 0 analytically with Cardano method.
     * */
    ArrayXd cubic_cardano_real_roots(const Vector4d &coefficients);

    std::variant<ArrayXd, std::nullopt_t> cubic_cardano_real_positive_roots(const Vector4d &coefficients);

    std::variant<ArrayXd, std::nullopt_t> select_polynomial_real_roots(const double
    &smallest_root, const double
    &greatest_root);

    std::variant<ArrayXd, std::nullopt_t> select_polynomial_real_positive_roots(const double &smallest_root, const
    double &greatest_root);

    std::variant<ArrayXd, std::nullopt_t> cubic_polynomial_roots(const Vector4d &coefficients);

}

#endif