#include <variant>
#include <optional>
#include <stdexcept>
#include <unsupported/Eigen/Polynomials>
#include <Eigen/Dense>
#include <vector>
#include <cmath>

#include "eos_cpp/solvers.hpp"
#include "eos_cpp/constants.hpp"

namespace eos::solvers {
    using namespace Eigen;

    std::variant<ArrayXd, std::nullopt_t> cubic_polynomial_real_positive_roots(const Vector4d &coefficients) {

        if (coefficients.size() != 4) {
            throw std::invalid_argument("This function is only valid for cubic polynomials. Please provide 4 "
                                        "coefficients only.");
        }

        bool has_a_real_root;

        PolynomialSolver<double, 3> solver;

        solver.compute(coefficients);

        auto smallest_real_root = solver.smallestRealRoot(has_a_real_root);
        auto greatest_real_root = solver.greatestRealRoot(has_a_real_root);

        if (has_a_real_root) {
            return select_polynomial_real_positive_roots(smallest_real_root, greatest_real_root);
        }
        else {
            return std::nullopt;
        }

    }

    std::variant<ArrayXd, std::nullopt_t> select_polynomial_real_roots(
            const double &smallest_root,
            const double &greatest_root) {
        if (smallest_root != greatest_root) {
            ArrayXd roots(2);
            roots << smallest_root, greatest_root;
            return roots;
        }
        else {
            ArrayXd root(1);
            root << greatest_root;
            return root;
        }
    }

    std::variant<ArrayXd, std::nullopt_t> select_polynomial_real_positive_roots(
            const double &smallest_root,
            const double &greatest_root) {
        if (greatest_root < 0) {
            return std::nullopt;
        }
        else if (smallest_root < 0) {
            ArrayXd root(1);
            root << greatest_root;
            return root;
        }
        else {
            if (smallest_root != greatest_root) {
                ArrayXd roots(2);
                roots << smallest_root, greatest_root;
                return roots;
            }
            else {
                ArrayXd root(1);
                root << greatest_root;
                return root;
            }
        }
    }

    std::variant<ArrayXd, std::nullopt_t> cubic_polynomial_roots(const Vector4d &coefficients) {

        if (coefficients.size() != 4) {
            throw std::invalid_argument("This function is only valid for cubic polynomials. Please provide 4 "
                                        "coefficients only.");
        }
        PolynomialSolver<double, 3> solver;

        solver.compute(coefficients);

        std::vector<double> real_roots;
        solver.realRoots(real_roots);
        std::vector<double> non_zero_real_roots;
        for (auto v : real_roots) {
            if (std::abs(v) > eos::constants::tol_zero)
                non_zero_real_roots.push_back(v);
        }

        const int number_of_real_roots = non_zero_real_roots.size();
        Map<ArrayXd> mapRealRoots(non_zero_real_roots.data(), number_of_real_roots);

        if (number_of_real_roots > 0)
            return mapRealRoots;
        else
            return std::nullopt;

    }

    ArrayXd cubic_cardano_real_roots(const Vector4d &coefficients) {

        if (coefficients.size() != 4) {
            throw std::invalid_argument("This function is only valid for cubic polynomials. Please provide 4 "
                                        "coefficients only.");
        }

        auto r = coefficients(0) / coefficients(3);
        auto q = coefficients(1) / coefficients(3);
        auto p = coefficients(2) / coefficients(3);

        // Normal form coefficients
        auto a = (1. / 3.) * (3 * q - p * p);
        auto b = (1. / 27.) * (2. * p * p * p - 9. * p * q + 27. * r);

        auto delta = b * b / 4. + a * a * a / 27.;

        std::vector<double> roots_vector;
        if (delta > 0) {
            auto A = std::cbrt(- b / 2. + std::sqrt(delta));
            auto B = std::cbrt(- b / 2. - std::sqrt(delta));
            roots_vector.push_back(A + B - p / 3.);
        }

        if (delta == 0) {
            ArrayXd roots(2);
            auto sqrt_minus_a_divided_by_three = std::sqrt(- a / 3.);  // to avoid same calculation and function calls
            if (b > 0) {
                roots_vector.push_back(-2 * sqrt_minus_a_divided_by_three - p / 3.);
                roots_vector.push_back(sqrt_minus_a_divided_by_three - p / 3.);
            }
            else if (b < 0) {
                roots_vector.push_back(2 * sqrt_minus_a_divided_by_three - p / 3.);
                roots_vector.push_back(-sqrt_minus_a_divided_by_three - p / 3.);
            }
            else {
                roots_vector.push_back(0. - p / 3.);
            }
        }

        if (delta < 0) {
            double phi;
            if (b > 0) {
                phi = -std::acos(std::sqrt((b * b / 4.) / (-a * a * a / 27)));
            }
            else {
                phi = std::acos(std::sqrt((b * b / 4.) / (-a * a * a / 27)));
            }

            auto sqrt_minus_a_divided_by_three = std::sqrt(- a / 3.);  // to avoid same calculation and function calls
            auto y_1 = 2 * sqrt_minus_a_divided_by_three * std::cos(phi / 3. + 2. * eos::constants::pi / 3.);
            auto y_2 = 2 * sqrt_minus_a_divided_by_three * std::cos(phi / 3. + 2. * 2. * eos::constants::pi / 3.);
            auto y_3 = 2 * sqrt_minus_a_divided_by_three * std::cos(phi / 3. + 2. * 3. * eos::constants::pi / 3.);

            roots_vector.push_back(y_1 - p / 3.);
            roots_vector.push_back(y_2 - p / 3.);
            roots_vector.push_back(y_3 - p / 3.);
        }

        const int number_of_real_roots = roots_vector.size();
        return Map<ArrayXd> (roots_vector.data(), number_of_real_roots);
    }

    std::variant<ArrayXd, std::nullopt_t> cubic_cardano_real_positive_roots(const Vector4d &coefficients) {

        auto roots = cubic_cardano_real_roots(coefficients);
        auto smallest_real_root = roots.minCoeff();
        auto greatest_real_root = roots.maxCoeff();

        return select_polynomial_real_positive_roots(smallest_real_root, greatest_real_root);
    }

}