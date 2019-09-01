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

}