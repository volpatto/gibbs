#ifndef EOS_CEOS_HPP
#define EOS_CEOS_HPP
#include <variant>
#include <optional>
#include <Eigen/Dense>
#include "eos_cpp/mixture.hpp"

namespace eos {

    using namespace Eigen;
    using namespace eos;
    using namespace mixture;

    /*
     * Generic Cubic EoS abstract class. It will serves as base class for derived
     * EoS classes.
     *
    */
    class CubicEos {

    public:

        CubicEos() = delete;

        explicit CubicEos(
                const Mixture &mixture,
                const ArrayXXd &bip,
                const double &Z_c,
                const double &Omega_a,
                const double &Omega_b)
                : mixture(mixture),
                bip(bip),
                _Z_c(Z_c),
                _Omega_a(Omega_a),
                _Omega_b(Omega_b){};

        virtual ~CubicEos() = default;

        inline int n_components() const {
            return this->mixture.Pc().size();
        };

        Mixture get_mixture() const;

        ArrayXXd get_bip() const;

        ArrayXd Tr(const double &T) const;

        ArrayXd Pr(const double &P) const;

        virtual ArrayXd m() const = 0;

        virtual ArrayXd alpha(const double &T) const = 0;

        ArrayXd calculate_a(const double &T) const;

        ArrayXd calculate_b() const;

        ArrayXd calculate_A(const double &P, const double &T) const;

        ArrayXd calculate_B(const double &P, const double &T) const;

        ArrayXXd calculate_A_ij(const double &P, const double &T) const;

        double calculate_A_mix(const double &P, const double &T, const ArrayXd &z) const;

        double calculate_B_mix(const double &P, const double &T, const ArrayXd &z) const;

        virtual Vector4d calculate_Z_cubic_polynomial_coeffs(const double &P, const double &T, const ArrayXd &z)
        const = 0;

        ArrayXd calculate_Z_factor(const double &P, const double &T, const ArrayXd &z) const;

        std::variant<double, std::nullopt_t> calculate_Z_minimal_energy(const double &P, const double &T, const ArrayXd &z)
        const;

        virtual ArrayXd calculate_fugacity_coefficients(const double &P, const double &T, const ArrayXd &z, const double
        &Z_factor) const = 0;

        ArrayXd calculate_fugacity(const double &P, const double &T, const ArrayXd &z, const double
        &Z_factor) const;

    protected:
        Mixture mixture;
        ArrayXXd bip;
        double _Z_c;
        double _Omega_a;
        double _Omega_b;

    };

    class PengRobinson : public CubicEos {

    public:

        explicit PengRobinson(
                const Mixture &mixture,
                const ArrayXXd &bip
                )
                : CubicEos(mixture, bip, Z_c, Omega_a, Omega_b) {};

        ~PengRobinson() = default;

        ArrayXd m() const override;

        ArrayXd alpha(const double &T) const override;

        Vector4d calculate_Z_cubic_polynomial_coeffs(const double &P, const double &T, const ArrayXd &z) const override;

        ArrayXd calculate_fugacity_coefficients(const double &P, const double &T, const ArrayXd &z, const double
        &Z_factor) const override ;

    protected:

        // Parameter values for Peng-Robinson EoS
        static constexpr double Z_c = 0.3074;
        static constexpr double Omega_a = 0.45724;
        static constexpr double Omega_b = 0.07780;

    };

    class PengRobinson78 : public PengRobinson {

    public:

        explicit PengRobinson78(
                const Mixture &mixture,
                const ArrayXXd &bip
        )
                : PengRobinson(mixture, bip) {};

        ~PengRobinson78() = default;

        ArrayXd m() const override;

    };

}

#endif //PETREOS_CEOS_HPP