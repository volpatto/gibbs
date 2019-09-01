#ifndef EOS_MIXTURE_HPP
#define EOS_MIXTURE_HPP

#include <stdexcept>
#include <Eigen/Dense>
#include <utility>

#include "eos_cpp/constants.hpp"

namespace mixture {

    using namespace Eigen;
    using namespace eos;

    class Mixture {

    private:
        ArrayXd _z;
        ArrayXd _Tc;
        ArrayXd _Pc;
        ArrayXd _omega;

    public:

        Mixture() = delete;

        explicit Mixture(
                ArrayXd z,
                ArrayXd Tc,
                ArrayXd Pc,
                ArrayXd omega
                ) : _z(std::move(z)), _Tc(std::move(Tc)), _Pc(std::move(Pc)), _omega(std::move(omega)) {

            check_overall_composition();
            validate_Pc();
            validate_Tc();
            validate_input_dimensions();

        }

        ~Mixture() = default;

        [[nodiscard]] ArrayXd z() const;

        [[nodiscard]] ArrayXd Tc() const;

        [[nodiscard]] ArrayXd Pc() const;

        [[nodiscard]] ArrayXd omega() const;

        void set_z(const ArrayXd &z);

        void set_Tc(const ArrayXd &Tc);

        void set_Pc(const ArrayXd &Pc);

        void set_omega(const ArrayXd &omega);

    private:

        inline void check_overall_composition() const {

            if (this->_z.sum() <= 1 - constants::tol_composition or this->_z.sum() >= 1 + constants::tol_composition) {
                throw std::invalid_argument("Overall composition must has summation equal/near to 1.");
            }

        }

        inline void validate_Tc() const {

            if ((this->_Tc < 0).any()) {
                throw std::invalid_argument("Temperature must be greater than zero.");
            }

        }

        inline void validate_Pc() const {

            if ((this->_Pc < 0).any()) {
                throw std::invalid_argument("Pressure must be greater than zero.");
            }

        }

        inline void validate_input_dimensions() const {

            bool size_omega_not_eq_Tc = this->_omega.size() != this->_Tc.size();
            bool size_omega_not_eq_Pc = this->_omega.size() != this->_Tc.size();
            bool size_Tc_not_eq_Pc = this->_Tc.size() != this->_Pc.size();
            bool size_z_not_eq_Pc = this->_z.size() != this->_Pc.size();

            if (size_omega_not_eq_Pc or size_omega_not_eq_Tc or size_Tc_not_eq_Pc or size_z_not_eq_Pc) {
                throw std::invalid_argument("Input values have incompatible dimensions.");
            }

        }

};

}

#endif //EOS_MIXTURE_HPP
