#include <Eigen/Dense>
#include "eos_cpp/mixture.hpp"

namespace mixture {

    using namespace Eigen;

    ArrayXd Mixture::z() const {
        return this->_z;
    }

    ArrayXd Mixture::Tc() const {
        return this->_Tc;
    }

    ArrayXd Mixture::Pc() const {
        return this->_Pc;
    }

    ArrayXd Mixture::omega() const {
        return this->_omega;
    }

    void Mixture::set_z(const ArrayXd &z) {
        this->_z = z;
    }

    void Mixture::set_Tc(const ArrayXd &Tc) {
        this->_Tc = Tc;
    }

    void Mixture::set_Pc(const ArrayXd &Pc) {
        this->_Pc = Pc;
    }

    void Mixture::set_omega(const ArrayXd &omega) {
        this->_omega = omega;
    }

}