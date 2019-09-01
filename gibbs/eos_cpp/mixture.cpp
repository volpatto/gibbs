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

}