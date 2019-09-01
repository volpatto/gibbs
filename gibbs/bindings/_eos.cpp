#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <memory>

#include <Eigen/Dense>
#include "eos_cpp/mixture.hpp"
#include "eos_cpp/solvers.hpp"
#include "eos_cpp/ceos.hpp"

using namespace Eigen;
namespace py = pybind11;

PYBIND11_MODULE(_eos, m) {
    m.doc() = "EoS C++ bindings"; // optional module docstring

    typedef mixture::Mixture Mix;
    py::class_<Mix, std::shared_ptr<Mix>> cMix (m, "Mixture", py::module_local());
    cMix.def(
            py::init<
                    const ArrayXd &,  // z
                    const ArrayXd &,  // Tc
                    const ArrayXd &,  // Pc
                    const ArrayXd &  // omega
                    >(),
            py::arg("z"),
            py::arg("Tc"),
            py::arg("Pc"),
            py::arg("omega")
    );

    cMix.def_property("z", &mixture::Mixture::z, &mixture::Mixture::set_z);
    cMix.def_property("Pc", &mixture::Mixture::Pc, &mixture::Mixture::set_Pc);
    cMix.def_property("Tc", &mixture::Mixture::Tc, &mixture::Mixture::set_Tc);
    cMix.def_property("omega", &mixture::Mixture::omega, &mixture::Mixture::set_omega);
    cMix.def(py::pickle(
            [](const Mix &p) { // __getstate__
                /* Return a tuple that fully encodes the state of the object */
                return py::make_tuple(p.z(), p.Pc(), p.Tc(), p.omega());
            },
            [](py::tuple t) { // __setstate__
                if (t.size() != 4)
                    throw std::runtime_error("Invalid state!");

                /* Create a new C++ instance */
                Mix p(t[0].cast<ArrayXd>(), t[1].cast<ArrayXd>(), t[2].cast<ArrayXd>(), t[3].cast<ArrayXd>());

                return p;
            }
            ));

    m.def(
            "_cubic_polynomial_real_positive_roots",
            &eos::solvers::cubic_polynomial_real_positive_roots,
            py::arg("coefficients"));

    m.def("_cubic_polynomial_roots", &eos::solvers::cubic_polynomial_roots, py::arg("coefficients"));

    typedef eos::CubicEos CEOS;
    py::class_<CEOS, std::shared_ptr<CEOS>> cCEOS (m, "CubicEos");
    cCEOS.def("Tr", &eos::CubicEos::Tr, py::arg("T"));
    cCEOS.def("Pr", &eos::CubicEos::Pr, py::arg("P"));
    cCEOS.def("m", &eos::CubicEos::m);
    cCEOS.def("alpha", &eos::CubicEos::alpha, py::arg("T"));
    cCEOS.def("calculate_a", &eos::CubicEos::calculate_a, py::arg("T"));
    cCEOS.def("calculate_b", &eos::CubicEos::calculate_b);
    cCEOS.def("calculate_A", &eos::CubicEos::calculate_A, py::arg("P"), py::arg("T"));
    cCEOS.def("calculate_B", &eos::CubicEos::calculate_B, py::arg("P"), py::arg("T"));
    cCEOS.def("calculate_A_ij", &eos::CubicEos::calculate_A_ij, py::arg("P"), py::arg("T"));
    cCEOS.def(
            "calculate_A_mix",
            &eos::CubicEos::calculate_A_mix,
            py::arg("P"),
            py::arg("T"),
            py::arg("z"));
    cCEOS.def(
            "calculate_B_mix",
            &eos::CubicEos::calculate_B_mix,
            py::arg("P"),
            py::arg("T"),
            py::arg("z"));
    cCEOS.def(
            "calculate_Z_cubic_polynomial_coeffs",
            &eos::CubicEos::calculate_Z_cubic_polynomial_coeffs,
            py::arg("P"),
            py::arg("T"),
            py::arg("z"));
    cCEOS.def(
            "calculate_Z_factor",
            &eos::CubicEos::calculate_Z_factor,
            py::arg("P"),
            py::arg("T"),
            py::arg("z"));
    cCEOS.def(
            "calculate_Z_minimal_energy",
            &eos::CubicEos::calculate_Z_minimal_energy,
            py::arg("P"),
            py::arg("T"),
            py::arg("z"));
    cCEOS.def(
            "calculate_fugacity_coefficients",
            &eos::CubicEos::calculate_fugacity_coefficients,
            py::arg("P"),
            py::arg("T"),
            py::arg("z"),
            py::arg("Z"));
    cCEOS.def(
            "calculate_fugacity",
            &eos::CubicEos::calculate_fugacity,
            py::arg("P"),
            py::arg("T"),
            py::arg("z"),
            py::arg("Z"));

    typedef eos::PengRobinson PR;
    py::class_<PR, CEOS, std::shared_ptr<PR>> cPR (m, "PengRobinson", py::module_local());
    cPR.def(
            py::init<
                    mixture::Mixture,  // mixture
                    ArrayXXd  // bip
                    >(),
             py::arg("mixture"),
             py::arg("bip"));
    cPR.def(py::pickle(
            [](const PR &p) { // __getstate__
                /* Return a tuple that fully encodes the state of the object */
                return py::make_tuple(p.get_mixture(), p.get_bip());
            },
            [](py::tuple t) { // __setstate__
                if (t.size() != 2)
                    throw std::runtime_error("Invalid state!");

                /* Create a new C++ instance */
                PR p(t[0].cast<Mix>(), t[1].cast<ArrayXXd>());

                return p;
            }
    ));

    typedef eos::PengRobinson78 PR78;
    py::class_<PR78, PR, std::shared_ptr<PR78>> cPR78 (m, "PengRobinson78", py::module_local());
    cPR78.def(
            py::init<
                    mixture::Mixture,  // mixture
                    ArrayXXd  // bip
            >(),
            py::arg("mixture"),
            py::arg("bip"));
    cPR78.def(py::pickle(
            [](const PR78 &p) { // __getstate__
                /* Return a tuple that fully encodes the state of the object */
                return py::make_tuple(p.get_mixture(), p.get_bip());
            },
            [](py::tuple t) { // __setstate__
                if (t.size() != 2)
                    throw std::runtime_error("Invalid state!");

                /* Create a new C++ instance */
                PR78 p(t[0].cast<Mix>(), t[1].cast<ArrayXXd>());

                return p;
            }
    ));

}
