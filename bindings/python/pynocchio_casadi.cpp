#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>

#include "casadi_pinocchio_bridge.hpp"

namespace py = pybind11;

PYBIND11_MODULE(pynocchio_casadi, m) {


    m.def("generate_inv_dyn", generate_inv_dyn)
     .def("generate_forward_kin", generate_forward_kin)
     .def("generate_jacobian", generate_jacobian);


}
