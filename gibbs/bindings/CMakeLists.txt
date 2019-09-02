project(_eos)
find_package(pybind11 REQUIRED)
#set(pybind11_DIR third_party/pybind11/tools)
#include(pybind11Tools)

if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    add_compile_options("-fsized-deallocation")
endif()

include_directories("${CMAKE_SOURCE_DIR}/gibbs")

# Generate Python module with pybind11
pybind11_add_module(
        _eos
        _eos.cpp
)

target_link_libraries(
        _eos
        PRIVATE
            eos
)
install(
        TARGETS _eos
        RUNTIME DESTINATION ${ARTIFACTS_PATH}
        LIBRARY DESTINATION ${ARTIFACTS_PATH}
        ARCHIVE DESTINATION ${ARTIFACTS_PATH}
)