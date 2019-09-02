cmake_minimum_required(VERSION 2.8.12)

# Identifying and locating conda dir
if (EXISTS $ENV{HOME}/miniconda )
    set(miniconda "miniconda")
elseif(EXISTS $ENV{HOME}/miniconda3 )
    set(miniconda "miniconda3")
endif ()

set(PYTHON_EXECUTABLE $ENV{HOME}/${miniconda}/envs/gibbs-py36/bin/python)
set(conda_env_path $ENV{HOME}/${miniconda})
set(INCLUDE_CONDA ${conda_env_path}/include)

# Better don't use the following set command, but to get the right compilers form conda envs, it's necessary,
# unfortunately.
set(CMAKE_CXX_COMPILER $ENV{HOME}/${miniconda}/envs/gibbs-py36/bin/g++)
set(CMAKE_CXX_STANDARD 17)

project(eos LANGUAGES CXX C)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
add_compile_options("-fPIC")  # Necessary for the linking

# Eigen is header-only, so it only needs to include the directory with the headers
set(Eigen3_INCLUDE_DIR $ENV{HOME}/${miniconda}/envs/gibbs-py36/include/eigen3)
if( NOT Eigen3_INCLUDE_DIR )
    message(FATAL_ERROR "Please point the environment variable Eigen3_INCLUDE_DIR to the include directory of your
    Eigen3 installation.")
endif()
include_directories("${Eigen3_INCLUDE_DIR}")

find_package(pybind11 REQUIRED)
#find_package(Boost REQUIRED)

set(ARTIFACTS_PATH ${CMAKE_SOURCE_DIR}/gibbs/_cpp_binding)

add_subdirectory(gibbs)
target_link_libraries(eos)

include_directories(gibbs/eos_cpp)

message("\n*** Dependencies and infos:")
message("-- miniconda: $ENV{HOME}/${miniconda}")
message("-- cmake project root: ${CMAKE_SOURCE_DIR}")
message("-- conda path: ${conda_env_path}")
message("-- conda include path: ${INCLUDE_CONDA}")
message("-- Eigen3 path: ${Eigen3_INCLUDE_DIR}")
message("-- pybind11_DIR: " ${pybind11_DIR})
message("-- Python executable: " ${PYTHON_EXECUTABLE})
if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    message("-- CXX compiler: Clang")
    add_compile_options("-fsized-deallocation")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    message("-- CXX compiler: GNU")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    message("-- CXX compiler: Intel")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    message("-- CXX compiler: MSVC")
endif()
message("***\n")
