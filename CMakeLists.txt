cmake_minimum_required(VERSION 3.4...3.18)
project(pybind11_analysis)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})

add_subdirectory(extern/pybind11)
pybind11_add_module(pybind11_analysis pybind11_extension/main.cpp)

target_compile_definitions(pybind11_analysis
                           PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})