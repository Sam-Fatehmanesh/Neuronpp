cmake_minimum_required(VERSION 3.17)
project(Neuron__)
include(CTest)
enable_testing()
set(CMAKE_CXX_STANDARD 20)

add_executable(Neuron__ main.cpp)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ltbb -std=c++20 -fopenmp")
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

find_package(ArrayFire)
target_link_libraries(Neuron__ ArrayFire::afcuda)