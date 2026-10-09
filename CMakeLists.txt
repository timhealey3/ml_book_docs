cmake_minimum_required(VERSION 4.1)
project(random_ml)

set(CMAKE_CXX_STANDARD 20)

add_executable(random_ml main.cpp
        Perceptron.cpp
        Perceptron.h)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

target_link_libraries(random_ml PRIVATE Eigen3::Eigen)