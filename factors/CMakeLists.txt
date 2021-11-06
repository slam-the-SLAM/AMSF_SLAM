cmake_minimum_required(VERSION 2.8.3)
project(ReprojectionOdometry)

set(CMAKE_BUILD_TYPE Release)

find_package(Eigen3 REQUIRED)

include_directories("${PROJECT_SOURCE_DIR}/include")

add_executable(testReprojectionOdometry src/testReprojectionFactor.cpp)
