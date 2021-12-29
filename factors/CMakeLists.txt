cmake_minimum_required(VERSION 2.8.3)
#project(ReprojectionOdometry)
project(LidarFeatureOdometry)

set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_BUILD_TYPE Release)

find_package(Eigen3 REQUIRED)
find_package(OpenCV 3.2 REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRECTORIES}
    "${PROJECT_SOURCE_DIR}/include")

#add_executable(testReprojectionOdometry src/testReprojectionFactor.cpp)
#add_executable(testReprojectionOdometry src/testReprojectionFactor_withCVKeypoints.cpp)
#target_link_libraries(testReprojectionOdometry ${OpenCV_LIBS})
add_executable(testLidarFeatureOdometry src/testLidarFeatureFactor.cpp)
