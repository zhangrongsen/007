cmake_minimum_required(VERSION 3.16)
project(outpost_estimation)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenCV
find_package(OpenCV REQUIRED)
# Eigen
find_package(Eigen3 REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

add_executable(outpost
    main.cpp
    filter.cpp
)
target_link_libraries(outpost ${OpenCV_LIBS})
