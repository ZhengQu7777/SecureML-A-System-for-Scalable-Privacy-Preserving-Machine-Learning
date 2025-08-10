cmake_minimum_required (VERSION 3.0)
project (Secure-ML)
set(NAME "Secure-ML")
set (CMAKE_CXX_STANDARD 11)
find_path(CMAKE_FOLDER NAMES cmake/emp-tool-config.cmake)
include(${CMAKE_FOLDER}/cmake/common.cmake)
include(${CMAKE_FOLDER}/cmake/source_of_randomness.cmake)
include(${CMAKE_FOLDER}/cmake/threading.cmake)
find_package(emp-ot REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
INCLUDE_DIRECTORIES(${EMP-OT_INCLUDE_DIRS})


add_executable(secure_ML
    src/secure_ML.cpp
    src/setup_phase.cpp
    src/util.cpp
    src/online_phase.cpp
    src/linear_regression.cpp
)

target_link_libraries(secure_ML ${EMP-OT_LIBRARIES} Eigen3::Eigen)