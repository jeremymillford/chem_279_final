cmake_minimum_required(VERSION 3.10)

# Project name
project(CNDO2_Overlap_Project)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find Armadillo
find_package(Armadillo REQUIRED)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/Include)

# Source files
set(SOURCES
    main.cpp
    Molecule.cpp
    Overlap.cpp
    SCF.cpp
)

# Create executable
add_executable(cndo2_overlap ${SOURCES})

# Link Armadillo
target_link_libraries(cndo2_overlap armadillo)
