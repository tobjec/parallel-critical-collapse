cmake_minimum_required(VERSION 3.16)
project(critical_collapse LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/build)

# Enable OpenMP
find_package(OpenMP REQUIRED)

# Find FFTW and LAPACKE
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW REQUIRED IMPORTED_TARGET fftw3)
pkg_check_modules(LAPACKE REQUIRED IMPORTED_TARGET lapacke)

# JSON (header-only library)
include(FetchContent)
FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.hpp
)
FetchContent_MakeAvailable(json)

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Source files
file(GLOB_RECURSE SOURCES
    ${PROJECT_SOURCE_DIR}/src/*.cpp
)

# Executable
add_executable(critical_collapse ${SOURCES})

# Link libraries
target_link_libraries(critical_collapse
    PkgConfig::FFTW
    PkgConfig::LAPACKE
    OpenMP::OpenMP_CXX
)

# Add JSON
target_include_directories(critical_collapse PRIVATE
    ${json_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/include
)

# Installation (optional)
install(TARGETS critical_collapse RUNTIME DESTINATION build)
