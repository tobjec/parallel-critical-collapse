cmake_minimum_required(VERSION 3.16)
project(critical_collapse LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---- dependencies ----------------------------------------------------------
find_package(OpenMP REQUIRED)
find_package(PkgConfig REQUIRED)

pkg_check_modules(FFTW    REQUIRED IMPORTED_TARGET fftw3)
pkg_check_modules(LAPACKE REQUIRED IMPORTED_TARGET lapacke)

# ---------- nlohmann/json ---------------------------------------------------
find_package(nlohmann_json 3.11 QUIET)
if (NOT TARGET nlohmann_json::nlohmann_json)
    add_library(nlohmann_json::nlohmann_json INTERFACE)
    target_include_directories(nlohmann_json::nlohmann_json INTERFACE
                               /usr/include)   
endif()

# ---- targets ---------------------------------------------------------------
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/src/*.cpp")
add_executable(critical_collapse ${SOURCES})

target_include_directories(critical_collapse PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(critical_collapse PRIVATE
    PkgConfig::FFTW
    PkgConfig::LAPACKE
    OpenMP::OpenMP_CXX
    nlohmann_json::nlohmann_json)

install(TARGETS critical_collapse RUNTIME DESTINATION bin)
