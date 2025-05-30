cmake_minimum_required(VERSION 3.20)
project(critical_collapse VERSION 1.0 LANGUAGES CXX)

# Enable compile command database
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Enable CTest testing
enable_testing()

# ---- Dependencies ----
find_package(OpenMP REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW REQUIRED IMPORTED_TARGET fftw3)
pkg_check_modules(LAPACKE REQUIRED IMPORTED_TARGET lapacke)
find_package(nlohmann_json 3.11 REQUIRED)

# ---- Compiler Flags ----
add_library(critical_collapse_flags INTERFACE)
target_compile_features(critical_collapse_flags INTERFACE cxx_std_17)

set(gcc_like_cxx "$<COMPILE_LANG_AND_ID:CXX,ARMClang,AppleClang,Clang,GNU,LCC>")
set(msvc_cxx "$<COMPILE_LANG_AND_ID:CXX,MSVC>")
target_compile_options(critical_collapse_flags INTERFACE
  "$<${gcc_like_cxx}:$<BUILD_INTERFACE:-Wall;-Wpedantic;-Wextra;-Wshadow;-Wformat=2;-Wunused>>"
  "$<${msvc_cxx}:$<BUILD_INTERFACE:-W3>>"
)

# ---- Include Directories ----
include_directories(${PROJECT_SOURCE_DIR}/include)

# ---- Library Components ----
add_library(InitialConditionGenerator ${PROJECT_SOURCE_DIR}/src/InitialConditionGenerator.cpp)
add_library(NewtonSolver              ${PROJECT_SOURCE_DIR}/src/NewtonSolver.cpp)
add_library(ODEStepper                ${PROJECT_SOURCE_DIR}/src/ODEStepper.cpp)
add_library(OutputWriter              ${PROJECT_SOURCE_DIR}/src/OutputWriter.cpp)
add_library(ShootingSolver            ${PROJECT_SOURCE_DIR}/src/ShootingSolver.cpp)
add_library(SpectralTransformer       ${PROJECT_SOURCE_DIR}/src/SpectralTransformer.cpp)
add_library(Common                    ${PROJECT_SOURCE_DIR}/src/common.cpp)

# ----  declare intra-project links  ----
target_link_libraries(ODEStepper       PUBLIC InitialConditionGenerator)
target_link_libraries(NewtonSolver     PUBLIC InitialConditionGenerator ODEStepper Common)
target_link_libraries(ShootingSolver   PUBLIC InitialConditionGenerator ODEStepper)

# Apply common dependencies to all components
foreach(lib IN ITEMS InitialConditionGenerator NewtonSolver ODEStepper OutputWriter ShootingSolver SpectralTransformer Common)
    target_link_libraries(${lib}
        PRIVATE
            PkgConfig::FFTW
            PkgConfig::LAPACKE
            OpenMP::OpenMP_CXX
            nlohmann_json::nlohmann_json
            critical_collapse_flags
    )
    target_include_directories(${lib} PUBLIC ${PROJECT_SOURCE_DIR}/include)
endforeach()

# ---- Main Executable ----
add_executable(${PROJECT_NAME} ${PROJECT_SOURCE_DIR}/src/main.cpp)
target_link_libraries(${PROJECT_NAME} PUBLIC
    InitialConditionGenerator
    NewtonSolver
    ODEStepper
    OutputWriter
    ShootingSolver
    SpectralTransformer
    Common
    PkgConfig::FFTW
    PkgConfig::LAPACKE
    OpenMP::OpenMP_CXX
    nlohmann_json::nlohmann_json
    critical_collapse_flags
)

# ---- Tests ----
add_subdirectory(test)