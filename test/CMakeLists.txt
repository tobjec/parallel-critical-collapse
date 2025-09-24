# =============================================================================
# CMakeLists.txt — Critical Collapse (HPC for self-similar GR collapse)
#
# Goal of this file
# -----------------
# Provide a *clear* and *modular* build that can produce four variants:
#   1) Serial        (no threading, no MPI)
#   2) OpenMP        (shared-memory parallelism)
#   3) MPI           (distributed-memory parallelism)
#   4) Hybrid        (MPI + OpenMP)
#
# How to use (typical invocations)
# --------------------------------
# mkdir build && cd build
#   - Default (build everything it can detect):
#       cmake .. && cmake --build .
#   - Only serial:
#       cmake -DENABLE_OPENMP=OFF -DENABLE_MPI=OFF -DENABLE_HYBRID=OFF ..
#   - Only OpenMP:
#       cmake -DENABLE_OPENMP=ON  -DENABLE_MPI=OFF -DENABLE_HYBRID=OFF ..
#   - Only MPI:
#       cmake -DENABLE_OPENMP=OFF -DENABLE_MPI=ON  -DENABLE_HYBRID=OFF ..
#   - Hybrid (MPI + OpenMP):
#       cmake -DENABLE_OPENMP=ON  -DENABLE_MPI=ON  -DENABLE_HYBRID=ON  ..
#
# Notes
# -----
# • FFTW and LAPACKE are discovered via pkg-config.
# • JSON is provided by nlohmann_json (CMake package).
# • We compile sources into object libraries per variant, then wrap them into
#   concrete static libraries and executables (cc_serial, cc_openmp, ...).
# • If a dependency is missing, the corresponding variant is skipped and a
#   friendly message is printed (no hard failure unless explicitly requested).
# =============================================================================

cmake_minimum_required(VERSION 3.20)
project(critical_collapse VERSION 1.0 LANGUAGES CXX)

# Export compile_commands.json for tooling (clangd, etc.)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ─────────────────────────── 1) User-visible build options ───────────────────
# Turn variants on/off here. We *also* always try to build a serial variant by
# default so that you can run locally even without parallel toolchains.
option(ENABLE_OPENMP  "Build OpenMP variant"                 ON)
option(ENABLE_MPI     "Build MPI variant"                    ON)
option(ENABLE_HYBRID  "Build MPI+OpenMP hybrid variant"      ON)
option(ENABLE_SERIAL  "Always build a pure serial variant"   ON)

# ─────────────────────────── 2) Dependencies ─────────────────────────────────
# We look for OpenMP/MPI *quietly* so the project still configures when absent.
find_package(OpenMP QUIET)
find_package(MPI    QUIET)

# FFTW3 + LAPACKE via pkg-config (required for any build)
find_package(PkgConfig       REQUIRED)
pkg_check_modules(FFTW    REQUIRED IMPORTED_TARGET fftw3)
pkg_check_modules(LAPACKE REQUIRED IMPORTED_TARGET lapacke)

# Header-only JSON
find_package(nlohmann_json   3.11 REQUIRED)

# ─────────────────────────── 3) Sanity checks ────────────────────────────────
# Fail early if the user explicitly asked for a variant but toolchain missing.
if(ENABLE_OPENMP AND NOT OpenMP_FOUND)
  message(FATAL_ERROR "ENABLE_OPENMP=ON but no OpenMP toolchain detected.")
endif()
if((ENABLE_MPI OR ENABLE_HYBRID) AND NOT MPI_CXX_FOUND)
  message(FATAL_ERROR "MPI requested but no MPI C++ compiler found.")
endif()

# ─────────────────────────── 4) Common compile flags ─────────────────────────
# Collect warnings/optimization flags in an INTERFACE target so all variants
# share the same baseline configuration.
add_library(critical_collapse_flags INTERFACE)
target_compile_features(critical_collapse_flags INTERFACE cxx_std_17)

# Nice warnings on GCC/Clang + reasonable defaults for MSVC.
set(gcc_like "$<COMPILE_LANG_AND_ID:CXX,ARMClang,AppleClang,Clang,GNU,LCC>")
set(msvc_like "$<COMPILE_LANG_AND_ID:CXX,MSVC>")
target_compile_options(critical_collapse_flags INTERFACE
  "$<${gcc_like}:-Wall;-Wextra;-Wpedantic;-Wshadow;-Wformat=2;-Wunused>"
  "$<${msvc_like}:/W3>"
  "$<$<AND:${gcc_like},$<CONFIG:Release>>:-O3;-DNDEBUG;-march=native>"
  "$<$<AND:${msvc_like},$<CONFIG:Release>>:/O2;/DNDEBUG>"
)

# ─────────────────────────── 5) Include path ─────────────────────────────────
# All headers live in include/
set(PROJECT_INCLUDE ${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_INCLUDE})

# ─────────────────────────── 6) Project sources ──────────────────────────────
# Single source list used by every variant/object library.
set(CC_SRC
    src/InitialConditionGenerator.cpp
    src/NewtonSolver.cpp
    src/ODEStepper.cpp
    src/OutputWriter.cpp
    src/ShootingSolver.cpp
    src/SpectralTransformer.cpp
    src/common.cpp)

# Helper: create one flavor of the *object* library (compiles sources once per
# variant for per-variant preprocessor defs and link flags).
function(make_core_objects out_target)
  add_library(${out_target} OBJECT ${CC_SRC})
  target_include_directories(${out_target} PUBLIC ${PROJECT_INCLUDE})
  target_link_libraries(${out_target} PUBLIC
        PkgConfig::FFTW PkgConfig::LAPACKE
        nlohmann_json::nlohmann_json
        critical_collapse_flags)
endfunction()

# ─────────────────────────── 6a) Object libraries per variant ────────────────
# Serial (no special defs)
make_core_objects(cc_core_serial)

# OpenMP flavor
if(ENABLE_OPENMP AND OpenMP_FOUND)
  make_core_objects(cc_core_openmp)
  target_compile_definitions(cc_core_openmp PUBLIC USE_OPENMP)
  target_link_libraries(cc_core_openmp PUBLIC OpenMP::OpenMP_CXX)
endif()

# MPI flavor
if(ENABLE_MPI AND MPI_CXX_FOUND)
  make_core_objects(cc_core_mpi)
  target_compile_definitions(cc_core_mpi PUBLIC USE_MPI)
  target_link_libraries(cc_core_mpi PUBLIC MPI::MPI_CXX)
endif()

# Hybrid (MPI + OpenMP) flavor
if(ENABLE_HYBRID AND OpenMP_FOUND AND MPI_CXX_FOUND)
  make_core_objects(cc_core_hybrid)
  target_compile_definitions(cc_core_hybrid PUBLIC USE_HYBRID)
  target_link_libraries  (cc_core_hybrid PUBLIC OpenMP::OpenMP_CXX MPI::MPI_CXX)
endif()

# ─────────────────────────── 7) Wrap object libs into static libs ────────────
# Usage: make_variant(<name> <object_libs...>)
# Rationale: create reusable static libraries that executables can link to.
function(make_variant variant_name)
  add_library(${variant_name} STATIC)
  target_link_libraries(${variant_name} PUBLIC ${ARGN}
        PkgConfig::FFTW PkgConfig::LAPACKE
        nlohmann_json::nlohmann_json
        critical_collapse_flags)
endfunction()

# ─────────────────────────── 8) Concrete variant libraries ───────────────────
if(ENABLE_SERIAL)
  make_variant(critical_collapse_serial cc_core_serial)
endif()

if(ENABLE_OPENMP AND OpenMP_FOUND)
  make_variant(critical_collapse_openmp cc_core_openmp)
endif()

if(ENABLE_MPI AND MPI_CXX_FOUND)
  make_variant(critical_collapse_mpi cc_core_mpi)
endif()

if(ENABLE_HYBRID AND OpenMP_FOUND AND MPI_CXX_FOUND)
  make_variant(critical_collapse_hybrid cc_core_hybrid)
endif()

# ─────────────────────────── 9) Executables ──────────────────────────────────
# Small helper to only add an executable if its corresponding library exists.
function(make_exe exe_name lib_name)
  if(TARGET ${lib_name})
    add_executable(${exe_name} src/main.cpp)
    target_link_libraries(${exe_name} PRIVATE ${lib_name})
  else()
    message(STATUS "Skipping '${exe_name}' — its library '${lib_name}' not built")
  endif()
endfunction()

make_exe(cc_serial  critical_collapse_serial)
make_exe(cc_openmp  critical_collapse_openmp)
make_exe(cc_mpi     critical_collapse_mpi)
make_exe(cc_hybrid  critical_collapse_hybrid)

# ─────────────────────────── 10) Tests ───────────────────────────────────────
# Add your tests under test/ (will be skipped if the directory is empty).
enable_testing()
add_subdirectory(test)

# ─────────────────────────── 11) Build summary ───────────────────────────────
message(STATUS "====== Variants built ======")
foreach(v serial openmp mpi hybrid)
  if(TARGET critical_collapse_${v})
    message(STATUS "  ✔  ${v}")
  endif()
endforeach()
