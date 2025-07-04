cmake_minimum_required(VERSION 3.20)
project(critical_collapse VERSION 1.0 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ─────────────────────────── 1. user-visible options ────────────────────────
option(ENABLE_OPENMP  "Build OpenMP variant"             ON)
option(ENABLE_MPI     "Build MPI variant"                ON)
option(ENABLE_HYBRID  "Build MPI+OpenMP hybrid variant"  ON)
option(ENABLE_SERIAL  "Always build a pure serial variant" ON)

# ─────────────────────────── 2. dependencies ────────────────────────────────
find_package(OpenMP QUIET)
find_package(MPI    QUIET)

find_package(PkgConfig       REQUIRED)
pkg_check_modules(FFTW    REQUIRED IMPORTED_TARGET fftw3)
pkg_check_modules(LAPACKE REQUIRED IMPORTED_TARGET lapacke)
find_package(nlohmann_json   3.11 REQUIRED)

# ─────────────────────────── 3. sanity checks  ──────────────────────────────
if(ENABLE_OPENMP AND NOT OpenMP_FOUND)
  message(FATAL_ERROR "ENABLE_OPENMP=ON but no OpenMP tool-chain detected")
endif()
if((ENABLE_MPI OR ENABLE_HYBRID) AND NOT MPI_CXX_FOUND)
  message(FATAL_ERROR "MPI requested but no MPI C++ compiler found")
endif()

# ─────────────────────────── 4. common flags  ───────────────────────────────
add_library(critical_collapse_flags INTERFACE)
target_compile_features(critical_collapse_flags INTERFACE cxx_std_17)

set(gcc_like "$<COMPILE_LANG_AND_ID:CXX,ARMClang,AppleClang,Clang,GNU,LCC>")
set(msvc_like "$<COMPILE_LANG_AND_ID:CXX,MSVC>")
target_compile_options(critical_collapse_flags INTERFACE
  "$<${gcc_like}:-Wall;-Wextra;-Wpedantic;-Wshadow;-Wformat=2;-Wunused>"
  "$<${msvc_like}:/W3>"
)

# ─────────────────────────── 5. include path  ───────────────────────────────
set(PROJECT_INCLUDE ${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_INCLUDE})

# ─────────────────────────── 6. project sources ─────────────────────────────
set(CC_SRC
    src/InitialConditionGenerator.cpp
    src/NewtonSolver.cpp
    src/ODEStepper.cpp
    src/OutputWriter.cpp
    src/ShootingSolver.cpp
    src/SpectralTransformer.cpp
    src/common.cpp)

# helper – build one flavour of the object library
function(make_core_objects out_target)
  add_library(${out_target} OBJECT ${CC_SRC})
  target_include_directories(${out_target} PUBLIC ${PROJECT_INCLUDE})
  target_link_libraries(${out_target} PUBLIC
        PkgConfig::FFTW PkgConfig::LAPACKE
        nlohmann_json::nlohmann_json
        critical_collapse_flags)
endfunction()

# ───────────────────────── 6a. object libraries  ────────────────────────────
make_core_objects(cc_core_serial)

if(ENABLE_OPENMP AND OpenMP_FOUND)
  make_core_objects(cc_core_openmp)
  target_compile_definitions(cc_core_openmp PUBLIC USE_OPENMP)
  target_link_libraries(cc_core_openmp PUBLIC OpenMP::OpenMP_CXX)
endif()

if(ENABLE_MPI AND MPI_CXX_FOUND)
  make_core_objects(cc_core_mpi)
  target_compile_definitions(cc_core_mpi PUBLIC USE_MPI)
  target_link_libraries(cc_core_mpi PUBLIC MPI::MPI_CXX)
endif()

if(ENABLE_HYBRID AND OpenMP_FOUND AND MPI_CXX_FOUND)
  make_core_objects(cc_core_hybrid)
  target_compile_definitions(cc_core_hybrid PUBLIC USE_HYBRID)
  target_link_libraries  (cc_core_hybrid PUBLIC
        OpenMP::OpenMP_CXX
        MPI::MPI_CXX)
endif()

# ────────────────────────── 7. helper to wrap a variant  ────────────────────
# Usage: make_variant name object_libs
function(make_variant variant_name)
  add_library(${variant_name} STATIC)
  target_link_libraries(${variant_name} PUBLIC ${ARGN}
        PkgConfig::FFTW PkgConfig::LAPACKE
        nlohmann_json::nlohmann_json
        critical_collapse_flags)
endfunction()

# ────────────────────────── 8. concrete variant libs  ───────────────────────
if(ENABLE_SERIAL)
  make_variant(critical_collapse_serial cc_core_serial)
endif()

if(ENABLE_OPENMP AND OpenMP_FOUND)
  make_variant(critical_collapse_openmp cc_core_openmp)
endif()

if(ENABLE_MPI AND MPI_CXX_FOUND)
  make_variant(critical_collapse_mpi    cc_core_mpi)
endif()

# hybrid now built from the new cc_core_hybrid objects
if(ENABLE_HYBRID AND OpenMP_FOUND AND MPI_CXX_FOUND)
  make_variant(critical_collapse_hybrid cc_core_hybrid)
endif()

# ────────────────────────── 9. executables  ────────────────────────────────
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

# ────────────────────────── 10. tests  ──────────────────────────────────────
enable_testing()
add_subdirectory(test)

# ────────────────────────── 11. summary  ────────────────────────────────────
message(STATUS "====== Variants built ======")
foreach(v serial openmp mpi hybrid)
  if(TARGET critical_collapse_${v})
    message(STATUS "  ✔  ${v}")
  endif()
endforeach()
