cmake_minimum_required(VERSION 3.20)
project(critical_collapse VERSION 1.0 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ─────────────────────────────── 1. User-visible build options ──────────────
option(ENABLE_OPENMP  "Build OpenMP variant"             ON)
option(ENABLE_MPI     "Build MPI variant"                ON)
option(ENABLE_CUDA    "Build CUDA variant"               OFF)
option(ENABLE_HYBRID  "Build MPI+OpenMP hybrid variant"  OFF)
option(ENABLE_SERIAL  "Always build a pure serial variant" ON)

# ─────────────────────────────── 2. Dependencies (once) ─────────────────────
find_package(OpenMP QUIET)             # OpenMP::OpenMP_CXX
find_package(MPI    QUIET)             # MPI::MPI_CXX
enable_language(CUDA OPTIONAL)

find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW    REQUIRED IMPORTED_TARGET fftw3)
pkg_check_modules(LAPACKE REQUIRED IMPORTED_TARGET lapacke)
find_package(nlohmann_json 3.11 REQUIRED)

# ─────────────────────────────── 3. Sanity checks (fail fast) ───────────────
if(ENABLE_OPENMP AND NOT OpenMP_FOUND)
  message(FATAL_ERROR "ENABLE_OPENMP=ON but no OpenMP tool-chain detected")
endif()
if((ENABLE_MPI OR ENABLE_HYBRID) AND NOT MPI_CXX_FOUND)
  message(FATAL_ERROR "MPI requested but no MPI C++ compiler found")
endif()
if(ENABLE_CUDA AND NOT CMAKE_CUDA_COMPILER)
  message(FATAL_ERROR "ENABLE_CUDA=ON but no CUDA compiler detected")
endif()

# ─────────────────────────────── 4. Warnings / features interface ───────────
add_library(critical_collapse_flags INTERFACE)
target_compile_features(critical_collapse_flags INTERFACE cxx_std_17)

set(gcc_like_cxx "$<COMPILE_LANG_AND_ID:CXX,ARMClang,AppleClang,Clang,GNU,LCC>")
set(msvc_cxx     "$<COMPILE_LANG_AND_ID:CXX,MSVC>")
target_compile_options(critical_collapse_flags INTERFACE
  "$<${gcc_like_cxx}:-Wall;-Wextra;-Wpedantic;-Wshadow;-Wformat=2;-Wunused>"
  "$<${msvc_cxx}:/W3>"
)

# ─────────────────────────────── 5. Include path (once) ─────────────────────
set(PROJECT_INCLUDE ${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_INCLUDE})

# ─────────────────────────────── 6. Core source list ────────────────────────
set(CC_SRC
    src/InitialConditionGenerator.cpp
    src/NewtonSolver.cpp
    src/ODEStepper.cpp
    src/OutputWriter.cpp
    src/ShootingSolver.cpp
    src/SpectralTransformer.cpp
    src/common.cpp)

# helper: build one flavour of the object library
function(make_core_objects out_target extra_opts)
  add_library(${out_target} OBJECT ${CC_SRC})
  target_include_directories(${out_target} PUBLIC ${PROJECT_INCLUDE})
  target_link_libraries(${out_target} PRIVATE
        PkgConfig::FFTW PkgConfig::LAPACKE
        nlohmann_json::nlohmann_json
        critical_collapse_flags)
  if(extra_opts)
    target_compile_options(${out_target} PRIVATE ${extra_opts})
  endif()
endfunction()

# ─────────────────────────────── 6a. Object libraries ───────────────────────
make_core_objects(cc_core_serial "")

if((ENABLE_OPENMP OR ENABLE_HYBRID) AND OpenMP_FOUND)
  make_core_objects(cc_core_openmp "")
  target_compile_definitions(cc_core_openmp PUBLIC USE_OPENMP)
  target_link_libraries  (cc_core_openmp PUBLIC OpenMP::OpenMP_CXX)
endif()

if((ENABLE_MPI OR ENABLE_HYBRID) AND MPI_CXX_FOUND)
  make_core_objects(cc_core_mpi "")
  target_compile_definitions(cc_core_mpi PUBLIC USE_MPI)
  target_link_libraries  (cc_core_mpi PUBLIC MPI::MPI_CXX)
endif()

# ─────────────────────────────── 7. Helper to wrap into a variant ───────────
# Usage:  make_variant name object_lib compile_defs backend_libs [extra_srcs…]
function(make_variant variant_name object_lib defs backend_libs)
  # create an empty static archive
  add_library(${variant_name} STATIC ${ARGN})

  # bring in the object files *and* their usage requirements
  target_link_libraries(${variant_name} PUBLIC ${object_lib})

  # propagate additional compile definitions
  if(defs)
    target_compile_definitions(${variant_name} PUBLIC ${defs})
  endif()

  # common external + warning flags (propagate to dependants)
  target_link_libraries(${variant_name} PUBLIC
        PkgConfig::FFTW PkgConfig::LAPACKE
        nlohmann_json::nlohmann_json
        critical_collapse_flags)

  # backend runtimes (also PUBLIC so executables inherit them)
  if(backend_libs)
    target_link_libraries(${variant_name} PUBLIC ${backend_libs})
  endif()
endfunction()

# ─────────────────────────────── 8. Build the variant libraries ─────────────
if(ENABLE_SERIAL)
  make_variant(critical_collapse_serial cc_core_serial "" "")
endif()

if(ENABLE_OPENMP AND OpenMP_FOUND)
  make_variant(critical_collapse_openmp cc_core_openmp
               "USE_OPENMP" OpenMP::OpenMP_CXX)
endif()

if(ENABLE_MPI AND MPI_CXX_FOUND)
  make_variant(critical_collapse_mpi    cc_core_mpi
               "USE_MPI" MPI::MPI_CXX)
endif()

if(ENABLE_HYBRID AND OpenMP_FOUND AND MPI_CXX_FOUND)
  make_variant(critical_collapse_hybrid cc_core_openmp
               "USE_MPI;USE_OPENMP"
               "MPI::MPI_CXX;OpenMP::OpenMP_CXX")
endif()

if(ENABLE_CUDA AND CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  make_variant(critical_collapse_cuda cc_core_serial
               "USE_CUDA" CUDA::cudart
               src/cuda/kernels.cu)
  set_target_properties(critical_collapse_cuda PROPERTIES
                        CUDA_ARCHITECTURES 60;70;80)
endif()

# ─────────────────────────────── 9. Executables (guarded) ───────────────────
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
make_exe(cc_cuda    critical_collapse_cuda)

# ─────────────────────────────── 10. Tests ─────────────────────────────────
enable_testing()
add_subdirectory(test)

# ─────────────────────────────── 11. Summary ───────────────────────────────
message(STATUS "====== Variants built ======")
foreach(v serial openmp mpi hybrid cuda)
  if(TARGET critical_collapse_${v})
    message(STATUS "  ✔  ${v}")
  endif()
endforeach()