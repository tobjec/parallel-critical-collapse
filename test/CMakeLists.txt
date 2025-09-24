# ────────────────────────────────────────────────────────────────
# 1.  Decide which back-end library the tests will link against
#     (serial → OpenMP → MPI → hybrid, in that order)
# ────────────────────────────────────────────────────────────────
set(TEST_BACKEND_LIB "")
foreach(cand serial openmp mpi hybrid)
  if(TARGET critical_collapse_${cand})
    set(TEST_BACKEND_LIB critical_collapse_${cand})
    break()
  endif()
endforeach()

if(NOT TEST_BACKEND_LIB)
  message(FATAL_ERROR
          "No critical-collapse variant library was built, "
          "so the tests cannot be linked.")
endif()
message(STATUS "Tests will link against '${TEST_BACKEND_LIB}'")

# ────────────────────────────────────────────────────────────────
# 2.  Helper for the repetitive boiler-plate
# ────────────────────────────────────────────────────────────────
function(make_test exe src)
  add_executable(${exe} ${src})

  # back-end (serial / OpenMP / …) – brings in all solver objects
  target_link_libraries(${exe} PRIVATE ${TEST_BACKEND_LIB})

  # external math libraries that the individual test uses
  if(${exe} STREQUAL test_fft)
    target_link_libraries(${exe} PRIVATE PkgConfig::FFTW)
  elseif(${exe} STREQUAL test_lapack)
    target_link_libraries(${exe} PRIVATE PkgConfig::LAPACKE)
  endif()

  # common compile flags, headers, etc.
  target_link_libraries(${exe} PRIVATE critical_collapse_flags)
  target_include_directories(${exe} PRIVATE ${PROJECT_SOURCE_DIR}/include)

  # register the test with CTest
  add_test(NAME ${exe} COMMAND ${exe})
endfunction()

# ────────────────────────────────────────────────────────────────
# 3.  Concrete test targets
# ────────────────────────────────────────────────────────────────
make_test(test_fft       test_fft.cpp)
make_test(test_lapack    test_lapack.cpp)
make_test(test_init_gen  test_init_gen.cpp)