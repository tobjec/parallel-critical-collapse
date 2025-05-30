add_executable(test_fft test_fft.cpp)
add_executable(test_lapack test_lapack.cpp)
add_executable(test_init_gen test_init_gen.cpp)

target_link_libraries(test_fft PRIVATE
    SpectralTransformer
    Common
    OpenMP::OpenMP_CXX
    critical_collapse_flags
)

target_link_libraries(test_lapack PRIVATE
    PkgConfig::LAPACKE
    Common
    OpenMP::OpenMP_CXX
    critical_collapse_flags
)

target_link_libraries(test_init_gen PRIVATE 
    SpectralTransformer
    Common
    InitialConditionGenerator
    OpenMP::OpenMP_CXX
    critical_collapse_flags

)

target_include_directories(test_fft PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(test_lapack PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(test_init_gen PRIVATE ${PROJECT_SOURCE_DIR}/include)

add_test(NAME SpectralTransformerTest COMMAND test_fft)
add_test(NAME LapackeSolverTest COMMAND test_lapack)
add_test(NAME InitGenTest COMMAND test_init_gen)