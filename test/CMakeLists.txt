add_executable(test_fft test_fft.cpp)

target_link_libraries(test_fft PRIVATE
    SpectralTransformer
    PkgConfig::FFTW
    PkgConfig::LAPACKE
    OpenMP::OpenMP_CXX
    nlohmann_json::nlohmann_json
    critical_collapse_flags
)

target_include_directories(test_fft PRIVATE ${PROJECT_SOURCE_DIR}/include)

add_test(NAME SpectralTransformerTest COMMAND test_fft)