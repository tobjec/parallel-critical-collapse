#pragma once

// ========== Standard Library ==========
#include <iostream>
#include <iomanip>
#include <cmath>
#include <complex>
#include <vector>
#include <array>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <functional>
#include <memory>
#include <cstring>
#include <string>
#include <sstream>
#include <fstream>
#include <filesystem>
#include <cassert>

// ========== Third-Party Libraries ==========
#include <nlohmann/json.hpp> // JSON for Modern C++

// ========== LAPACK ==========
#include <lapacke.h>        // LAPACK C interface

// ========== FFTW ==========
#include <fftw3.h>

// ========== Parallelism ==========
#ifdef _OPENMP
#include <omp.h>
#endif

// ========== MKL (Optional LAPACK backend) ==========
#ifdef USE_MKL
#include <mkl.h>
#endif

// ========== Project Headers ================

// ========== ENUM CLASSES ==========
enum class Scheme { IRK2, IRK3, RKF45 };

// ========== Aliases ===============
using real_t = double;
using complex_t = std::complex<real_t>;
using vec_real = std::vector<real_t>;
using vec_complex = std::vector<complex_t>;
using mat_real = std::vector<std::vector<real_t>>;
using mat_complex = std::vector<std::vector<complex_t>>;
using json = nlohmann::json;

