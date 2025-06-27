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
#include <thread>

// ========== Third-Party Libraries ==========
#include <nlohmann/json.hpp> // JSON for Modern C++

// ========== LAPACK ==========
#include <lapacke.h>        // LAPACK C interface

// ========== FFTW ==========
#include <fftw3.h>

// ========== Parallelism ==========
#ifdef USE_OPENMP
#include <omp.h>
#endif

#ifdef USE_MPI
#include <mpi.h>
#endif

#ifdef USE_HYBRID
#include <mpi.h>
#include <omp.h>
#endif

// ========== ENUM CLASSES ==========
enum class Scheme { IRK1, IRK2, IRK3 };

// ========== Aliases ===============
using real_t = double;
using complex_t = std::complex<real_t>;
using vec_real = std::vector<real_t>;
using vec_complex = std::vector<complex_t>;
using mat_real = std::vector<std::vector<real_t>>;
using mat_complex = std::vector<std::vector<complex_t>>;
using json = nlohmann::json;


// ============ Common Functions =======

bool almost_equal(complex_t a, complex_t b, double tol = 1e-15);

bool almost_equal(double a, double b, double tol = 1e-15);

void print_vec(const vec_real& vec);

void print_vec(const vec_complex& vec);

void write_mat(std::string filename, mat_real& mat);

void write_vec(std::string filename, vec_real& vec);

void write_vec(std::string filename, vec_complex& vec);
