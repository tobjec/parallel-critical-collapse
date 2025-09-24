//==============================================================================
// test_lapack_solve.cpp
// Minimal correctness test for LAPACK(e) dense linear solver.
//
// We solve A x = b for a crafted 10×10 system whose true solution is
// x = [1,2,3,4,5,6,7,8,9,10]^T. We call LAPACKE_dgesv in ROW-MAJOR mode,
// which performs an LU factorization with partial pivoting and overwrites
// the right-hand side `b` with the solution. The test asserts that the
// returned vector matches the reference to ~1e-12.
//==============================================================================

#include <iostream>
#include <vector>
#include <cstring>   // std::memcpy
#include <lapacke.h> // LAPACKE_dgesv
#include <cassert>   // assert
// NOTE: If your compiler complains about std::abs(double), include <cmath>.

// Convenience aliases for readability
using vec_real = std::vector<double>;
using mat_real = std::vector<std::vector<double>>;

int main()
{
    // Problem size
    constexpr size_t N = 10;

    // Right-hand side b (will be overwritten by LAPACKE_dgesv with the solution x)
    vec_real b   = {6, 25, -11, 15, 10, 12, -1, 14, 7, 5};

    // Known exact solution x_ref = [1, 2, ..., 10]^T for the matrix below.
    vec_real ref = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    // Coefficient matrix A (row-major, stored here as vector<vector<double>>)
    mat_real mat = {
        {10, -1,  2,  0,  0,  3,  0,  0,  0,  1},
        {-2, 11, -1,  3,  0,  0,  0,  0,  1,  0},
        { 2, -1, 10, -1,  2,  0,  0,  0,  0,  0},
        { 0,  3, -1,  8, -1, -1,  0,  0,  0,  0},
        { 0,  0,  2, -1,  7,  0,  0,  0,  1,  0},
        { 3,  0,  0, -1,  0, 12, -2,  0,  0,  0},
        { 0,  0,  0,  0,  0, -2,  9,  1,  0,  0},
        { 0,  0,  0,  0,  0,  0,  1,  6, -1,  0},
        { 0,  1,  0,  0,  1,  0,  0, -1,  5, -2},
        { 1,  0,  0,  0,  0,  0,  0,  0, -2,  8}
    };

    // Flatten A into a single contiguous buffer for LAPACKE (ROW-MAJOR layout).
    vec_real mat_flat(N * N);
    std::vector<lapack_int> ipiv(N); // pivot indices from LU factorization

    // Copy each row of `mat` into the flat buffer `mat_flat`
    for (size_t i = 0; i < N; ++i)
        std::memcpy(&mat_flat[i * N], mat[i].data(), N * sizeof(double));

    // Solve A x = b in-place:
    //  - On entry: mat_flat = A, b = RHS
    //  - On exit : mat_flat contains the LU factors; b contains the solution x
    lapack_int info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, static_cast<lapack_int>(N), 1,
                                    mat_flat.data(), static_cast<lapack_int>(N),
                                    ipiv.data(), b.data(), 1);

    // info < 0: i-th argument had an illegal value
    // info > 0: U(i,i) == 0 => singular matrix, no solution
    if (info != 0)
    {
        std::cerr << "LAPACKE_dgesv failed with code " << info << std::endl;
        return EXIT_FAILURE;
    }

    // Verify solution against the known reference vector.
    for (size_t i = 0; i < N; ++i)
    {
        assert(std::abs(b[i] - ref[i]) < 1e-12);
    }

    return 0;
}
