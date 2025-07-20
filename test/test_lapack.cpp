#include <iostream>
#include <vector>
#include <cstring>
#include <lapacke.h>
#include <cassert>

using vec_real = std::vector<double>;
using mat_real = std::vector<std::vector<double>>;

int main()
{
    constexpr size_t N = 10;

    vec_real b = {6, 25, -11, 15, 10, 12, -1, 14, 7, 5};
    vec_real ref = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
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

    vec_real mat_flat(N * N);
    std::vector<lapack_int> ipiv(N);

    for (size_t i = 0; i < N; ++i)
        std::memcpy(&mat_flat[i * N], mat[i].data(), N * sizeof(double));

    lapack_int info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, static_cast<lapack_int>(N), 1,
                                    mat_flat.data(), static_cast<lapack_int>(N),
                                    ipiv.data(), b.data(), 1);

    if (info != 0)
    {
        std::cerr << "LAPACKE_dgesv failed with code " << info << std::endl;
        return EXIT_FAILURE;
    }

    for (size_t i = 0; i < N; ++i)
    {
        assert(std::abs(b[i] - ref[i]) < 1e-12);
    }

    return 0;
}