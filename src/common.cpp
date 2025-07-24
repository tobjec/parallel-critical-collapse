#include "common.hpp"

bool almost_equal(complex_t a, complex_t b, double tol)
{
    return std::abs(a.real() - b.real()) < tol && std::abs(a.imag() - b.imag()) < tol;
}

bool almost_equal(double a, double b, double tol)
{
    return std::abs(a - b) < tol;
}

void print_vec(const vec_real& vec)
{
    std::cout << std::setprecision(5);
    for (size_t i=0; i<vec.size(); ++i)
    {
        if ((i+1)%10 == 0)
        {
            std::cout << vec[i] << "," << std::endl;
        }
        else if (i == vec.size()-1)
        {
            std::cout << vec[i] << std::endl;
        }
        else
        {
            std::cout << vec[i] << ", ";
        }
    }
}

void print_vec(const std::vector<std::string>& vec)
{
    std::cout << std::setprecision(5);
    for (size_t i=0; i<vec.size(); ++i)
    {
        if ((i+1)%10 == 0)
        {
            std::cout << vec[i] << "," << std::endl;
        }
        else if (i == vec.size()-1)
        {
            std::cout << vec[i] << std::endl;
        }
        else
        {
            std::cout << vec[i] << ", ";
        }
    }
}

void print_vec(const vec_complex& vec)
{
    std::cout << std::setprecision(16);
    for (size_t i=0; i<vec.size(); ++i)
    {
        if ((i+1)%3 == 0)
        {
            std::cout << "{" << vec[i].real() << ", ";
            std::cout << vec[i].imag() << "}," << std::endl;
        }
        else if (i == vec.size()-1)
        {
            std::cout << "{" << vec[i].real() << ", ";
            std::cout << vec[i].imag() << "}" << std::endl;
        }
        else
        {
            std::cout << "{" << vec[i].real() << ", " << vec[i].imag() << "}, ";
        }
    }
}

void write_mat(std::string filename, mat_real& mat)
{

    std::ofstream outfile(filename, std::ios::out);

    outfile << std::setprecision(16);
    for (size_t i=0; i<mat.size(); ++i)
    {
        for (size_t j=0; j<mat[i].size()-1; ++j)
        {
            outfile << mat[i][j] << ", ";
        }
        outfile << mat[i][mat[i].size()-1] << std::endl;
    }

}

void write_vec(std::string filename, vec_real& vec)
{
    std::ofstream outfile(filename, std::ios::out);

    outfile << std::setprecision(16);

    for (size_t i=0; i<vec.size()-1; ++i)
    {
        outfile << vec[i] << std::endl;
    }
    outfile << vec[vec.size()-1] << std::endl;
}

void write_vec(std::string filename, vec_complex& vec)
{
    std::ofstream outfile(filename, std::ios::out);

    outfile << std::setprecision(16);

    for (size_t i=0; i<vec.size()-1; ++i)
    {
        outfile <<"(" << vec[i].real() << ", " << vec[i].imag() << "), " << std::endl;
    }
    outfile <<"(" << vec[vec.size()-1].real() << ", " << vec[vec.size()-1].imag() << ")" << std::endl;
}

vec_real build_design_matrix(real_t x1, real_t x2, real_t x3)
{
    vec_real design_matrix(3*3, 0);
    
    design_matrix[0] = x1*x1;
    design_matrix[1] = x1;
    design_matrix[2] = 1.0;


    design_matrix[3] = x2*x2;
    design_matrix[4] = x2;
    design_matrix[5] = 1.0;


    design_matrix[6] = x3*x3;
    design_matrix[7] = x3;
    design_matrix[8] = 1.0;

    return design_matrix;
}

vec_real fit_quadratic_least_squares(const vec_real& x_vals, const vec_real& y_vals)
{
    const lapack_int m = 3; 
    const lapack_int n = 3; 
    const lapack_int nrhs = 1;
    const lapack_int lda = n;
    const lapack_int ldb = nrhs;

    vec_real A = build_design_matrix(x_vals[0], x_vals[1], x_vals[2]);
    vec_real b = y_vals;

    b.resize(n, 0.0);

    // Solve A * coeffs = b using least-squares
    lapack_int info = LAPACKE_dgels(LAPACK_ROW_MAJOR, 'N', m, n, nrhs, A.data(), lda, b.data(), ldb);
    if (info != 0)
    {
        throw std::runtime_error("LAPACKE_dgels failed with info = " + std::to_string(info));
    }

    // a, b, c 
    return {b[0], b[1], b[2]};
}