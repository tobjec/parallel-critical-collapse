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