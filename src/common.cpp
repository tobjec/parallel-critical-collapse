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
    std::cout << std::setprecision(16);
    for (size_t i=0; i<vec.size(); ++i)
    {
        if ((i+1)%3 == 0)
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