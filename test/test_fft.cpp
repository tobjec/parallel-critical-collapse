#include "common.hpp"
#include "SpectralTransformer.hpp"

bool almost_equal(complex_t a, complex_t b, double tol = 1e-15)
{
    return std::abs(a.real() - b.real()) < tol && std::abs(a.imag() - b.imag()) < tol;
}

bool almost_equal(double a, double b, double tol = 1e-15)
{
    return std::abs(a - b) < tol;
}

int main()
{

    constexpr int N{16};

    SpectralTransformer fft(N, 1.0);

    vec_real in_real(N), ref_in_real(N), ref_diff_out(N), ref_diff(N), ref_int(N);
    vec_complex in_comp(N), ref_in_comp(N), out_real(N/2+1), out_comp(N), diff_real(N/2+1), int_real(N/2+1);
    vec_complex halve_real(N/4+1), double_real(N/2+1), halve_comp(N/2), double_comp(N);

    vec_complex ref_out_real = {
        {0,0},
        {0,-0.5},
        {0,0},
        {0,0},
        {0,0},
        {0,0},
        {0,0},
        {0,0},
        {0,0},
    };

    vec_complex ref_out_comp = {
        {0, 0},
        {1.0, 0},
        {0, 0},
        {0, 0},
        {0, 0},
        {0, 0},
        {0, 0},
        {0, 0},
        {0, 0},
        {0, 0},
        {0, 0},
        {0, 0},
        {0, 0},
        {0, 0},
        {0, 0},
        {0, 0}
    };
    
    for (size_t i=0; i<in_real.size(); ++i)
    {
        in_real[i] = std::sin(2*M_PI/static_cast<real_t>(N)*static_cast<real_t>(i));
        ref_diff[i] = std::cos(2*M_PI/static_cast<real_t>(N)*static_cast<real_t>(i));
        in_comp[i] = std::polar(1.0, 2*M_PI/static_cast<real_t>(N)*static_cast<real_t>(i)); 
    }

    fft.forwardFFT(in_real, out_real);
    fft.differentiate(out_real, diff_real);
    fft.lamIntegrate(diff_real, int_real, complex_t(0.0));
    fft.forwardFFTComplex(in_comp, out_comp);
    fft.inverseFFT(out_real, ref_in_real);
    fft.inverseFFT(diff_real, ref_diff_out);
    fft.inverseFFT(int_real, ref_int);
    fft.inverseFFTComplex(out_comp, ref_in_comp);

    for (size_t i=0; i<out_comp.size(); ++i)
    {
        if (i<out_real.size())
        {
            assert(almost_equal(out_real[i], ref_out_real[i]));
        }

        assert(almost_equal(out_comp[i], ref_out_comp[i]));
        assert(almost_equal(in_real[i], ref_in_real[i]));
        assert(almost_equal(in_comp[i], ref_in_comp[i]));

    }

    fft.halveModes(out_real, halve_real);
    fft.doubleModes(halve_real, double_real);
    fft.halveModes(double_real, double_real);

    fft.halveModes(out_comp, halve_comp);
    fft.doubleModes(halve_comp, double_comp);
    fft.halveModes(double_comp, double_comp);

    for (size_t i=0; i<double_comp.size(); ++i)
    {
        if (i<double_real.size())
        {
            assert(almost_equal(halve_real[i], double_real[i]));
        }

        assert(almost_equal(halve_comp[i], double_comp[i]));
    }

    /* for (size_t i=0; i<in_real.size(); ++i)
    {
        assert(almost_equal(in_real[i], ref_int[i]));
        assert(almost_equal(ref_diff[i], ref_diff_out[i]));
    } */



    return 0;
}