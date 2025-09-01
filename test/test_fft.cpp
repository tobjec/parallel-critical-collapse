#include "common.hpp"
#include "SpectralTransformer.hpp"

int main()
{

    constexpr int N{16};

    SpectralTransformer fft(N, 1.0);

    vec_real in_real(N), ref_in_real(N);
    vec_complex in_comp(N), ref_in_comp(N), out_real(N), out_comp(N);
    vec_complex halve_real(N/2), double_real(N), halve_comp(N/2), double_comp(N);

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
        in_comp[i] = std::polar(1.0, 2*M_PI/static_cast<real_t>(N)*static_cast<real_t>(i)); 
    }

    fft.forwardFFT(in_real, out_real);
    fft.forwardFFT(in_comp, out_comp);
    fft.backwardFFT(out_real, ref_in_real);
    fft.backwardFFT(out_comp, ref_in_comp);

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

    return 0;
}