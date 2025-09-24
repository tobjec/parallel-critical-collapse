//==============================================================================
// test_spectral_transformer.cpp
// Minimal sanity test for SpectralTransformer:
//   1) Forward+backward FFT round-trips for real and complex inputs.
//   2) Known-reference spectra for a sine wave and a unit-modulus complex wave.
//   3) Mode decimation/expansion (halveModes/doubleModes) round-trip.
// Uses `almost_equal` helpers and plain `assert` for checks.
//==============================================================================

#include "common.hpp"
#include "SpectralTransformer.hpp"

int main()
{
    // Fixed small size so reference spectra are easy to reason about.
    constexpr int N{16};

    SpectralTransformer fft(N, 1.0);  // period = 1.0 → k0 = 2π

    // Input/output buffers
    vec_real    in_real(N), ref_in_real(N);
    vec_complex in_comp(N), ref_in_comp(N), out_real(N), out_comp(N);
    vec_complex halve_real(N/2), double_real(N), halve_comp(N/2), double_comp(N);

    // -------------------------------------------------------------------------
    // Reference spectra
    // For the real input sin(2π i/N), the unitary-scaled DFT has only ±1 modes.
    // With our sign/scale conventions the expected coefficients below are zeros
    // except the k=1 entry having purely imaginary part -1/2 (and conjugate in
    // the mirrored half; here we keep only the first N entries for the test).
    // -------------------------------------------------------------------------
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

    // For the complex input e^{i 2π i/N}, the DFT is a Kronecker delta at k=1
    // (scaled by 1), i.e. out[1] = 1 and all others 0.
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
    
    // Build inputs:
    //  • in_real[i]  = sin(2π i/N)
    //  • in_comp[i]  = e^{i 2π i/N}
    for (size_t i=0; i<in_real.size(); ++i)
    {
        in_real[i] = std::sin(2*M_PI/static_cast<real_t>(N)*static_cast<real_t>(i));
        in_comp[i] = std::polar(1.0, 2*M_PI/static_cast<real_t>(N)*static_cast<real_t>(i)); 
    }

    // FFTs and inverse FFTs
    fft.forwardFFT(in_real, out_real);
    fft.forwardFFT(in_comp, out_comp);
    fft.backwardFFT(out_real, ref_in_real);
    fft.backwardFFT(out_comp, ref_in_comp);

    // Check forward spectra against references; then verify round-trip.
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

    // Mode decimation/expansion on real-spectrum path
    fft.halveModes(out_real, halve_real);
    fft.doubleModes(halve_real, double_real);
    fft.halveModes(double_real, double_real); // return to half-size for compare

    // Same on complex-spectrum path
    fft.halveModes(out_comp, halve_comp);
    fft.doubleModes(halve_comp, double_comp);
    fft.halveModes(double_comp, double_comp);

    // Both paths should match their respective half-resolution representations.
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
