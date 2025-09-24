//==============================================================================
// SpectralTransformer.cpp
// Thin wrapper around FFTW for periodic spectral transforms and utilities.
// Features:
//   • Forward/backward FFTs for real/complex data (FFTW complex-to-complex).
//   • Spectral differentiation with 2π/period wave numbers (odd Nyquist-safe).
//   • λ-integration in Fourier space: solve (λ - i m k0) Ĝ = ˆf mode-wise.
//   • Mode decimation/expansion (halveModes/doubleModes) for Newton packing.
//   • Inhomogeneous first-order ODE solve via integrating factor in τ.
// Notes:
//   • FFTW uses e^{-i k x} for FORWARD; we invert directions to match DFT
//     conventions used elsewhere, hence the BACKWARD/ FORWARD swap in plans.
//   • Keep the user's one-line loop/if braces `{}` intact.
//==============================================================================

#include "SpectralTransformer.hpp"

//------------------------------------------------------------------------------
// Ctor: allocate FFTW work buffers and build forward/backward plans.
// We deliberately swap BACKWARD/FORWARD flags to match our DFT sign convention.
//------------------------------------------------------------------------------
SpectralTransformer::SpectralTransformer(size_t N_, real_t period_)
    : N(N_), period(period_), k0(2.0*M_PI / period_)
{
    forward_data = fftw_alloc_complex(N_);
    backward_data = fftw_alloc_complex(N_);

    // Changing FFT forward and backward flag due to DFT definition of FFTW3 (-exp(...) vs. exp(...))
    #ifdef USE_OPENMP
    // Guard FFTW planner (not thread-safe) with a critical section
    #pragma omp critical(fftw_planner)
    {
        forward_plan  = fftw_plan_dft_1d(static_cast<int>(N_), forward_data, backward_data, FFTW_BACKWARD, FFTW_ESTIMATE);
        backward_plan = fftw_plan_dft_1d(static_cast<int>(N_), backward_data, forward_data, FFTW_FORWARD,  FFTW_ESTIMATE);
    }
    #else
    forward_plan  = fftw_plan_dft_1d(static_cast<int>(N_), forward_data, backward_data, FFTW_BACKWARD, FFTW_ESTIMATE);
    backward_plan = fftw_plan_dft_1d(static_cast<int>(N_), backward_data, forward_data, FFTW_FORWARD,  FFTW_ESTIMATE);
    #endif
}

//------------------------------------------------------------------------------
// Dtor: free plans and work arrays.
//------------------------------------------------------------------------------
SpectralTransformer::~SpectralTransformer()
{
    fftw_destroy_plan(forward_plan);
    fftw_destroy_plan(backward_plan);
    fftw_free(forward_data);
    fftw_free(backward_data);
}

//------------------------------------------------------------------------------
// forwardFFT (real → complex): pack real input into complex buffer, execute,
// and scale by 1/N to obtain unitary-like convention.
//------------------------------------------------------------------------------
void SpectralTransformer::forwardFFT(const vec_real& in, vec_complex& out)
{
    for (size_t i=0; i<N; ++i)
    {
        forward_data[i][0] = in[i];
        forward_data[i][1] = 0.0;
    }

    fftw_execute(forward_plan);

    out.resize(N);
    for (size_t i=0; i<N; ++i)
    {
        out[i] = complex_t(backward_data[i][0], backward_data[i][1]) / static_cast<real_t>(N);
    }
}

//------------------------------------------------------------------------------
// backwardFFT (complex → real): inverse transform and extract real part.
//------------------------------------------------------------------------------
void SpectralTransformer::backwardFFT(const vec_complex& in, vec_real& out)
{
    for (size_t i=0; i<N; ++i)
    {
        backward_data[i][0] = in[i].real();
        backward_data[i][1] = in[i].imag();
    }

    fftw_execute(backward_plan);

    out.resize(N);
    for (size_t i=0; i<N; ++i)
    {
        out[i] = forward_data[i][0];
    }
}

//------------------------------------------------------------------------------
// forwardFFT (complex → complex): complex-to-complex forward with 1/N scaling.
//------------------------------------------------------------------------------
void SpectralTransformer::forwardFFT(const vec_complex& in, vec_complex& out)
{
    for (size_t i=0; i<N; ++i)
    {
        forward_data[i][0] = in[i].real();
        forward_data[i][1] = in[i].imag();
    }

    fftw_execute(forward_plan);

    out.resize(N);
    for (size_t i=0; i<N; ++i)
    {
        out[i] = complex_t(backward_data[i][0], backward_data[i][1]) / static_cast<real_t>(N);
    }
}

//------------------------------------------------------------------------------
// backwardFFT (complex → complex): inverse complex transform.
//------------------------------------------------------------------------------
void SpectralTransformer::backwardFFT(const vec_complex& in, vec_complex& out)
{
    for (size_t i=0; i<N; ++i)
    {
        backward_data[i][0] = in[i].real();
        backward_data[i][1] = in[i].imag();
    }

    fftw_execute(backward_plan);

    out.resize(N);
    for (size_t i=0; i<N; ++i)
    {
        out[i] = complex_t(forward_data[i][0], forward_data[i][1]);
    }
}

//------------------------------------------------------------------------------
// differentiate: spectral derivative in τ.
// Uses m = k for k < N/2 and m = k - N for k ≥ N/2 (wrap-around), and
// multiplies by -i m k0c. The Nyquist (k=N/2) is set to 0 for stability.
// Optionally override the period (changes k0c).
//------------------------------------------------------------------------------
void SpectralTransformer::differentiate(const vec_complex& in, vec_complex& out, real_t period_)
{
    real_t k0c = (period_ == 0.0) ? k0 : 2*M_PI / period_;
    size_t size = in.size();
        
    out.resize(size);
    for (size_t k=0; k<size; ++k)
    {
        if (k != size/2)
        {
            int m = (k<size/2) ? static_cast<int>(k) : static_cast<int>(k) - static_cast<int>(size);
            out[k] = -complex_t(0.0, m*k0c) * in[k];
        }
        else
        {
            out[size/2] = complex_t(0.0);
        }   
    }    
}

//------------------------------------------------------------------------------
// lamIntegrate: solve (λ - i m k0) * Ĝ_m = f̂_m for each Fourier mode m.
// This is the frequency-domain form of integrating factor solves. The DC and
// Nyquist modes are treated carefully to avoid division by ~0.
//------------------------------------------------------------------------------
void SpectralTransformer::lamIntegrate(const vec_complex& fk, vec_complex& out, complex_t lambda, real_t period_)
{
    real_t k0c = (period_== 0.0) ? k0 : 2*M_PI / period_;
    size_t size = fk.size();

    out.resize(size); 

    // m = 0 mode
    out[0] = lambda != complex_t(0.0) ? fk[0] / lambda : complex_t(0.0);

    for (size_t k=1; k<size; ++k)
    {
        if (k != size/2)
        {
            int m = (k < size/2) ? static_cast<int>(k) : static_cast<int>(k) - static_cast<int>(size);
            complex_t denom = lambda - complex_t(0.0, m*k0c);
            out[k] = (std::abs(denom) < 1e-14) ? complex_t(0.0) : fk[k] / denom;
        }
        else
        {
            // Stabilized Nyquist treatment
            out[size/2] = lambda / (std::pow(lambda,2) + std::pow(k0c*size/2.0, 2)) * fk[size/2]; 
        }   
    }
}

//------------------------------------------------------------------------------
// halveModes: downsample complex spectrum by 2, folding high/low halves and
// summing the two cosine Nyquist partners into the new center bin.
// Layout: [0 .. N/2-1 | N/2 | N/2+1 .. N-1]  → size→N/2 array.
//------------------------------------------------------------------------------
void SpectralTransformer::halveModes(const vec_complex& in, vec_complex& out)
{
    size_t N_in  = in.size();
    size_t N_out = N_in/2;

    vec_complex tmp(N_out);

    for (size_t k=0; k<N_out/2; ++k)
    {
        tmp[k] = in[k];
    }

    // Fold the two Nyquist partners
    tmp[N_out/2] = in[N_out/2] + in[3*N_out/2];

    for (size_t k=N_out/2+1; k<N_out; ++k)
    {
        tmp[k] = in[N_out+k];
    }

    out.swap(tmp);
}

//------------------------------------------------------------------------------
// doubleModes: upsample complex spectrum by 2, splitting the cosine Nyquist
// into two half-amplitude copies and zeroing interstitial high-freq slots.
// Result size is 2*N_in with appropriate zero padding.
//------------------------------------------------------------------------------
void SpectralTransformer::doubleModes(const vec_complex& in, vec_complex& out)
{
    size_t N_in  = in.size();
    size_t N_out = 2*N_in;

    out.resize(N_out);

    // Low frequencies (strictly below Nyquist)
    for (size_t k=0; k<N_in/2; ++k)
    {
        out[k] = in[k];
    }

    // Split the Nyquist cosine into two symmetric bins
    out[N_in/2]   = 0.5*in[N_in/2];
    out[3*N_in/2] = 0.5*in[N_in/2];

    // Upper half copied to the high-frequency end
    for (size_t k=N_in/2; k<N_in; ++k)
    {
        out[N_in+k] = in[k];
    }

    // Zero the guard band between the split Nyquist bins
    for (size_t k=N_in/2+1; k<3*N_in/2; ++k)
    {
        out[k] = complex_t(0.0);
    }
}

//------------------------------------------------------------------------------
// solveInhom: solve the scalar first-order inhomogeneous ODE
//       X' + f(τ) X = g(τ)
// on a periodic domain using integrating factors in spectral space.
// Steps:
//   1) F' = f  →  F via spectral "integration" with λ=0 (DC handled).
//   2) h = −g e^{F}.  Solve Y' − f Y = h  in Fourier: (i m k0) Ŷ_m = ˆh_m
//      implemented via lamIntegrate with λ = f̂_0 (consistent with step 1).
//   3) X = e^{−F} Y.
//------------------------------------------------------------------------------
void SpectralTransformer::solveInhom(const vec_real& f, const vec_real& g, vec_real& x, real_t period_)
{
    // Step 1: F from f (F' = f)
    vec_complex f_hat, F_hat;
    forwardFFT(f, f_hat);
    lamIntegrate(f_hat, F_hat, complex_t(0.0), period_);
    backwardFFT(F_hat, x);

    vec_real F_real = x;
    vec_real h(N);

    // Step 2: h(τ) = −g(τ) e^{F(τ)}
    for (size_t i=0; i<N; ++i)
    {
        h[i] = -g[i] * std::exp(F_real[i]);
    }

    // Solve for Y in spectral domain, then invert
    vec_complex h_hat, H_hat;
    forwardFFT(h, h_hat);
    lamIntegrate(h_hat, H_hat, f_hat[0], period_);
    backwardFFT(H_hat, x);

    // Step 3: X = e^{-F} Y
    for (size_t i=0; i<N; ++i)
    {
        x[i] *= std::exp(-F_real[i]);
    }
}
