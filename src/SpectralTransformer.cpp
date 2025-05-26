#include "SpectralTransformer.hpp"

SpectralTransformer::SpectralTransformer(int N_, real_t period_)
    : N(N_), N_freq(N_/2 + 1), period(period_), k0(2.0*M_PI / period_)
{
    real_data = fftw_alloc_real(N);
    freq_data = fftw_alloc_complex(N_freq);

    forward_plan  = fftw_plan_dft_r2c_1d(N, real_data, freq_data, FFTW_MEASURE);
    backward_plan = fftw_plan_dft_c2r_1d(N, freq_data, real_data, FFTW_MEASURE);
}

SpectralTransformer::~SpectralTransformer()
{
    fftw_destroy_plan(forward_plan);
    fftw_destroy_plan(backward_plan);
    fftw_free(real_data);
    fftw_free(freq_data);
}

void SpectralTransformer::forwardFFT(const vec_real& in, vec_complex& out)
{
    std::copy(in.begin(), in.end(), real_data);
    fftw_execute(forward_plan);

    out.resize(N_freq);
    for (int i = 0; i < N_freq; ++i)
    {
        out[i] = complex_t(freq_data[i][0], freq_data[i][1]) / static_cast<real_t>(N);
    }

}

void SpectralTransformer::inverseFFT(const vec_complex& in, vec_real& out)
{
    for (int i = 0; i < N_freq; ++i)
    {
        freq_data[i][0] = in[i].real();
        freq_data[i][1] = in[i].imag();
    }

    fftw_execute(backward_plan);
    out.resize(N);
    for (int i = 0; i < N; ++i)
    {
        out[i] = real_data[i];
    }

}

void SpectralTransformer::differentiate(const vec_complex& in, vec_complex& out)
{
    out.resize(N_freq);

    for (int k = 0; k < N_freq-1; ++k)
    {
        out[k] = -complex_t(0.0, k*k0) * in[k];
    }

    out[N_freq-1] = complex_t(0.0);

}

void SpectralTransformer::lamIntegrate(const vec_complex& fk, vec_complex& out, complex_t lambda)
{
    out.resize(N_freq);

    out[0] = lambda != complex_t(0.0) ? fk[0] / lambda : complex_t(0.0);

    for (int k = 1; k < N_freq-1; ++k)
    {
        complex_t denom = lambda - complex_t(0.0, k * k0);
        out[k] = (std::abs(denom) < 1e-14) ? complex_t(0.0) : fk[k] / denom;
    }

    out[N_freq-1] = lambda / (std::pow(lambda,2) - std::pow(k0*(N_freq-1),2)) * fk[N_freq-1];
}

real_t SpectralTransformer::interpolate(const vec_complex& fk, real_t x)
{
    complex_t result = 0.0;

    for (int k = 0; k < N_freq; ++k)
    {
        result += fk[k] * std::exp(complex_t(0.0, k * k0 * x));
    }

    return result.real();
}

void SpectralTransformer::halveModes(const vec_complex& in, vec_complex& out)
{
    int N_in = static_cast<int>(in.size());
    int N_out = N_in/2 + 1;

    for (int k = 0; k < N_out; ++k)
    {
        out[k] = in[k];
    }

    out[N_out] = 2.0 * in[N_out];

    out.resize(N_out);

}

void SpectralTransformer::doubleModes(const vec_complex& in, vec_complex& out)
{
    int N_in = static_cast<int>(in.size());
    int N_out = 2 * (N_in-1);

    out.resize(N_out);

    for (int k = 0; k < N_in; ++k)
    {
        out[k] = in[k];
    }

    out[N_in] = 0.5 * in[N_in];

}

void SpectralTransformer::solveInhom(const vec_real& f, const vec_real& g, vec_real& x)
{
    vec_complex f_hat, F_hat;
    forwardFFT(f, f_hat);
    lamIntegrate(f_hat, F_hat, complex_t(0.0));
    inverseFFT(F_hat, x);

    vec_real F_real = x;
    vec_real h(N);

    for (int i = 0; i < N; ++i)
    {
        h[i] = -g[i] * std::exp(F_real[i]);
    }

    vec_complex h_hat, H_hat;
    forwardFFT(h, h_hat);
    lamIntegrate(h_hat, H_hat, f_hat[0]);
    inverseFFT(H_hat, x);

    for (int i = 0; i < N; ++i)
    {
        x[i] *= std::exp(-F_real[i]);
    }

}