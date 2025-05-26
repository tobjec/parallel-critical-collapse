#include "SpectralTransformer.hpp"

SpectralTransformer::SpectralTransformer(int N_, real_t period) : N(N_), L(period)
{
    fft_in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    fft_out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);

    forward_plan = fftw_plan_dft_1d(N, fft_in, fft_out, FFTW_FORWARD, FFTW_MEASURE);
    backward_plan = fftw_plan_dft_1d(N, fft_in, fft_out, FFTW_BACKWARD, FFTW_MEASURE);
}

SpectralTransformer::~SpectralTransformer()
{
    fftw_destroy_plan(forward_plan);
    fftw_destroy_plan(backward_plan);
    fftw_free(fft_in);
    fftw_free(fft_out);
}

void SpectralTransformer::forward_fft(const vec_real& in, vec_complex& out)
{
    for (int i = 0; i < N; ++i)
    {
        fft_in[i][0] = in[i];
        fft_in[i][1] = 0.0;
    }

    fftw_execute(forward_plan);
    out.resize(N);

    for (int i = 0; i < N; ++i)
    {
        out[i] = complex_t(fft_out[i][0], fft_out[i][1]) / static_cast<real_t>(N);
    }
}

void SpectralTransformer::inverse_fft(const vec_complex& in, vec_real& out)
{
    for (int i = 0; i < N; ++i)
    {
        fft_in[i][0] = in[i].real();
        fft_in[i][1] = in[i].imag();
    }

    fftw_execute(backward_plan);
    out.resize(N);

    for (int i = 0; i < N; ++i)
    {
        out[i] = fft_out[i][0];
    }
}

void SpectralTransformer::differentiate(const vec_complex& in, vec_complex& out)
{
    const real_t k0 = 2 * M_PI / L;
    out.resize(N);

    for (int k = 0; k < N; ++k)
    {
        int m = (k <= N / 2) ? k : k - N;
        out[k] = complex_t(0.0, m * k0) * in[k];
    }
}

void SpectralTransformer::lamint(const vec_complex& fk, vec_complex& out, complex_t lambda)
{
    const real_t k0 = 2 * M_PI / L;
    out.resize(N);

    for (int k = 0; k < N; ++k)
    {
        int m = (k <= N / 2) ? k : k - N;
        complex_t denom = lambda - complex_t(0.0, m * k0);
        out[k] = (std::abs(denom) < 1e-14) ? complex_t(0.0) : fk[k] / denom;
    }
}

real_t SpectralTransformer::interpolate(const vec_complex& fk, real_t x)
{
    const real_t k0 = 2 * M_PI / L;
    complex_t result = 0.0;

    for (int k = 0; k < N; ++k)
    {
        int m = (k <= N / 2) ? k : k - N;
        result += fk[k] * std::exp(complex_t(0.0, m * k0 * x));
    }

    return result.real();
}

void SpectralTransformer::resample_modes(const vec_complex& in, vec_complex& out, int N_out)
{
    const int N_in = static_cast<int>(in.size());
    const int n_copy = std::min(N_in, N_out);

    vec_complex temp(N_out, complex_t(0.0));

    for (int k = 0; k < n_copy / 2; ++k)
        temp[k] = in[k];

    for (int k = 0; k < n_copy / 2; ++k)
        temp[N_out - k - 1] = in[N_in - k - 1];

    out = std::move(temp);
}

void SpectralTransformer::solve_inhomogeneous(const vec_real& f, const vec_real& g, vec_real& x)
{
    vec_complex f_hat, F_hat, F;
    forward_fft(f, f_hat);
    lamint(f_hat, F_hat, complex_t(0.0));
    inverse_fft(F_hat, x);

    F.resize(N);
    for (int i = 0; i < N; ++i)
        F[i] = complex_t(x[i]);

    vec_real h(N);
    for (int i = 0; i < N; ++i)
        h[i] = -g[i] * std::exp(F[i].real());

    vec_complex h_hat, H_hat;
    forward_fft(h, h_hat);
    lamint(h_hat, H_hat, f_hat[0]);
    inverse_fft(H_hat, x);

    for (int i = 0; i < N; ++i)
        x[i] *= std::exp(-F[i].real());
}