#include "SpectralTransformer.hpp"

SpectralTransformer::SpectralTransformer(int N_, real_t period_)
    : N(N_), N_freq(N_/2 + 1), period(period_), k0(2.0*M_PI / period_)
{
    real_data = fftw_alloc_real(N_);
    freq_data = fftw_alloc_complex(N_/2 + 1);
    complex_data = fftw_alloc_complex(N_);
    freq_data_complex = fftw_alloc_complex(N_);

    forward_plan  = fftw_plan_dft_r2c_1d(N_, real_data, freq_data, FFTW_MEASURE);
    backward_plan = fftw_plan_dft_c2r_1d(N_, freq_data, real_data, FFTW_MEASURE);
    forward_plan_complex  = fftw_plan_dft_1d(N_, complex_data, freq_data_complex, FFTW_FORWARD, FFTW_MEASURE);
    backward_plan_complex = fftw_plan_dft_1d(N_, freq_data_complex, complex_data, FFTW_BACKWARD, FFTW_MEASURE);
}

SpectralTransformer::~SpectralTransformer()
{
    fftw_destroy_plan(forward_plan);
    fftw_destroy_plan(backward_plan);
    fftw_destroy_plan(forward_plan_complex);
    fftw_destroy_plan(backward_plan_complex);
    fftw_free(real_data);
    fftw_free(freq_data);
    fftw_free(complex_data);
    fftw_free(freq_data_complex);
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

void SpectralTransformer::forwardFFTComplex(const vec_complex& in, vec_complex& out)
{
    for (int i = 0; i < N; ++i)
    {
        complex_data[i][0] = in[i].real();
        complex_data[i][1] = in[i].imag();
    }

    fftw_execute(forward_plan_complex);

    out.resize(N);
    for (int i = 0; i < N; ++i)
    {
        out[i] = complex_t(freq_data_complex[i][0], freq_data_complex[i][1]) / static_cast<real_t>(N);
    }

}

void SpectralTransformer::inverseFFTComplex(const vec_complex& in, vec_complex& out)
{
    for (int i = 0; i < N; ++i)
    {
        freq_data_complex[i][0] = in[i].real();
        freq_data_complex[i][1] = in[i].imag();
    }

    fftw_execute(backward_plan_complex);
    out.resize(N);
    for (int i = 0; i < N; ++i)
    {
        out[i] = complex_t(complex_data[i][0], complex_data[i][1]);
    }

}

void SpectralTransformer::differentiate(const vec_complex& in, vec_complex& out, real_t period_)
{
    real_t k0c = (period_ == 0.0) ? k0 : 2*M_PI / period_;
    int size = static_cast<int>(in.size());

    if (size == N_freq)
    {
        
        out.resize(size);

        for (int k = 0; k < size-1; ++k)
        {
            out[k] = complex_t(0.0, k*k0c) * in[k];
        }

        out[size-1] = complex_t(0.0);
    }
    else
    {
        
        out.resize(size);

        for (int k = 0; k < size-1; ++k)
        {
            if (k != size/2)
            {
                int m = (k < size/2) ? k : k-size;
                out[k] = complex_t(0.0, m*k0c) * in[k];
            }
            else
            {
                out[size/2] = complex_t(0.0);
            }   
        }
    }    
}

void SpectralTransformer::lamIntegrate(const vec_complex& fk, vec_complex& out, complex_t lambda, real_t period_)
{

    real_t k0c = (period_ == 0.0) ? k0 : 2*M_PI / period_;
    int size = static_cast<int>(fk.size());

    if (size == N_freq)
    {
        out.resize(size); 

        out[0] = lambda != complex_t(0.0) ? fk[0] / lambda : complex_t(0.0);

        for (int k = 1; k < size-1; ++k)
        {
            complex_t denom = lambda + complex_t(0.0, k * k0c);
            out[k] = (std::abs(denom) < 1e-14) ? complex_t(0.0) : fk[k] / denom;
        }

        out[size-1] = lambda / (std::pow(lambda,2) + std::pow(k0c*(size-1),2)) * fk[size-1];
    }
    else
    {
        out.resize(size); 

        out[0] = lambda != complex_t(0.0) ? fk[0] / lambda : complex_t(0.0);

        for (int k = 1; k < size-1; ++k)
        {
            if (k != size/2)
            {
                int m = (k < size/2) ? k : k-size;
                complex_t denom = lambda + complex_t(0.0, m * k0c);
                out[k] = (std::abs(denom) < 1e-14) ? complex_t(0.0) : fk[k] / denom;
            }
            else
            {
                out[size/2-1] = lambda / (std::pow(lambda,2) + std::pow(k0c*(size/2-1),2)) * fk[size/2-1];
            }   
        }
    }
}

real_t SpectralTransformer::interpolate(const vec_complex& fk, real_t x, real_t period_)
{
    real_t result = 0.0;
    real_t k0c = (period_ == 0.0) ? k0 : 2*M_PI / period_;
    int size = static_cast<int>(fk.size());

    if (size == N_freq)
    {
        result += fk[0].real();
        for (int k = 1; k < size-1; ++k)
        {
            real_t phase = k * k0c * x;
            result += 2.0 * (fk[k] * std::polar(1.0, phase)).real();
        }
        real_t phase = (size-1) * k0c * x;
        result += fk[size-1].real() * std::cos(phase);
    }
    else
    {
        result += fk[0].real();
        for (int k = 1; k < size/2; ++k)
        {
            real_t phase = k * k0c * x;
            result += (fk[k] * std::polar(-1.0, phase) + fk[size-k] * std::polar(1.0, phase)).real();
        }
        real_t phase = size/2 * k0c * x;
        result += fk[size/2].real() * std::cos(phase);

    }   

    return result;
}

void SpectralTransformer::halveModes(const vec_complex& in, vec_complex& out, int Nsize)
{
    int N_in = (Nsize > 0 ) ? Nsize : static_cast<int>(in.size());

    if (N_in == N_freq)
    {
        int N_out = N_in/2 + 1;

        for (int k = 0; k < N_out-1; ++k)
        {
            out[k] = in[k];
        }

        out[N_out-1] = in[N_out-1] + std::conj(in[N_out-1]); // unsure whether this is true

        out.resize(N_out);
    }
    else
    {
        int N_out = N_in/2;

        for (int k = 0; k < N_out/2; ++k)
        {
            out[k] = in[k];
        }

        out[N_out/2] = in[N_out/2] + in[3*N_out/2];

        for (int k = N_out/2+1; k < N_out; ++k)
        {
            out[k] = in[N_out+k];
        }

        out.resize(N_out);
    }
    
}

void SpectralTransformer::doubleModes(const vec_complex& in, vec_complex& out, int Nsize)
{
    int N_in = (Nsize > 0 ) ? Nsize : static_cast<int>(in.size());

    if (N_in == N_freq/2+1)
    {
        int N_out = 2 * N_in - 1;

        out.resize(N_out);

        for (int k = 0; k < N_in-1; ++k)
        {
            out[k] = in[k];
        }

        out[N_in-1] = 0.5 * in[N_in-1];

        for (int k = N_in; k < N_out; ++k)
        {
            out[k] = complex_t(0.0);
        }
    }
    else
    {
        int N_out = 2 * N_in;

        out.resize(N_out);

        for (int k = 0; k < N_in/2; ++k)
        {
            out[k] = in[k];
        }

        for (int k = N_in/2; k < N_in; ++k)
        {
            out[N_in+k] = in[k];
        }

        out[N_in/2] = 0.5 * out[3*N_in/2];
        out[3*N_in/2] *= 0.5;
        
        for (int k = N_in/2+1; k < 3*N_in/2; ++k)
        {
            out[k] = complex_t(0.0);
        }

    }
    

}

void SpectralTransformer::solveInhom(const vec_real& f, const vec_real& g, vec_real& x, real_t period_)
{
    vec_complex f_hat, F_hat;
    forwardFFT(f, f_hat);
    lamIntegrate(f_hat, F_hat, complex_t(0.0), period_);
    inverseFFT(F_hat, x);

    vec_real F_real = x;
    vec_real h(N);

    for (int i = 0; i < N; ++i)
    {
        h[i] = -g[i] * std::exp(F_real[i]);
    }

    vec_complex h_hat, H_hat;
    forwardFFT(h, h_hat);
    lamIntegrate(h_hat, H_hat, f_hat[0], period_);
    inverseFFT(H_hat, x);

    for (int i = 0; i < N; ++i)
    {
        x[i] *= std::exp(-F_real[i]);
    }

}