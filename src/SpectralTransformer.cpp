#include "SpectralTransformer.hpp"

SpectralTransformer::SpectralTransformer(size_t N_, real_t period_)
    : N(N_), period(period_), k0(2.0*M_PI / period_)
{
    forward_data = fftw_alloc_complex(N_);
    backward_data = fftw_alloc_complex(N_);

    // Changing FFT forward and backward flag due to DFT definition of FFTW3 (-exp(...) vs. exp(...))
    #ifdef USE_OPENMP

    #pragma omp critical(fftw_planner)
    {
        forward_plan  = fftw_plan_dft_1d(static_cast<int>(N_), forward_data, backward_data, FFTW_BACKWARD, FFTW_MEASURE);
        backward_plan = fftw_plan_dft_1d(static_cast<int>(N_), backward_data, forward_data, FFTW_FORWARD, FFTW_MEASURE);
    }
    
    #else

    forward_plan  = fftw_plan_dft_1d(static_cast<int>(N_), forward_data, backward_data, FFTW_BACKWARD, FFTW_MEASURE);
    backward_plan = fftw_plan_dft_1d(static_cast<int>(N_), backward_data, forward_data, FFTW_FORWARD, FFTW_MEASURE);

    #endif

}

SpectralTransformer::~SpectralTransformer()
{
    fftw_destroy_plan(forward_plan);
    fftw_destroy_plan(backward_plan);
    fftw_free(forward_data);
    fftw_free(backward_data);
}

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

void SpectralTransformer::inverseFFT(const vec_complex& in, vec_real& out)
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

void SpectralTransformer::inverseFFT(const vec_complex& in, vec_complex& out)
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

void SpectralTransformer::differentiate(const vec_complex& in, vec_complex& out, real_t period_)
{
    real_t k0c = (period_ == 0.0) ? k0 : 2*M_PI / period_;
    size_t size = in.size();
        
    out.resize(size);
    for (size_t k=0; k<size; ++k)
    {
        if (k != size/2)
        {
            int m = (k<size/2) ? k : k-size;
            out[k] = -complex_t(0.0, m*k0c) * in[k];
        }
        else
        {
            out[size/2] = complex_t(0.0);
        }   
    }    
}

void SpectralTransformer::lamIntegrate(const vec_complex& fk, vec_complex& out, complex_t lambda, real_t period_)
{

    real_t k0c = (period_== 0.0) ? k0 : 2*M_PI / period_;
    size_t size = fk.size();

    out.resize(size); 
    out[0] = lambda != complex_t(0.0) ? fk[0] / lambda : complex_t(0.0);
    for (size_t k=1; k<size; ++k)
    {
        if (k != size/2)
        {
            int m = (k < size/2) ? k : k-size;
            complex_t denom = lambda - complex_t(0.0, m*k0c);
            out[k] = (std::abs(denom) < 1e-14) ? complex_t(0.0) : fk[k] / denom;
        }
        else
        {
            out[size/2] = lambda / (std::pow(lambda,2) + std::pow(k0c*size/2.0, 2)) * fk[size/2]; 
        }   
    }
}

real_t SpectralTransformer::interpolate(const vec_complex& fk, real_t x, real_t period_)
{
    real_t result = 0.0;
    real_t k0c = (period_ == 0.0) ? k0 : 2*M_PI / period_;
    size_t size = fk.size();

    result += fk[0].real();
    for (size_t k=1; k<size/2; ++k)
    {
        real_t phase = k*k0c*x;
        result += (fk[k]*std::polar(1.0, -phase) + fk[size-k]*std::polar(1.0, phase)).real();
    }
    real_t phase = size/2.0*k0c*x;
    result += fk[size/2].real() * std::cos(phase);   

    return result;
}

void SpectralTransformer::halveModes(const vec_complex& in, vec_complex& out)
{
    size_t N_in = in.size();
    size_t N_out = N_in/2;

    vec_complex tmp(N_out);

    for (size_t k=0; k<N_out/2; ++k)
    {
        tmp[k] = in[k];
    }

    tmp[N_out/2] = in[N_out/2] + in[3*N_out/2];

    for (size_t k=N_out/2+1; k<N_out; ++k)
    {
        tmp[k] = in[N_out+k];
    }

    out.swap(tmp);
    
}

void SpectralTransformer::doubleModes(const vec_complex& in, vec_complex& out)
{
    size_t N_in = in.size();
    size_t N_out = 2*N_in;

    out.resize(N_out);
    for (size_t k=0; k<N_in/2; ++k)
    {
        out[k] = in[k];
    }

    out[N_in/2] = 0.5*in[N_in/2];
    out[3*N_in/2] = 0.5*in[N_in/2];

    for (size_t k=N_in/2; k<N_in; ++k)
    {
        out[N_in+k] = in[k];
    }

    for (size_t k=N_in/2+1; k<3*N_in/2; ++k)
    {
        out[k] = complex_t(0.0);
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

    for (size_t i=0; i<N; ++i)
    {
        h[i] = -g[i] * std::exp(F_real[i]);
    }

    vec_complex h_hat, H_hat;
    forwardFFT(h, h_hat);
    lamIntegrate(h_hat, H_hat, f_hat[0], period_);
    inverseFFT(H_hat, x);

    for (size_t i=0; i<N; ++i)
    {
        x[i] *= std::exp(-F_real[i]);
    }

}