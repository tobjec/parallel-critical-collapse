#pragma once
#include "common.hpp"

class SpectralTransformer
{
    private:
        int N;                          // Number of real-space grid points
        int N_freq;                     // N/2 + 1
        real_t period;                  // Period (Delta)
        real_t k0;
        fftw_plan forward_plan, forward_plan_complex;
        fftw_plan backward_plan, backward_plan_complex;
        real_t* real_data;
        fftw_complex *complex_data, *freq_data, *freq_data_complex;

    public:
        explicit SpectralTransformer(int N, real_t period);
        ~SpectralTransformer();

        void forwardFFT(const vec_real& in, vec_complex& out);
        void inverseFFT(const vec_complex& in, vec_real& out);
        void forwardFFTComplex(const vec_complex& in, vec_complex& out);
        void inverseFFTComplex(const vec_complex& in, vec_complex& out);

        void differentiate(const vec_complex& in, vec_complex& out, real_t period_=0.0);
        void lamIntegrate(const vec_complex& fk, vec_complex& out, complex_t lambda, real_t period_=0.0);
        real_t interpolate(const vec_complex& fk, real_t x, real_t period_=0.0);

        void halveModes(const vec_complex& in, vec_complex& out, int Nsize=0);
        void doubleModes(const vec_complex& in, vec_complex& out, int Nsize=0);
        void solveInhom(const vec_real& f, const vec_real& g, vec_real& x, real_t period_=0.0);
};