#pragma once
#include "common.hpp"

class SpectralTransformer
{
    private:
        size_t N;                          // Number of real-space grid points
        real_t period;                  // Period (Delta)
        real_t k0;
        fftw_plan forward_plan;
        fftw_plan backward_plan;
        fftw_complex *forward_data, *backward_data;

    public:
        explicit SpectralTransformer(size_t N, real_t period);
        ~SpectralTransformer();

        void forwardFFT(const vec_real& in, vec_complex& out);
        void inverseFFT(const vec_complex& in, vec_real& out);
        void forwardFFT(const vec_complex& in, vec_complex& out);
        void inverseFFT(const vec_complex& in, vec_complex& out);

        void differentiate(const vec_complex& in, vec_complex& out, real_t period_=0.0);
        void lamIntegrate(const vec_complex& fk, vec_complex& out, complex_t lambda=complex_t(0.0), real_t period_=0.0);
        real_t interpolate(const vec_complex& fk, real_t x, real_t period_=0.0);

        void halveModes(const vec_complex& in, vec_complex& out);
        void doubleModes(const vec_complex& in, vec_complex& out);
        void solveInhom(const vec_real& f, const vec_real& g, vec_real& x, real_t period_=0.0);
};