#pragma once
#include "common.hpp"

class SpectralTransformer
{
    private:
        int N;              // number of real-space grid points
        real_t L;           // period (Delta)
        fftw_plan forward_plan;
        fftw_plan backward_plan;
        fftw_complex* fft_in;
        fftw_complex* fft_out;

    public:
        explicit SpectralTransformer(int N, real_t period);
        ~SpectralTransformer();

        void forward_fft(const vec_real& in, vec_complex& out);
        void inverse_fft(const vec_complex& in, vec_real& out);

        void differentiate(const vec_complex& in, vec_complex& out);
        void lamint(const vec_complex& fk, vec_complex& out, complex_t lambda);
        real_t interpolate(const vec_complex& fk, real_t x);

        void resample_modes(const vec_complex& in, vec_complex& out, int N_out);
        void solve_inhomogeneous(const vec_real& f, const vec_real& g, vec_real& x);
};