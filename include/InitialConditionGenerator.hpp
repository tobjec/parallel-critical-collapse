#pragma once
#include "common.hpp"
#include "SpectralTransformer.hpp"

class InitialConditionGenerator
{

    private:
        int NumGridPts;       // Number of grid points
        real_t Dim;           // Physical dimension (e.g., 4.0)
        real_t Delta;         // Period (used in FFT)
        SpectralTransformer fft; // Fourier engine      

    public:
        InitialConditionGenerator(int NumGridPts, real_t Dim, real_t Delta);

        // Compute Taylor expansion and return packed vector Y
        // isLeft = true  → x ≈ 0 expansion (order 5)
        // isLeft = false → x ≈ 1 expansion (order 3)
        void generateInitialCondition(
            const vec_real& Fc, const vec_real& Psic, const vec_real& Up,
            real_t X, bool isLeft,
            vec_real& Y, bool PrintDiagnostics = false);
        
        void computeLeftExpansion(
            const vec_real& Fc, const vec_real& Psic, real_t X,
            vec_real& U, vec_real& V, vec_real& F, bool PrintDiagnostics);

        void computeRightExpansion(
            const vec_real& Up, real_t X,
            vec_real& U, vec_real& V, vec_real& F, bool PrintDiagnostics);

        void packSpectralFields(
            const vec_real& U, const vec_real& V, const vec_real& F,
            vec_real& Y);

        void unpackSpectralFields(const vec_real& Y,
            vec_real& U, vec_real& V, vec_real& F);

};