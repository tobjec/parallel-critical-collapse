#pragma once

#include "common.hpp"
#include "SpectralTransformer.hpp"

class InitialConditionGenerator
{

    private:
        int Ntau, Nnewton;       // Number of grid points
        real_t Dim;           // Physical dimension (e.g., 4.0)
        real_t Delta;         // Period (used in FFT)
        SpectralTransformer fft; // Fourier engine      

    public:
        InitialConditionGenerator(int Ntau_, real_t Dim_, real_t Delta_);

        // Compute Taylor expansion and return packed vector Y
        // isLeft = true  → x ≈ 0 expansion (order 5)
        // isLeft = false → x ≈ 1 expansion (order 3)
        
        void computeLeftExpansion(
            real_t XLeft, const vec_real& fc, const vec_real& psic,
            vec_complex& Y, real_t DeltaIn, bool PrintDiagnostics);

        void computeRightExpansion(
            real_t XRight, const vec_real& Up, 
            vec_complex& Y, real_t DeltaIn, bool PrintDiagnostics);

        void packSpectralFields(
            const vec_real& Odd1, const vec_real& Odd2, const vec_real& Even,
            vec_real& Z);

        void unpackSpectralFields(const vec_real& Z,
            vec_real& Odd1, vec_real& Odd2, vec_real& Even);
        
        void FieldsToStateVector(const vec_real& U, const vec_real& V,
            const vec_real& F, vec_complex& Y);

        void StateVectorToFields(vec_complex& Y, vec_real& U, vec_real& V,
            vec_real& F, vec_real& IA2, vec_real& dUdt, vec_real& dVdt, vec_real& dFdt, real_t X);
        
        void StateVectorToFields(vec_complex& Y, vec_real& U, vec_real& V,
            vec_real& F, real_t X);

};