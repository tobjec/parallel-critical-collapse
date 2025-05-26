#pragma once
#include "common.hpp"
#include "SpectralTransformer.hpp"
#include "InitialConditionGenerator.hpp"
#include "ODEStepper.hpp"

class ShootingSolver
{
    private:
        int numGridPts;
        real_t dim, delta, precision;
        InitialConditionGenerator initGen;
        std::unique_ptr<ODEStepper> stepper;
    
        void integrateToMidpoint(
            const vec_real& yInit, const vec_real& xGrid,
            int startIdx, int endIdx, bool forward,
            vec_real& yFinal);
    
        void computeMismatch(const vec_real& yLeft, const vec_real& yRight, vec_real& mismatchOut);
            
    public:
        ShootingSolver(int numGridPts, real_t dim, real_t delta, real_t precision);

        void shoot(
            const vec_real& fc, const vec_real& psic, const vec_real& up,
            const vec_real& gridX,  // x-grid
            int iLeft, int iRight, int iMid,
            vec_real& mismatchOut);

};