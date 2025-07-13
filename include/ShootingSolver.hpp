#pragma once

#include "common.hpp"
#include "SpectralTransformer.hpp"
#include "InitialConditionGenerator.hpp"
#include "ODEStepper.hpp"

class ShootingSolver
{
    private:
        int Ntau;
        real_t Dim, Precision;
        InitialConditionGenerator& initGen;
        std::unique_ptr<ODEStepper> stepper;
        bool converged=false;
        int itsReached, maxIts;
    
        void integrateToMidpoint(
            const vec_complex& yInit, const vec_real& xGrid,
            size_t startIdx, size_t endIdx, bool forward,
            vec_complex& yFinal, bool Debug=false, json* fieldVals=nullptr);
    
        void computeMismatch(const vec_complex& yLeft, const vec_complex& yRight, vec_complex& mismatchOut);
            
    public:
        ShootingSolver(int Ntau_, real_t Dim_, real_t precision_, InitialConditionGenerator& initGen_, int maxIts_);
        

        void shoot(vec_complex& YLeft, vec_complex& YRight, const vec_real& gridX,
                   size_t iLeft, size_t iRight, size_t iMid, vec_complex& mismatchOut, bool =false, json* fieldVals=nullptr);

};