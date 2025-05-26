#pragma once
#include "common.hpp"
#include "InitialConditionGenerator.hpp"

class ODEStepper
{
    private:
        int Ntau;
        real_t Dim;
        real_t precision;
        Scheme scheme;
        InitialConditionGenerator& initGen;
        mat_real a;
        vec_real b, c;

        void computeDerivatives(vec_real& Yreal, vec_real& dYreal, real_t x);

        void stepIRK(vec_complex& Yin, vec_complex& Yout,
                    real_t Xin, real_t Xout, int& itsReached,
                    bool& converged, int maxIts);

    public:

        ODEStepper(int numVars, real_t dim, real_t precision, Scheme method, InitialConditionGenerator& initGen);

        void integrate(vec_complex& Yin, vec_complex& Yout,
                    real_t Xin, real_t Xout,
                    bool& converged, int& itsReached,
                    int maxIts);

};