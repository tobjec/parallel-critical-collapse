#pragma once
#include "common.hpp"
#include "InitialConditionGenerator.hpp"

class ODEStepper
{
    private:
        int Ntau;
        real_t Dim;
        real_t precision;
        InitialConditionGenerator& initGen;
        Scheme scheme;
        mat_real a;
        vec_real b, c;

        void computeDerivatives(const vec_complex& Y, vec_complex& dY, real_t x);

        void stepIRK(const vec_complex& Yin, vec_complex& Yout,
                    real_t Xin, real_t Xout, int& itsReached,
                    bool& converged, int maxIts);

        void stepRKF45(const vec_complex& Yin, vec_complex& Yout,
                    real_t Xin, real_t Xout,
                    real_t& dxNext, bool& converged, int maxIts);

    public:

        ODEStepper(int numVars, real_t dim, real_t precision, Scheme method, InitialConditionGenerator& initGen);

        void integrate(const vec_complex& Yin, vec_complex& Yout,
                    real_t Xin, real_t Xout,
                    bool& converged, int& itsReached,
                    int maxIts);

};