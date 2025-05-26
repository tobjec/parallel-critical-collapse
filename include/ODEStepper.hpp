#pragma once
#include "common.hpp"

class ODEStepper
{
    private:
        int numVars;
        real_t dim;
        real_t precision;
        Scheme scheme;
        DerivativeFunc computeDerivatives;

        void stepIRK(const vec_real& yin, vec_real& yout,
                    real_t xin, real_t xout,
                    int nu, int& itsReached, bool& converged);

        void stepRKF45(const vec_real& yin, vec_real& yout,
                    real_t xin, real_t xout,
                    real_t& dxNext, bool& converged);

    public:

        ODEStepper(int numVars, real_t dim, real_t precision, Scheme method, DerivativeFunc deriv);

        void integrate(const vec_real& yin, vec_real& yout,
                    real_t xin, real_t xout,
                    bool& converged, int& itsReached,
                    int maxIts = 100);

};