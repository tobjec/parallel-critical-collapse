#include "ODEStepper.hpp"

ODEStepper::ODEStepper(int Ntau_, real_t Dim_, real_t precision_, Scheme method_, InitialConditionGenerator& initGen_)
    : Ntau(Ntau_), Dim(Dim_), precision(precision_), scheme(method_), initGen(initGen_)
{
    size_t stage {};

    switch (scheme)
    {
        case Scheme::IRK1:
            stage = 1;
            a.assign(stage, vec_real(stage));
            b.resize(stage);
            c.resize(stage);

            a[0][0] = 0.5;
            b[0] = 1.0;
            c[0] = 0.5;
            
            break;

        case Scheme::IRK2:
            stage = 2;
            a.assign(stage, vec_real(stage));
            b.resize(stage);
            c.resize(stage);

            a[0][0] = 0.25;
            a[0][1] = 0.25 - 0.5 / std::sqrt(3.0);
            a[1][0] = 0.25 + 0.5 / std::sqrt(3.0);
            a[1][1] = 0.25;
            b[0] = b[1] = 0.5;
            c[0] = 0.5 - 0.5 / std::sqrt(3.0);
            c[1] = 0.5 + 0.5 / std::sqrt(3.0);

            break;

        case Scheme::IRK3:
            stage = 3;
            a.assign(stage, vec_real(stage));
            b.resize(stage);
            c.resize(stage);

            a[0][0] = 5.0 / 36.0;
            a[0][1] = 2.0 / 9.0 - 1.0 / std::sqrt(15.0);
            a[0][2] = 5.0 / 36.0 - 0.5 / std::sqrt(15.0);
            a[1][0] = 5.0 / 36.0 + std::sqrt(15.0) / 24.0;
            a[1][1] = 2.0 / 9.0; 
            a[1][2] = 5.0 / 36.0 - std::sqrt(15.0) / 24.0;
            a[2][0] = 5.0 / 36.0 + 0.5 / std::sqrt(15.0);
            a[2][1] = 2.0 / 9.0 + 1.0 / std::sqrt(15.0);
            a[2][2] = 5.0 / 36.0;
            b[0] = b[2] = 5.0 / 18.0;
            b[1] = 4.0 / 9.0;
            c[0] = 0.5 - std::sqrt(15.0) / 10.0;
            c[1] = 0.5;
            c[2] = 0.5 + std::sqrt(15.0) / 10.0;

            break;
    }
}

void ODEStepper::computeDerivatives(vec_real& Yreal, vec_real& dYreal, real_t x)
{
    vec_real U(Ntau), V(Ntau), F(Ntau), IA2(Ntau), dUdt(Ntau), dVdt(Ntau), dFdt(Ntau),
             dUdx(Ntau), dVdx(Ntau), dFdx(Ntau);
    
    vec_complex Y(Ntau/2), dYdx(Ntau);

    for (int j=0; j<Ntau/2; ++j)
    {
        Y[j] = complex_t(Yreal[2*j], Yreal[2*j+1]);
    }

    initGen.StateVectorToFields(Y, U, V, F, IA2, dUdt, dVdt, dFdt, x);

    #ifdef USE_HYBRID
    #pragma omp parallel for schedule(static)
    #endif
    for (int j=0; j<Ntau; ++j)
    {
        dUdx[j] = (F[j] * ((Dim - 2.0) * V[j]
                    - (2.0 * (Dim - 3.0) / IA2[j] - (Dim - 2.0)) * U[j])
                    - 2.0 * x * dUdt[j]) 
                  / (2.0 * x * (F[j] + x));

        dVdx[j] = (F[j] * ((Dim - 2.0) * U[j]
                    - (2.0 * (Dim - 3.0) / IA2[j] - (Dim - 2.0)) * V[j])
                    + 2.0 * x * dVdt[j]) 
                  / (2.0 * x * (F[j] - x));

        dFdx[j] = (Dim - 3.0) * F[j] * (1.0 - IA2[j]) / (IA2[j] * x);
    }

    initGen.FieldsToStateVector(dUdx, dVdx, dFdx, dYdx);

    dYdx[2] = complex_t(0.0, dYdx[2].imag());

    for (int j=0; j<Ntau/2; ++j)
    {
        dYreal[2*j] = dYdx[j].real();
        dYreal[2*j+1] = dYdx[j].imag();
    }

}

void ODEStepper::integrate(vec_complex& Yin, vec_complex& Yout,
                           real_t Xin, real_t Xout,
                           bool& converged, int& itsReached,
                           int maxIts)
{
    switch (scheme)
    {
        case Scheme::IRK1:
            stepIRK(Yin, Yout, Xin, Xout, itsReached, converged, maxIts);
            break;
        case Scheme::IRK2:
            stepIRK(Yin, Yout, Xin, Xout, itsReached, converged, maxIts);
            break;
        case Scheme::IRK3:
            stepIRK(Yin, Yout, Xin, Xout, itsReached, converged, maxIts);
            break;

    }
}

// === IRK
void ODEStepper::stepIRK(vec_complex& Yin, vec_complex& Yout,
                         real_t Xin, real_t Xout, int& itsReached,
                         bool& converged, int maxIts)
{
    const real_t dx = Xout - Xin;
    int stage = static_cast<int>(b.size());

    vec_real x(stage);
    vec_real y(Ntau);
    mat_real f(stage, vec_real(Ntau)), yK1(stage, vec_real(Ntau)),
             yK2(stage, vec_real(Ntau));
    
    itsReached = 0;
    converged = false;

    // Initialization of y
    for (int i=0; i<Ntau/2; ++i)
    {
        y[2*i] = Yin[i].real();
        y[2*i+1] = Yin[i].imag();
    }

    // Collocation points
    for (int i=0; i<stage; ++i)
    {
        x[i] = Xin + dx*c[i];
    }

    // Zeroth-order estimates
    for (int i=0; i<stage; ++i)
    {
        for (int j=0; j<Ntau; ++j)
        {
            yK2[i][j] = y[j];
        }
    }

    // Fixed-point iteration
    for (int its=0; its<maxIts; ++its)
    {
        ++itsReached;
        yK1 = yK2;

        // Calculation of derivatives
        for (int i=0; i<stage; ++i)
        {
            computeDerivatives(yK2[i], f[i], x[i]);
        }

        // Compute new stage
        #ifdef USE_HYBRID
        #pragma omp parallel for collapse(2) schedule(static)
        #endif
        for (int i=0; i<stage; ++i)
        {
            for (int j=0; j<Ntau; ++j)
            {
                real_t tmp = y[j];
                for (int k=0; k<stage; ++k)
                {
                    tmp += dx * a[i][k] * f[k][j];
                }
                yK2[i][j] = tmp;
            }
        }

        real_t norm2 = 0.0;
        #ifdef USE_HYBRID
        #pragma omp parallel for collapse(2) reduction(+:norm2)
        #endif
        for (int i=0; i<stage; ++i)
        {
            for (int j=0; j<Ntau; ++j)
            {
                norm2 += std::pow(yK1[i][j] - yK2[i][j], 2);
            }
        }

        norm2 = std::sqrt(norm2 / (Ntau*stage));

        /* if (its >= maxIts/2)
        {
            precision *= 10.0 * (2.0*static_cast<double>(its)/static_cast<double>(maxIts) - 1.0);
        } */
        
        if (norm2 < precision)
        {
            converged = true;
            break;
            
            /* // Checking inf-norm of difference
            real_t normInf = 0.0;
            #ifdef USE_HYBRID
            #pragma omp parallel for collapse(2) reduction(+:norm2)
            #endif
            for (int i=0; i<stage; ++i)
            {
                for (int j=0; j<Ntau; ++j)
                {
                    real_t pointNorm = std::abs(yK1[i][j] - yK2[i][j]);
                    normInf = std::max(normInf, pointNorm);
                }
            }

            if (normInf < 10.0*precision)
            {
                converged = true;
                break;
            } */

        }
        
    }

    for (int i=0; i<stage; ++i)
    {
        for (int j=0; j<Ntau; ++j)
        {
            y[j] += dx * b[i] * f[i][j];
        }
    }
    
    // Assigning to final vector
    for (int i=0; i<Ntau/2; ++i)
    {
        Yout[i] = complex_t(y[2*i], y[2*i+1]);
    }
}
