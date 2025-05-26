#include "ODEStepper.hpp"

ODEStepper::ODEStepper(int Ntau_, real_t Dim_, real_t precision_, Scheme method_, InitialConditionGenerator& initGen_)
    : Ntau(Ntau_), Dim(Dim_), precision(precision_), scheme(method_), initGen(initGen_)
    {
        int stage {};

        switch (scheme)
        {
            case Scheme::IRK2:
                stage = 2;
                a.assign(stage, vec_real(3));
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
                a.assign(stage, vec_real(3));
                b.resize(stage);
                c.resize(stage);

                a[0][0] = 5.0 / 36.0;
                a[0][1] = 2.0 / 9.0 - 1 / std::sqrt(15.0);
                a[0][2] = 5.0 / 36.0 - 0.5 / std::sqrt(15.0);
                a[1][0] = 5.0 / 36.0 + std::sqrt(15.0) / 24.0;
                a[1][1] = 2.0 / 9.0; 
                a[1][2] = 5.0 / 36.0 + std::sqrt(15.0) / 24.0;
                a[2][0] = 5.0 / 36.0 + 0.5 / std::sqrt(15.0);
                a[2][1] = 2.0 / 9.0 + 1.0 / std::sqrt(15.0);
                a[2][2] = 5.0 / 36.0;
                b[0] = b[2] = 5.0 / 18.0;
                b[1] = 4.0 / 9.0;
                c[0] = 0.5 - std::sqrt(15.0) / 10.0;
                c[1] = 0.5;
                c[2] = 0.5 + std::sqrt(15.0) / 10.0;

                break;

            case Scheme::RKF45:
                break;
        }
    }

void ODEStepper::computeDerivatives(const vec_complex& Y, vec_complex& dY, real_t x)
{
    vec_real U(Ntau), V(Ntau), F(Ntau), IA2(Ntau), dU(Ntau), dV(Ntau), dF(Ntau),
             dUdx(Ntau), dVdx(Ntau), dFdx(Ntau); 
    initGen.StateVectorToFields(Y, U, V, F, IA2, dU, dV, dF, x);

    for (int j; j<Ntau; ++j)
    {
        dUdx[j] = (F[j] * ((Dim - 2.0) * V[j]
                    - (2.0 * (Dim - 3.0) / IA2[j] - (Dim - 2.0)) * U[j])
                    - 2.0 * x * dU[j]) 
                  / (2.0 * x * (F[j] + x));

        dVdx[j] = (F[j] * ((Dim - 2.0) * U[j]
                    - (2.0 * (Dim - 3.0) / IA2[j] - (Dim - 2.0)) * V[j])
                    + 2.0 * x * dV[j]) 
                  / (2.0 * x * (F[j] - x));

        dFdx[j] = (Dim - 3.0) * F[j] * (1.0 - IA2[j]) / (IA2[j] * x);
    }

    initGen.FieldsToStateVector(dUdx, dVdx, dFdx, dY);

    dY[2] = complex_t(0.0, dY[2].imag());

}

void ODEStepper::integrate(const vec_complex& Yin, vec_complex& Yout,
                           real_t Xin, real_t Xout,
                           bool& converged, int& itsReached,
                           int maxIts)
{
    switch (scheme)
    {
        case Scheme::IRK2:
            stepIRK(Yin, Yout, Xin, Xout, itsReached, converged, maxIts);
            break;
        case Scheme::IRK3:
            stepIRK(Yin, Yout, Xin, Xout, itsReached, converged, maxIts);
            break;
        case Scheme::RKF45:
            real_t dxTry = Xout - Xin;
            stepRKF45(Yin, Yout, Xin, Xout, dxTry, converged, maxIts);
            itsReached = 1;
            break;
    }
}

// === IRK
void ODEStepper::stepIRK(const vec_complex& Yin, vec_complex& Yout,
                         real_t Xin, real_t Xout, int& itsReached,
                         bool& converged, int maxIts)
{
    const real_t dx = Xout - Xin;
    int stage = b.size();

    vec_real x(stage);
    vec_real y(Ntau);
    mat_complex f(stage, vec_complex(Ntau, 0.0)), yK1(stage, vec_complex(Ntau, 0.0)),
             yK2(stage, vec_complex(Ntau, 0.0));
    
    itsReached = 0;
    converged = false;

    // Initialization of y
    for (int i=0; i<Ntau/2; ++i)
    {
        y[i] = Yin[i].real();
        y[i+1] = Yin[i].imag();
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
    for (int its = 0; its < maxIts; ++its)
    {
        yK1 = yK2;

        // Calculation of derivatives
        for (int i=0; i<stage; ++i)
        {
            computeDerivatives(yK2[i], f[i], x[i]);
        }

        // Compute new stage
        for (int i=0; i<Ntau; ++i)
        {
            for (int j=0; j<stage; ++j)
            {
                for (int k=0; k<stage; ++k)
                {
                    yK2[i][j] = yK2[i][j] + dx * a[j][k] * f[i][k];
                }
            }
        }

        real_t norm2 = 0.0;
        for (int i=0; i<stage; ++i)
        {
            for (int j=0; j<Ntau; ++j)
            {
                norm2 += std::pow(yK1[i][j].real() - yK2[i][j].real(), 2);
            }
        }

        norm2 = std::sqrt(norm2 / (Ntau*stage));
        if (norm2 < precision)
        {
            converged = true;
            break;
        }
    }

    for (int i=0; i<stage; ++i)
    {
        for (int j=0; j<Ntau; ++j)
        {
            y[j] += dx * b[i] * f[i][j].real();
        }
    }
    
    // Assigning to final vector
    for (int i=0; i<Ntau/2; ++i)
    {
        Yout[i] = complex_t(y[i], y[i+1]);
    }
}

// === RKF45 (adaptive)
void ODEStepper::stepRKF45(const vec_complex& yin, vec_complex& yout,
                           real_t xin, real_t xout,
                           real_t& dxNext, bool& converged, int maxIts)
{}
