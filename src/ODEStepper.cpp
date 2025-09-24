//==============================================================================
// ODEStepper.cpp
// Implicit Runge–Kutta (Gauss–Legendre) stepper for the reduced ODE system
// produced by the shooting formulation. Supports IRK1/IRK2/IRK3 collocation.
// Responsibilities:
//   • Hold Butcher tableau (a,b,c) for chosen IRK scheme.
//   • Map spectral state <-> physical fields via InitialConditionGenerator.
//   • Evaluate RHS dY/dx from Einstein-scalar equations in (U,V,F,IA2) form.
//   • Do one IRK step with fixed-point iteration on stage values.
// Parallel notes:
//   • USE_HYBRID enables OpenMP over stages×modes in the inner loops.
//==============================================================================

#include "ODEStepper.hpp"

//------------------------------------------------------------------------------
// Ctor: choose IRK scheme and build its Butcher tableau.
//  - IRK1: 1-stage Gauss (midpoint)
//  - IRK2: 2-stage Gauss (order 4)
//  - IRK3: 3-stage Gauss (order 6)
//------------------------------------------------------------------------------
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

            // Gauss–Legendre s=1
            a[0][0] = 0.5;
            b[0]    = 1.0;
            c[0]    = 0.5;
            break;

        case Scheme::IRK2:
            stage = 2;
            a.assign(stage, vec_real(stage));
            b.resize(stage);
            c.resize(stage);

            // Gauss–Legendre s=2
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

            // Gauss–Legendre s=3
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
            b[1]        = 4.0 / 9.0;
            c[0] = 0.5 - std::sqrt(15.0) / 10.0;
            c[1] = 0.5;
            c[2] = 0.5 + std::sqrt(15.0) / 10.0;
            break;
    }
}

//------------------------------------------------------------------------------
// computeDerivatives
// Map spectral state Yreal → fields (U,V,F,IA2) and time-derivatives (dU/dt,…)
// using InitialConditionGenerator; then compute spatial derivatives d/dx of
// (U,V,F) from the PDE system at a specific x. Finally, pack dY/dx back to
// spectral representation dYreal.
// Notes:
//   • Y is stored as Ntau/2 complex modes → 2*Ntau/2 = Ntau real entries.
//   • We zero the real part of dYdx[2] to keep Δ (stored there) constant.
//------------------------------------------------------------------------------
void ODEStepper::computeDerivatives(vec_real& Yreal, vec_real& dYreal, real_t x)
{
    vec_real U(Ntau), V(Ntau), F(Ntau), IA2(Ntau), dUdt(Ntau), dVdt(Ntau), dFdt(Ntau),
             dUdx(Ntau), dVdx(Ntau), dFdx(Ntau);
    
    vec_complex Y(Ntau/2), dYdx(Ntau);

    // Unpack real vector → complex Fourier/Spectral representation
    for (int j=0; j<Ntau/2; ++j)
    {
        Y[j] = complex_t(Yreal[2*j], Yreal[2*j+1]);
    }

    // Spectral → fields and τ-derivatives at spatial position x
    initGen.StateVectorToFields(Y, U, V, F, IA2, dUdt, dVdt, dFdt, x);

    // Spatial RHS from reduced Einstein–scalar equations (characteristic form)
    for (int j=0; j<Ntau; ++j)
    {
        dUdx[j] = ( F[j] * ( (Dim - 2.0) * V[j]
                    - (2.0 * (Dim - 3.0) / IA2[j] - (Dim - 2.0)) * U[j])
                    - 2.0 * x * dUdt[j]) 
                  / (2.0 * x * (F[j] + x));

        dVdx[j] = ( F[j] * ( (Dim - 2.0) * U[j]
                    - (2.0 * (Dim - 3.0) / IA2[j] - (Dim - 2.0)) * V[j])
                    + 2.0 * x * dVdt[j]) 
                  / (2.0 * x * (F[j] - x));

        dFdx[j] = (Dim - 3.0) * F[j] * (1.0 - IA2[j]) / (IA2[j] * x);
    }

    // Fields → spectral derivative vector
    initGen.FieldsToStateVector(dUdx, dVdx, dFdx, dYdx);

    // Keep Δ (stored in Re(Y[2])) fixed along x
    dYdx[2] = complex_t(0.0, dYdx[2].imag());

    // Pack complex derivative back to interleaved real storage
    for (int j=0; j<Ntau/2; ++j)
    {
        dYreal[2*j]   = dYdx[j].real();
        dYreal[2*j+1] = dYdx[j].imag();
    }
}

//------------------------------------------------------------------------------
// integrate
// Single-step integration Yin@Xin → Yout@Xout using selected IRK scheme.
// Uses a fixed-point iteration on stage states with tolerance `precision`.
// Returns convergence flag and number of iterations performed.
//------------------------------------------------------------------------------
void ODEStepper::integrate(vec_complex& Yin, vec_complex& Yout,
                           real_t Xin, real_t Xout,
                           bool& converged, int& itsReached,
                           int maxIts)
{
    stepIRK(Yin, Yout, Xin, Xout, itsReached, converged, maxIts);
}

//------------------------------------------------------------------------------
// stepIRK
// Perform one implicit Gauss–Legendre step with `stage = b.size()` collocation
// points. We iterate a fixed-point map on the stage values yK2 until the
// stage-difference norm falls below an adaptive threshold.
// Implementation details:
//   • y holds the interleaved real-imag parts of Yin.
//   • f[i] keeps RHS at stage abscissa x[i].
//   • yK2[i] are stage states; yK1 is the previous iterate for convergence test.
//   • After convergence, do the final combination: y += dx * sum_i b[i] * f[i].
//------------------------------------------------------------------------------
void ODEStepper::stepIRK(vec_complex& Yin, vec_complex& Yout,
                         real_t Xin, real_t Xout, int& itsReached,
                         bool& converged, int maxIts)
{
    const real_t dx = Xout - Xin;
    int stage = static_cast<int>(b.size());

    vec_real x(stage);                 // collocation abscissae
    vec_real y(Ntau);                  // interleaved state at the start of the step
    mat_real f(stage, vec_real(Ntau)), // RHS evaluated at each stage
             yK1(stage, vec_real(Ntau)),
             yK2(stage, vec_real(Ntau));
    
    itsReached = 0;
    converged = false;

    // Initialize y from Yin (complex → interleaved real array)
    for (int i=0; i<Ntau/2; ++i)
    {
        y[2*i]   = Yin[i].real();
        y[2*i+1] = Yin[i].imag();
    }

    // Stage abscissae x_i = Xin + c_i * dx
    for (int i=0; i<stage; ++i)
    {
        x[i] = Xin + dx*c[i];
    }

    // Zeroth-order guess: all stages start at y
    for (int i=0; i<stage; ++i)
    {
        for (int j=0; j<Ntau; ++j)
        {
           yK2[i][j] = y[j];
        }
    }

    // Fixed-point iterations on stage states
    for (int its=0; its<maxIts; ++its)
    {
        ++itsReached;
        yK1 = yK2;
        real_t norm2 = 0.0;

        // Evaluate RHS at current stage guesses
        for (int i=0; i<stage; ++i)
        {
            computeDerivatives(yK2[i], f[i], x[i]);
        }

        // Update stage states: y_i = y + dx * Σ_k a_{ik} f_k
        #ifdef USE_HYBRID
        #pragma omp parallel
        {
            #pragma omp for collapse(2) schedule(static)
            for (int i=0; i<stage; ++i)
            {
                for (int j=0; j<Ntau; ++j)
                {
                    real_t tmp = y[j];
                    for (int k=0; k<stage; ++k)
                        tmp += dx * a[i][k] * f[k][j];
                    yK2[i][j] = tmp;
                }
            }

            // Convergence metric: RMS of stage-wise differences
            #pragma omp for collapse(2) reduction(+:norm2)
            for (int i=0; i<stage; ++i)
            {
                for (int j=0; j<Ntau; ++j)
                {
                   norm2 += std::pow(yK1[i][j] - yK2[i][j], 2);
                }
            }
        }
        #else
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
        for (int i=0; i<stage; ++i)
        {
            for (int j=0; j<Ntau; ++j)
            {
               norm2 += std::pow(yK1[i][j] - yK2[i][j], 2);
            }
        }
        #endif

        norm2 = std::sqrt(norm2 / (Ntau*stage));

        // Slightly relax tolerance late in the iteration window to avoid stalls
        real_t precision10 = precision;
        if (its > maxIts/2)
        {
            precision10 = precision * 10.0 * (2.0*static_cast<real_t>(its)/static_cast<real_t>(maxIts) - 0.5);
        }

        if (norm2 < precision10)
        {
            converged = true;
            break;
        }
    }

    // Final combination: y^{n+1} = y + dx * Σ_i b_i f_i
    for (int i=0; i<stage; ++i)
    {
        for (int j=0; j<Ntau; ++j)
        {
           y[j] += dx * b[i] * f[i][j];
        }
    }

    // Pack back to complex spectral Yout
    for (int i=0; i<Ntau/2; ++i)
    {
        Yout[i] = complex_t(y[2*i], y[2*i+1]);
    }
}
