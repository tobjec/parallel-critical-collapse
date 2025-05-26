#include "ShootingSolver.hpp"

ShootingSolver::ShootingSolver(int Ntau_, real_t Dim_, real_t precision_, InitialConditionGenerator& initGen_, int maxIts_)
    : Ntau(Ntau_), Dim(Dim_), Precision(precision_), initGen(initGen_), maxIts(maxIts_)
{
    stepper = std::make_unique<ODEStepper>(Ntau_, Dim_, precision_, Scheme::IRK2, initGen);
}

void ShootingSolver::shoot(vec_complex& YLeft, vec_complex& YRight, const vec_real& gridX,
    int iLeft, int iRight, int iMid, vec_real& mismatchOut)
{

    integrateToMidpoint(YLeft, gridX, iLeft, iMid, true, YLeft);
    integrateToMidpoint(YRight, gridX, iRight, iMid, false, YRight);

    computeMismatch(YLeft, YRight, mismatchOut);
}

void ShootingSolver::integrateToMidpoint(
    const vec_complex& Yinit, const vec_real& xGrid,
    int startIdx, int endIdx, bool forward,
    vec_complex& Yfinal)
{
    vec_complex Y1=Yinit, Y2;

    if (forward)
    {
        for (int i=startIdx; i<endIdx; ++i)
        {
            stepper->integrate(Y1, Y2, xGrid[i], xGrid[i+1], converged, itsReached, maxIts);
        }
        Yfinal = Y2;
        
    }
    else
    {
        for (int i=startIdx; i>endIdx; --i)
        {
            stepper->integrate(Y1, Y2, xGrid[i], xGrid[i-1], converged, itsReached, maxIts);
        }
        Yfinal = Y2;

    }
}

void ShootingSolver::computeMismatch(
    const vec_complex& yLeft, const vec_complex& yRight, vec_real& mismatchOut)
{
    
    mismatchOut.resize(Ntau);
    for (int i=0; i<Ntau/2; ++i)
    {
        mismatchOut[i] = yRight[i].real() - yLeft[i].real();
        mismatchOut[i+1] = yRight[i].imag() - yLeft[i].imag();
    }

    // Enforce consistency of Delta value
    mismatchOut[4] = yLeft[2].real();  // Fortran index y(5)
}
