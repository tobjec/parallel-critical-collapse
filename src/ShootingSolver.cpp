#include "ShootingSolver.hpp"

ShootingSolver::ShootingSolver(int Ntau_, real_t Dim_, real_t precision_, InitialConditionGenerator& initGen_, int maxIts_)
    : Ntau(Ntau_), Dim(Dim_), Precision(precision_), initGen(initGen_), maxIts(maxIts_)
{
    stepper = std::make_unique<ODEStepper>(Ntau_, Dim_, precision_, Scheme::IRK2, initGen);
}

void ShootingSolver::shoot(vec_complex& YLeft, vec_complex& YRight, const vec_real& gridX,
    size_t iLeft, size_t iRight, size_t iMid, vec_complex& mismatchOut)
{
    integrateToMidpoint(YLeft, gridX, iLeft, iMid, true, YLeft);
    integrateToMidpoint(YRight, gridX, iRight, iMid, false, YRight);

    computeMismatch(YLeft, YRight, mismatchOut);
}

void ShootingSolver::integrateToMidpoint(
    const vec_complex& Yinit, const vec_real& xGrid,
    size_t startIdx, size_t endIdx, bool forward,
    vec_complex& Yfinal)
{
    vec_complex Y1=Yinit, Y2=Yinit;

    if (forward)
    {
        for (size_t i=startIdx; i<endIdx; ++i)
        {
            stepper->integrate(Y1, Y2, xGrid[i], xGrid[i+1], converged, itsReached, maxIts);

            if (!converged)
            {
                std::cerr << "ERROR: No convergence between grid point " << xGrid[i] << " (" << i << ") ";
                std::cerr << " and " << xGrid[i+1] << " (" << i+1 << ") " << std::endl;
                std::exit(EXIT_FAILURE);
            }

            Y1 = Y2;
        }
        Yfinal = Y1;
        
    }
    else
    {
        for (size_t i=startIdx; i>endIdx; --i)
        {
            stepper->integrate(Y1, Y2, xGrid[i], xGrid[i-1], converged, itsReached, maxIts);

            if (!converged)
            {
                std::cerr << "ERROR: No convergence between grid point " << xGrid[i] << " (" << i << ") ";
                std::cerr << " and " << xGrid[i-1] << " (" << i-1 << ") " << std::endl;
                std::exit(EXIT_FAILURE);
            }

            Y1 = Y2;
        }
        Yfinal = Y1;

    }
}

void ShootingSolver::computeMismatch(
    const vec_complex& yLeft, const vec_complex& yRight, vec_complex& mismatchOut)
{
    size_t size = yLeft.size();
    mismatchOut.resize(size);
    for (size_t i=0; i<size; ++i)
    {
        mismatchOut[i] = yRight[i] - yLeft[i];
    }

    if (mismatchOut[2].real() > Precision)
    {
        std::cout << "Mismatch in Delta from left and right side!!!" << std::endl;
    }

    // Enforce consistency of Delta value
    mismatchOut[2] = complex_t(yLeft[2].real(), mismatchOut[2].imag());
}
