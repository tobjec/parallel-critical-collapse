#include "common.hpp"
#include "ShootingSolver.hpp"
#include "ODEStepper.hpp" // Contains `implicitStep`

static void derivs(real_t x, const vec_real& y, vec_real& dydx)
{
    // Placeholder — replicate the logic from derivs.f
    // dydx = f(y, x, d, Delta, etc.)
    dydx = y; // ← for now, identity just for structure
}


ShootingSolver::ShootingSolver(int numGridPts_, real_t dim_, real_t delta_, real_t precision_)
    : numGridPts(numGridPts_), dim(dim_), delta(delta_),
      precision(precision_), initGen(numGridPts_, dim_, delta_)
{
    stepper = std::make_unique<ODEStepper>(numGridPts_, dim_, precision_, Scheme::IRK2, derivs);
}

void ShootingSolver::shoot(
    const vec_real& fc, const vec_real& psic, const vec_real& up,
    const vec_real& gridX, int iLeft, int iRight, int iMid,
    vec_real& mismatchOut)
{
    const int ny = numGridPts;
    const int n3 = 3 * ny / 4;

    vec_real yLeft, yRight;
    initGen.generateInitialCondition(fc, psic, up, gridX[iLeft], true, yLeft);
    initGen.generateInitialCondition(fc, psic, up, gridX[iRight], false, yRight);

    integrateToMidpoint(yLeft, gridX, iLeft, iMid, true, yLeft);
    integrateToMidpoint(yRight, gridX, iRight, iMid, false, yRight);

    computeMismatch(yLeft, yRight, mismatchOut);
}

void ShootingSolver::integrateToMidpoint(
    const vec_real& yInit, const vec_real& xGrid,
    int startIdx, int endIdx, bool forward,
    vec_real& yFinal)
{
    const int direction = forward ? 1 : -1;
    const int nSteps = std::abs(endIdx - startIdx);

    vec_real y = yInit;

    for (int i = 0; i < nSteps; ++i)
    {
        int idx = forward ? (startIdx + i) : (startIdx - i);
        real_t x0 = xGrid[idx];
        real_t x1 = xGrid[idx + direction];

        bool converged = false;
        int itsReached = 0;

        vec_real yNext;
        stepper->integrate(y, yNext, x0, x1, converged, itsReached);

        if (!converged)
        {
            std::cerr << "ERROR: ODE step failed to converge at x = " << x0 << std::endl;
            std::exit(EXIT_FAILURE);
        }

        y = yNext;  // advance solution
    }

    yFinal = y;
}

void ShootingSolver::computeMismatch(
    const vec_real& yLeft, const vec_real& yRight, vec_real& mismatchOut)
{
    const int ny = numGridPts;

    mismatchOut.resize(ny);
    for (int i = 0; i < ny; ++i)
    {
        mismatchOut[i] = yRight[i] - yLeft[i];
    }

    // Enforce consistency of Delta value
    mismatchOut[4] = delta;  // Fortran index y(5)
}
