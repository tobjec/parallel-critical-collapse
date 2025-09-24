//==============================================================================
// ShootingSolver.cpp
// Two-sided shooting integrator to match left/right boundary expansions at a
// midpoint. Uses an IRK ODE stepper to march along the x-grid and returns the
// spectral mismatch vector used by the Newton solver.
// Responsibilities:
//   • Hold an ODEStepper (IRK2 by default) configured with precision/Dim.
//   • March YLeft forward to X_mid, and YRight backward to X_mid.
//   • Optionally dump intermediate fields (A,U,V,f) for diagnostics.
//   • Compute mismatch = Y_right_at_mid − Y_left_at_mid and enforce Δ-consistency.
// Notes:
//   • Please keep the user's one-line for/if bodies with braces `{}` intact.
//==============================================================================

#include "ShootingSolver.hpp"

//------------------------------------------------------------------------------
// Ctor: initialize IRK stepper (Gauss–Legendre s=2) with shared initGen.
//------------------------------------------------------------------------------
ShootingSolver::ShootingSolver(int Ntau_, real_t Dim_, real_t precision_, InitialConditionGenerator& initGen_, int maxIts_, int SchemeIRK)
    : Ntau(Ntau_), Dim(Dim_), Precision(precision_), initGen(initGen_), maxIts(maxIts_)
{
    switch (SchemeIRK)
    {
        case 1:
            SchemeIRK = Scheme::IRK1;
            break;
        case 2:
            SchemeIRK = Scheme::IRK2;
            break;
        case 3:
            SchemeIRK = Scheme::IRK3;
            break;
        default:
            throw std::invalid_argument("No known IRK Scheme supplied!");
    }

    // Use IRK2 (order 4) as a good balance of stability/accuracy for stiff RHS
    stepper = std::make_unique<ODEStepper>(Ntau_, Dim_, precision_, SchemeIRK, initGen);
}

//------------------------------------------------------------------------------
// shoot
// Integrate left and right boundary states to the match point and form the
// spectral mismatch that Newton will drive to zero.
// If Debug=true and fieldVals!=nullptr, store snapshots every ~100 steps.
//------------------------------------------------------------------------------
void ShootingSolver::shoot(vec_complex& YLeft, vec_complex& YRight, const vec_real& gridX,
    size_t iLeft, size_t iRight, size_t iMid, vec_complex& mismatchOut, bool Debug, json* fieldVals)
{
    // March left → mid (increasing x)
    integrateToMatchPoint(YLeft,  gridX, iLeft,  iMid,  true,  YLeft,  Debug, fieldVals);
    // March right → mid (decreasing x)
    integrateToMatchPoint(YRight, gridX, iRight, iMid,  false, YRight, Debug, fieldVals);

    // Difference at the midpoint
    computeMismatch(YLeft, YRight, mismatchOut);
}

//------------------------------------------------------------------------------
// integrateToMatchPoint
// March a boundary state Yinit along xGrid from startIdx to endIdx.
//  • forward=true  : integrate x[i] → x[i+1]
//  • forward=false : integrate x[i] → x[i-1]
// On failure to converge inside a step, abort (MPI_Abort or std::exit).
// When Debug is enabled, export diagnostic fields at selected grid points.
//------------------------------------------------------------------------------
void ShootingSolver::integrateToMatchPoint(
    const vec_complex& Yinit, const vec_real& xGrid,
    size_t startIdx, size_t endIdx, bool forward,
    vec_complex& Yfinal, bool Debug, json* fieldVals)
{
    vec_complex Y1=Yinit, Y2=Yinit;      // rolling input/output per step
    vec_real A(Ntau), U(Ntau), V(Ntau), f(Ntau), dUdt(Ntau), dVdt(Ntau), dFdt(Ntau); 
    
    if (forward)
    {
        // Forward sweep: startIdx .. endIdx-1, stepping to larger x
        for (size_t i=startIdx; i<endIdx; ++i)
        {
            stepper->integrate(Y1, Y2, xGrid[i], xGrid[i+1], converged, itsReached, maxIts);

            // Hard failure if stage fixed-point does not converge
            if (!converged)
            {
                #if defined(USE_MPI) || defined(USE_HYBRID)
                int rank;
                MPI_Comm_rank(MPI_COMM_WORLD, &rank);
                std::cerr << "ERROR for rank:" << rank << ", no convergence between grid point " << xGrid[i] << " (" << i << ") ";
                std::cerr << " and " << xGrid[i+1] << " (" << i+1 << ") " << std::endl;
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                #else
                std::cerr << "ERROR: No convergence between grid point " << xGrid[i] << " (" << i << ") ";
                std::cerr << " and " << xGrid[i+1] << " (" << i+1 << ") " << std::endl;
                std::exit(EXIT_FAILURE);
                #endif
            }

            // Optional diagnostics every ~100 steps and at the last point
            if (Debug && (i%100==0 || i==endIdx-1) && fieldVals)
            {
                // Convert current spectral state → fields at xGrid[i]
                initGen.StateVectorToFields(Y1, U, V, f, A, dUdt, dVdt, dFdt, xGrid[i]); 
                
                // Store lapse-like A = 1/sqrt(IA2)
                std::for_each(A.begin(), A.end(), [](auto& ele){ele=std::sqrt(1/ele);});

                (*fieldVals)[std::to_string(xGrid[i])]["A"] = A;
                (*fieldVals)[std::to_string(xGrid[i])]["U"] = U;
                (*fieldVals)[std::to_string(xGrid[i])]["V"] = V;
                (*fieldVals)[std::to_string(xGrid[i])]["f"] = f;
            }

            // Advance rolling state
            Y1 = Y2;
        }
        Yfinal = Y1;
        
    }
    else
    {
        // Backward sweep: startIdx .. endIdx+1, stepping to smaller x
        for (size_t i=startIdx; i>endIdx; --i)
        {
            stepper->integrate(Y1, Y2, xGrid[i], xGrid[i-1], converged, itsReached, maxIts);

            if (!converged)
            {
                #if defined(USE_MPI) || defined(USE_HYBRID)
                int rank;
                MPI_Comm_rank(MPI_COMM_WORLD, &rank);
                std::cerr << "ERROR for rank:" << rank << ", no convergence between grid point " << xGrid[i] << " (" << i << ") ";
                std::cerr << " and " << xGrid[i-1] << " (" << i-1 << ") " << std::endl;
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                #else
                std::cerr << "ERROR: No convergence between grid point " << xGrid[i] << " (" << i << ") ";
                std::cerr << " and " << xGrid[i-1] << " (" << i-1 << ") " << std::endl;
                std::exit(EXIT_FAILURE);
                #endif
            }

            // Optional diagnostics every ~100 steps and at the bounds
            if (Debug && (i%100==0 || i==endIdx+1 || i==startIdx) && fieldVals)
            {
                initGen.StateVectorToFields(Y1, U, V, f, A, dUdt, dVdt, dFdt, xGrid[i]); 
                
                std::for_each(A.begin(), A.end(), [](auto& ele){ele=std::sqrt(1/ele);});

                (*fieldVals)[std::to_string(xGrid[i])]["A"] = A;
                (*fieldVals)[std::to_string(xGrid[i])]["U"] = U;
                (*fieldVals)[std::to_string(xGrid[i])]["V"] = V;
                (*fieldVals)[std::to_string(xGrid[i])]["f"] = f;
            }

            Y1 = Y2;
        }
        Yfinal = Y1;

    }
}

//------------------------------------------------------------------------------
// computeMismatch
// Form the spectral mismatch at the match point, component-wise.
// Also enforce that the real parts storing Δ are identical (within Precision);
// if violated, abort, since Δ must be single-valued across both integrations.
// Finally, overwrite mismatchOut[2].real() with the agreed Δ value.
//------------------------------------------------------------------------------
void ShootingSolver::computeMismatch(
    const vec_complex& yLeft, const vec_complex& yRight, vec_complex& mismatchOut)
{
    size_t size = yLeft.size();
    mismatchOut.resize(size);

    // yRight - yLeft (vectorized)
    for (size_t i=0; i<size; ++i)
    {
        mismatchOut[i] = yRight[i] - yLeft[i];
    }

    // Guard: Δ mismatch beyond tolerance is a hard error
    if (mismatchOut[2].real() > Precision)
    {
         #if defined(USE_MPI) || defined(USE_HYBRID)
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        std::cerr << "ERROR for rank:" << rank << ", mismatch in Delta from left and right side!!!" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        #else
        std::cerr << "Mismatch in Delta from left and right side!!!" << std::endl;
        std::exit(EXIT_FAILURE);
        #endif
    }

    // Enforce consistency of Delta value: keep Re part equal to left's Δ
    mismatchOut[2] = complex_t(yLeft[2].real(), mismatchOut[2].imag());
}
