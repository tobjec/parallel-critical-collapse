#include "ShootingSolver.hpp"

ShootingSolver::ShootingSolver(int Ntau_, real_t Dim_, real_t precision_, InitialConditionGenerator& initGen_, int maxIts_)
    : Ntau(Ntau_), Dim(Dim_), Precision(precision_), initGen(initGen_), maxIts(maxIts_)
{
    stepper = std::make_unique<ODEStepper>(Ntau_, Dim_, precision_, Scheme::IRK3, initGen);
}

void ShootingSolver::shoot(vec_complex& YLeft, vec_complex& YRight, const vec_real& gridX,
    size_t iLeft, size_t iRight, size_t iMid, vec_complex& mismatchOut, bool Debug, json* fieldVals)
{
    integrateToMatchPoint(YLeft, gridX, iLeft, iMid, true, YLeft, Debug, fieldVals);
    integrateToMatchPoint(YRight, gridX, iRight, iMid, false, YRight, Debug, fieldVals);

    computeMismatch(YLeft, YRight, mismatchOut);
}

void ShootingSolver::integrateToMatchPoint(
    const vec_complex& Yinit, const vec_real& xGrid,
    size_t startIdx, size_t endIdx, bool forward,
    vec_complex& Yfinal, bool Debug, json* fieldVals)
{
    vec_complex Y1=Yinit, Y2=Yinit;
    vec_real A(Ntau), U(Ntau), V(Ntau), f(Ntau), dUdt(Ntau), dVdt(Ntau), dFdt(Ntau); 
    
    if (forward)
    {
        for (size_t i=startIdx; i<endIdx; ++i)
        {
            stepper->integrate(Y1, Y2, xGrid[i], xGrid[i+1], converged, itsReached, maxIts);

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

            if (Debug && fieldVals)
            {
                initGen.StateVectorToFields(Y1, U, V, f, A, dUdt, dVdt, dFdt, xGrid[i]); 
                
                std::for_each(A.begin(), A.end(), [](auto& ele){ele=std::sqrt(1/ele);});

                real_t U_max = computeMaxNorm(U);
                real_t V_max = computeMaxNorm(V);
                real_t f_max = computeMaxNorm(f);
                real_t A_max = computeMaxNorm(A);

                real_t U_l2 = computeL2Norm(U);
                real_t V_l2 = computeL2Norm(V);
                real_t f_l2 = computeL2Norm(f);
                real_t A_l2 = computeL2Norm(A);

                (*fieldVals)[std::to_string(xGrid[i])]["A_max"] = A_max;
                (*fieldVals)[std::to_string(xGrid[i])]["U_max"] = U_max;
                (*fieldVals)[std::to_string(xGrid[i])]["V_max"] = V_max;
                (*fieldVals)[std::to_string(xGrid[i])]["f_max"] = f_max;

                (*fieldVals)[std::to_string(xGrid[i])]["A_l2"] = A_l2;
                (*fieldVals)[std::to_string(xGrid[i])]["U_l2"] = U_l2;
                (*fieldVals)[std::to_string(xGrid[i])]["V_l2"] = V_l2;
                (*fieldVals)[std::to_string(xGrid[i])]["f_l2"] = f_l2;
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

            if (Debug && fieldVals)
            {
                initGen.StateVectorToFields(Y1, U, V, f, A, dUdt, dVdt, dFdt, xGrid[i]); 
                
                std::for_each(A.begin(), A.end(), [](auto& ele){ele=std::sqrt(1/ele);});

                real_t U_max = computeMaxNorm(U);
                real_t V_max = computeMaxNorm(V);
                real_t f_max = computeMaxNorm(f);
                real_t A_max = computeMaxNorm(A);

                real_t U_l2 = computeL2Norm(U);
                real_t V_l2 = computeL2Norm(V);
                real_t f_l2 = computeL2Norm(f);
                real_t A_l2 = computeL2Norm(A);

                (*fieldVals)[std::to_string(xGrid[i])]["A_max"] = A_max;
                (*fieldVals)[std::to_string(xGrid[i])]["U_max"] = U_max;
                (*fieldVals)[std::to_string(xGrid[i])]["V_max"] = V_max;
                (*fieldVals)[std::to_string(xGrid[i])]["f_max"] = f_max;

                (*fieldVals)[std::to_string(xGrid[i])]["A_l2"] = A_l2;
                (*fieldVals)[std::to_string(xGrid[i])]["U_l2"] = U_l2;
                (*fieldVals)[std::to_string(xGrid[i])]["V_l2"] = V_l2;
                (*fieldVals)[std::to_string(xGrid[i])]["f_l2"] = f_l2;
            
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

    // Enforce consistency of Delta value
    mismatchOut[2] = complex_t(yLeft[2].real(), mismatchOut[2].imag());
}

real_t ShootingSolver::computeL2Norm(const vec_real& vc)
{
    real_t sum = std::transform_reduce(vc.cbegin(), vc.cend(), 0.0, std::plus{},
                                       [](auto x){return x*x;});
    return std::sqrt(sum / static_cast<real_t>(vc.size()));
}

real_t ShootingSolver::computeMaxNorm(const vec_real& vc)
{
    return *(std::max_element(vc.begin(), vc.end()));
}
