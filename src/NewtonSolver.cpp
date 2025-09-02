#include "NewtonSolver.hpp"

NewtonSolver::NewtonSolver(SimulationConfig configIn, std::filesystem::path dataPathIn, bool benchmarkIn)
    : config(configIn), Ntau(configIn.Ntau), Nnewton(3*configIn.Ntau/4), maxIts(configIn.MaxIterNewton), Dim(configIn.Dim),
      Delta(configIn.Delta), slowErr(configIn.SlowError), EpsNewton(configIn.EpsNewton), TolNewton(configIn.PrecisionNewton),
      XLeft(configIn.XLeft), XMid(configIn.XMid), XRight(configIn.XRight), NLeft(configIn.NLeft), NRight(configIn.NRight),
      iL(0), iM(configIn.NLeft), iR(configIn.NLeft+configIn.NRight), Debug(configIn.Debug), Verbose(configIn.Verbose),
      Converged(configIn.Converged), initGen(configIn.Ntau, configIn.Dim, configIn.Delta), baseFolder(dataPathIn), benchmark(benchmarkIn)
{
    fc = configIn.fc;
    psic = configIn.psic;
    Up = configIn.Up;
    mismatchOut.resize(Ntau);
    in0.resize(3*configIn.Ntau/4);
    out0.resize(3*configIn.Ntau/4);
    XGrid.resize(configIn.NLeft + configIn.NRight + 1);

    YLeft.resize(Ntau);
    YRight.resize(Ntau);

    shooter = std::make_unique<ShootingSolver>(configIn.Ntau, configIn.Dim,
                                               configIn.PrecisionIRK, initGen, configIn.MaxIterIRK);

    #if defined(USE_MPI) || defined(USE_HYBRID)
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Type_contiguous(Nnewton, MPI_DOUBLE, &mpi_contiguous_t);
    MPI_Type_commit(&mpi_contiguous_t);
    #endif
}

#if defined(USE_MPI) || defined(USE_HYBRID)
NewtonSolver::~NewtonSolver()
{
    MPI_Type_free(&mpi_contiguous_t);
}

json NewtonSolver::run(json* benchmark_result)
{
    if (!Converged)
    {
        real_t err = 1.0;
        real_t fac = 1.0;
        real_t errOld = 1.0;
        real_t overallTimeStart{}, overallTimeEnd{};
        real_t newtonTimeStart{}, newtonTimeEnd{};
        
        if (rank==0 && benchmark)
        {
            overallTimeStart = MPI_Wtime();    
        }

        initGen.packSpectralFields(Up, psic, fc, in0);
        in0[2*Nnewton/3+2] = Delta;
        generateGrid();

        vec_real in0old = in0;

        for (size_t its=0; its<maxIts; ++its)
        {
            if (rank==0)
            {
                if (benchmark)
                {
                    newtonTimeStart = MPI_Wtime();
                }
                
                std::cout << "Newton iteration: " << its+1 << std::endl;

                errOld = err;

                if (config.Debug)
                {
                    Debug = config.Debug;
                    json fieldVals;
                    shoot(in0, out0, &fieldVals);
                    auto fieldPath = baseFolder / ("fields_"+std::to_string(Dim)+"_"+std::to_string(its)+".json");
                    OutputWriter::writeJsonToFile(fieldPath.c_str(), fieldVals);
                    Debug = false;
                }
                else
                {
                    shoot(in0, out0);
                }

                err = computeL2Norm(out0);
                std::cout << "Mismatch norm: " << err << std::endl;

                if (err<TolNewton)
                {
                    std::cout << "The solution has converged!" << std::endl << std::endl;
                }

            }
            else
            {
                errOld = err;
            }

            MPI_Bcast(&err, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            if (err<TolNewton)
            {
                Converged=true;
                MPI_Bcast(in0.data(), 1, mpi_contiguous_t, 0, MPI_COMM_WORLD);
                Delta = in0[2*Nnewton/3+2];
                in0[2*Nnewton/3+2] = 0.0;
                initGen.unpackSpectralFields(in0, Up, psic, fc);
                writeFinalOutput(its, err);
                if (rank==0 && benchmark)
                {
                    overallTimeEnd = MPI_Wtime();
                    (*benchmark_result)["OverallTime"] = overallTimeEnd - overallTimeStart;   
                }
                break;
            }
            else
            {
                MPI_Bcast(in0.data(), 1, mpi_contiguous_t, 0, MPI_COMM_WORLD);
                MPI_Bcast(out0.data(), 1, mpi_contiguous_t, 0, MPI_COMM_WORLD);
            }

            if (its >= 2 &&
                std::log10(err) >
                std::log10(errOld))
            {
                Converged = true;
                MPI_Bcast(in0old.data(), 1, mpi_contiguous_t, 0, MPI_COMM_WORLD);
                Delta = in0old[2*Nnewton/3+2];
                in0old[2*Nnewton/3+2] = 0.0;
                initGen.unpackSpectralFields(in0old, Up, psic, fc);
                writeFinalOutput(its, errOld);
                if (rank==0)
                {
                    std::cerr << "Mismatch increased – terminating Newton iteration." << std::endl;
                }
                break;                    
            }

            mat_real J(Nnewton, vec_real(Nnewton));
            assembleJacobian(in0, out0, J);        

            if (rank==0)
            {
                if (config.Debug)
                {
                    auto jacPath = baseFolder / ("jacobian_"+std::to_string(Dim)+"_"+std::to_string(its+1)+".csv");
                    OutputWriter::writeMatrix(jacPath.c_str(), J);
                }

                // Solving J dx = -out0
                vec_real dx(Nnewton);
                vec_real rhs = out0;
                std::for_each(rhs.begin(), rhs.end(), [](auto& ele){ele*=-1.0;});

                solveLinearSystem(J, rhs, dx);

                // Damping factor based on mismatch reduction
                fac = std::min(1.0, slowErr / err);
                
                in0old = in0;

                for (size_t i=0; i<Nnewton; ++i)
                {
                    in0[i] += fac * dx[i];
                }


                if (benchmark)
                {
                    newtonTimeEnd = MPI_Wtime();
                    (*benchmark_result)["NewtonStep"+std::to_string(its+1)] = newtonTimeEnd - newtonTimeStart;
                }
            }
            
        }
        if (rank==0 && !Converged)
        {
            std::cerr << "Newton method did not converge in " << maxIts << " iterations. For Dim: " << Dim << std::endl;
        }
        if (!Converged)
        {
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        
    }
    else
    {
        real_t err{};

        if (rank==0)
        {
            std::cout << "The solution is already marked as converged!" << std::endl << std::endl;
            initGen.packSpectralFields(Up, psic, fc, in0);
            in0[2*Nnewton/3+2] = Delta;
            generateGrid();
            shoot(in0, out0);
            err = computeL2Norm(out0);
               
        }

        MPI_Bcast(&err, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        writeFinalOutput(0, err);

    }

    return resultDict;

}

#else
json NewtonSolver::run(json* benchmark_result)
{
    if (!Converged)
    {
        real_t fac = 1.0;
        real_t err = 1.0;
        real_t errOld = 1.0;
        std::chrono::_V2::system_clock::time_point overallTimeStart{}, overallTimeEnd{};
        std::chrono::_V2::system_clock::time_point newtonTimeStart{}, newtonTimeEnd{};

        if (benchmark)
        {
            overallTimeStart = std::chrono::high_resolution_clock::now();
        }

        initGen.packSpectralFields(Up, psic, fc, in0);
        in0[2*Nnewton/3+2] = Delta;
        generateGrid();

        vec_real in0old = in0;

        for (size_t its=0; its<maxIts; ++its)
        {
            if (benchmark)
            {
                newtonTimeStart = std::chrono::high_resolution_clock::now();
            }
            
            std::cout << "Newton iteration: " << its+1 << std::endl;

            errOld = err;

            if (config.Debug)
            {
                Debug = config.Debug;
                json fieldVals;
                shoot(in0, out0, &fieldVals);
                auto fieldPath = baseFolder / ("fields_"+std::to_string(Dim)+"_"+std::to_string(its)+".json");
                OutputWriter::writeJsonToFile(fieldPath.c_str(), fieldVals);
                Debug = false;
            }
            else
            {
                shoot(in0, out0);
            }

            err = computeL2Norm(out0);
            std::cout << "Mismatch norm: " << err << std::endl;

            if (err<TolNewton)
            {
                Converged = true;
                std::cout << "The solution has converged!" << std::endl << std::endl;
                writeFinalOutput(its, err);
                if (benchmark)
                {
                    overallTimeEnd = std::chrono::high_resolution_clock::now();
                    (*benchmark_result)["OverallTime"] = static_cast<double>((overallTimeEnd - overallTimeStart).count()) / 1e9; 
                }
                break;
            }

            if (its >= 2 &&
                std::log10(err) >
                std::log10(errOld))
            {
                Converged = true;
                Delta = in0old[2*Nnewton/3+2];
                in0old[2*Nnewton/3+2] = 0.0;
                initGen.unpackSpectralFields(in0old, Up, psic, fc);
                writeFinalOutput(its, errOld);
                std::cerr << "Mismatch increased – terminating Newton.\n";
                break;                    
            }

            mat_real J(Nnewton, vec_real(Nnewton));
            assembleJacobian(in0, out0, J);

            if (config.Debug)
            {
                auto jacPath = baseFolder / ("jacobian_"+std::to_string(Dim)+"_"+std::to_string(its+1)+".csv");
                OutputWriter::writeMatrix(jacPath.c_str(), J);
            }

            // Solving J dx = -out0
            vec_real dx(Nnewton);
            vec_real rhs = out0;
            std::for_each(rhs.begin(), rhs.end(), [](auto& ele){ele*=-1.0;});

            solveLinearSystem(J, rhs, dx);

            // Damping factor based on mismatch reduction
            fac = std::min(1.0, slowErr / err);
            
            in0old = in0;

            for (size_t i=0; i<Nnewton; ++i)
            {
                in0[i] += fac * dx[i];
            }


            if (benchmark)
            {
                newtonTimeEnd = std::chrono::high_resolution_clock::now();
                (*benchmark_result)["NewtonStep"+std::to_string(its+1)] = 
                static_cast<double>((newtonTimeEnd - newtonTimeStart).count()) / 1e9;
            }
        }

        if (!Converged)
        {
            std::cerr << "Newton method did not converge in " << maxIts << " iterations. For dimension: " << Dim << std::endl;
            std::exit(EXIT_FAILURE);
        }
        
    }
    else
    {
        initGen.packSpectralFields(Up, psic, fc, in0);
        in0[2*Nnewton/3+2] = Delta;
        generateGrid();
        shoot(in0, out0);
        real_t err = computeL2Norm(out0);
        std::cout << "The solution is already marked as converged!" << std::endl;
        writeFinalOutput(0, err);
    }

    return resultDict;
}

#endif

void NewtonSolver::shoot(vec_real& inputVec, vec_real& outputVec, json* fieldVals)
{
    vec_real U(Ntau), V(Ntau), F(Ntau);

    Delta = inputVec[2*Nnewton/3+2];
    inputVec[2*Nnewton/3+2] = 0.0;

    initGen.unpackSpectralFields(inputVec, Up, psic, fc);

    inputVec[2*Nnewton/3+2] = Delta;

    // Initialize input for shooting
    initializeInput(Verbose);

    // Shooting from left and right side
    shooter->shoot(YLeft, YRight, XGrid, iL, iR, iM, mismatchOut, Debug, fieldVals);
    
    // Packing state vector to fields
    initGen.StateVectorToFields(mismatchOut, U, V, F);
    
    // Packing resulting fields to out vector
    initGen.packSpectralFields(U, V, F, outputVec);

}

void NewtonSolver::initializeInput(bool printDiagnostics)
{
    initGen.computeLeftExpansion(XLeft, fc, psic, YLeft, Delta, printDiagnostics);
    initGen.computeRightExpansion(XRight, Up, YRight, Delta, printDiagnostics);
}

#ifdef USE_OPENMP
void NewtonSolver::shoot(vec_real& inputVec, vec_real& outputVec, ShootingSolver& shooter_local,
                   InitialConditionGenerator& initGen_local, vec_complex& YLeft_local, vec_complex& YRight_local,
                   vec_complex& mismatchOut_local)
{
    vec_real U(Ntau), V(Ntau), F(Ntau), Up_local(Ntau), psic_local(Ntau), fc_local(Ntau);
    real_t Delta_local {};

    Delta_local = inputVec[2*Nnewton/3+2];
    inputVec[2*Nnewton/3+2] = 0.0;

    initGen_local.unpackSpectralFields(inputVec, Up_local, psic_local, fc_local);

    inputVec[2*Nnewton/3+2] = Delta_local;

    // Initialize input for shooting
    initializeInput(initGen_local, YLeft_local, YRight_local, Up_local, psic_local, fc_local, Delta_local, Verbose);

    // Shooting from left and right side
    shooter_local.shoot(YLeft_local, YRight_local, XGrid, iL, iR, iM, mismatchOut_local, false);
    
    // Packing state vector to fields
    initGen_local.StateVectorToFields(mismatchOut_local, U, V, F);
    
    // Packing resulting fields to out vector
    initGen_local.packSpectralFields(U, V, F, outputVec);

}

void NewtonSolver::initializeInput(InitialConditionGenerator& initGen_local, 
                                   vec_complex& YLeft_local, vec_complex& YRight_local,
                                   vec_real& Up_local, vec_real& psic_local, vec_real& fc_local,
                                   real_t Delta_local, bool printDiagnostics)
{
    initGen_local.computeLeftExpansion(XLeft, fc_local, psic_local, YLeft_local, Delta_local, printDiagnostics);
    initGen_local.computeRightExpansion(XRight, Up_local, YRight_local, Delta_local, printDiagnostics);
}

#endif

void NewtonSolver::generateGrid()
{
    XGrid[iL] = XLeft;
    XGrid[iM] = XMid;
    XGrid[iR] = XRight;

    // Uniform grid in z
    real_t ZLeft = std::log(XLeft) - std::log(1.0-XLeft);
    real_t ZMid = std::log(XMid) - std::log(1.0-XMid);
    real_t ZRight = std::log(XRight) - std::log(1.0-XRight);

    real_t dZL = (ZMid - ZLeft) / static_cast<real_t>(NLeft);
    real_t dZR = (ZRight - ZMid) / static_cast<real_t>(NRight);

    #ifdef USE_HYBRID
    #pragma omp parallel for schedule(static)
    #endif
    for (size_t j=iL+1; j<iM; ++j)
    {
        real_t exponent = ZLeft + static_cast<real_t>(j-iL) * dZL;
        XGrid[j] = std::exp(exponent) / (1.0 + std::exp(exponent));
    }

    #ifdef USE_HYBRID
    #pragma omp parallel for schedule(static)
    #endif
    for (size_t j=iM+1; j<iR; ++j)
    {
        real_t exponent = ZMid + static_cast<real_t>(j-iM) * dZR;
        XGrid[j] = std::exp(exponent) / (1.0 + std::exp(exponent));
    }

}

#if defined(USE_MPI) || defined(USE_HYBRID)
void NewtonSolver::assembleJacobian(const vec_real& baseInput, const vec_real& baseOutput, mat_real& jacobian)
{
    if (rank==0)
    {
        std::cout << "Starting to assemble Jacobian: " << std::endl << std::endl;
    }

    Verbose = false;
    
    auto toc_outer = std::chrono::high_resolution_clock::now();

    std::vector<int> indices;
    std::vector<MPI_Request> requests;

    for (size_t i=rank; i<Nnewton; i+=size)
    {
        auto toc = std::chrono::high_resolution_clock::now();

        indices.push_back(i);
        if (rank!=0) requests.emplace_back();
        vec_real perturbedInput = baseInput;
        perturbedInput[i] += EpsNewton;

        vec_real perturbedOutput(Nnewton);
        shoot(perturbedInput, perturbedOutput);

        // Transposed version of Jacobian
        #ifdef USE_HYBRID
        #pragma omp parallel for schedule(static)
        #endif
        for (size_t j=0; j<Nnewton; ++j)
        {
            jacobian[i][j] = (perturbedOutput[j] - baseOutput[j]) / EpsNewton;
        }
        
        if (config.Verbose)
        {
            auto tic = std::chrono::high_resolution_clock::now();
            std::cout << "Varying parameter: " << i+1 << "/" << Nnewton;
            std::cout << " by rank " << rank;
            std::cout << " in " << static_cast<real_t>((tic-toc).count()) / 1e9;
            std::cout << " s." << std::endl;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank!=0)
    {
        for (size_t i=0; i<indices.size(); ++i)
        {
            size_t idx = indices[i];
            MPI_Isend(jacobian[idx].data(), 1, mpi_contiguous_t, 0, static_cast<int>(idx), MPI_COMM_WORLD, &requests[i]);
        }

        MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
    }
    else
    {
        for (size_t i=0; i<Nnewton; ++i)
        {
            if (i%size!=0)
            {
                MPI_Recv(jacobian[i].data(), 1, mpi_contiguous_t, MPI_ANY_SOURCE, static_cast<int>(i), MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            
        }

    }

    MPI_Barrier(MPI_COMM_WORLD);

    auto tic_outer = std::chrono::high_resolution_clock::now();

    if (rank==0)
    {
        std::cout << "Time for Newton Iteration: " << static_cast<real_t>((tic_outer-toc_outer).count()) / 1e9;
        std::cout << " s." << std::endl << std::endl;
    }

    Verbose = config.Verbose;
    
}

#else

void NewtonSolver::assembleJacobian(const vec_real& baseInput, const vec_real& baseOutput, mat_real& jacobian)
{
    Verbose = false;

    #if defined(USE_OPENMP)
    std::cout << "Starting to assemble Jacobian: " << std::endl << std::endl;

    #pragma omp parallel
    {
        vec_complex YLeft_local(Ntau), YRight_local(Ntau), mismatchOut_local(Ntau);
        InitialConditionGenerator initGen_local(Ntau, Dim, Delta);
        ShootingSolver shooter_local(Ntau, Dim, config.PrecisionIRK, initGen_local, config.MaxIterIRK);

        auto toc_outer = std::chrono::high_resolution_clock::now();

        #pragma omp for schedule(static)
        for (size_t i=0; i<Nnewton; ++i)
        {
            auto toc_inner = std::chrono::high_resolution_clock::now();
            vec_real perturbedInput = baseInput;
            perturbedInput[i] += EpsNewton;

            vec_real perturbedOutput(Nnewton);
            shoot(perturbedInput, perturbedOutput, shooter_local, initGen_local, YLeft_local, YRight_local, mismatchOut_local);

            for (size_t j=0; j<Nnewton; ++j)
            {
                jacobian[j][i] = (perturbedOutput[j] - baseOutput[j]) / EpsNewton;
            }

            if (config.Verbose)
            {
                auto tic_inner = std::chrono::high_resolution_clock::now();
                #pragma omp critical
                {
                    std::cout << "Varying parameter: " << i+1 << "/" << Nnewton;
                    std::cout << " by thread " << omp_get_thread_num();
                    std::cout << " in " << static_cast<real_t>((tic_inner-toc_inner).count()) / 1e9;
                    std::cout << " s." << std::endl;
                }
                
            }
        }

        auto tic_outer = std::chrono::high_resolution_clock::now();
        if (omp_get_thread_num() == 0)
        {
            std::cout << "Time for Newton Iteration: " << static_cast<real_t>((tic_outer-toc_outer).count()) / 1e9;
            std::cout << " s." << std::endl << std::endl; 
        }

    }

    #else
    std::cout << "Starting to assemble Jacobian: " << std::endl << std::endl;

    auto toc_outer = std::chrono::high_resolution_clock::now();

    for (size_t i=0; i<Nnewton; ++i)
    {
        auto toc_inner = std::chrono::high_resolution_clock::now();

        vec_real perturbedInput = baseInput;
        perturbedInput[i] += EpsNewton;

        vec_real perturbedOutput(Nnewton);
        shoot(perturbedInput, perturbedOutput);

        for (size_t j=0; j<Nnewton; ++j)
        {
            jacobian[j][i] = (perturbedOutput[j] - baseOutput[j]) / EpsNewton;
        }

        if (config.Verbose)
        {
            auto tic_inner = std::chrono::high_resolution_clock::now();
            std::cout << "Varying parameter: " << i+1 << "/" << Nnewton;
            std::cout << " in " << static_cast<real_t>((tic_inner-toc_inner).count()) / 1e9;
            std::cout << " s." << std::endl;
        }
        
    }

    auto tic_outer = std::chrono::high_resolution_clock::now();

    std::cout << "Time for Newton Iteration: " << static_cast<real_t>((tic_outer-toc_outer).count()) / 1e9;
    std::cout << " s." << std::endl << std::endl;

    #endif
    Verbose = config.Verbose;
}
#endif


#if defined(USE_MPI) || defined(USE_HYBRID)
void NewtonSolver::solveLinearSystem(mat_real& A_in, vec_real& rhs, vec_real& dx)
{
    vec_real A_flat(Nnewton*Nnewton);
    for (size_t i=0; i<Nnewton; ++i)
    {
        for (size_t j=0; j<Nnewton; ++j)
        {
            A_flat[j*Nnewton+i] = A_in[i][j];
        }
    }

    std::vector<lapack_int> ipiv(Nnewton);

    lapack_int info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, static_cast<int32_t>(Nnewton), 1, A_flat.data(),
                                    static_cast<int32_t>(Nnewton), ipiv.data(), rhs.data(), 1);
    if (info != 0)
    {
        std::cerr << "ERROR: LAPACKE_dgesv failed with error code " << info << std::endl;
        std::exit(EXIT_FAILURE);
    }
    // Solution vector
    dx = rhs;
}

#else
void NewtonSolver::solveLinearSystem(const mat_real& A_in, vec_real& rhs, vec_real& dx)
{
    // Convert matrix to row-major double* for LAPACKE
    vec_real A_flat(Nnewton * Nnewton);
    for (size_t i=0; i<Nnewton; ++i)
    {
        std::memcpy(&A_flat[i * Nnewton], A_in[i].data(), Nnewton * sizeof(real_t));
    }

    std::vector<lapack_int> ipiv(Nnewton);

    lapack_int info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, static_cast<int32_t>(Nnewton), 1, A_flat.data(),
                                    static_cast<int32_t>(Nnewton), ipiv.data(), rhs.data(), 1);
    if (info != 0)
    {
        std::cerr << "ERROR: LAPACKE_dgesv failed with error code " << info << std::endl;
        std::exit(EXIT_FAILURE);
    }
    // Solution vector
    dx = rhs;
}
#endif

real_t NewtonSolver::computeL2Norm(const vec_real& vc)
{
    real_t sum = std::transform_reduce(vc.cbegin(), vc.cend(), 0.0, std::plus{},
                                       [](auto x){return x*x;});
    return std::sqrt(sum / static_cast<real_t>(vc.size()));
}

void NewtonSolver::writeFinalOutput(size_t newtonIts, real_t mismatchNorm)
{
    resultDict["Ntau"] = Ntau;
    resultDict["Dim"] = Dim;
    resultDict["XLeft"] = XLeft;
    resultDict["XMid"] = XMid;
    resultDict["XRight"] = XRight;
    resultDict["EpsNewton"] = EpsNewton;
    resultDict["PrecisionNewton"] = TolNewton;
    resultDict["SlowError"] = slowErr;
    resultDict["MaxIterNewton"] = maxIts;
    resultDict["Verbose"] = Verbose;
    resultDict["Debug"] = Debug;
    resultDict["Converged"] = true;
    resultDict["NLeft"] = NLeft;
    resultDict["NRight"] = NRight;
    resultDict["PrecisionIRK"] = config.PrecisionIRK;
    resultDict["MaxIterIRK"] = config.MaxIterIRK;
    resultDict["IterNewton"] = newtonIts;
    resultDict["mismatchNorm"] = mismatchNorm;
    resultDict["Initial_Conditions"]["Delta"] = Delta;
    resultDict["Initial_Conditions"]["fc"] = fc;
    resultDict["Initial_Conditions"]["psic"] = psic;
    resultDict["Initial_Conditions"]["Up"] = Up;

    #if defined(USE_MPI) || defined(USE_HYBRID)
    if (rank==0)
    {
        std::cout << "Final result stored in simulation dictionary for dimension D = " << Dim << "." << std::endl << std::endl;
    }
    #else
    std::cout << "Final result stored in simulation dictionary for dimension D = " << Dim << "." << std::endl << std::endl;
    #endif
}