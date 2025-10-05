//==============================================================================
// NewtonSolver.cpp
// Newton–Kantorovich solver for the critical-collapse boundary value problem.
// Responsibilities:
//   • Hold simulation config and working buffers (spectral/state/grids).
//   • Build left/right Taylor initial data and call the ShootingSolver.
//   • Assemble finite-difference Jacobian via repeated “shoot” evaluations.
//   • Solve J·dx = −res with LAPACK (row-major), apply line damping, iterate.
//   • Support Serial / OpenMP / MPI / Hybrid backends and benchmarking.
//==============================================================================

#include "NewtonSolver.hpp"

//------------------------------------------------------------------------------
// Ctor: cache config scalars & sizes, allocate work arrays, construct helpers.
// If MPI/Hybrid, also create a contiguous MPI type to send/recv vectors.
//------------------------------------------------------------------------------
NewtonSolver::NewtonSolver(SimulationConfig configIn, std::filesystem::path dataPathIn, bool benchmarkIn)
    : config(configIn), Ntau(configIn.Ntau), Nnewton(3*configIn.Ntau/4), maxIts(configIn.MaxIterNewton), Dim(configIn.Dim),
      Delta(configIn.Delta), slowErr(configIn.SlowError), EpsNewton(configIn.EpsNewton), TolNewton(configIn.PrecisionNewton),
      XLeft(configIn.XLeft), XMid(configIn.XMid), XRight(configIn.XRight), NLeft(configIn.NLeft), NRight(configIn.NRight),
      iL(0), iM(configIn.NLeft), iR(configIn.NLeft+configIn.NRight), Debug(configIn.Debug), Verbose(configIn.Verbose),
      Converged(configIn.Converged), initGen(configIn.Ntau, configIn.Dim, configIn.Delta), baseFolder(dataPathIn), benchmark(benchmarkIn)
{
    // Initial field data (may be empty for extrapolation-driven runs)
    fc   = configIn.fc;
    psic = configIn.psic;
    Up   = configIn.Up;

    // Work buffers
    mismatchOut.resize(Ntau);
    in0.resize(3*configIn.Ntau/4);
    out0.resize(3*configIn.Ntau/4);
    XGrid.resize(configIn.NLeft + configIn.NRight + 1);

    // State vectors at the left/right boundaries
    YLeft.resize(Ntau);
    YRight.resize(Ntau);

    // Shooting ODE integrator (IRK) wrapper
    shooter = std::make_unique<ShootingSolver>(configIn.Ntau, configIn.Dim,
                                               configIn.PrecisionIRK, initGen, configIn.MaxIterIRK, configIn.SchemeIRK);

    #if defined(USE_MPI) || defined(USE_HYBRID)
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Contiguous datatype to move entire R^Nnewton rows/cols efficiently
    MPI_Type_contiguous(Nnewton, MPI_DOUBLE, &mpi_contiguous_t);
    MPI_Type_commit(&mpi_contiguous_t);
    #endif
}

#if defined(USE_MPI) || defined(USE_HYBRID)
//------------------------------------------------------------------------------
// Dtor: free the custom MPI datatype.
//------------------------------------------------------------------------------
NewtonSolver::~NewtonSolver()
{
    MPI_Type_free(&mpi_contiguous_t);
}

//------------------------------------------------------------------------------
// run (MPI/Hybrid)
// Orchestrates Newton iterations; root rank logs and writes results.
// Optional `benchmark_result` JSON collects timing per step.
//------------------------------------------------------------------------------
json NewtonSolver::run(json* benchmark_result)
{
    real_t overallTimeStart{}, overallTimeEnd{};
    real_t newtonTimeStart{}, newtonTimeEnd{};
    real_t assembleTimeStart{}, assembleTimeEnd{};

    if (rank==0 && benchmark) overallTimeStart = MPI_Wtime();    

    if (!Converged)
    {
        // Newton scalars
        real_t err = 1.0;       // current residual norm
        real_t fac = 1.0;       // damping factor
        real_t errOld = 1.0;    // previous residual norm

        // Pack (Up, psic, fc, Δ) → in0 and build spatial grid
        initGen.packSpectralFields(Up, psic, fc, in0);
        in0[2*Nnewton/3+2] = Delta;
        generateGrid();

        vec_real in0old = in0;  // best-so-far input for backtracking

        //==============================
        // Newton iteration loop
        //==============================
        for (size_t its=0; its<maxIts; ++its)
        {
            if (rank==0)
            {
                if (benchmark) newtonTimeStart = MPI_Wtime();
                std::cout << "Newton iteration: " << its+1 << std::endl;

                errOld = err;

                // Optionally dump fields (for debugging/inspection)
                if (config.Debug)
                {
                    Debug = config.Debug;
                    json fieldVals;
                    fieldVals["Delta"] = Delta;
                    shoot(in0, out0, &fieldVals);
                    auto fieldPath = baseFolder / ("fields_"+std::to_string(Dim)+"_"+std::to_string(its)+".json");
                    OutputWriter::writeJsonToFile(fieldPath.c_str(), fieldVals);
                    Debug = false;
                }
                else
                {
                    shoot(in0, out0);
                }

                // Residual norm
                err = computeL2Norm(out0);
                std::cout << "Mismatch norm: " << err << std::endl;

                if (err<TolNewton)
                {
                    std::cout << "The solution has converged!" << std::endl << std::endl;
                }
            }
            else
            {
                // keep local copy of previous err for monotonicity check
                errOld = err;
            }

            // Broadcast residual to all ranks for common termination criteria
            MPI_Bcast(&err, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            // Convergence: unpack final state and write output
            if (err<TolNewton)
            {
                Converged=true;
                MPI_Bcast(in0.data(), 1, mpi_contiguous_t, 0, MPI_COMM_WORLD);
                Delta = in0[2*Nnewton/3+2];
                in0[2*Nnewton/3+2] = 0.0;
                initGen.unpackSpectralFields(in0, Up, psic, fc);
                writeFinalOutput(its, err);
                break;
            }
            else
            {
                // Share current in/out across ranks (for Jacobian assembly)
                MPI_Bcast(in0.data(), 1, mpi_contiguous_t, 0, MPI_COMM_WORLD);
                MPI_Bcast(out0.data(), 1, mpi_contiguous_t, 0, MPI_COMM_WORLD);
            }

            // Monotonicity safeguard: if residual worsens for ≥1 step, backtrack
            if (its >= 2 && std::log10(err) > std::log10(errOld))
            {
                Converged = true;
                MPI_Bcast(in0old.data(), 1, mpi_contiguous_t, 0, MPI_COMM_WORLD);
                Delta = in0old[2*Nnewton/3+2];
                in0old[2*Nnewton/3+2] = 0.0;
                initGen.unpackSpectralFields(in0old, Up, psic, fc);
                writeFinalOutput(its, errOld);
                if (rank==0)
                    std::cerr << "Mismatch increased – terminating Newton iteration." << std::endl;
                break;                    
            }

            // Assemble finite-difference Jacobian (column-by-column)
            mat_real J(Nnewton, vec_real(Nnewton));
            if (rank==0 && benchmark) assembleTimeStart = MPI_Wtime();
            assembleJacobian(in0, out0, J);
            if (rank==0 && benchmark)
            {
                assembleTimeEnd = MPI_Wtime();
                (*benchmark_result)["AssembleStep_"+std::to_string(its+1)] = assembleTimeEnd - assembleTimeStart;
            }        

            // Solve J·dx = −res and update with damping on root rank
            if (rank==0)
            {
                if (config.Debug)
                {
                    auto jacPath = baseFolder / ("jacobian_"+std::to_string(Dim)+"_"+std::to_string(its+1)+".csv");
                    OutputWriter::writeMatrix(jacPath.c_str(), J);
                }

                vec_real dx(Nnewton);
                vec_real rhs = out0;
                std::for_each(rhs.begin(), rhs.end(), [](auto& e){ e *= -1.0; });

                solveLinearSystem(J, rhs, dx);

                // Line damping based on target “slowErr”
                fac = std::min(1.0, slowErr / err);

                in0old = in0;                 // store trial point
                for (size_t i=0; i<Nnewton; ++i)
                {
                    in0[i] += fac * dx[i];
                }

                if (benchmark)
                {
                    newtonTimeEnd = MPI_Wtime();
                    (*benchmark_result)["NewtonStep_"+std::to_string(its+1)] = newtonTimeEnd - newtonTimeStart;
                }
            }
        }

        // Out-of-iterations handling
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
        // Already converged: compute and store residual once for the record
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

    if (rank==0 && benchmark)
    {
        overallTimeEnd = MPI_Wtime();
        (*benchmark_result)["OverallTime"] = overallTimeEnd - overallTimeStart;   
    }

    return resultDict;
}
#else
//------------------------------------------------------------------------------
// run (Serial/OpenMP)
// Same control flow as MPI, but with std::chrono for timing and no broadcasting.
//------------------------------------------------------------------------------
json NewtonSolver::run(json* benchmark_result)
{
    std::chrono::_V2::system_clock::time_point overallTimeStart{}, overallTimeEnd{};
    std::chrono::_V2::system_clock::time_point newtonTimeStart{}, newtonTimeEnd{};
    std::chrono::_V2::system_clock::time_point assembleTimeStart{}, assembleTimeEnd{};

    if (benchmark) overallTimeStart = std::chrono::high_resolution_clock::now();

    if (!Converged)
    {
        real_t fac = 1.0, err = 1.0, errOld = 1.0;

        initGen.packSpectralFields(Up, psic, fc, in0);
        in0[2*Nnewton/3+2] = Delta;
        generateGrid();

        vec_real in0old = in0;

        for (size_t its=0; its<maxIts; ++its)
        {
            if (benchmark) newtonTimeStart = std::chrono::high_resolution_clock::now();

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
                break;
            }

            // Safeguard against divergence
            if (its >= 2 && std::log10(err) > std::log10(errOld))
            {
                Converged = true;
                Delta = in0old[2*Nnewton/3+2];
                in0old[2*Nnewton/3+2] = 0.0;
                initGen.unpackSpectralFields(in0old, Up, psic, fc);
                writeFinalOutput(its, errOld);
                std::cerr << "Mismatch increased – terminating Newton.\n";
                break;                    
            }

            // Assemble J via finite differences
            mat_real J(Nnewton, vec_real(Nnewton));
            if (benchmark) assembleTimeStart = std::chrono::high_resolution_clock::now();
            assembleJacobian(in0, out0, J);
            if (benchmark)
            {
                assembleTimeEnd = std::chrono::high_resolution_clock::now();
                (*benchmark_result)["AssembleStep_"+std::to_string(its+1)] = 
                    static_cast<double>((assembleTimeEnd - assembleTimeStart).count()) / 1e9;
            }

            if (config.Debug)
            {
                auto jacPath = baseFolder / ("jacobian_"+std::to_string(Dim)+"_"+std::to_string(its+1)+".csv");
                OutputWriter::writeMatrix(jacPath.c_str(), J);
            }

            // Solve J·dx = −res and update input with damping
            vec_real dx(Nnewton);
            vec_real rhs = out0;
            std::for_each(rhs.begin(), rhs.end(), [](auto& e){ e *= -1.0; });

            solveLinearSystem(J, rhs, dx);

            fac = std::min(1.0, slowErr / err);
            in0old = in0;
            for (size_t i=0; i<Nnewton; ++i) in0[i] += fac * dx[i];

            if (benchmark)
            {
                newtonTimeEnd = std::chrono::high_resolution_clock::now();
                (*benchmark_result)["NewtonStep_"+std::to_string(its+1)] = 
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
        // Already converged: recompute residual for logging and store output
        initGen.packSpectralFields(Up, psic, fc, in0);
        in0[2*Nnewton/3+2] = Delta;
        generateGrid();
        shoot(in0, out0);
        real_t err = computeL2Norm(out0);
        std::cout << "The solution is already marked as converged!" << std::endl;
        writeFinalOutput(0, err);
    }

    if (benchmark)
    {
        overallTimeEnd = std::chrono::high_resolution_clock::now();
        (*benchmark_result)["OverallTime"] = static_cast<double>((overallTimeEnd - overallTimeStart).count()) / 1e9; 
    }

    return resultDict;
}
#endif

//------------------------------------------------------------------------------
// shoot
// 1) Extract (Up, ψc, fc, Δ) from packed input vector (spectral).
// 2) Build left/right boundary data (Taylor) and call IRK shooter.
// 3) Transform mismatch state → fields → pack back to spectral output.
// If `fieldVals` is provided, the shooter exports intermediate field slices.
//------------------------------------------------------------------------------
void NewtonSolver::shoot(vec_real& inputVec, vec_real& outputVec, json* fieldVals)
{
    vec_real U(Ntau), V(Ntau), F(Ntau);

    // Temporarily extract Δ stored in the “extra” slot of in0
    Delta = inputVec[2*Nnewton/3+2];
    inputVec[2*Nnewton/3+2] = 0.0;

    initGen.unpackSpectralFields(inputVec, Up, psic, fc);
    inputVec[2*Nnewton/3+2] = Delta;

    // Produce boundary Taylor data for both sides
    initializeInput(Verbose);

    // Two-sided shooting (left→mid and right→mid), get spectral mismatch
    shooter->shoot(YLeft, YRight, XGrid, iL, iR, iM, mismatchOut, Debug, fieldVals);

    // Transform mismatch state to fields and re-pack as output
    initGen.StateVectorToFields(mismatchOut, U, V, F);
    initGen.packSpectralFields(U, V, F, outputVec);
}

//------------------------------------------------------------------------------
// initializeInput (convenience): build boundary Taylor series on both ends.
//------------------------------------------------------------------------------
void NewtonSolver::initializeInput(bool printDiagnostics)
{
    initGen.computeLeftExpansion(XLeft, fc, psic, YLeft, Delta, printDiagnostics);
    initGen.computeRightExpansion(XRight, Up, YRight, Delta, printDiagnostics);
}

#ifdef USE_OPENMP
//------------------------------------------------------------------------------
// OpenMP-specialized “shoot”: accepts thread-local shooters/generators/buffers
// to avoid contention when assembling the Jacobian in parallel.
//------------------------------------------------------------------------------
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

    // Thread-local boundary Taylor data
    initializeInput(initGen_local, YLeft_local, YRight_local, Up_local, psic_local, fc_local, Delta_local, Verbose);

    // Local shoot and pack
    shooter_local.shoot(YLeft_local, YRight_local, XGrid, iL, iR, iM, mismatchOut_local, false);
    initGen_local.StateVectorToFields(mismatchOut_local, U, V, F);
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

//------------------------------------------------------------------------------
// generateGrid
// Build a monotone grid in x by uniform spacing in the logit variable
//   z = log(x) - log(1-x).
// Left block: [XLeft, XMid] with NLeft segments; Right: [XMid, XRight].
//------------------------------------------------------------------------------
void NewtonSolver::generateGrid()
{
    XGrid[iL] = XLeft;
    XGrid[iM] = XMid;
    XGrid[iR] = XRight;

    // Uniform in z
    real_t ZLeft  = std::log(XLeft)  - std::log(1.0 - XLeft);
    real_t ZMid   = std::log(XMid)   - std::log(1.0 - XMid);
    real_t ZRight = std::log(XRight) - std::log(1.0 - XRight);

    real_t dZL = (ZMid   - ZLeft ) / static_cast<real_t>(NLeft );
    real_t dZR = (ZRight - ZMid  ) / static_cast<real_t>(NRight);

    #ifdef USE_HYBRID
    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (size_t j=iL+1; j<iM; ++j)
        {
            real_t exponent = ZLeft + static_cast<real_t>(j-iL) * dZL;
            XGrid[j] = std::exp(exponent) / (1.0 + std::exp(exponent));
        }
        #pragma omp for schedule(static)
        for (size_t j=iM+1; j<iR; ++j)
        {
            real_t exponent = ZMid + static_cast<real_t>(j-iM) * dZR;
            XGrid[j] = std::exp(exponent) / (1.0 + std::exp(exponent));
        }
    }
    #else
    for (size_t j=iL+1; j<iM; ++j)
    {
        real_t exponent = ZLeft + static_cast<real_t>(j-iL) * dZL;
        XGrid[j] = std::exp(exponent) / (1.0 + std::exp(exponent));
    }
    for (size_t j=iM+1; j<iR; ++j)
    {
        real_t exponent = ZMid + static_cast<real_t>(j-iM) * dZR;
        XGrid[j] = std::exp(exponent) / (1.0 + std::exp(exponent));
    }
    #endif
}

#if defined(USE_MPI) || defined(USE_HYBRID)
//------------------------------------------------------------------------------
// assembleJacobian (MPI/Hybrid)
// Finite-difference columns in a round-robin fashion across ranks.
// Non-root ranks compute local columns and send them to rank 0.
//------------------------------------------------------------------------------
void NewtonSolver::assembleJacobian(const vec_real& baseInput, const vec_real& baseOutput, mat_real& jacobian)
{
    if (rank==0) std::cout << "Starting to assemble Jacobian: " << std::endl << std::endl;

    Verbose = false; // silence inner shoots
    auto toc_outer = std::chrono::high_resolution_clock::now();

    std::vector<int> indices;                 // which columns this rank computed
    std::vector<MPI_Request> requests;

    // Distribute i = rank, rank+size, ...
    for (size_t i=rank; i<Nnewton; i+=size)
    {
        auto toc = std::chrono::high_resolution_clock::now();

        indices.push_back(static_cast<int>(i));
        if (rank!=0) requests.emplace_back();

        // Perturb i-th input and re-shoot
        vec_real perturbedInput = baseInput;
        perturbedInput[i] += EpsNewton;

        vec_real perturbedOutput(Nnewton);
        shoot(perturbedInput, perturbedOutput);

        // Column i of J ≈ (f(u+εe_i) − f(u)) / ε
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
            std::cout << "Varying parameter: " << i+1 << "/" << Nnewton
                      << " by rank " << rank
                      << " in " << static_cast<real_t>((tic-toc).count()) / 1e9
                      << " s." << std::endl;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Non-root: send computed columns to root; Root: receive missing columns
    if (rank!=0)
    {
        for (size_t k=0; k<indices.size(); ++k)
        {
            size_t idx = static_cast<size_t>(indices[k]);
            MPI_Isend(jacobian[idx].data(), 1, mpi_contiguous_t, 0,
                      static_cast<int>(idx), MPI_COMM_WORLD, &requests[k]);
        }
        MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
    }
    else
    {
        for (size_t i=0; i<Nnewton; ++i)
        {
            if (i%size!=0)
            {
               MPI_Recv(jacobian[i].data(), 1, mpi_contiguous_t, MPI_ANY_SOURCE,
                         static_cast<int>(i), MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    auto tic_outer = std::chrono::high_resolution_clock::now();

    if (rank==0)
    {
        std::cout << "Time for Newton Iteration: " << static_cast<real_t>((tic_outer-toc_outer).count()) / 1e9
                  << " s." << std::endl << std::endl;
    }

    Verbose = config.Verbose; // restore verbosity
}
#else
//------------------------------------------------------------------------------
// assembleJacobian (Serial/OpenMP)
// Serial: loop over columns; OpenMP: thread-local shooters/generators and
//         private buffers to build columns in parallel safely.
// Note: we fill J by columns (jacobian[j][i]) for LAPACK row-major layout later.
//------------------------------------------------------------------------------
void NewtonSolver::assembleJacobian(const vec_real& baseInput, const vec_real& baseOutput, mat_real& jacobian)
{
    Verbose = false;

    #if defined(USE_OPENMP)
    std::cout << "Starting to assemble Jacobian: " << std::endl << std::endl;

    #pragma omp parallel
    {
        // Thread-local resources (avoid contention)
        vec_complex YLeft_local(Ntau), YRight_local(Ntau), mismatchOut_local(Ntau);
        InitialConditionGenerator initGen_local(Ntau, Dim, Delta);
        ShootingSolver shooter_local(Ntau, Dim, config.PrecisionIRK, initGen_local, config.MaxIterIRK, config.SchemeIRK);

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
                    std::cout << "Varying parameter: " << i+1 << "/" << Nnewton
                              << " by thread " << omp_get_thread_num()
                              << " in " << static_cast<real_t>((tic_inner-toc_inner).count()) / 1e9
                              << " s." << std::endl;
                }
            }
        }

        auto tic_outer = std::chrono::high_resolution_clock::now();
        if (omp_get_thread_num() == 0)
        {
            std::cout << "Time for Newton Iteration: " << static_cast<real_t>((tic_outer-toc_outer).count()) / 1e9
                      << " s." << std::endl << std::endl; 
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
            std::cout << "Varying parameter: " << i+1 << "/" << Nnewton
                      << " in " << static_cast<real_t>((tic_inner-toc_inner).count()) / 1e9
                      << " s." << std::endl;
        }
    }

    auto tic_outer = std::chrono::high_resolution_clock::now();
    std::cout << "Time for Newton Iteration: " << static_cast<real_t>((tic_outer-toc_outer).count()) / 1e9
              << " s." << std::endl << std::endl;
    #endif

    Verbose = config.Verbose;
}
#endif

#if defined(USE_MPI) || defined(USE_HYBRID)
//------------------------------------------------------------------------------
// solveLinearSystem (MPI/Hybrid)
// Build a row-major dense copy and solve with LAPACK dgesv in-place on rhs.
// `dx` receives the solution.
//------------------------------------------------------------------------------
void NewtonSolver::solveLinearSystem(mat_real& A_in, vec_real& rhs, vec_real& dx)
{
    // Column-major to row-major flattening that matches dgesv expectations
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
    dx = rhs;  // solution
}
#else
//------------------------------------------------------------------------------
// solveLinearSystem (Serial/OpenMP)
// Same LAPACK call; row-major dense copy made by memcpy per row.
//------------------------------------------------------------------------------
void NewtonSolver::solveLinearSystem(const mat_real& A_in, vec_real& rhs, vec_real& dx)
{
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
    dx = rhs;
}
#endif

//------------------------------------------------------------------------------
// computeL2Norm: sqrt( mean( v.^2 ) ). Used as mismatch metric.
//------------------------------------------------------------------------------
real_t NewtonSolver::computeL2Norm(const vec_real& vc)
{
    real_t sum = std::transform_reduce(vc.cbegin(), vc.cend(), 0.0, std::plus{},
                                       [](auto x){return x*x;});
    return std::sqrt(sum / static_cast<real_t>(vc.size()));
}

//------------------------------------------------------------------------------
// writeFinalOutput
// Fill `resultDict` with solver metadata, final mismatch, and initial data.
// On parallel builds only rank 0 prints the confirmation message.
//------------------------------------------------------------------------------
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
    resultDict["SchemeIRK"] = int(config.SchemeIRK)+1;
    resultDict["PrecisionIRK"] = config.PrecisionIRK;
    resultDict["MaxIterIRK"] = config.MaxIterIRK;
    resultDict["IterNewton"] = newtonIts;
    resultDict["mismatchNorm"] = mismatchNorm;

    // Store (Δ, fc, ψc, Up) that produced the final state
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
