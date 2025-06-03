#include "NewtonSolver.hpp"

NewtonSolver::NewtonSolver(SimulationConfig configIn)
    : config(configIn), Ntau(configIn.Ntau), Nnewton(3*configIn.Ntau/4), maxIts(configIn.MaxIterNewton), Dim(configIn.Dim),
      Delta(configIn.Delta), slowErr(configIn.SlowError), EpsNewton(configIn.EpsNewton), TolNewton(configIn.PrecisionNewton),
      XLeft(configIn.XLeft), XMid(configIn.XMid), XRight(configIn.XRight), NLeft(configIn.NLeft), NRight(configIn.NRight),
      iL(0), iM(configIn.NLeft), iR(configIn.NLeft+configIn.NRight-1), initGen(configIn.Ntau, configIn.Dim, configIn.Delta)
{
    fc = configIn.fc;
    psic = configIn.psic;
    Up = configIn.Up;
    mismatchOut.resize(Ntau);
    in0.resize(3*configIn.Ntau/4);
    out0.resize(3*configIn.Ntau/4);
    XGrid.resize(configIn.NLeft + configIn.NRight);

    YLeft.resize(Ntau);
    YRight.resize(Ntau);

    shooter = std::make_unique<ShootingSolver>(configIn.Ntau, configIn.Dim,
                                               configIn.PrecisionIRK, initGen, configIn.MaxIterIRK);

    #ifdef USE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Type_contiguous(Nnewton, MPI_DOUBLE, &mpi_contiguous_t);
    MPI_Type_vector(Nnewton, 1, Nnewton, MPI_DOUBLE, &col);
    MPI_Type_commit(&col);
    MPI_Type_create_resized(col, 0, sizeof(double), &mpi_column_t);
    MPI_Type_commit(&mpi_contiguous_t);
    MPI_Type_commit(&mpi_column_t);
    MPI_Type_free(&col);
    #endif
}

#ifdef USE_MPI

NewtonSolver::~NewtonSolver()
{
    MPI_Type_free(&mpi_contiguous_t);
    MPI_Type_free(&mpi_column_t);
}

void NewtonSolver::run()
{
    real_t err = 1.0;
    real_t fac = 1.0;

    initGen.packSpectralFields(Up, psic, fc, in0);
    in0[2*Nnewton/3+2] = Delta;
    generateGrid();

    for (size_t its=0; its<maxIts; ++its)
    {
        if (rank==0)
        {
            std::cout << "Newton iteration: " << its + 1 << std::endl;

            shoot(in0, out0);

            err = computeL2Norm(out0);
            std::cout << "Mismatch norm: " << err << std::endl;

            if (err<TolNewton)
            {
                std::cout << "Converged!" << std::endl;
                writeFinalOutput();
            }

        }

        MPI_Bcast(&err, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (err<TolNewton)
        {
            return;
        }
        else
        {
            MPI_Bcast(in0.data(), 1, mpi_contiguous_t, 0, MPI_COMM_WORLD);
            MPI_Bcast(out0.data(), 1, mpi_contiguous_t, 0, MPI_COMM_WORLD);
        }

        vec_real J(Nnewton*Nnewton);
        assembleJacobian(in0, out0, J);

        if (rank==0)
        {
            // Solving J dx = -out0
            vec_real dx(Nnewton);
            vec_real rhs = out0;
            std::for_each(rhs.begin(), rhs.end(), [](auto& ele){ele*=-1.0;});

            solveLinearSystem(J, rhs, dx);

            // Damping factor based on mismatch reduction
            fac = std::min(1.0, slowErr / err);

            for (size_t i=0; i<Nnewton; ++i)
            {
                in0[i] += fac * dx[i];
            }
        }
        
    }

    std::cerr << "Newton method did not converge in " << maxIts << " iterations." << std::endl;
    std::exit(EXIT_FAILURE);
}

#else
void NewtonSolver::run()
{
    real_t err = 1.0;
    real_t fac = 1.0;

    initGen.packSpectralFields(Up, psic, fc, in0);
    in0[2*Nnewton/3+2] = Delta;
    generateGrid();

    for (size_t its=0; its<maxIts; ++its)
    {
        std::cout << "Newton iteration: " << its + 1 << std::endl;

        shoot(in0, out0);

        err = computeL2Norm(out0);
        std::cout << "Mismatch norm: " << err << std::endl;

        if (err<TolNewton)
        {
            std::cout << "Converged!" << std::endl;
            writeFinalOutput();
            return;
        }

        mat_real J(Nnewton, vec_real(Nnewton));
        assembleJacobian(in0, out0, J);

        // Solving J dx = -out0
        vec_real dx(Nnewton);
        vec_real rhs = out0;
        std::for_each(rhs.begin(), rhs.end(), [](auto& ele){ele*=-1.0;});

        solveLinearSystem(J, rhs, dx);

        // Damping factor based on mismatch reduction
        fac = std::min(1.0, slowErr / err);

        for (size_t i=0; i<Nnewton; ++i)
        {
            in0[i] += fac * dx[i];
        }
    }

    std::cerr << "Newton method did not converge in " << maxIts << " iterations." << std::endl;
    std::exit(EXIT_FAILURE);
}

#endif

void NewtonSolver::shoot(vec_real& inputVec, vec_real& outputVec)
{
    vec_real U(Ntau), V(Ntau), F(Ntau);

    Delta = inputVec[2*Nnewton/3+2];
    inputVec[2*Nnewton/3+2] = 0.0;

    initGen.unpackSpectralFields(inputVec, Up, psic, fc);

    inputVec[2*Nnewton/3+2] = Delta;

    // Initialize input for shooting
    initializeInput();

    // Shooting from left and right side
    shooter->shoot(YLeft, YRight, XGrid, iL, iR, iM, mismatchOut);
    
    // Packing state vector to fields
    initGen.StateVectorToFields(mismatchOut, U, V, F);
    
    // Packing resulting fields to out vector
    initGen.packSpectralFields(U, V, F, outputVec);

}

void NewtonSolver::initializeInput()
{
    initGen.computeLeftExpansion(XLeft, fc, psic, YLeft, Delta, false);
    initGen.computeRightExpansion(XRight, Up, YRight, Delta, false);
}

#ifdef USE_OPENMP
void NewtonSolver::shoot(vec_real& inputVec, vec_real& outputVec, ShootingSolver& shooter_local,
                   InitialConditionGenerator& initGen_local, vec_complex& YLeft_local, vec_complex& YRight_local,
                   vec_complex& mismatchOut_local)
{
    vec_real U(Ntau), V(Ntau), F(Ntau), Up_local(Ntau), psic_local(Ntau), fc_local(Ntau);
    real_t Delta_local {0};

    Delta_local = inputVec[2*Nnewton/3+2];
    inputVec[2*Nnewton/3+2] = 0.0;

    initGen_local.unpackSpectralFields(inputVec, Up_local, psic_local, fc_local);

    inputVec[2*Nnewton/3+2] = Delta_local;

    // Initialize input for shooting
    initializeInput(initGen_local, YLeft_local, YRight_local, Up_local, psic_local, fc_local, Delta_local);

    // Shooting from left and right side
    shooter_local.shoot(YLeft_local, YRight_local, XGrid, iL, iR, iM, mismatchOut_local);
    
    // Packing state vector to fields
    initGen_local.StateVectorToFields(mismatchOut_local, U, V, F);
    
    // Packing resulting fields to out vector
    initGen_local.packSpectralFields(U, V, F, outputVec);

}

void NewtonSolver::initializeInput(InitialConditionGenerator& initGen_local, 
                                   vec_complex& YLeft_local, vec_complex& YRight_local,
                                   vec_real& Up_local, vec_real& psic_local, vec_real& fc_local,
                                   real_t Delta_local)
{
    initGen_local.computeLeftExpansion(XLeft, fc_local, psic_local, YLeft_local, Delta_local, false);
    initGen_local.computeRightExpansion(XRight, Up_local, YRight_local, Delta_local, false);
}

#endif

void NewtonSolver::generateGrid()
{
    if (config.UseLogGrid)
    {
        XGrid[iL] = XLeft;
        XGrid[iR] = XRight;

        // Uniform grid in z
        real_t ZLeft = std::log(XLeft/(1-XLeft));
        real_t ZMid = std::log(XMid/(1-XMid));
        real_t ZRight = std::log(XRight/(1-XRight));

        real_t dZL = (ZMid - ZLeft) / static_cast<real_t>(NLeft);
        real_t dZR = (ZRight - ZMid) / static_cast<real_t>(NRight);

        for (size_t j=iL+1; j<iM; ++j)
        {
            real_t exponent = ZLeft + static_cast<real_t>(j-iL) * dZL;
            XGrid[j] = std::exp(exponent) / (1.0 + std::exp(exponent));
        }

        XGrid[iM] = XMid;

        for (size_t j=iM+1; j<iR; ++j)
        {
            real_t exponent = ZMid + static_cast<real_t>(j-iM) * dZR;
            XGrid[j] = std::exp(exponent) / (1.0 + std::exp(exponent));
        }

    }

}

#ifdef USE_MPI
void NewtonSolver::assembleJacobian(const vec_real& baseInput, const vec_real& baseOutput, vec_real& jacobian)
{
    if (rank==0)
    {
        std::cout << "Starting to assemble Jacobian: " << std::endl;
    }
    
    auto toc_outer = std::chrono::high_resolution_clock::now();

    std::vector<int> indices;

    for (size_t i=rank; i<Nnewton; i+=size)
    {
        auto toc = std::chrono::high_resolution_clock::now();

        indices.push_back(i);
        vec_real perturbedInput = baseInput;
        perturbedInput[i] += EpsNewton;

        vec_real perturbedOutput(Nnewton);
        shoot(perturbedInput, perturbedOutput);

        for (size_t j=0; j<Nnewton; ++j)
        {
            jacobian[j*Nnewton+i] = (perturbedOutput[j] - baseOutput[j]) / EpsNewton;
        }
       
        auto tic = std::chrono::high_resolution_clock::now();

        std::cout << "Varying parameter: " << i+1 << "/" << Nnewton;
        std::cout << " by rank " << rank;
        std::cout << " in " << static_cast<real_t>((tic-toc).count()) / 1e9;
        std::cout << " s." << std::endl;
    }

    if (rank!=0)
    {
        for (size_t i=0; i<indices.size(); ++i)
        {
            size_t idx = indices[i];
            MPI_Send(&jacobian[idx], 1, mpi_column_t, 0, static_cast<int>(idx), MPI_COMM_WORLD);
        }
    }
    else
    {
        for (size_t i=0; i<Nnewton; ++i)
        {
            if (i%size!=0)
            {
                MPI_Recv(&jacobian[i], 1, mpi_column_t, MPI_ANY_SOURCE, static_cast<int>(i), MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    auto tic_outer = std::chrono::high_resolution_clock::now();

    if (rank==0)
    {
        std::cout << "Time for Newton Iteration: " << static_cast<real_t>((tic_outer-toc_outer).count()) / 1e9;
        std::cout << " s." << std::endl;
    }
    
}

#else

void NewtonSolver::assembleJacobian(const vec_real& baseInput, const vec_real& baseOutput, mat_real& jacobian)
{

    #if defined(USE_OPENMP)
    std::cout << "Starting to assemble Jacobian: " << std::endl;

    #pragma omp parallel
    {
        vec_complex YLeft_local(Ntau), YRight_local(Ntau), mismatchOut_local(Ntau);
        InitialConditionGenerator initGen_local(Ntau, Dim, Delta);
        ShootingSolver shooter_local(Ntau, Dim, config.PrecisionIRK, initGen_local, config.MaxIterIRK);

        auto toc = std::chrono::high_resolution_clock::now();

        #pragma omp for schedule(static,8)
        for (size_t i=0; i<Nnewton; ++i)
        {
            vec_real perturbedInput = baseInput;
            perturbedInput[i] += EpsNewton;

            vec_real perturbedOutput(Nnewton);
            shoot(perturbedInput, perturbedOutput, shooter_local, initGen_local, YLeft_local, YRight_local, mismatchOut_local);

            for (size_t j=0; j<Nnewton; ++j)
            {
                jacobian[j][i] = (perturbedOutput[j] - baseOutput[j]) / EpsNewton;
            }
        }

        auto tic = std::chrono::high_resolution_clock::now();
        if (omp_get_thread_num() == 0)
        {
            std::cout << "Time for Newton Iteration: " << static_cast<real_t>((tic-toc).count()) / 1e9;
            std::cout << " s." << std::endl; 
        }

    }

    #else
    std::cout << "Starting to assemble Jacobian: " << std::endl;

    auto toc_outer = std::chrono::high_resolution_clock::now();

    for (size_t i=0; i<Nnewton; ++i)
    {
        auto toc = std::chrono::high_resolution_clock::now();

        vec_real perturbedInput = baseInput;
        perturbedInput[i] += EpsNewton;

        vec_real perturbedOutput(Nnewton);
        shoot(perturbedInput, perturbedOutput);

        for (size_t j=0; j<Nnewton; ++j)
        {
            jacobian[j][i] = (perturbedOutput[j] - baseOutput[j]) / EpsNewton;
        }

        auto tic = std::chrono::high_resolution_clock::now();

        std::cout << "Varying parameter: " << i+1 << "/" << Nnewton;
        std::cout << " in " << static_cast<real_t>((tic-toc).count()) / 1e9;
        std::cout << " s." << std::endl;
    }

    auto tic_outer = std::chrono::high_resolution_clock::now();

    std::cout << "Time for Newton Iteration: " << static_cast<real_t>((tic_outer-toc_outer).count()) / 1e9;
    std::cout << " s." << std::endl;

    #endif
}
#endif


#ifdef USE_MPI
void NewtonSolver::solveLinearSystem(vec_real& A_in, vec_real& rhs, vec_real& dx)
{
    std::vector<lapack_int> ipiv(Nnewton);

    lapack_int info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, static_cast<int32_t>(Nnewton), 1, A_in.data(),
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

void NewtonSolver::writeFinalOutput()
{
    // Append all to a structured JSON file
    OutputWriter::appendResultToJson("data/results.json", Dim, Delta, fc, psic, Up);

    std::cout << "Final result stored under dimension d = " << Dim
              << " in data/results.json\n";
}