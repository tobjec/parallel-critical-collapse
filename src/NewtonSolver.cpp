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

void NewtonSolver::assembleJacobian(const vec_real& baseInput, const vec_real& baseOutput, mat_real& jacobian)
{

    std::cout << "Starting to assemble Jacobian: " << std::endl;

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
}

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