#include "NewtonSolver.hpp"

NewtonSolver::NewtonSolver(SimulationConfig configIn)
    : config(configIn), Ntau(configIn.Ntau), Nnewton(3*configIn.Ntau/4), Dim(configIn.Dim),
      Delta(configIn.Delta), XLeft(configIn.XLeft), XMid(configIn.XMid), XRight(configIn.XRight),
      EpsNewton(configIn.EpsNewton), TolNewton(configIn.PrecisionNewton),
      initGen(configIn.Ntau, configIn.Dim, configIn.Delta)
{
    fc = configIn.fc;
    psic = configIn.psic;
    Up = configIn.Up;
    mismatchOut.resize(Ntau);
    in0.resize(3*configIn.Ntau/4);
    out0.resize(3*configIn.Ntau/4);
    NLeft = configIn.NLeft;
    NRight = configIn.NRight;
    iL = 0;
    iM = NLeft;
    iR = NLeft + NRight - 1;

    YLeft.resize(Ntau);
    YRight.resize(Ntau);

    shooter = std::make_unique<ShootingSolver>(configIn.Ntau, configIn.Dim,
                                               configIn.PrecisionIRK, initGen, configIn.MaxIterIRK);
}

void NewtonSolver::run(int maxIts)
{
    real_t err = 1.0;
    real_t fac = 1.0;

    initGen.packSpectralFields(Up, psic, fc, in0);
    generateGrid();

    for (int its = 0; its < maxIts; ++its)
    {
        std::cout << "Newton iteration: " << its << std::endl;

        shoot(in0, out0);

        err = computeL2Norm(out0);
        std::cout << "Mismatch norm: " << err << std::endl;

        if (err < TolNewton)
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
        fac = std::min(1.0, err / 1e-3);
        
        for (int i = 0; i < Nnewton; ++i)
        {
            in0[i] += fac * dx[i];
        }
    }

    std::cerr << "Newton method did not converge in " << maxIts << " iterations." << std::endl;
    std::exit(EXIT_FAILURE);
}

void NewtonSolver::shoot(const vec_real& inputVec, vec_real& outputVec)
{
    vec_real U(Ntau), V(Ntau), F(Ntau), IA2(Ntau), dummy(Ntau);

    initGen.unpackSpectralFields(inputVec, Up, psic, fc);

    // Initialize input for shooting
    initializeInput();

    // Shooting from left and right side
    shooter->shoot(YLeft, YRight, XGrid, iL, iR, iM, mismatchOut);
    
    // Packing state vector to fields
    //initGen.StateVectorToFields(mismatchOut, U, V, F, IA2, dummy, dummy, dummy, XMid);
    
    // Packing resulting fields to out vector
    initGen.packSpectralFields(U, V, F, outputVec);

}

void NewtonSolver::initializeInput()
{
    initGen.computeLeftExpansion(XLeft, fc, psic, YLeft, false);
    initGen.computeRightExpansion(XRight, Up, YRight, false);
}

void NewtonSolver::generateGrid()
{
    if (config.UseLogGrid)
    {
        XGrid.push_back(XLeft);
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
            XGrid.push_back(std::exp(exponent) / (1.0 + std::exp(exponent)));
        }

        XGrid.push_back(XMid);

        for (size_t j=iM+1; j<iR; ++j)
        {
            real_t exponent = ZMid + static_cast<size_t>(j-iM) * dZR;
            XGrid.push_back(std::exp(exponent) / (1.0 + std::exp(exponent)));
        }

        XGrid.push_back(XRight);

    }
}

void NewtonSolver::assembleJacobian(const vec_real& baseInput, const vec_real& baseOutput, mat_real& jacobian)
{

    for (int i = 0; i < Nnewton; ++i)
    {
        vec_real perturbedInput = baseInput;
        perturbedInput[i] += EpsNewton;

        vec_real perturbedOutput(Nnewton);
        shoot(perturbedInput, perturbedOutput);

        for (int j = 0; j < Nnewton; ++j)
        {
            jacobian[j][i] = (perturbedOutput[j] - baseOutput[j]) / EpsNewton;
        }
    }
}

void NewtonSolver::solveLinearSystem(const mat_real& A_in, vec_real& rhs, vec_real& dx)
{
    const int N = static_cast<int>(rhs.size());

    // Convert matrix to row-major double* for LAPACKE
    std::vector<real_t> A_flat(N * N);
    for (int i = 0; i < N; ++i)
    {
        std::memcpy(&A_flat[i * N], A_in[i].data(), N * sizeof(real_t));
    }

    std::vector<lapack_int> ipiv(N);

    lapack_int info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, N, 1, A_flat.data(), N, ipiv.data(), rhs.data(), 1);
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
    return std::sqrt(sum / vc.size());
}

void NewtonSolver::writeFinalOutput()
{
    // Append all to a structured JSON file
    OutputWriter::appendResultToJson("data/results.json", Dim, Delta, fc, psic, Up);

    std::cout << "Final result stored under dimension d = " << Dim
              << " in data/results.json\n";
}