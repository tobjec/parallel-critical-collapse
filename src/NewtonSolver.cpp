#include "NewtonSolver.hpp"
#include "OutputWriter.hpp"

NewtonSolver::NewtonSolver(int numGridPts, real_t dim, real_t delta,
                           real_t epsNewton, real_t tolNewton)
    : ny(numGridPts), Dim(dim), Delta(delta),
      EpsNewton(epsNewton), TolNewton(tolNewton),
      initGen(numGridPts, dim, delta)
{
    n3 = 3 * ny / 4;
    in0.resize(n3);
    out0.resize(n3);
    change.resize(n3);

    shooter = std::make_unique<ShootingSolver>(ny, dim, delta);
}

void NewtonSolver::run(int maxIts)
{
    initializeInput();

    real_t err = 1.0;
    vec_real inOld(n3);
    real_t fac = 1.0;

    for (int its = 0; its < maxIts; ++its)
    {
        std::cout << ">> Newton iteration " << its << std::endl;

        // Save current input in case we need to restore it
        inOld = in0;

        // Evaluate current mismatch
        shoot(in0, out0);
        err = computeL2Norm(out0);
        std::cout << "   mismatch norm: " << err << std::endl;

        if (err < TolNewton)
        {
            std::cout << "✓ Converged!" << std::endl;
            writeFinalOutput();
            return;
        }

        mat_real J(n3, vec_real(n3));
        assembleJacobian(in0, J);

        // Solve J dx = -out0
        vec_real dx(n3);
        vec_real rhs = out0;
        for (auto& r : rhs) r *= -1.0;

        solveLinearSystem(J, rhs, dx);

        // Damping factor based on mismatch reduction
        fac = std::min(1.0, err / 1e-3); // configurable slowerr

        for (int i = 0; i < n3; ++i)
        {
            in0[i] += fac * dx[i];
        }
    }

    std::cerr << "✗ Newton method did not converge in " << maxIts << " iterations." << std::endl;
    std::exit(EXIT_FAILURE);
}

void NewtonSolver::initializeInput()
{
    const real_t XLeft = 1e-3;
    const real_t XRight = 1.0 - 1e-3;

    vec_real Fc(ny, 1.0);    // Initial guess
    vec_real Psic(ny, 0.5);  // Initial guess
    vec_real Up(ny, 0.1);    // Initial guess

    vec_real U(ny), V(ny), F(ny);

    initGen.computeRightExpansion(Up, XRight, U, V, F, false);

    initGen.computeLeftExpansion(Fc, Psic, XLeft, U, V, F, false);

    initGen.packSpectralFields(U, V, F, in0);

    in0[ny / 2 + 2] = Delta;
}

void NewtonSolver::shoot(const vec_real& input, vec_real& output)
{
    vec_real fc(ny, 1.0);
    vec_real psic(ny, 0.5);
    vec_real up(ny, 0.1);

    // Build dummy x-grid
    vec_real gridX(ny + 1);
    real_t dx = 1.0 / ny;
    for (int i = 0; i <= ny; ++i)
        gridX[i] = i * dx;

    const int iLeft = 0;
    const int iRight = ny;
    const int iMid = ny / 2;

    shooter->shoot(fc, psic, up, gridX, iLeft, iRight, iMid, output);
}

void NewtonSolver::assembleJacobian(const vec_real& baseInput, mat_real& jacobian)
{
    vec_real baseOutput(n3);
    shoot(baseInput, baseOutput);

    for (int i = 0; i < n3; ++i)
    {
        vec_real perturbedInput = baseInput;
        perturbedInput[i] += EpsNewton;

        vec_real perturbedOutput(n3);
        shoot(perturbedInput, perturbedOutput);

        for (int j = 0; j < n3; ++j)
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
        std::memcpy(&A_flat[i * N], A_in[i].data(), N * sizeof(real_t));

    std::vector<lapack_int> ipiv(N);

    lapack_int info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, N, 1, A_flat.data(), N, ipiv.data(), rhs.data(), 1);
    if (info != 0)
    {
        std::cerr << "ERROR: LAPACKE_dgesv failed with error code " << info << std::endl;
        std::exit(EXIT_FAILURE);
    }

    dx = rhs; // solution vector
}

real_t NewtonSolver::computeL2Norm(const vec_real& v)
{
    real_t sum = 0.0;
    for (const auto& x : v)
        sum += x * x;
    return std::sqrt(sum / v.size());
}

void NewtonSolver::writeFinalOutput()
{
    vec_real up(ny), psic(ny), fc(ny);
    vec_real inCopy = in0;

    // Remove Delta from spectral input for unpacking
    inCopy[ny / 2 + 2] = 0.0;
    initGen.unpackSpectralFields(inCopy, up, psic, fc);

    real_t finalDelta = in0[ny / 2 + 2];

    // Append all to a structured JSON file
    OutputWriter::appendResultToJson("data/results.json", Dim, finalDelta, fc, psic, up);

    std::cout << "✔ Final result stored under dimension d = " << Dim
              << " in data/results.json\n";
}