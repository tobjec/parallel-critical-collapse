#include "ODEStepper.hpp"

ODEStepper::ODEStepper(int numVars_, real_t dim_, real_t precision_, Scheme method, DerivativeFunc deriv)
    : numVars(numVars_), dim(dim_), precision(precision_), scheme(method), computeDerivatives(std::move(deriv)) {}

void ODEStepper::integrate(const vec_real& yin, vec_real& yout,
                           real_t xin, real_t xout,
                           bool& converged, int& itsReached,
                           int maxIts)
{
    switch (scheme) {
        case Scheme::IRK2:
            stepIRK(yin, yout, xin, xout, 2, itsReached, converged);
            break;
        case Scheme::IRK3:
            stepIRK(yin, yout, xin, xout, 3, itsReached, converged);
            break;
        case Scheme::RKF45:
            real_t dxTry = xout - xin;
            stepRKF45(yin, yout, xin, xout, dxTry, converged);
            itsReached = 1;
            break;
    }
}

// === IRK
void ODEStepper::stepIRK(const vec_real& yin, vec_real& yout,
                         real_t xin, real_t xout,
                         int nu, int& itsReached, bool& converged)
{
    const real_t dx = xout - xin;
    const int N = numVars;

    mat_real a(nu, vec_real(nu, 0.0));
    vec_real b(nu, 0.0), c(nu, 0.0);

    if (nu == 2) {
        a[0][0] = 0.25;
        a[0][1] = 0.25 - 0.5 / std::sqrt(3.0);
        a[1][0] = 0.25 + 0.5 / std::sqrt(3.0);
        a[1][1] = 0.25;
        b[0] = b[1] = 0.5;
        c[0] = 0.5 - 0.5 / std::sqrt(3.0);
        c[1] = 0.5 + 0.5 / std::sqrt(3.0);
    }

    std::vector<vec_real> x(nu, vec_real(N, 0.0));
    std::vector<vec_real> f(nu, vec_real(N, 0.0));
    yout = yin;

    int maxIts = 100;
    converged = false;

    for (int its = 0; its < maxIts; ++its) {
        auto xOld = x;

        for (int i = 0; i < nu; ++i) {
            real_t xi = xin + dx * c[i];
            for (int j = 0; j < N; ++j) {
                x[i][j] = yin[j];
                for (int k = 0; k < nu; ++k)
                    x[i][j] += dx * a[i][k] * f[k][j];
            }
            computeDerivatives(xi, x[i], f[i]);
        }

        real_t norm2 = 0.0;
        for (int i = 0; i < nu; ++i)
            for (int j = 0; j < N; ++j)
                norm2 += std::pow(x[i][j] - xOld[i][j], 2.0);

        norm2 = std::sqrt(norm2 / (nu * N));
        if (norm2 < precision) {
            converged = true;
            break;
        }
    }

    for (int j = 0; j < N; ++j)
        for (int i = 0; i < nu; ++i)
            yout[j] += dx * b[i] * f[i][j];
}

// === RKF45 (adaptive)
void ODEStepper::stepRKF45(const vec_real& yin, vec_real& yout,
                           real_t xin, real_t xout,
                           real_t& dxNext, bool& converged)
{
    static const real_t A[6][5] = {
        {0},
        {1.0 / 4.0},
        {3.0 / 32.0, 9.0 / 32.0},
        {1932.0 / 2197.0, -7200.0 / 2197.0, 7296.0 / 2197.0},
        {439.0 / 216.0, -8.0, 3680.0 / 513.0, -845.0 / 4104.0},
        {-8.0 / 27.0, 2.0, -3544.0 / 2565.0, 1859.0 / 4104.0, -11.0 / 40.0}
    };
    static const real_t B[6] = {25.0 / 216.0, 0, 1408.0 / 2565.0,
                                2197.0 / 4104.0, -1.0 / 5.0, 0};
    static const real_t BHAT[6] = {16.0 / 135.0, 0, 6656.0 / 12825.0,
                                   28561.0 / 56430.0, -9.0 / 50.0, 2.0 / 55.0};
    static const real_t C[6] = {0, 0.25, 0.375, 12.0 / 13.0, 1.0, 0.5};

    const int N = numVars;
    const real_t dx = xout - xin;

    mat_real k(6, vec_real(N));
    vec_real yTemp(N, 0.0);
    yout = yin;

    // compute k_i
    for (int i = 0; i < 6; ++i) {
        vec_real yEval = yin;
        for (int j = 0; j < i; ++j)
            for (int n = 0; n < N; ++n)
                yEval[n] += dx * A[i][j] * k[j][n];
        computeDerivatives(xin + C[i] * dx, yEval, k[i]);
    }

    vec_real y4(N), y5(N);
    for (int i = 0; i < N; ++i) {
        y4[i] = y5[i] = yin[i];
        for (int j = 0; j < 6; ++j) {
            y4[i] += dx * B[j] * k[j][i];
            y5[i] += dx * BHAT[j] * k[j][i];
        }
    }

    // Error estimate
    real_t errNorm = 0.0;
    for (int i = 0; i < N; ++i) {
        real_t scale = std::abs(yin[i]) + std::abs(dx * k[0][i]) + 1e-10;
        real_t diff = std::abs(y4[i] - y5[i]) / scale;
        errNorm = std::max(errNorm, diff);
    }

    errNorm /= precision;
    if (errNorm > 1.0) {
        dxNext = dx * std::max(0.1, 0.9 * std::pow(errNorm, -0.25));
        converged = false;
    } else {
        yout = y4;
        dxNext = (errNorm > 1e-4)
                 ? dx * std::min(5.0, 0.9 * std::pow(errNorm, -0.2))
                 : dx * 5.0;
        converged = true;
    }
}
