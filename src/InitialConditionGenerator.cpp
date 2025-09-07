#include "InitialConditionGenerator.hpp"

InitialConditionGenerator::InitialConditionGenerator(size_t Ntau_, real_t Dim_, real_t Delta_)
    : Ntau(Ntau_), Nnewton(3*Ntau_/4), Dim(Dim_), Delta(Delta_), fft(Ntau_, Delta_) {}

void InitialConditionGenerator::computeLeftExpansion(
    real_t XLeft, const vec_real& fc, const vec_real& psic,
            vec_complex& Y, real_t DeltaIn, bool PrintDiagnostics)
{

    Delta = DeltaIn;

    // Step 1: Compute f0 and its derivatives
    const vec_real& f0 = fc;
    const vec_real& psi0 = psic;

    vec_complex F0Hat, DfHat, D2fHat, D3fHat;
    vec_real df0dtau, d2f0dtau2, d3f0dtau3;

    fft.forwardFFT(f0, F0Hat);
    fft.differentiate(F0Hat, DfHat, Delta);
    fft.backwardFFT(DfHat, df0dtau);

    fft.forwardFFT(df0dtau, D2fHat);
    fft.differentiate(D2fHat, D2fHat, Delta);
    fft.backwardFFT(D2fHat, d2f0dtau2);

    fft.forwardFFT(d2f0dtau2, D3fHat);
    fft.differentiate(D3fHat, D3fHat, Delta);
    fft.backwardFFT(D3fHat, d3f0dtau3);

    // Step 2: Solve inhomogeneous ODE for u1
    vec_real Coeff1(Ntau, 1.0), Coeff2(Ntau), u1;
    #ifdef USE_HYBRID
    #pragma omp parallel for schedule(static)
    #endif
    for (size_t j=0; j<Ntau; ++j)
    {
        Coeff2[j] = -(Dim - 1.0) * psi0[j] * f0[j];
    }
    fft.solveInhom(Coeff1, Coeff2, u1, Delta);

    // Step 3: Compute derivatives of u1
    vec_complex U1Hat, dU1Hat, d2U1Hat, d3U1Hat, d4U1Hat;
    vec_real du1dtau, d2u1dtau2, d3u1dtau3, d4u1dtau4;

    fft.forwardFFT(u1, U1Hat);
    fft.differentiate(U1Hat, dU1Hat, Delta);
    fft.backwardFFT(dU1Hat, du1dtau);

    fft.forwardFFT(du1dtau, d2U1Hat);
    fft.differentiate(d2U1Hat, d2U1Hat, Delta);
    fft.backwardFFT(d2U1Hat, d2u1dtau2);

    fft.forwardFFT(d2u1dtau2, d3U1Hat);
    fft.differentiate(d3U1Hat, d3U1Hat, Delta);
    fft.backwardFFT(d3U1Hat, d3u1dtau3);

    fft.forwardFFT(d3u1dtau3, d4U1Hat);
    fft.differentiate(d4U1Hat, d4U1Hat, Delta);
    fft.backwardFFT(d4U1Hat, d4u1dtau4);

    // Step 4: Taylor expansions
    vec_real f2(Ntau), f4(Ntau),
             u2(Ntau), u3(Ntau), u4(Ntau), u5(Ntau),
             v1(Ntau), v2(Ntau), v3(Ntau), v4(Ntau), v5(Ntau),
             U(Ntau), V(Ntau), F(Ntau);

    #ifdef USE_HYBRID
    #pragma omp parallel for schedule(static)
    #endif
    for (size_t j=0; j<Ntau; ++j)
    {
        const real_t f = f0[j], psi = psi0[j], d = Dim;
        const real_t u1j = u1[j], du1 = du1dtau[j], d2u1 = d2u1dtau2[j], d3u1 = d3u1dtau3[j], d4u1 = d4u1dtau4[j];
        const real_t df = df0dtau[j], d2f = d2f0dtau2[j], d3f = d3f0dtau3[j];

        // Order 1 and Order 2
        v1[j] = u1j;

        f2[j] = ((d - 3.0) * std::pow(d - 2.0, 3) * f * std::pow(u1j, 2)) / (8.0 * (d - 1.0));
        u2[j] = -psi;
        v2[j] = psi;

        // Order 3
        const real_t common3 = ((d - 3.0) * std::pow(d - 2.0, 3) * std::pow(f, 3) * std::pow(u1j, 3)
                            + 4.0 * df * (u1j + du1)
                            - 4.0 * f * (2.0 * u1j + 3.0 * du1 + d2u1));
        u3[j] = common3 / (8.0 * (1.0 - d) * std::pow(f, 3));
        v3[j] = u3[j];

        // Order 4
        f4[j] = -((d - 3.0) * std::pow(d - 2.0, 3) *
                ((std::pow(d - 2.0, 3) * (5.0 - 6.0 * d + d * d) * std::pow(f, 3) * std::pow(u1j, 4))
                + 8.0 * (d - 1.0) * df * u1j * (du1 + u1j)
                - 8.0 * f * (std::pow(du1, 2)
                + ((d - 1.0) * d2u1 + (-1.0 + 3.0 * d) * du1) * u1j
                + (-1.0 + 2.0 * d) * std::pow(u1j, 2))))
                / (128.0 * std::pow(d - 1.0, 2) * (1.0 + d) * std::pow(f, 2));

        u4[j] = (
            -192.0 * (d - 1.0) * std::pow(df, 2) * du1 * std::pow(f, 2)
            + 192.0 * (d - 1.0) * d2u1 * df * std::pow(f, 3)
            + 64.0 * (d - 1.0) * d2f * du1 * std::pow(f, 3)
            + 640.0 * (d - 1.0) * df * du1 * std::pow(f, 3)
            - 64.0 * (d - 1.0) * (6.0 * d2u1 + d3u1) * std::pow(f, 4)
            - 704.0 * (d - 1.0) * du1 * std::pow(f, 4)
            - 192.0 * (d - 1.0) * std::pow(df, 2) * std::pow(f, 2) * u1j
            + 64.0 * (d - 1.0) * d2f * std::pow(f, 3) * u1j
            + 448.0 * (d - 1.0) * df * std::pow(f, 3) * u1j
            - 384.0 * (d - 1.0) * std::pow(f, 4) * u1j
            + 32.0 * std::pow(d - 2.0, 3) * (3.0 - 7.0 * d + 2.0 * d * d)
            * std::pow(f, 6) * std::pow(u1j, 2) * (du1 + u1j)
        ) / (128.0 * std::pow(d - 1.0, 2) * (1.0 + d) * std::pow(f, 7));

        v4[j] = -u4[j];

        // Order 5
        u5[j] = (
            -64.0 * (d - 1.0) * d2f * d2u1 * std::pow(f, 2)
            - 16.0 * (d - 1.0) * (15.0 * d2f + d3f) * du1 * std::pow(f, 2)
            - 32.0 * (d - 1.0) * df * (20.0 * d2u1 + 3.0 * d3u1 + 40.0 * du1) * std::pow(f, 2)
            + 560.0 * (d - 1.0) * d2u1 * std::pow(f, 3)
            + 160.0 * (d - 1.0) * d3u1 * std::pow(f, 3)
            + 16.0 * (d - 1.0) * d4u1 * std::pow(f, 3)
            + 800.0 * (d - 1.0) * du1 * std::pow(f, 3)
            - 16.0 * (d - 1.0) * (11.0 * d2f + d3f) * std::pow(f, 2) * u1j
            - 736.0 * (d - 1.0) * df * std::pow(f, 2) * u1j
            + 384.0 * (d - 1.0) * std::pow(f, 3) * u1j
            - 8.0 * std::pow(d - 2.0, 3) * (3.0 - 13.0 * d + 4.0 * d * d)
            * std::pow(du1, 2) * std::pow(f, 5) * u1j
            + 8.0 * std::pow(d - 2.0, 3) * (3.0 - 13.0 * d + 4.0 * d * d)
            * df * du1 * std::pow(f, 4) * std::pow(u1j, 2)
            - 8.0 * std::pow(d - 2.0, 3) * (3.0 - 13.0 * d + 4.0 * d * d)
            * (d2u1 + 5.0 * du1) * std::pow(f, 5) * std::pow(u1j, 2)
            + 8.0 * std::pow(d - 2.0, 3) * (3.0 - 13.0 * d + 4.0 * d * d)
            * df * std::pow(f, 4) * std::pow(u1j, 3)
            - 24.0 * std::pow(d - 2.0, 3) * (3.0 - 13.0 * d + 4.0 * d * d)
            * std::pow(f, 5) * std::pow(u1j, 3)
            + std::pow(d - 2.0, 6) * (3.0 + 29.0 * d - 19.0 * d * d + 3.0 * d * d * d)
            * std::pow(f, 7) * std::pow(u1j, 5)
            - 240.0 * (d - 1.0) * std::pow(df, 3) * (du1 + u1j)
            + 80.0 * (d - 1.0) * df * f * (
                2.0 * d2f * du1
                + df * (3.0 * d2u1 + 11.0 * du1)
                + 2.0 * (d2f + 4.0 * df) * u1j
            )
        ) / (128.0 * std::pow(d - 1.0, 2) * (1.0 + d) * std::pow(f, 7));

        v5[j] = u5[j];

        // Compose final Taylor expansion
        U[j] = XLeft * u1j + std::pow(XLeft, 2) * u2[j] + std::pow(XLeft, 3) * u3[j] + std::pow(XLeft, 4) * u4[j] + std::pow(XLeft, 5) * u5[j];
        V[j] = XLeft * v1[j] + std::pow(XLeft, 2) * v2[j] + std::pow(XLeft, 3) * v3[j] + std::pow(XLeft, 4) * v4[j] + std::pow(XLeft, 5) * v5[j];
        F[j] = f + std::pow(XLeft, 2) * f2[j] + std::pow(XLeft, 4) * f4[j];
    }

    FieldsToStateVector(U,V,F,Y);
    
    Y[2] = complex_t(Delta, Y[2].imag());

    if (PrintDiagnostics)
    {
        std::cout << "[INFO] Left-side 5th-order Taylor expansion at x = " << XLeft << " completed.\n";

        real_t f20max = 0.0, f42max = 0.0;
        real_t u21max = 0.0, u32max = 0.0, u43max = 0.0, u54max = 0.0;
        real_t f0norm = 0.0, f2norm = 0.0, f4norm = 0.0;
        real_t u1norm = 0.0, u2norm = 0.0, u3norm = 0.0, u4norm = 0.0, u5norm = 0.0;

        for (size_t j = 0; j < Ntau; ++j)
        {
            f20max = std::max(f20max, std::abs(std::pow(XLeft, 2) * f2[j] / f0[j]));
            f42max = std::max(f42max, std::abs(std::pow(XLeft, 2) * f4[j] / f2[j]));
            u21max = std::max(u21max, std::abs(XLeft * u2[j] / u1[j]));
            u32max = std::max(u32max, std::abs(XLeft * u3[j] / u2[j]));
            u43max = std::max(u43max, std::abs(XLeft * u4[j] / u3[j]));
            u54max = std::max(u54max, std::abs(XLeft * u5[j] / u4[j]));

            f0norm += f0[j] * f0[j];
            f2norm += f2[j] * f2[j];
            f4norm += f4[j] * f4[j];

            u1norm += u1[j] * u1[j];
            u2norm += u2[j] * u2[j];
            u3norm += u3[j] * u3[j];
            u4norm += u4[j] * u4[j];
            u5norm += u5[j] * u5[j];
        }

        f0norm = std::sqrt(f0norm);
        f2norm = std::sqrt(f2norm);
        f4norm = std::sqrt(f4norm);
        u1norm = std::sqrt(u1norm);
        u2norm = std::sqrt(u2norm);
        u3norm = std::sqrt(u3norm);
        u4norm = std::sqrt(u4norm);
        u5norm = std::sqrt(u5norm);

        std::cout << "***************************************\n";
        std::cout << " INFO: Taylor expansion at xleft = " << XLeft << "\n";
        std::cout << "***************************************\n";

        std::cout << "max(x^2 f2 / f0) = " << f20max << "\n";
        std::cout << "max(x^2 f4 / f2) = " << f42max << "\n\n";
        std::cout << "max(x * u2 / u1) = " << u21max << "\n";
        std::cout << "max(x * u3 / u2) = " << u32max << "\n";
        std::cout << "max(x * u4 / u3) = " << u43max << "\n";
        std::cout << "max(x * u5 / u4) = " << u54max << "\n\n";

        std::cout << "(x^2 f2norm) / f0norm = " << (std::pow(XLeft, 2) * f2norm / f0norm) << "\n";
        std::cout << "(x^2 f4norm) / f2norm = " << (std::pow(XLeft, 2) * f4norm / f2norm) << "\n\n";

        std::cout << "(x * u2norm) / u1norm = " << (XLeft * u2norm / u1norm) << "\n";
        std::cout << "(x * u3norm) / u2norm = " << (XLeft * u3norm / u2norm) << "\n";
        std::cout << "(x * u4norm) / u3norm = " << (XLeft * u4norm / u3norm) << "\n";
        std::cout << "(x * u5norm) / u4norm = " << (XLeft * u5norm / u4norm) << "\n";
    }

}

void InitialConditionGenerator::computeRightExpansion(
    real_t XRight, const vec_real& Up, vec_complex& Y, real_t DeltaIn, bool PrintDiagnostics)
{
    Delta = DeltaIn;
    const real_t Xp = 1.0;
    const real_t d = Dim;
    const real_t dx = XRight - Xp;

    const vec_real& u0 = Up;
    vec_real du0, d2u0;

    vec_complex U0Hat, dU0Hat, d2U0Hat;

    fft.forwardFFT(u0, U0Hat);
    fft.differentiate(U0Hat, dU0Hat, Delta);
    fft.backwardFFT(dU0Hat, du0);

    fft.differentiate(dU0Hat, d2U0Hat, Delta);
    fft.backwardFFT(d2U0Hat, d2u0);

    vec_real coeff1(Ntau), coeff2(Ntau), ia20, f0(Ntau, 1.0);
    vec_real v0;
    vec_real f1(Ntau), u1(Ntau), ia21(Ntau), v1(Ntau);
    vec_real f2(Ntau), ia22(Ntau), u2(Ntau), du2(Ntau), v2(Ntau);
    vec_complex U2Hat, dU2Hat;
    vec_real f3(Ntau), ia23(Ntau), u3(Ntau), v3(Ntau);
    vec_real F(Ntau), U(Ntau), V(Ntau);

    #ifdef USE_HYBRID
    #pragma omp parallel
    {

        #pragma omp for schedule(static)
        for (size_t j=0; j<Ntau; ++j)
        {
            coeff1[j] = 3.0 - d - std::pow(d - 2.0, 3) * u0[j] * u0[j] / 4.0;
            coeff2[j] = d - 3.0;
        }

        #pragma omp single
        {
            fft.solveInhom(coeff1, coeff2, ia20, Delta);
        }   

        
        #pragma omp for schedule(static)
        for (size_t j=0; j<Ntau; ++j)
        {
            coeff1[j] = (6.0 - 2.0 * d + (d - 2.0) * ia20[j]) / (2.0 * ia20[j]);
            coeff2[j] = (d - 2.0) * u0[j] / 2.0;
        }

        #pragma omp single
        {
            fft.solveInhom(coeff1, coeff2, v0, Delta);
        }

        #pragma omp for schedule(static)
        for (size_t j=0; j<Ntau; ++j)
        {
            f1[j] = 3.0 - d + (d - 3.0) / ia20[j];

            ia21[j] = - ia20[j] * (8.0 * (d - 3.0)
                    + std::pow(d - 2.0, 3) * (u0[j] * u0[j] + v0[j] * v0[j]))
                    / 8.0 + d - 3.0;

            u1[j] = ((d - 2.0 - 2.0 * (d - 3.0) / ia20[j]) * u0[j]
                    + (d - 2.0) * v0[j]) / 4.0 - du0[j] / 2.0;
        }

        #pragma omp for schedule(static)
        for (size_t j=0; j<Ntau; ++j)
        {
            coeff1[j] = (6.0 - 2.0 * d + d * ia20[j] - 2.0 * f1[j] * ia20[j])
                        / (2.0 * ia20[j]);

            coeff2[j] = (
                2.0 * (3.0 - d) * (f1[j] - 1.0) * ia20[j] * v0[j]
            + 2.0 * (d - 3.0) * ia21[j] * v0[j]
            + (d - 2.0) * std::pow(ia20[j], 2) *
                ((f1[j] - 1.0) * u0[j] + u1[j] + (f1[j] - 1.0) * v0[j])
            ) / (2.0 * std::pow(ia20[j], 2));
        }

        #pragma omp single
        {
            fft.solveInhom(coeff1, coeff2, v1, Delta);
        }

        #pragma omp for schedule(static)
        for (size_t j=0; j<Ntau; ++j)
        {
            f2[j] = (d - 3.0) * ((f1[j] - 1.0) * (1.0 - ia20[j]) * ia20[j] - ia21[j]) /
                    (2.0 * std::pow(ia20[j], 2));

            ia22[j] = (3.0 - d
                    - (ia21[j] - ia20[j]) * (8.0 * (d - 3.0)
                    + std::pow(d - 2.0, 3) * (std::pow(u0[j], 2) + std::pow(v0[j], 2))) / 8.0
                    - std::pow(d - 2.0, 3) * ia20[j] * (u0[j] * u1[j] + v0[j] * v1[j]) / 4.0) / 2.0;

            const real_t A = 4.0 * (3.0 - d) * (d - 6.0 + f1[j]) * ia20[j];
            const real_t B = (d - 2.0) * (d - 8.0 + 2.0 * f1[j]) * std::pow(ia20[j], 2);
            const real_t C = 4.0 * (d - 3.0) * (d - 3.0 + 2.0 * ia21[j]);
            const real_t D = -(d - 3.0) * std::pow(d - 2.0, 3) * ia20[j] * std::pow(u0[j], 3);
            const real_t E = (4.0 * (6.0 - 2.0 * d + (d - 2.0) * ia20[j]) * u1[j]
                            + (d - 2.0) * v0[j] * (6.0 - 2.0 * d + (d - 8.0 + 2.0 * f1[j]) * ia20[j])
                            - 8.0 * ia20[j] * v1[j] + 4.0 * d * ia20[j] * v1[j]
                            - 12.0 * du0[j] + 4.0 * d * du0[j]
                            + 8.0 * ia20[j] * du0[j] - 2.0 * d * ia20[j] * du0[j]
                            + 4.0 * f1[j] * ia20[j] * du0[j]
                            + 4.0 * ia20[j] * d2u0[j]) * ia20[j];

            u2[j] = (A + B + C) * u0[j] + D + E;
            u2[j] /= 32.0 * std::pow(ia20[j], 2);
        }

        #pragma omp single
        {
            fft.forwardFFT(u2, U2Hat);
            fft.differentiate(U2Hat, dU2Hat, Delta);
            fft.backwardFFT(dU2Hat, du2);
        }

        #pragma omp for schedule(static)
        for (size_t j=0; j<Ntau; ++j)
        {
            coeff1[j] = 1.0 + d / 2.0 - 2.0 * f1[j] + (3.0 - d) / ia20[j];

            coeff2[j] =
                (2.0 * (3.0 - d) * std::pow(ia21[j], 2) * v0[j]
                + 2.0 * (d - 3.0) * std::pow(ia20[j], 2) *
                    ((f1[j] - 1.0 - f2[j]) * v0[j] - (f1[j] - 1.0) * v1[j])
                + ((d - 2.0) *
                        ((1.0 - f1[j] + f2[j]) * u0[j]
                        + (f1[j] - 1.0) * u1[j]
                        + u2[j]
                        + (1.0 - f1[j] + f2[j]) * v0[j])
                    + ((d - 2.0) * (f1[j] - 1.0) - 2.0 * f2[j]) * v1[j])
                    * std::pow(ia20[j], 3)
                + 2.0 * (d - 3.0) * ia20[j] *
                    (ia22[j] * v0[j]
                    + ia21[j] * ((f1[j] - 1.0) * v0[j] + v1[j])))
                / (2.0 * std::pow(ia20[j], 3));
        }

        #pragma omp single
        {
            fft.solveInhom(coeff1, coeff2, v2, Delta);
        }
        
        #pragma omp for schedule(static)
        for (size_t j=0; j<Ntau; ++j)
        {
            f3[j] = ((-3.0 + d) * ((1.0 - f1[j] + f2[j]) * std::pow(ia20[j], 2)
                + (-1.0 + f1[j] - f2[j]) * std::pow(ia20[j], 3)
                + std::pow(ia21[j], 2)
                - ia20[j] * ((-1.0 + f1[j]) * ia21[j] + ia22[j])))
                / (3.0 * std::pow(ia20[j], 3));

            ia23[j] = (-3.0 + d 
                + ( -((ia20[j] - ia21[j] + ia22[j]) * (8.0 * (-3.0 + d) 
                    + std::pow(d - 2.0, 3) * (u0[j] * u0[j] + v0[j] * v0[j])))
                    + 2.0 * std::pow(d - 2.0, 3) * (ia20[j] - ia21[j]) 
                    * (u0[j] * u1[j] + v0[j] * v1[j]) 
                    - std::pow(d - 2.0, 3) * ia20[j] 
                    * (std::pow(u1[j], 2) + 2.0 * u0[j] * u2[j] 
                    + std::pow(v1[j], 2) + 2.0 * v0[j] * v2[j])) / 8.0)
                / 3.0;

            u3[j] = -(16.0 * (-3.0 + d) * std::pow(ia21[j], 2) * u0[j] 
                + 4.0 * (-3.0 + d) * ia20[j]
                    * ((-3.0 + d + f1[j] * (-3.0 + d - 2.0 * ia21[j]) 
                    + 6.0 * ia21[j] - 4.0 * ia22[j]) * u0[j] 
                    - 4.0 * ia21[j] * u1[j])
                - (d - 3.0) * std::pow(ia20[j], 2)
                    * (4.0 * (-10.0 + d + f1[j] * (-1.0 + d + f1[j]) - 2.0 * f2[j]) * u0[j] 
                    + std::pow(d - 2.0, 3) * (1.0 + f1[j]) * std::pow(u0[j], 3) 
                    + 2.0 * (-4.0 * (-3.0 + f1[j]) * u1[j] - 8.0 * u2[j] 
                    + (1.0 + f1[j]) * (-2.0 * du0[j] + (d - 2.0) * v0[j])))
                + std::pow(ia20[j], 3)
                    * (8.0 * du0[j] - 2.0 * d * du0[j] 
                    + 12.0 * du0[j] * f1[j] - 2.0 * d * du0[j] * f1[j] 
                    + 4.0 * du0[j] * std::pow(f1[j], 2) 
                    + 4.0 * d2u0[j] * (1.0 + f1[j]) 
                    - 8.0 * du0[j] * f2[j] 
                    + (d - 2.0) * (-16.0 + d + f1[j] * (2.0 + d + 2.0 * f1[j]) - 4.0 * f2[j]) * u0[j] 
                    - 4.0 * (d - 2.0) * (-3.0 + f1[j]) * u1[j] 
                    + 16.0 * u2[j] - 8.0 * d * u2[j] 
                    + 32.0 * v0[j] - 18.0 * d * v0[j] + std::pow(d, 2) * v0[j] 
                    - 4.0 * f1[j] * v0[j] + std::pow(d, 2) * f1[j] * v0[j] 
                    - 4.0 * std::pow(f1[j], 2) * v0[j] 
                    + 2.0 * d * std::pow(f1[j], 2) * v0[j] 
                    + 8.0 * f2[j] * v0[j] - 4.0 * d * f2[j] * v0[j] 
                    - 24.0 * v1[j] + 12.0 * d * v1[j] 
                    + 8.0 * f1[j] * v1[j] - 4.0 * d * f1[j] * v1[j] 
                    + 16.0 * v2[j] - 8.0 * d * v2[j] 
                    + 16.0 * du2[j]))
                / (96.0 * std::pow(ia20[j], 3));
            
            coeff1[j] = 2.0 + d / 2.0 - 3.0 * f1[j] + (3.0 - d) / ia20[j];

            coeff2[j] = ((d - 3.0) * std::pow(ia21[j], 3) * v0[j]) / std::pow(ia20[j], 4)
                    - ((d - 3.0) * ia21[j]
                    * (2.0 * ia22[j] * v0[j]
                    + ia21[j] * ((-1.0 + f1[j]) * v0[j] + v1[j])))
                    / std::pow(ia20[j], 3)
                    - ((d - 3.0) * ((-1.0 + f1[j] - f2[j] + f3[j]) * v0[j]
                    + (1.0 - f1[j] + f2[j]) * v1[j]
                    + (-1.0 + f1[j]) * v2[j])) / ia20[j]
                    + ((d - 2.0) * (-1.0 + f1[j] - f2[j] + f3[j]) * u0[j]
                    - (d - 2.0) * (-1.0 + f1[j] - f2[j]) * u1[j]
                    + 2.0 * u2[j] - d * u2[j]
                    - 2.0 * f1[j] * u2[j] + d * f1[j] * u2[j]
                    - 2.0 * u3[j] + d * u3[j]
                    + 2.0 * v0[j] - d * v0[j]
                    - 2.0 * f1[j] * v0[j] + d * f1[j] * v0[j]
                    + 2.0 * f2[j] * v0[j] - d * f2[j] * v0[j]
                    - 2.0 * f3[j] * v0[j] + d * f3[j] * v0[j]
                    - 2.0 * v1[j] + d * v1[j]
                    + 2.0 * f1[j] * v1[j] - d * f1[j] * v1[j]
                    - 2.0 * f2[j] * v1[j] + d * f2[j] * v1[j]
                    - 2.0 * f3[j] * v1[j]
                    + ((d - 2.0) * (-1.0 + f1[j]) - 4.0 * f2[j]) * v2[j]) / 2.0
                    + ((d - 3.0) * (ia23[j] * v0[j]
                    + ia22[j] * ((-1.0 + f1[j]) * v0[j] + v1[j])
                    + ia21[j] * ((1.0 - f1[j] + f2[j]) * v0[j]
                    + (-1.0 + f1[j]) * v1[j] + v2[j]))) / std::pow(ia20[j], 2);
        }

        #pragma omp single
        {
            fft.solveInhom(coeff1, coeff2, v3, Delta);
        }

        #pragma omp for schedule(static)
        for (size_t j=0; j<Ntau; ++j)
        {
            F[j] = f0[j] + dx * f1[j] + dx * dx * f2[j] + dx * dx * dx * f3[j];
            U[j] = u0[j] + dx * u1[j] + dx * dx * u2[j] + dx * dx * dx * u3[j];
            V[j] = v0[j] + dx * v1[j] + dx * dx * v2[j] + dx * dx * dx * v3[j];
        }
    }

    #else

    for (size_t j=0; j<Ntau; ++j)
    {
        coeff1[j] = 3.0 - d - std::pow(d - 2.0, 3) * u0[j] * u0[j] / 4.0;
        coeff2[j] = d - 3.0;
    }
    fft.solveInhom(coeff1, coeff2, ia20, Delta);

    // Step 3: Solve for v0
    for (size_t j=0; j<Ntau; ++j)
    {
        coeff1[j] = (6.0 - 2.0 * d + (d - 2.0) * ia20[j]) / (2.0 * ia20[j]);
        coeff2[j] = (d - 2.0) * u0[j] / 2.0;
    }
    fft.solveInhom(coeff1, coeff2, v0, Delta);

    // f0 is already set to 1.0
    for (size_t j=0; j<Ntau; ++j)
    {
        f1[j] = 3.0 - d + (d - 3.0) / ia20[j];

        ia21[j] = - ia20[j] * (8.0 * (d - 3.0)
                + std::pow(d - 2.0, 3) * (u0[j] * u0[j] + v0[j] * v0[j]))
                / 8.0 + d - 3.0;

        u1[j] = ((d - 2.0 - 2.0 * (d - 3.0) / ia20[j]) * u0[j]
                + (d - 2.0) * v0[j]) / 4.0 - du0[j] / 2.0;
    }

    // Solve linear inhomogeneous ODE for v1
    for (size_t j=0; j<Ntau; ++j)
    {
        coeff1[j] = (6.0 - 2.0 * d + d * ia20[j] - 2.0 * f1[j] * ia20[j])
                    / (2.0 * ia20[j]);

        coeff2[j] = (
            2.0 * (3.0 - d) * (f1[j] - 1.0) * ia20[j] * v0[j]
        + 2.0 * (d - 3.0) * ia21[j] * v0[j]
        + (d - 2.0) * std::pow(ia20[j], 2) *
            ((f1[j] - 1.0) * u0[j] + u1[j] + (f1[j] - 1.0) * v0[j])
        ) / (2.0 * std::pow(ia20[j], 2));
    }

    fft.solveInhom(coeff1, coeff2, v1, Delta);

    // Order 2
    for (size_t j=0; j<Ntau; ++j)
    {
        f2[j] = (d - 3.0) * ((f1[j] - 1.0) * (1.0 - ia20[j]) * ia20[j] - ia21[j]) /
                (2.0 * std::pow(ia20[j], 2));
    }

    // Step 2: Compute ia22
    for (size_t j=0; j<Ntau; ++j)
    {
        ia22[j] = (3.0 - d
                - (ia21[j] - ia20[j]) * (8.0 * (d - 3.0)
                + std::pow(d - 2.0, 3) * (std::pow(u0[j], 2) + std::pow(v0[j], 2))) / 8.0
                - std::pow(d - 2.0, 3) * ia20[j] * (u0[j] * u1[j] + v0[j] * v1[j]) / 4.0) / 2.0;
    }

    // Step 3: Compute u2
    for (size_t j=0; j<Ntau; ++j)
    {
        const real_t A = 4.0 * (3.0 - d) * (d - 6.0 + f1[j]) * ia20[j];
        const real_t B = (d - 2.0) * (d - 8.0 + 2.0 * f1[j]) * std::pow(ia20[j], 2);
        const real_t C = 4.0 * (d - 3.0) * (d - 3.0 + 2.0 * ia21[j]);
        const real_t D = -(d - 3.0) * std::pow(d - 2.0, 3) * ia20[j] * std::pow(u0[j], 3);
        const real_t E = (4.0 * (6.0 - 2.0 * d + (d - 2.0) * ia20[j]) * u1[j]
                        + (d - 2.0) * v0[j] * (6.0 - 2.0 * d + (d - 8.0 + 2.0 * f1[j]) * ia20[j])
                        - 8.0 * ia20[j] * v1[j] + 4.0 * d * ia20[j] * v1[j]
                        - 12.0 * du0[j] + 4.0 * d * du0[j]
                        + 8.0 * ia20[j] * du0[j] - 2.0 * d * ia20[j] * du0[j]
                        + 4.0 * f1[j] * ia20[j] * du0[j]
                        + 4.0 * ia20[j] * d2u0[j]) * ia20[j];

        u2[j] = (A + B + C) * u0[j] + D + E;
        u2[j] /= 32.0 * std::pow(ia20[j], 2);
    }

    // Step 4: Differentiate u2
    
    fft.forwardFFT(u2, U2Hat);
    fft.differentiate(U2Hat, dU2Hat, Delta);
    fft.backwardFFT(dU2Hat, du2);

    // Step 5: Compute v2 via linear inhomogeneous solve
    for (size_t j=0; j<Ntau; ++j)
    {
        coeff1[j] = 1.0 + d / 2.0 - 2.0 * f1[j] + (3.0 - d) / ia20[j];

        coeff2[j] =
            (2.0 * (3.0 - d) * std::pow(ia21[j], 2) * v0[j]
            + 2.0 * (d - 3.0) * std::pow(ia20[j], 2) *
                ((f1[j] - 1.0 - f2[j]) * v0[j] - (f1[j] - 1.0) * v1[j])
            + ((d - 2.0) *
                    ((1.0 - f1[j] + f2[j]) * u0[j]
                    + (f1[j] - 1.0) * u1[j]
                    + u2[j]
                    + (1.0 - f1[j] + f2[j]) * v0[j])
                + ((d - 2.0) * (f1[j] - 1.0) - 2.0 * f2[j]) * v1[j])
                * std::pow(ia20[j], 3)
            + 2.0 * (d - 3.0) * ia20[j] *
                (ia22[j] * v0[j]
                + ia21[j] * ((f1[j] - 1.0) * v0[j] + v1[j])))
            / (2.0 * std::pow(ia20[j], 3));
    }

    // Solve for v2
    fft.solveInhom(coeff1, coeff2, v2, Delta);

    // Order 3
    for (size_t j=0; j<Ntau; ++j)
    {
        f3[j] = ((-3.0 + d) * ((1.0 - f1[j] + f2[j]) * std::pow(ia20[j], 2)
            + (-1.0 + f1[j] - f2[j]) * std::pow(ia20[j], 3)
            + std::pow(ia21[j], 2)
            - ia20[j] * ((-1.0 + f1[j]) * ia21[j] + ia22[j])))
            / (3.0 * std::pow(ia20[j], 3));

        ia23[j] = (-3.0 + d 
            + ( -((ia20[j] - ia21[j] + ia22[j]) * (8.0 * (-3.0 + d) 
                + std::pow(d - 2.0, 3) * (u0[j] * u0[j] + v0[j] * v0[j])))
                + 2.0 * std::pow(d - 2.0, 3) * (ia20[j] - ia21[j]) 
                * (u0[j] * u1[j] + v0[j] * v1[j]) 
                - std::pow(d - 2.0, 3) * ia20[j] 
                * (std::pow(u1[j], 2) + 2.0 * u0[j] * u2[j] 
                + std::pow(v1[j], 2) + 2.0 * v0[j] * v2[j])) / 8.0)
            / 3.0;

        u3[j] = -(16.0 * (-3.0 + d) * std::pow(ia21[j], 2) * u0[j] 
            + 4.0 * (-3.0 + d) * ia20[j]
                * ((-3.0 + d + f1[j] * (-3.0 + d - 2.0 * ia21[j]) 
                + 6.0 * ia21[j] - 4.0 * ia22[j]) * u0[j] 
                - 4.0 * ia21[j] * u1[j])
            - (d - 3.0) * std::pow(ia20[j], 2)
                * (4.0 * (-10.0 + d + f1[j] * (-1.0 + d + f1[j]) - 2.0 * f2[j]) * u0[j] 
                + std::pow(d - 2.0, 3) * (1.0 + f1[j]) * std::pow(u0[j], 3) 
                + 2.0 * (-4.0 * (-3.0 + f1[j]) * u1[j] - 8.0 * u2[j] 
                + (1.0 + f1[j]) * (-2.0 * du0[j] + (d - 2.0) * v0[j])))
            + std::pow(ia20[j], 3)
                * (8.0 * du0[j] - 2.0 * d * du0[j] 
                + 12.0 * du0[j] * f1[j] - 2.0 * d * du0[j] * f1[j] 
                + 4.0 * du0[j] * std::pow(f1[j], 2) 
                + 4.0 * d2u0[j] * (1.0 + f1[j]) 
                - 8.0 * du0[j] * f2[j] 
                + (d - 2.0) * (-16.0 + d + f1[j] * (2.0 + d + 2.0 * f1[j]) - 4.0 * f2[j]) * u0[j] 
                - 4.0 * (d - 2.0) * (-3.0 + f1[j]) * u1[j] 
                + 16.0 * u2[j] - 8.0 * d * u2[j] 
                + 32.0 * v0[j] - 18.0 * d * v0[j] + std::pow(d, 2) * v0[j] 
                - 4.0 * f1[j] * v0[j] + std::pow(d, 2) * f1[j] * v0[j] 
                - 4.0 * std::pow(f1[j], 2) * v0[j] 
                + 2.0 * d * std::pow(f1[j], 2) * v0[j] 
                + 8.0 * f2[j] * v0[j] - 4.0 * d * f2[j] * v0[j] 
                - 24.0 * v1[j] + 12.0 * d * v1[j] 
                + 8.0 * f1[j] * v1[j] - 4.0 * d * f1[j] * v1[j] 
                + 16.0 * v2[j] - 8.0 * d * v2[j] 
                + 16.0 * du2[j]))
            / (96.0 * std::pow(ia20[j], 3));
    }

    // v3 via solving inhomogeneous equation
    for (size_t j=0; j<Ntau; ++j)
    {
        coeff1[j] = 2.0 + d / 2.0 - 3.0 * f1[j] + (3.0 - d) / ia20[j];

        coeff2[j] = ((d - 3.0) * std::pow(ia21[j], 3) * v0[j]) / std::pow(ia20[j], 4)
                - ((d - 3.0) * ia21[j]
                * (2.0 * ia22[j] * v0[j]
                + ia21[j] * ((-1.0 + f1[j]) * v0[j] + v1[j])))
                / std::pow(ia20[j], 3)
                - ((d - 3.0) * ((-1.0 + f1[j] - f2[j] + f3[j]) * v0[j]
                + (1.0 - f1[j] + f2[j]) * v1[j]
                + (-1.0 + f1[j]) * v2[j])) / ia20[j]
                + ((d - 2.0) * (-1.0 + f1[j] - f2[j] + f3[j]) * u0[j]
                - (d - 2.0) * (-1.0 + f1[j] - f2[j]) * u1[j]
                + 2.0 * u2[j] - d * u2[j]
                - 2.0 * f1[j] * u2[j] + d * f1[j] * u2[j]
                - 2.0 * u3[j] + d * u3[j]
                + 2.0 * v0[j] - d * v0[j]
                - 2.0 * f1[j] * v0[j] + d * f1[j] * v0[j]
                + 2.0 * f2[j] * v0[j] - d * f2[j] * v0[j]
                - 2.0 * f3[j] * v0[j] + d * f3[j] * v0[j]
                - 2.0 * v1[j] + d * v1[j]
                + 2.0 * f1[j] * v1[j] - d * f1[j] * v1[j]
                - 2.0 * f2[j] * v1[j] + d * f2[j] * v1[j]
                - 2.0 * f3[j] * v1[j]
                + ((d - 2.0) * (-1.0 + f1[j]) - 4.0 * f2[j]) * v2[j]) / 2.0
                + ((d - 3.0) * (ia23[j] * v0[j]
                + ia22[j] * ((-1.0 + f1[j]) * v0[j] + v1[j])
                + ia21[j] * ((1.0 - f1[j] + f2[j]) * v0[j]
                + (-1.0 + f1[j]) * v1[j] + v2[j]))) / std::pow(ia20[j], 2);
    }

    fft.solveInhom(coeff1, coeff2, v3, Delta);

    // Construct final fields
    for (size_t j=0; j<Ntau; ++j)
    {
        F[j] = f0[j] + dx * f1[j] + dx * dx * f2[j] + dx * dx * dx * f3[j];
        U[j] = u0[j] + dx * u1[j] + dx * dx * u2[j] + dx * dx * dx * u3[j];
        V[j] = v0[j] + dx * v1[j] + dx * dx * v2[j] + dx * dx * dx * v3[j];
    }

    #endif

    FieldsToStateVector(U, V, F, Y);

    Y[2] = complex_t(Delta, Y[2].imag());

    if (PrintDiagnostics)
    {
        std::cout << "[INFO] Right-side 3rd-order Taylor expansion at x = " << XRight << " completed.\n";
        real_t f10max = 0.0, f21max = 0.0;
        real_t u10max = 0.0, u21max = 0.0;
        real_t v10max = 0.0, v21max = 0.0;
        real_t f0norm = 0.0, f1norm = 0.0, f2norm = 0.0;
        real_t u0norm = 0.0, u1norm = 0.0, u2norm = 0.0;
        real_t v0norm = 0.0, v1norm = 0.0, v2norm = 0.0;

        for (size_t j = 0; j < Ntau; ++j)
        {
            f10max = std::max(f10max, std::abs(dx * f1[j] / f0[j]));
            f21max = std::max(f21max, std::abs(dx * f2[j] / f1[j]));
            u10max = std::max(u10max, std::abs(dx * u1[j] / u0[j]));
            u21max = std::max(u21max, std::abs(dx * u2[j] / u1[j]));
            v10max = std::max(v10max, std::abs(dx * v1[j] / v0[j]));
            v21max = std::max(v21max, std::abs(dx * v2[j] / v1[j]));

            f0norm += f0[j] * f0[j];
            f1norm += f1[j] * f1[j];
            f2norm += f2[j] * f2[j];
            u0norm += u0[j] * u0[j];
            u1norm += u1[j] * u1[j];
            u2norm += u2[j] * u2[j];
            v0norm += v0[j] * v0[j];
            v1norm += v1[j] * v1[j];
            v2norm += v2[j] * v2[j];
        }

        f0norm = std::sqrt(f0norm);
        f1norm = std::sqrt(f1norm);
        f2norm = std::sqrt(f2norm);
        u0norm = std::sqrt(u0norm);
        u1norm = std::sqrt(u1norm);
        u2norm = std::sqrt(u2norm);
        v0norm = std::sqrt(v0norm);
        v1norm = std::sqrt(v1norm);
        v2norm = std::sqrt(v2norm);

        std::cout << "***************************************\n";
        std::cout << " INFO: Taylor expansion at xright = " << XRight << "\n";
        std::cout << "***************************************\n";
        std::cout << "max((x-1) * f1 / f0) = " << f10max << "\n";
        std::cout << "max((x-1) * f2 / f1) = " << f21max << "\n\n";
        std::cout << "max((x-1) * u1 / u0) = " << u10max << "\n";
        std::cout << "max((x-1) * u2 / u1) = " << u21max << "\n\n";
        std::cout << "max((x-1) * v1 / v0) = " << v10max << "\n";
        std::cout << "max((x-1) * v2 / v1) = " << v21max << "\n\n";

        std::cout << "((x-1) * f1norm) / f0norm = " << dx * f1norm / f0norm << "\n";
        std::cout << "((x-1) * f2norm) / f1norm = " << dx * f2norm / f1norm << "\n\n";

        std::cout << "((x-1) * u1norm) / u0norm = " << dx * u1norm / u0norm << "\n";
        std::cout << "((x-1) * u2norm) / u1norm = " << dx * u2norm / u1norm << "\n\n";

        std::cout << "((x-1) * v1norm) / v0norm = " << dx * v1norm / v0norm << "\n";
        std::cout << "((x-1) * v2norm) / v1norm = " << dx * v2norm / v1norm << "\n";
        
    }
}

void InitialConditionGenerator::packSpectralFields(
    const vec_real& Odd1, const vec_real& Odd2, const vec_real& Even,
    vec_real& Z)
{
    vec_complex Odd1F, Odd2F, EvenF;
    fft.forwardFFT(Odd1, Odd1F);
    fft.forwardFFT(Odd2, Odd2F);
    fft.forwardFFT(Even, EvenF);
    fft.halveModes(Odd1F, Odd1F);
    fft.halveModes(Odd2F, Odd2F);
    fft.halveModes(EvenF, EvenF);

    Z.resize(Nnewton);

    //#ifdef USE_HYBRID
    //#pragma omp parallel for schedule(static)
    //#endif
    for (size_t j=0; j<Nnewton/6; ++j)
    {
        Z[2*j] = Odd1F[2*j+1].real();
        Z[2*j+1] = Odd1F[2*j+1].imag();
        Z[2*j+Nnewton/3] = Odd2F[2*j+1].real();
        Z[2*j+1+Nnewton/3] = Odd2F[2*j+1].imag();
        Z[2*j+2*Nnewton/3] = EvenF[2*j].real();
        Z[2*j+1+2*Nnewton/3] = EvenF[2*j].imag();
    }

    Z[2*Nnewton/3+1] = EvenF[Nnewton/3].real();

}

void InitialConditionGenerator::unpackSpectralFields(const vec_real& Z,
    vec_real& Odd1, vec_real& Odd2, vec_real& Even)
{

    vec_complex Odd1F(Ntau/2, complex_t(0.0));
    vec_complex Odd2F(Ntau/2, complex_t(0.0));
    vec_complex EvenF(Ntau/2, complex_t(0.0));

    //#ifdef USE_HYBRID
    //#pragma omp parallel for schedule(static)
    //#endif
    for (size_t j=0; j<Nnewton/6; ++j)
    {
        Odd1F[2*j+1] = complex_t(Z[2*j], Z[2*j+1]);
        Odd1F[Ntau/2-2*j-1] = std::conj(Odd1F[2*j+1]);
        Odd2F[2*j+1] = complex_t(Z[2*j+Nnewton/3], Z[2*j+1+Nnewton/3]);
        Odd2F[Ntau/2-2*j-1] = std::conj(Odd2F[2*j+1]);
        EvenF[2*j] = complex_t(Z[2*j+2*Nnewton/3], Z[2*j+1+2*Nnewton/3]);
        if (j!=0)
        {
            EvenF[Ntau/2-2*j] = std::conj(EvenF[2*j]);
        } 

    }

    EvenF[0] = complex_t(EvenF[0].real());

    fft.doubleModes(Odd1F, Odd1F);
    fft.doubleModes(Odd2F, Odd2F);
    fft.doubleModes(EvenF, EvenF);

    // Restore high-frequency cosine
    EvenF[Nnewton/3] = complex_t(Z[2*Nnewton/3+1]) / 2.0;
    EvenF[Nnewton] = complex_t(Z[2*Nnewton/3+1]) / 2.0;

    fft.backwardFFT(Odd1F, Odd1);
    fft.backwardFFT(Odd2F, Odd2);
    fft.backwardFFT(EvenF, Even);

}

void InitialConditionGenerator::FieldsToStateVector(const vec_real& U, const vec_real& V,
    const vec_real& F, vec_complex& Y)
{
    Y.resize(Ntau);

    //#ifdef USE_HYBRID
    //#pragma omp parallel for schedule(static)
    //#endif
    for (size_t j=0; j<Ntau; ++j)
    {
        Y[j] = complex_t(U[j], V[j] + F[j]);
    }

    fft.forwardFFT(Y, Y);
    fft.halveModes(Y, Y);

}

void InitialConditionGenerator::StateVectorToFields(vec_complex& Y, vec_real& U, vec_real& V,
     vec_real& F, vec_real& IA2, vec_real& dUdt, vec_real& dVdt, vec_real& dFdt, real_t X)
{
    vec_complex compVec1, compVec2;
    vec_real Coeff1(Ntau), Coeff2(Ntau);
    Delta = Y[2].real();
    fft.doubleModes(Y, compVec1);
    compVec1[2] = complex_t(-compVec1[compVec1.size()-2].real(), compVec1[2].imag());

    fft.differentiate(compVec1, compVec2, Delta);
    fft.backwardFFT(compVec1, compVec1);
    fft.backwardFFT(compVec2, compVec2);    

    #ifdef USE_HYBRID
    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (size_t j=0; j<Ntau/2; ++j)
        {
            U[j] = 0.5 * (compVec1[j].real() - compVec1[j+Ntau/2].real());
            V[j] = 0.5 * (compVec1[j].imag() - compVec1[j+Ntau/2].imag());
            F[j] = 0.5 * (compVec1[j].imag() + compVec1[j+Ntau/2].imag());
            dUdt[j] = 0.5 * (compVec2[j].real() - compVec2[j+Ntau/2].real());
            dVdt[j] = 0.5 * (compVec2[j].imag() - compVec2[j+Ntau/2].imag());
            dFdt[j] = 0.5 * (compVec2[j].imag() + compVec2[j+Ntau/2].imag());
            U[j+Ntau/2] = - U[j];
            V[j+Ntau/2] = - V[j];
            F[j+Ntau/2] = F[j];
            dUdt[j+Ntau/2] = - dUdt[j];
            dVdt[j+Ntau/2] = - dVdt[j];
            dFdt[j+Ntau/2] = dFdt[j];
        }

        #pragma omp for schedule(static)
        for (size_t j=0; j<Ntau; ++j)
        {
            Coeff1[j] = - ((X+F[j]) * U[j]*U[j] + (X-F[j]) * V[j]*V[j])
                        * std::pow((Dim - 2.0),3) / (8.0 * X) - (Dim - 3.0);
            Coeff2[j] = Dim - 3.0; 
        }
    }
    
    
    fft.solveInhom(Coeff1, Coeff2, IA2, Delta);
    #else

    Delta = Y[2].real();
    fft.doubleModes(Y, compVec1);
    compVec1[2] = complex_t(-compVec1[compVec1.size()-2].real(), compVec1[2].imag());

    fft.differentiate(compVec1, compVec2, Delta);

    fft.backwardFFT(compVec1, compVec1);

    for (size_t j=0; j<Ntau/2; ++j)
    {
        U[j] = 0.5 * (compVec1[j].real() - compVec1[j+Ntau/2].real());
        V[j] = 0.5 * (compVec1[j].imag() - compVec1[j+Ntau/2].imag());
        F[j] = 0.5 * (compVec1[j].imag() + compVec1[j+Ntau/2].imag());
    }

    // Shifting symmetries
    for (size_t j=0; j<Ntau/2; ++j)
    {
        U[j+Ntau/2] = - U[j];
        V[j+Ntau/2] = - V[j];
        F[j+Ntau/2] = F[j];
    }

    // Derivatives
    fft.backwardFFT(compVec2, compVec2);
    for (size_t j=0; j<Ntau/2; ++j)
    {
        dUdt[j] = 0.5 * (compVec2[j].real() - compVec2[j+Ntau/2].real());
        dVdt[j] = 0.5 * (compVec2[j].imag() - compVec2[j+Ntau/2].imag());
        dFdt[j] = 0.5 * (compVec2[j].imag() + compVec2[j+Ntau/2].imag());
    }

    // Shifting symmetries
    for (size_t j=0; j<Ntau/2; ++j)
    {
        dUdt[j+Ntau/2] = - dUdt[j];
        dVdt[j+Ntau/2] = - dVdt[j];
        dFdt[j+Ntau/2] = dFdt[j];
    }

    //IA2 from constraint
    for (size_t j=0; j<Ntau; ++j)
    {
        Coeff1[j] = - ((X+F[j]) * U[j]*U[j] + (X-F[j]) * V[j]*V[j])
                    * std::pow((Dim - 2.0),3) / (8.0 * X) - (Dim - 3.0);
        Coeff2[j] = Dim - 3.0; 
    }
    
    fft.solveInhom(Coeff1, Coeff2, IA2, Delta);
    #endif

}

void InitialConditionGenerator::StateVectorToFields(vec_complex& Y, vec_real& U, vec_real& V,
     vec_real& F)
{
    vec_complex compVec1;
    Delta = Y[2].real();
    fft.doubleModes(Y, compVec1);
    compVec1[2] = complex_t(-compVec1[compVec1.size()-2].real(), compVec1[2].imag());

    fft.backwardFFT(compVec1, compVec1);

    #ifdef USE_HYBRID
    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (size_t j=0; j<Ntau/2; ++j)
        {
            U[j] = 0.5 * (compVec1[j].real() - compVec1[j+Ntau/2].real());
            V[j] = 0.5 * (compVec1[j].imag() - compVec1[j+Ntau/2].imag());
            F[j] = 0.5 * (compVec1[j].imag() + compVec1[j+Ntau/2].imag());
            U[j+Ntau/2] = - U[j];
            V[j+Ntau/2] = - V[j];
            F[j+Ntau/2] = F[j];
        }
    }
    #else

    for (size_t j=0; j<Ntau/2; ++j)
    {
        U[j] = 0.5 * (compVec1[j].real() - compVec1[j+Ntau/2].real());
        V[j] = 0.5 * (compVec1[j].imag() - compVec1[j+Ntau/2].imag());
        F[j] = 0.5 * (compVec1[j].imag() + compVec1[j+Ntau/2].imag());
    }

    // Shifting symmetries
    for (size_t j=0; j<Ntau/2; ++j)
    {
        U[j+Ntau/2] = - U[j];
        V[j+Ntau/2] = - V[j];
        F[j+Ntau/2] = F[j];
    }
    #endif

}
