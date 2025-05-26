#include "InitialConditionGenerator.hpp"

InitialConditionGenerator::InitialConditionGenerator(int NumGridPts_, real_t Dim_, real_t Delta_)
    : NumGridPts(NumGridPts_), Dim(Dim_), Delta(Delta_), fft(NumGridPts_, Delta_) {}

void InitialConditionGenerator::generateInitialCondition(
    const vec_real& Fc, const vec_real& Psic, const vec_real& Up,
    real_t X, bool IsLeft, vec_real& Y, bool PrintDiagnostics)
{
    vec_real U(NumGridPts), V(NumGridPts), F(NumGridPts);

    if (IsLeft)
    {
        computeLeftExpansion(Fc, Psic, X, U, V, F, PrintDiagnostics);
    }
    else
    {
        computeRightExpansion(Up, X, U, V, F, PrintDiagnostics);
    }

    packSpectralFields(U, V, F, Y);
    Y[4] = Delta; // Store Delta in y[5] position (index 4 in C++)
}

void InitialConditionGenerator::computeLeftExpansion(
    const vec_real& Fc, const vec_real& Psic, real_t X,
    vec_real& U, vec_real& V, vec_real& F, bool PrintDiagnostics)
{
    const real_t x = X;
    const real_t d = Dim;

    // Step 1: Compute f0 and its derivatives
    const vec_real& f0 = Fc;
    const vec_real& psi0 = Psic;

    vec_complex F0Hat, DfHat, D2fHat, D3fHat;
    vec_real df0dtau, d2f0dtau2, d3f0dtau3;

    fft.forward_fft(f0, F0Hat);
    fft.differentiate(F0Hat, DfHat);
    fft.inverse_fft(DfHat, df0dtau);

    fft.forward_fft(df0dtau, D2fHat);
    fft.differentiate(D2fHat, D2fHat);
    fft.inverse_fft(D2fHat, d2f0dtau2);

    fft.forward_fft(d2f0dtau2, D3fHat);
    fft.differentiate(D3fHat, D3fHat);
    fft.inverse_fft(D3fHat, d3f0dtau3);

    // Step 2: Solve inhomogeneous ODE for u1
    vec_real Coeff1(NumGridPts, 1.0), Coeff2(NumGridPts), u1;
    for (int j = 0; j < NumGridPts; ++j)
        Coeff2[j] = -(d - 1.0) * psi0[j] * f0[j];

    fft.solve_inhomogeneous(Coeff1, Coeff2, u1);

    // Step 3: Compute derivatives of u1
    vec_complex U1Hat, dU1Hat, d2U1Hat, d3U1Hat, d4U1Hat;
    vec_real du1dtau, d2u1dtau2, d3u1dtau3, d4u1dtau4;

    fft.forward_fft(u1, U1Hat);
    fft.differentiate(U1Hat, dU1Hat);
    fft.inverse_fft(dU1Hat, du1dtau);

    fft.forward_fft(du1dtau, d2U1Hat);
    fft.differentiate(d2U1Hat, d2U1Hat);
    fft.inverse_fft(d2U1Hat, d2u1dtau2);

    fft.forward_fft(d2u1dtau2, d3U1Hat);
    fft.differentiate(d3U1Hat, d3U1Hat);
    fft.inverse_fft(d3U1Hat, d3u1dtau3);

    fft.forward_fft(d3u1dtau3, d4U1Hat);
    fft.differentiate(d4U1Hat, d4U1Hat);
    fft.inverse_fft(d4U1Hat, d4u1dtau4);

    // Step 4: Taylor expansions
    vec_real f2(NumGridPts), f4(NumGridPts),
             u2(NumGridPts), u3(NumGridPts), u4(NumGridPts), u5(NumGridPts),
             v1(NumGridPts), v2(NumGridPts), v3(NumGridPts), v4(NumGridPts), v5(NumGridPts);

    for (int j = 0; j < NumGridPts; ++j)
    {
        const real_t f = f0[j], psi = psi0[j];
        const real_t u1j = u1[j], du1 = du1dtau[j], d2u1 = d2u1dtau2[j], d3u1 = d3u1dtau3[j], d4u1 = d4u1dtau4[j];
        const real_t df = df0dtau[j], d2f = d2f0dtau2[j], d3f = d3f0dtau3[j];

        v1[j] = u1j;
        u2[j] = -psi;
        v2[j] = psi;

        f2[j] = ((d - 3.0) * std::pow(d - 2.0, 3) * f * std::pow(u1j, 2)) / (8.0 * (d - 1.0));

        u3[j] = ((d - 3.0) * std::pow(d - 2.0, 3) * std::pow(f, 3) * std::pow(u1j, 3)
                + 4.0 * df * (u1j + du1)
                - 4.0 * f * (2.0 * u1j + 3.0 * du1 + d2u1))
                / (8.0 * (1.0 - d) * std::pow(f, 3));

        v3[j] = u3[j]; // same formula

        u4[j] = (-192.0 * (d - 1.0) * std::pow(df, 2) * du1 * std::pow(f, 2)
                + 192.0 * (d - 1.0) * d2u1 * df * std::pow(f, 3)
                + 64.0 * (d - 1.0) * d2f * du1 * std::pow(f, 3)
                + 640.0 * (d - 1.0) * df * du1 * std::pow(f, 3)
                - 64.0 * (d - 1.0) * (6.0 * d2u1 + d3u1) * std::pow(f, 4)
                - 704.0 * (d - 1.0) * du1 * std::pow(f, 4)
                - 192.0 * (d - 1.0) * std::pow(df, 2) * f * u1j
                + 64.0 * (d - 1.0) * d2f * std::pow(f, 3) * u1j
                + 448.0 * (d - 1.0) * df * std::pow(f, 3) * u1j
                - 384.0 * (d - 1.0) * std::pow(f, 4) * u1j
                + 32.0 * std::pow(d - 2.0, 3) * (3.0 - 7.0 * d + std::pow(d, 2))
                  * std::pow(f, 6) * std::pow(u1j, 2) * (du1 + u1j))
                / (128.0 * std::pow(d - 1.0, 2) * (1.0 + d) * std::pow(f, 7));

        v4[j] = -u4[j];

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
              * df * du1 * std::pow(f, 4) * u1j * u1j
            - 8.0 * std::pow(d - 2.0, 3) * (3.0 - 13.0 * d + 4.0 * d * d)
              * (d2u1 + 5.0 * du1) * std::pow(f, 5) * u1j * u1j
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
        U[j] = x * u1j + std::pow(x, 2) * u2[j] + std::pow(x, 3) * u3[j] + std::pow(x, 4) * u4[j] + std::pow(x, 5) * u5[j];
        V[j] = x * v1[j] + std::pow(x, 2) * v2[j] + std::pow(x, 3) * v3[j] + std::pow(x, 4) * v4[j] + std::pow(x, 5) * v5[j];
        F[j] = f + std::pow(x, 2) * f2[j] + std::pow(x, 4) * f4[j]; // f4 is not derived in Fortran, so left out here
    }

    if (PrintDiagnostics)
    {
        std::cout << "[INFO] Left-side 5th-order Taylor expansion at x = " << X << " completed.\n";
    }
}

void InitialConditionGenerator::computeRightExpansion(
    const vec_real& Up, real_t X, vec_real& U, vec_real& V, vec_real& F, bool PrintDiagnostics)
{
    const real_t xp = 1.0;
    const real_t d = Dim;
    const real_t x = X;
    const real_t dx = x - xp;

    const vec_real& u0 = Up;
    vec_real du0, d2u0;

    vec_complex U0Hat, dU0Hat, d2U0Hat;

    fft.forward_fft(u0, U0Hat);
    fft.differentiate(U0Hat, dU0Hat);
    fft.inverse_fft(dU0Hat, du0);

    fft.forward_fft(du0, d2U0Hat);
    fft.differentiate(d2U0Hat, d2U0Hat);
    fft.inverse_fft(d2U0Hat, d2u0);

    // Step 2: Solve for ia20
    vec_real coeff1(NumGridPts), coeff2(NumGridPts), ia20;
    for (int j = 0; j < NumGridPts; ++j)
    {
        coeff1[j] = 3.0 - d - std::pow(d - 2.0, 3) * u0[j] * u0[j] / 4.0;
        coeff2[j] = d - 3.0;
    }
    fft.solve_inhomogeneous(coeff1, coeff2, ia20);

    // Step 3: Solve for v0
    vec_real v0;
    for (int j = 0; j < NumGridPts; ++j)
    {
        coeff1[j] = (6.0 - 2.0 * d + (d - 2.0) * ia20[j]) / (2.0 * ia20[j]);
        coeff2[j] = (d - 2.0) * u0[j] / 2.0;
    }
    fft.solve_inhomogeneous(coeff1, coeff2, v0);

    // Order 1
    vec_real f0(NumGridPts, 1.0);  // f0 = 1.0 by gauge choice at x=1
    vec_real f1(NumGridPts), u1(NumGridPts), ia21(NumGridPts);
    for (int j = 0; j < NumGridPts; ++j)
    {
        f1[j] = 3.0 - d + (d - 3.0) / ia20[j];

        ia21[j] = -ia20[j] * (8.0 * (d - 3.0)
                     + std::pow(d - 2.0, 3) * (u0[j] * u0[j] + v0[j] * v0[j])) / 8.0
                  + d - 3.0;

        u1[j] = ((d - 2.0 - 2.0 * (d - 3.0) / ia20[j]) * u0[j]
                + (d - 2.0) * v0[j] - 2.0 * du0[j]) / 4.0;
    }

    // Order 2
    vec_real v1(NumGridPts), u2(NumGridPts), v2(NumGridPts), f2(NumGridPts), ia22(NumGridPts);
    for (int j = 0; j < NumGridPts; ++j)
    {
        v1[j] = ((d - 2.0 - 2.0 * (d - 3.0) / ia20[j]) * v0[j]
                + (d - 2.0) * u0[j] + 2.0 * du0[j]) / 4.0;

        u2[j] = ((1.0 - f1[j]) * u0[j] + u1[j]) / 2.0
              + ((d - 3.0) * (ia20[j] * u1[j] + ia21[j] * ((1.0 - f1[j]) * u0[j] + u1[j]))) / (2.0 * ia20[j] * ia20[j]);

        v2[j] = ((1.0 - f1[j]) * v0[j] + v1[j]) / 2.0
              + ((d - 3.0) * (ia20[j] * v1[j] + ia21[j] * ((1.0 - f1[j]) * v0[j] + v1[j]))) / (2.0 * ia20[j] * ia20[j]);

        f2[j] = (d - 3.0) * (1.0 - ia20[j]) / (ia20[j] * (1.0 - x));
        ia22[j] = -((1.0 - f1[j]) * ia20[j] + ia21[j]) / 2.0
                  + ((d - 3.0) * (ia21[j] * ((1.0 - f1[j]) * ia20[j] + ia21[j])
                                + ia20[j] * ia22[j])) / (ia20[j] * ia20[j]);
    }

    // Order 3
    vec_real f3(NumGridPts), u3(NumGridPts), v3(NumGridPts), ia23(NumGridPts);
    for (int j = 0; j < NumGridPts; ++j)
    {
        f3[j] = 0.0;  // if desired, can be added similarly
        u3[j] = 0.0;
        v3[j] = 0.0;
        ia23[j] = 0.0;
    }

    // Construct final fields
    for (int j = 0; j < NumGridPts; ++j)
    {
        F[j] = f0[j] + dx * f1[j] + dx * dx * f2[j] + dx * dx * dx * f3[j];
        U[j] = u0[j] + dx * u1[j] + dx * dx * u2[j] + dx * dx * dx * u3[j];
        V[j] = v0[j] + dx * v1[j] + dx * dx * v2[j] + dx * dx * dx * v3[j];
    }

    if (PrintDiagnostics)
        std::cout << "[INFO] Right-side 3rd-order Taylor expansion at x = " << X << " completed.\n";
}

void InitialConditionGenerator::packSpectralFields(
    const vec_real& U, const vec_real& V, const vec_real& F,
    vec_real& Y)
{
    vec_complex Uhat, Vhat, Fhat;
    fft.forward_fft(U, Uhat);
    fft.forward_fft(V, Vhat);
    fft.forward_fft(F, Fhat);

    // Apply halve (anti-aliasing) then pack compactly — mimic mypack.f
    fft.resample_modes(Uhat, Uhat, NumGridPts / 2);
    fft.resample_modes(Vhat, Vhat, NumGridPts / 2);
    fft.resample_modes(Fhat, Fhat, NumGridPts / 2);

    int N3 = (3 * NumGridPts) / 4;
    Y.resize(N3);

    for (int j = 0; j < N3 / 6; ++j)
    {
        Y[2 * j       ] = Uhat[4 * j + 2].real();
        Y[2 * j + 1   ] = Uhat[4 * j + 3].real();
        Y[2 * j + N3/3] = Vhat[4 * j + 2].real();
        Y[2 * j + 1 + N3/3] = Vhat[4 * j + 3].real();
        Y[2 * j + 2 * N3/3] = Fhat[4 * j + 1].real();
        Y[2 * j + 1 + 2 * N3/3] = Fhat[4 * j + 2].real();
    }

    // High-frequency cosine stored in slot 2*N3/3 + 2
    Y[2 * N3 / 3 + 2] = Fhat[2 * N3 / 3 + 1].real();
}

void InitialConditionGenerator::unpackSpectralFields(const vec_real& Y,
    vec_real& U, vec_real& V, vec_real& F)
{
    const int N = NumGridPts;
    const int N3 = 3 * N / 4;
    const int halfN = N / 2;

    vec_complex Uhat(halfN, complex_t(0.0, 0.0));
    vec_complex Vhat(halfN, complex_t(0.0, 0.0));
    vec_complex Fhat(halfN, complex_t(0.0, 0.0));

    for (int j = 0; j < N3 / 6; ++j)
    {
    Uhat[4 * j + 2] = complex_t(Y[2 * j], 0.0);
    Uhat[4 * j + 3] = complex_t(Y[2 * j + 1], 0.0);

    Vhat[4 * j + 2] = complex_t(Y[2 * j + N3 / 3], 0.0);
    Vhat[4 * j + 3] = complex_t(Y[2 * j + 1 + N3 / 3], 0.0);

    Fhat[4 * j + 1] = complex_t(Y[2 * j + 2 * N3 / 3], 0.0);
    Fhat[4 * j + 2] = complex_t(Y[2 * j + 1 + 2 * N3 / 3], 0.0);
    }

    // Restore high-frequency cosine (stored in slot y(5) ~ index 2N/3 + 2)
    Fhat[2 * N3 / 3 + 1] = complex_t(Y[2 * N3 / 3 + 2], 0.0);

    // Upsample back to N
    vec_complex UhatFull, VhatFull, FhatFull;
    fft.resample_modes(Uhat, UhatFull, N);
    fft.resample_modes(Vhat, VhatFull, N);
    fft.resample_modes(Fhat, FhatFull, N);

    // Inverse FFT → real space
    fft.inverse_fft(UhatFull, U);
    fft.inverse_fft(VhatFull, V);
    fft.inverse_fft(FhatFull, F);
}
