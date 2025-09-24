#pragma once
/**
 * @file InitialConditionGenerator.hpp
 * @brief Construction of near-boundary initial data and conversions between
 *        field representations for the critical collapse solver.
 *
 * @details
 * The InitialConditionGenerator creates high-order Taylor expansions of the
 * DSS fields near the left boundary (x≈0) and right boundary (x≈1). It also
 * provides utilities to pack/unpack parity-separated spectral fields and to
 * convert between
 *   - real-valued field triples (U,V,F)(τ), and
 *   - the complex spectral state vector Y( k ) used by the ODE/Shooting solvers.
 *
 * Internally, derivatives and inhomogeneous solves in τ are performed via a
 * SpectralTransformer (FFTW-based). The echoing period Δ is carried alongside
 * the transforms; by convention the implementation stores Δ in a component of
 * Y (see @ref FieldsToStateVector / @ref StateVectorToFields notes).
 *
 * Conventions:
 *  - Ntau real samples represent periodic functions over one echoing period τ∈[0,Δ).
 *  - Vectors `fc`, `psic`, `Up` have length Ntau.
 *  - Parity: two odd fields (Odd1, Odd2) and one even field (Even) appear in
 *    several packing utilities; in the usual gauge these correspond to
 *    Odd1≡U, Odd2≡V, Even≡F.
 */

#include "common.hpp"
#include "SpectralTransformer.hpp"

/**
 * @class InitialConditionGenerator
 * @brief Builds boundary expansions and performs field/state conversions.
 *
 * @section responsibilities Responsibilities
 * - Left (x≈0) expansion up to 5th order using input (f_c, ψ_c).
 * - Right (x≈1) expansion up to 3rd order using input (U_p).
 * - Convert (U,V,F) ↔ spectral state vector Y.
 * - Pack/unpack parity-split spectral arrays to flat storage.
 *
 * @section requirements Requirements
 * - Ntau ≥ 8 and even (typical spectral constraints).
 * - Input arrays must have length Ntau.
 * - The echoing period Δ must be known (constructor or provided per call).
 */
class InitialConditionGenerator
{
  private:
    /// Number of τ grid points per period.
    size_t Ntau;

    /// Internal Newton vector length used by the shooting/Newton scheme (typically 3*Ntau/4).
    size_t Nnewton;

    /// Physical (rational) spacetime dimension D.
    real_t Dim;

    /// Echoing period Δ used by spectral operators (can be updated per call).
    real_t Delta;

    /// FFTW-based spectral engine (derivatives, integrals, solves).
    SpectralTransformer fft;

  public:
    /**
     * @brief Construct an InitialConditionGenerator.
     * @param Ntau_   Number of τ samples per period.
     * @param Dim_    Physical dimension D.
     * @param Delta_  Echoing period Δ.
     *
     * @note The internal SpectralTransformer is initialized with (Ntau_, Δ).
     */
    InitialConditionGenerator(size_t Ntau_, real_t Dim_, real_t Delta_);

    /**
     * @brief Compute the left-boundary (x≈0) Taylor expansion and output state vector.
     *
     * @details
     * Builds a 5th-order expansion consistent with the left regularity
     * conditions using the supplied periodic inputs:
     *  - f_c(τ)   : even field
     *  - ψ_c(τ)   : auxiliary combination entering the left hierarchy
     *
     * The routine evaluates derivatives in τ spectrally, solves required
     * inhomogeneous equations, assembles the series at x = XLeft, and packs
     * the resulting (U,V,F) into the complex spectral state Y. The echoing
     * period used by spectral operators is set to @p DeltaIn for this call.
     *
     * @param XLeft            Small positive evaluation point near x=0 (e.g., 1e-3).
     * @param fc               f_c(τ) samples, length Ntau.
     * @param psic             ψ_c(τ) samples, length Ntau.
     * @param[out] Y           Output spectral state vector (size implementation-dependent).
     * @param DeltaIn          Echoing period Δ to use for this expansion.
     * @param PrintDiagnostics If true, prints norms/ratios of series terms to assess truncation.
     *
     * @pre  fc.size()==Ntau, psic.size()==Ntau.
     * @post Y encodes the expanded fields; by convention, the real part of Y[2]
     *       stores Δ (echoing period) after packing.
     * @throws std::runtime_error on internal spectral failures.
     */
    void computeLeftExpansion(
        real_t XLeft, const vec_real& fc, const vec_real& psic,
        vec_complex& Y, real_t DeltaIn, bool PrintDiagnostics);

    /**
     * @brief Compute the right-boundary (x≈1) Taylor expansion and output state vector.
     *
     * @details
     * Builds a 3rd-order expansion consistent with the right boundary hierarchy
     * from the periodic input:
     *  - U_p(τ) : right-side seed function
     *
     * The routine evaluates τ-derivatives spectrally, solves the associated
     * inhomogeneous equations, assembles the series at x = XRight, and packs
     * the resulting (U,V,F) into Y. The echoing period used is set to @p DeltaIn.
     *
     * @param XRight           Evaluation point near x=1 (e.g., 1 - 1e-3).
     * @param Up               U_p(τ) samples, length Ntau.
     * @param[out] Y           Output spectral state vector.
     * @param DeltaIn          Echoing period Δ to use for this expansion.
     * @param PrintDiagnostics If true, prints per-order diagnostics.
     *
     * @pre  Up.size()==Ntau.
     * @post Y encodes the expanded fields; by convention, the real part of Y[2]
     *       stores Δ (echoing period) after packing.
     */
    void computeRightExpansion(
        real_t XRight, const vec_real& Up,
        vec_complex& Y, real_t DeltaIn, bool PrintDiagnostics);

    /**
     * @brief Pack three parity-separated real fields into a single flat vector.
     *
     * @details
     * The common convention is Odd1≡U, Odd2≡V, Even≡F. The packing order is
     * implementation-defined but stable across @ref packSpectralFields and
     * @ref unpackSpectralFields.
     *
     * @param Odd1  Real samples of first odd field (length Ntau).
     * @param Odd2  Real samples of second odd field (length Ntau).
     * @param Even  Real samples of even field (length Ntau).
     * @param[out] Z  Output flat storage of size 3*Ntau.
     *
     * @pre Odd1.size()==Odd2.size()==Even.size()==Ntau.
     */
    void packSpectralFields(
        const vec_real& Odd1, const vec_real& Odd2, const vec_real& Even,
        vec_real& Z);

    /**
     * @brief Inverse of @ref packSpectralFields.
     * @param Z      Packed storage vector of size 3*Ntau.
     * @param[out] Odd1  First odd field (size Ntau).
     * @param[out] Odd2  Second odd field (size Ntau).
     * @param[out] Even  Even field (size Ntau).
     */
    void unpackSpectralFields(const vec_real& Z,
                              vec_real& Odd1, vec_real& Odd2, vec_real& Even);

    /**
     * @brief Pack physical fields (U,V,F) into complex spectral state Y.
     *
     * @details
     * Performs real→spectral transforms and arranges coefficients into the
     * solver’s state vector layout. By convention the real part of Y[2] stores
     * Δ after packing so downstream components can read the echoing period
     * without separate plumbing.
     *
     * @param U    Real samples of U(τ), length Ntau.
     * @param V    Real samples of V(τ), length Ntau.
     * @param F    Real samples of F(τ), length Ntau.
     * @param[out] Y  Complex spectral state vector.
     */
    void FieldsToStateVector(const vec_real& U, const vec_real& V,
                             const vec_real& F, vec_complex& Y);

    /**
     * @brief Unpack complex spectral state Y to fields and selected τ-derivatives.
     *
     * @details
     * Inverse of @ref FieldsToStateVector with optional recovery of τ-derivatives
     * (computed spectrally) at a given spatial location X. Also returns IA2 if
     * present in the chosen representation (auxiliary even quantity).
     *
     * @param[in,out] Y  Spectral state vector (may be updated to enforce Δ convention).
     * @param[out] U     U(τ), length Ntau.
     * @param[out] V     V(τ), length Ntau.
     * @param[out] F     F(τ), length Ntau.
     * @param[out] IA2   Auxiliary even field a²(τ) or its inverse variant (length Ntau).
     * @param[out] dUdt  τ-derivative of U(τ), length Ntau.
     * @param[out] dVdt  τ-derivative of V(τ), length Ntau.
     * @param[out] dFdt  τ-derivative of F(τ), length Ntau.
     * @param X          Spatial location where the expansion/interpretation applies.
     */
    void StateVectorToFields(vec_complex& Y, vec_real& U, vec_real& V,
                             vec_real& F, vec_real& IA2,
                             vec_real& dUdt, vec_real& dVdt, vec_real& dFdt,
                             real_t X);

    /**
     * @brief Unpack complex spectral state Y to fields (no derivatives).
     * @param[in,out] Y  Spectral state vector.
     * @param[out] U     U(τ).
     * @param[out] V     V(τ).
     * @param[out] F     F(τ).
     */
    void StateVectorToFields(vec_complex& Y, vec_real& U, vec_real& V,
                             vec_real& F);
};
