#pragma once
/**
 * @file SpectralTransformer.hpp
 * @brief Thin wrapper around FFTW to perform Fourier transforms and spectral operations.
 *
 * @details
 * Provides Fourier transforms (real/complex, forward/backward) and
 * additional utilities needed in the critical collapse solver:
 * - Differentiate in spectral space.
 * - Integrate with λ-shift.
 * - Interpolate Fourier series at arbitrary x.
 * - Mode truncation/doubling (halve/double resolution).
 * - Solve simple inhomogeneous equations in Fourier space.
 *
 * This class owns FFTW plans and work arrays. RAII ensures cleanup.
 */

#include "common.hpp"

/**
 * @class SpectralTransformer
 * @brief Encapsulates Fourier-based spectral operations.
 *
 * @section usage Usage
 * Construct with number of modes and period, then call forwardFFT/backwardFFT
 * to convert between time/τ domain and frequency domain. Use additional
 * helpers for differentiation, interpolation, etc.
 */
class SpectralTransformer
{
  private:
    size_t N;               ///< Number of real-space grid points.
    real_t period;          ///< Period (echoing period Δ).
    real_t k0;              ///< Fundamental frequency (2π/Δ).
    fftw_plan forward_plan {nullptr};   ///< FFTW plan: forward.
    fftw_plan backward_plan {nullptr};  ///< FFTW plan: backward.
    fftw_complex *forward_data {nullptr}, *backward_data {nullptr}; ///< Work arrays.

  public:
    /**
     * @brief Construct transformer for N points with given period.
     * @param N      Number of real-space samples.
     * @param period Period of the signal (Δ).
     */
    explicit SpectralTransformer(size_t N, real_t period);

    /// Destructor: destroys FFTW plans and frees memory.
    ~SpectralTransformer();

    /// Forward FFT (real → complex).
    void forwardFFT(const vec_real& in, vec_complex& out);

    /// Backward FFT (complex → real).
    void backwardFFT(const vec_complex& in, vec_real& out);

    /// Forward FFT (complex → complex).
    void forwardFFT(const vec_complex& in, vec_complex& out);

    /// Backward FFT (complex → complex).
    void backwardFFT(const vec_complex& in, vec_complex& out);

    /**
     * @brief Differentiate a Fourier series.
     * @param in      Input Fourier coefficients.
     * @param out     Output differentiated Fourier coefficients.
     * @param period_ Optional override of period (defaults to ctor value).
     */
    void differentiate(const vec_complex& in, vec_complex& out, real_t period_=0.0);

    /**
     * @brief λ-integration in Fourier space.
     * @param fk      Input Fourier coefficients.
     * @param out     Output integrated coefficients.
     * @param lambda  Integration constant (shift).
     * @param period_ Optional override of period.
     */
    void lamIntegrate(const vec_complex& fk, vec_complex& out,
                      complex_t lambda=complex_t(0.0), real_t period_=0.0);

    /**
     * @brief Truncate Fourier modes by factor 2 (half resolution).
     * @param in  Input coefficients.
     * @param out Output truncated coefficients.
     */
    void halveModes(const vec_complex& in, vec_complex& out);

    /**
     * @brief Double Fourier modes by zero-padding (double resolution).
     * @param in  Input coefficients.
     * @param out Output upsampled coefficients.
     */
    void doubleModes(const vec_complex& in, vec_complex& out);

    /**
     * @brief Solve inhomogeneous spectral equation of form Ax=f,g.
     * @param f Input RHS vector f.
     * @param g Input RHS vector g.
     * @param x Output solution vector.
     * @param period_ Optional override of period.
     */
    void solveInhom(const vec_real& f, const vec_real& g, vec_real& x, real_t period_=0.0);
};
