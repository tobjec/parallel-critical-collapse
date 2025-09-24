#pragma once
/**
 * @file ODEStepper.hpp
 * @brief Implicit Runge–Kutta (IRK) integrator for the ODE system in
 *        discretely self-similar scalar field collapse.
 *
 * @details
 * The ODEStepper advances the spectral state vector Y(x) across the radial
 * domain using an implicit Runge–Kutta scheme. The equations correspond
 * to the reduced Einstein–scalar PDE system, reformulated as ODEs in x
 * with τ treated spectrally. Supported methods are IRK1, IRK2, IRK3.
 *
 * Responsibilities:
 *  - Hold Butcher tableau coefficients (a,b,c) for selected IRK scheme.
 *  - Evaluate τ-derivatives of fields via InitialConditionGenerator.
 *  - Perform nonlinear solve at each step with Newton iterations.
 *  - Report convergence status and number of iterations used.
 */

#include "common.hpp"
#include "InitialConditionGenerator.hpp"

/**
 * @class ODEStepper
 * @brief Implicit Runge–Kutta solver for advancing the DSS state vector.
 *
 * @section usage Usage
 * - Construct with Ntau, dimension, precision, scheme, and reference to
 *   an InitialConditionGenerator.
 * - Call integrate() to advance Y from X_in → X_out.
 * - Pass in/out state vectors (vec_complex) representing spectral fields.
 *
 * @section notes Notes
 * - Precision parameter controls Newton tolerance inside each IRK step.
 * - Supported schemes: IRK1 (backward Euler), IRK2, IRK3 (higher order).
 * - The nonlinear system inside each step is solved iteratively.
 */
class ODEStepper
{
  private:
    int Ntau;                       ///< Number of τ modes (problem dimension).
    real_t Dim;                     ///< Physical dimension D.
    real_t precision;               ///< Tolerance for Newton solve inside IRK.
    Scheme scheme;                  ///< Selected implicit Runge–Kutta scheme.
    InitialConditionGenerator& initGen; ///< Reference to initial condition/spectral transformer.

    mat_real a; ///< IRK Butcher tableau coefficients (matrix).
    vec_real b; ///< IRK weights.
    vec_real c; ///< IRK nodes.

    /**
     * @brief Compute ODE right-hand side dY/dx at given x.
     * @param[in]  Yreal   Current state vector in real form.
     * @param[out] dYreal  Output derivative dY/dx.
     * @param[in]  x       Current spatial location.
     *
     * @details
     * Converts spectral complex vector → real, evaluates RHS using
     * InitialConditionGenerator to supply τ-derivatives, then fills dY/dx.
     */
    void computeDerivatives(vec_real& Yreal, vec_real& dYreal, real_t x);

    /**
     * @brief Perform one IRK step between Xin and Xout.
     * @param[in]  Yin        Input state vector at Xin.
     * @param[out] Yout       Output state vector at Xout.
     * @param[in]  Xin        Start location.
     * @param[in]  Xout       End location.
     * @param[out] itsReached Number of Newton iterations performed.
     * @param[out] converged  True if nonlinear solver converged.
     * @param[in]  maxIts     Maximum Newton iterations allowed.
     *
     * @details
     * Builds and solves the IRK stage equations using the selected scheme.
     * If convergence fails, converged=false and output may be invalid.
     */
    void stepIRK(vec_complex& Yin, vec_complex& Yout,
                 real_t Xin, real_t Xout, int& itsReached,
                 bool& converged, int maxIts);

  public:
    /**
     * @brief Construct ODEStepper with given parameters.
     * @param numVars   Number of τ modes / variables.
     * @param dim       Physical dimension D.
     * @param precision Tolerance for IRK Newton solve.
     * @param method    Selected IRK scheme (IRK1/2/3).
     * @param initGen   Reference to InitialConditionGenerator.
     */
    ODEStepper(int numVars, real_t dim, real_t precision,
               Scheme method, InitialConditionGenerator& initGen);

    /**
     * @brief Integrate ODE system from Xin → Xout.
     * @param[in]  Yin        Input state vector at Xin.
     * @param[out] Yout       Output state vector at Xout.
     * @param[in]  Xin        Start location.
     * @param[in]  Xout       End location.
     * @param[out] converged  True if all steps converged.
     * @param[out] itsReached Number of Newton iterations used in last step.
     * @param[in]  maxIts     Maximum Newton iterations per step.
     *
     * @details
     * Calls stepIRK() repeatedly to advance over the domain. Reports if
     * solver converged within tolerance at each step.
     */
    void integrate(vec_complex& Yin, vec_complex& Yout,
                   real_t Xin, real_t Xout,
                   bool& converged, int& itsReached,
                   int maxIts);
};
