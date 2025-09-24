#pragma once
/**
 * @file ShootingSolver.hpp
 * @brief Implements the shooting method to connect left and right boundary
 *        expansions at a matching point in the DSS collapse equations.
 *
 * @details
 * The ShootingSolver integrates the ODE system from both boundaries
 * (x≈0, x≈1) toward a chosen match point Xmid. Using ODEStepper for
 * each integration, it computes the mismatch between left- and right-
 * integrated states. This mismatch serves as the residual for the
 * NewtonSolver.
 *
 * Responsibilities:
 *  - Initialize ODEStepper with given Ntau, dimension, precision.
 *  - Integrate spectral state vector Y along a radial grid.
 *  - Return mismatch vector between left and right states.
 */

#include "common.hpp"
#include "SpectralTransformer.hpp"
#include "InitialConditionGenerator.hpp"
#include "ODEStepper.hpp"

/**
 * @class ShootingSolver
 * @brief Shooting integration engine for DSS boundary value problem.
 *
 * @section workflow Workflow
 * - Construct with Ntau, Dim, precision, InitialConditionGenerator, maxIts.
 * - Given YLeft and YRight boundary states and radial grid X,
 *   integrate inward/outward toward Xmid.
 * - Compute mismatch at match point.
 */
class ShootingSolver
{
  private:
    int Ntau;                          ///< Number of τ modes.
    real_t Dim;                        ///< Physical dimension D.
    real_t Precision;                  ///< Newton tolerance for IRK steps.
    InitialConditionGenerator& initGen;///< Reference to spectral/tau operator.
    std::unique_ptr<ODEStepper> stepper; ///< ODE integrator (IRK scheme).
    bool converged = false;            ///< True if last integration converged.
    int itsReached;                    ///< Iterations used in last step.
    int maxIts;                        ///< Maximum iterations per IRK step.

    /**
     * @brief Integrate state vector from startIdx → endIdx on grid.
     * @param yInit     Initial spectral state vector at start.
     * @param xGrid     Radial grid points.
     * @param startIdx  Index to start integration.
     * @param endIdx    Index to stop integration.
     * @param forward   If true, integrate forward (increasing x); else backward.
     * @param[out] yFinal  State vector at endIdx after integration.
     * @param Debug     If true, prints diagnostics during integration.
     * @param fieldVals Optional JSON to collect full field values along path.
     *
     * @details
     * Calls ODEStepper::integrate step by step across the subgrid.
     * Handles convergence checks and optional debug output.
     */
    void integrateToMatchPoint(
        const vec_complex& yInit, const vec_real& xGrid,
        size_t startIdx, size_t endIdx, bool forward,
        vec_complex& yFinal, bool Debug=false, json* fieldVals=nullptr);

    /**
     * @brief Compute mismatch vector between left- and right-integrated states.
     * @param yLeft       Spectral state at match point from left.
     * @param yRight      Spectral state at match point from right.
     * @param[out] mismatchOut Difference vector encoding residual.
     *
     * @details
     * The mismatch is defined component-wise (yLeft - yRight) in the spectral
     * representation, restricted to Newton unknowns.
     */
    void computeMismatch(const vec_complex& yLeft,
                         const vec_complex& yRight,
                         vec_complex& mismatchOut);

  public:
    /**
     * @brief Construct ShootingSolver.
     * @param Ntau_     Number of τ modes.
     * @param Dim_      Physical dimension D.
     * @param precision_ Newton tolerance for IRK steps.
     * @param initGen_  Reference to InitialConditionGenerator.
     * @param maxIts_   Maximum Newton iterations per IRK step.
     * @param SchemeIRK_   Scheme of IRK method.
     * 
     * @throws std::invalid_argument if supplied integer is not linked to IRK Scheme.
     */
    ShootingSolver(int Ntau_, real_t Dim_, real_t precision_,
                   InitialConditionGenerator& initGen_, int maxIts_ , int SchemeIRK_=2);

    /**
     * @brief Perform bidirectional shooting and compute mismatch at Xmid.
     * @param[in,out] YLeft   Left boundary state vector (x≈0).
     * @param[in,out] YRight  Right boundary state vector (x≈1).
     * @param gridX           Radial grid points.
     * @param iLeft           Index of left boundary.
     * @param iRight          Index of right boundary.
     * @param iMid            Index of match point.
     * @param[out] mismatchOut Output mismatch vector at Xmid.
     * @param Debug           If true, enable verbose diagnostics.
     * @param fieldVals       Optional JSON to collect full field values.
     *
     * @details
     * - Integrates YLeft forward to Xmid.
     * - Integrates YRight backward to Xmid.
     * - Computes mismatch = YLeft(Xmid) - YRight(Xmid).
     */
    void shoot(vec_complex& YLeft, vec_complex& YRight, const vec_real& gridX,
               size_t iLeft, size_t iRight, size_t iMid,
               vec_complex& mismatchOut, bool Debug=false,
               json* fieldVals=nullptr);
};
