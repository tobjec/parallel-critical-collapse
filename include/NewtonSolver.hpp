#pragma once
/**
 * @file NewtonSolver.hpp
 * @brief Nonlinear solver (Newton–Raphson) for the boundary value problem
 *        arising in discretely self-similar scalar field collapse.
 *
 * @details
 * The NewtonSolver iteratively adjusts the initial data vector to enforce
 * matching conditions between left- and right-boundary Taylor expansions.
 * It drives the mismatch residual to zero using a Jacobian-based Newton
 * method. Parallel variants exist with OpenMP, MPI, or hybrid execution.
 *
 * Responsibilities:
 *  - Initialize near-boundary data via InitialConditionGenerator.
 *  - Perform shooting integration across the domain via ShootingSolver.
 *  - Assemble mismatch residuals at the matching point XMid.
 *  - Construct and solve the Newton system J·dx = -F.
 *  - Track convergence and write results/output files.
 */

#include "common.hpp"
#include "ShootingSolver.hpp"
#include "InitialConditionGenerator.hpp"
#include "SimulationConfig.hpp"
#include "OutputWriter.hpp"

/**
 * @class NewtonSolver
 * @brief Newton–Raphson driver enforcing boundary matching in DSS collapse.
 *
 * @section workflow Workflow
 * - Construct from a SimulationConfig and output folder.
 * - Generate τ-grid and boundary data.
 * - Run Newton iterations until mismatch < TolNewton or maxIts reached.
 * - Return results as JSON; optionally benchmark execution.
 *
 * @section parallelism Parallelism
 * - OpenMP: replicate local data structures for thread-level parallel shooting.
 * - MPI: distribute Jacobian assembly across ranks, collective solves.
 * - Hybrid: combine MPI+OpenMP.
 */
class NewtonSolver
{
  private:
    // ===== Simulation state =====
    SimulationConfig config;      ///< Simulation parameters (dimension, Ntau, Δ, etc.)
    size_t Ntau;                  ///< Number of τ samples per period.
    size_t Nnewton;               ///< Number of Newton unknowns (typically 3*Ntau/4).
    size_t maxIts;                ///< Maximum Newton iterations.
    real_t Dim;                   ///< Physical spacetime dimension D.
    real_t Delta;                 ///< Echoing period Δ.
    real_t slowErr;               ///< Accumulated error metric (slowly updated).
    real_t EpsNewton;             ///< Step length damping or regularization.
    real_t TolNewton;             ///< Convergence tolerance on mismatch norm.
    real_t XLeft, XMid, XRight;   ///< Domain decomposition: left/mid/right match points.
    size_t NLeft, NRight;         ///< Number of grid cells left/right.
    size_t iL, iM, iR;            ///< Indices of left, mid, right points.

    bool Debug;                   ///< Debug flag for verbose diagnostics.
    bool Verbose;                 ///< Verbose printing flag.
    bool Converged;               ///< True if Newton converged.

    // ===== Components =====
    InitialConditionGenerator initGen; ///< Generates near-boundary Taylor expansions.
    vec_complex YLeft, YRight;         ///< Packed state vectors at left/right boundaries.
    vec_complex mismatchOut;           ///< Mismatch residual at match point.
    vec_real fc, psic, Up;             ///< Boundary input functions (τ-dependent).
    vec_real XGrid;                    ///< Radial grid points (left → right).
    vec_real in0, out0;                ///< Working vectors for shooting I/O.
    json resultDict;                   ///< Stores results for output/benchmark.
    std::filesystem::path baseFolder;  ///< Path where outputs are written.
    bool benchmark;                    ///< If true, run in benchmark mode.

    std::unique_ptr<ShootingSolver> shooter; ///< Integrator for evolution between boundaries.

    // ===== Internal helpers =====
    /**
     * @brief Initialize left/right expansions and state vectors.
     * @param printDiagnostics If true, print truncation and norm checks.
     */
    void initializeInput(bool printDiagnostics=false);

    /// Build spatial/radial grid (fills XGrid, indices).
    void generateGrid();

    /**
     * @brief Perform shooting integration from left/right to mid, produce mismatch.
     * @param inputVec  Newton unknown vector.
     * @param outputVec Output mismatch vector.
     * @param fieldVals Optional JSON container for storing fields along integration.
     */
    void shoot(vec_real& inputVec, vec_real& outputVec, json* fieldVals=nullptr);

    #ifdef USE_OPENMP
    /// Thread-local variant of initializeInput.
    void initializeInput(InitialConditionGenerator& initGen_local,
                         vec_complex& YLeft_local, vec_complex& YRight_local,
                         vec_real& Up_local, vec_real& psic_local, vec_real& fc_local,
                         real_t Delta_local, bool printDiagnostics=false);

    /// Thread-local variant of shoot for parallel evaluation.
    void shoot(vec_real& inputVec, vec_real& outputVec,
               ShootingSolver& shooter_local,
               InitialConditionGenerator& initGen_local,
               vec_complex& YLeft_local, vec_complex& YRight_local,
               vec_complex& mismatchOut_local);
    #endif

    #if defined(USE_MPI) || defined(USE_HYBRID)
    int size;                     ///< MPI world size.
    int rank;                     ///< MPI rank ID.
    MPI_Datatype mpi_contiguous_t;///< Custom MPI datatype for real_t arrays.
    #endif

    /**
     * @brief Assemble finite-difference Jacobian of mismatch residuals.
     * @param baseInput  Current Newton input vector.
     * @param baseOutput Corresponding mismatch vector.
     * @param[out] jacobian Jacobian matrix J = ∂F/∂x.
     */
    void assembleJacobian(const vec_real& baseInput, const vec_real& baseOutput,
                          mat_real& jacobian);

    /**
     * @brief Solve linear system A·dx = rhs.
     * @param[in,out] A   Matrix (modified in-place if LAPACK factorization used).
     * @param[in,out] rhs Right-hand side vector (overwritten with solution).
     * @param[out] dx     Solution increment.
     */
    #if defined(USE_MPI) || defined(USE_HYBRID)
    void solveLinearSystem(mat_real& A, vec_real& rhs, vec_real& dx);
    #else
    void solveLinearSystem(const mat_real& A, vec_real& rhs, vec_real& dx);
    #endif

    /**
     * @brief Compute L² norm of a vector.
     * @param vc Input vector.
     * @return sqrt(Σ vᵢ²).
     */
    real_t computeL2Norm(const vec_real& vc);

  public:
    /**
     * @brief Construct NewtonSolver with simulation config and output path.
     * @param configIn   SimulationConfig object (Ntau, Dim, Δ, etc.).
     * @param dataPathIn Base path for output files.
     * @param benchmarkIn If true, enable benchmark mode.
     */
    NewtonSolver(SimulationConfig configIn, std::filesystem::path dataPathIn, bool benchmarkIn=false);

    #if defined(USE_MPI) || defined(USE_HYBRID)
    /// Finalize MPI datatype if created.
    ~NewtonSolver();
    #endif

    /**
     * @brief Run Newton solver until convergence or max iterations.
     * @param benchmark_result Optional JSON to collect benchmark statistics.
     * @return JSON result dictionary with fields (Converged, NewtonIts, Delta, etc.).
     */
    json run(json* benchmark_result=nullptr);

    /**
     * @brief Write final converged output and metadata to files.
     * @param newtonIts     Number of Newton iterations performed.
     * @param mismatchNorm  Final mismatch L² norm.
     */
    void writeFinalOutput(size_t newtonIts, real_t mismatchNorm);
};
