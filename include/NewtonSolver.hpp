#pragma once

#include "common.hpp"
#include "ShootingSolver.hpp"
#include "InitialConditionGenerator.hpp"
#include "SimulationConfig.hpp"
#include "OutputWriter.hpp"

class NewtonSolver
{
    private:
        SimulationConfig config;
        size_t Ntau, Nnewton, maxIts;
        real_t Dim, Delta, slowErr;
        real_t EpsNewton, TolNewton;
        real_t XLeft, XMid, XRight;
        size_t NLeft, NRight, iL, iM, iR;
        bool Debug, Verbose, Converged;
    
        InitialConditionGenerator initGen;
        vec_complex YLeft, YRight, mismatchOut;
        vec_real fc, psic, Up, XGrid, in0, out0;
        json resultDict;
        std::filesystem::path baseFolder;
    
        std::unique_ptr<ShootingSolver> shooter;
    
        void initializeInput(bool printDiagnostics=false);
        void generateGrid();
        void shoot(vec_real& inputVec, vec_real& outputVec, json* fieldVals=nullptr);
        
        #ifdef USE_OPENMP
        void initializeInput(InitialConditionGenerator& initGen_local, vec_complex& YLeft_local, vec_complex& YRight_local,
                             vec_real& Up_local, vec_real& psic_local, vec_real& fc_local, real_t Delta_local, bool printDiagnostics=false);
        void shoot(vec_real& inputVec, vec_real& outputVec, ShootingSolver& shooter_local,
                   InitialConditionGenerator& initGen_local, vec_complex& YLeft_local, vec_complex& YRight_local,
                   vec_complex& mismatchOut_local);
        #endif

        #if defined(USE_MPI) || defined(USE_HYBRID)
        int size, rank;
        MPI_Datatype mpi_contiguous_t;
        void assembleJacobian(const vec_real& baseInput, const vec_real& baseOutput,
                              mat_real& jacobian);
        void solveLinearSystem(mat_real& A, vec_real& rhs, vec_real& dx);
        #else
        void assembleJacobian(const vec_real& baseInput, const vec_real& baseOutput,
                              mat_real& jacobian);
        void solveLinearSystem(const mat_real& A, vec_real& rhs, vec_real& dx);
        #endif
        
        real_t computeL2Norm(const vec_real& vc);

    public:
        NewtonSolver(SimulationConfig configIn, std::filesystem::path dataPathIn);

        #if defined(USE_MPI) || defined(USE_HYBRID)
        ~NewtonSolver();
        #endif

        json run();
        void writeFinalOutput();

};